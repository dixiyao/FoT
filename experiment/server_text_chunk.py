"""
Chunked Text-Based Insight Aggregation Server

Extends TextBasedInsightAggregationServer with automatic chunking when the
collected reasoning traces exceed the model's context window.

Pipeline (when chunking is needed):
1. Collect Insight Books (same as server_text.py)
2. Dynamic rolling merge with adaptive chunk sizes:
   - Chunk 1: up to 0.75 * token_limit of traces → Step 2 + Step 3 → insights_1
   - Chunk 2: traces that fit in (0.75 * token_limit - tokens(insights_1))
              → combined with insights_1 → Step 2 + Step 3 → insights_2
   - Chunk 3: traces that fit in (0.75 * token_limit - tokens(insights_2))
              → combined with insights_2 → Step 2 + Step 3 → insights_3
   - ... until all traces are consumed

Chunk sizes adapt dynamically: after each round, the accumulated insights
consume part of the budget, so subsequent chunks hold fewer raw traces.

When the traces fit within budget, falls back to non-chunked behavior.

Usage:
  python server_text_chunk.py -i math_output -o encyclopedia_output -d cuda \\
      -m deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --token-limit 16384
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional

from server_text import TextBasedInsightAggregationServer


class ChunkedTextInsightAggregationServer(TextBasedInsightAggregationServer):
    """Text-based insight aggregation with dynamic chunking and rolling merge.

    When the total reasoning traces exceed the model's context budget, traces
    are consumed in dynamically-sized chunks via a rolling merge. Each round
    the budget for new traces is: ``0.75 * token_limit - tokens(accumulated_insights)``.
    This way chunk sizes adapt automatically — as accumulated insights grow,
    each subsequent chunk holds fewer raw traces, but the combined input
    always stays within budget.

    Args:
        token_limit: Total token budget for the insight-store portion of
            the prompt. The usable budget is ``0.75 * token_limit``.
            Default 16384.
        All other arguments are forwarded to the parent class.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device: Optional[str] = None,
        input_dirs: Optional[List[str]] = None,
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
        num_insights: Optional[int] = None,
        max_files: Optional[int] = None,
        custom_prompt_section: str = "",
        token_limit: int = 16384,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            input_dirs=input_dirs,
            use_api=use_api,
            api_key=api_key,
            api_provider=api_provider,
            num_insights=num_insights,
            max_files=max_files,
            custom_prompt_section=custom_prompt_section,
        )
        self.token_limit = token_limit
        self.budget = int(token_limit * 0.5)

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in *text*.

        Uses the HuggingFace tokenizer when available; falls back to a rough
        word-count estimate for Gemini.
        """
        if self.use_api:
            return len(re.findall(r"\S+", text))

        self._load_model()
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return len(token_ids)

    # ------------------------------------------------------------------
    # Dynamic chunk consumption
    # ------------------------------------------------------------------
    def _take_next_chunk(
        self,
        remaining: List[tuple],
        max_tokens: int,
    ) -> tuple:
        """Consume traces from *remaining* up to *max_tokens*.

        Args:
            remaining: list of (name, desc) pairs not yet processed.
            max_tokens: token budget available for this chunk's traces.

        Returns:
            (chunk_dict, num_consumed) — the traces that fit plus how many
            items were popped from the front of *remaining*.
        """
        chunk: Dict[str, str] = {}
        used_tokens = 0
        consumed = 0

        while remaining:
            name, desc = remaining[0]
            entry_text = f"- {name}: {desc}\n"
            entry_tokens = self._count_tokens(entry_text)

            # A single oversized trace: include it alone so we never drop.
            if entry_tokens > max_tokens and not chunk:
                chunk[name] = desc
                remaining.pop(0)
                consumed += 1
                break

            if used_tokens + entry_tokens > max_tokens:
                break

            chunk[name] = desc
            used_tokens += entry_tokens
            remaining.pop(0)
            consumed += 1

        return chunk, consumed

    # ------------------------------------------------------------------
    # Main orchestration (overrides parent)
    # ------------------------------------------------------------------
    def aggregate_and_build_encyclopedia(
        self,
        json_files: Optional[List[str]] = None,
        output_dir: str = "math_output",
    ) -> Dict:
        """Chunk-aware aggregation pipeline.

        Overrides the parent method to add automatic chunking when the
        insight store exceeds the model's token budget.

        Args:
            json_files: JSON files to collect insights from.
            output_dir: Directory for loading existing encyclopedia and
                saving results.

        Returns:
            Result dictionary with the same structure as the parent class.
        """
        # ==============================================================
        # Step 1: Collect Insight Books (same as parent)
        # ==============================================================
        print("\n" + "=" * 80)
        print("STEP 1: Collecting Insight Books")
        print("=" * 80)
        collection_result = self.collect_insight_books(json_files)
        time.sleep(1)

        if not self.insight_store:
            files_processed = collection_result.get("files_processed", 0)
            print(f"Warning: No insights found in {files_processed} collected files!")
            return {
                "error": "No insights found",
                "files_processed": files_processed,
                "collection_result": collection_result,
                "aggregation_steps": self.aggregation_steps,
            }

        # ==============================================================
        # Check total token count of insight store
        # ==============================================================
        insight_text = "\n".join(
            f"- {name}: {desc}" for name, desc in self.insight_store.items()
        )
        total_tokens = self._count_tokens(insight_text)
        print(f"\nInsight store: {len(self.insight_store)} traces, "
              f"{total_tokens} tokens (budget: {self.budget} = 0.75 * {self.token_limit})")

        # ==============================================================
        # If within budget → delegate to parent's normal pipeline
        # ==============================================================
        if total_tokens <= self.budget:
            print("Insight store fits within budget — using standard pipeline.")
            return self._run_single_pass(output_dir)

        # ==============================================================
        # Dynamic rolling merge with adaptive chunk sizes
        #
        # Round 1: chunk_1 budget = full budget (no accumulated insights yet)
        #   chunk_1 traces → Step 2+3 → insights_1
        # Round 2: chunk_2 budget = budget - tokens(insights_1)
        #   insights_1 + chunk_2 traces → Step 2+3 → insights_2
        # Round 3: chunk_3 budget = budget - tokens(insights_2)
        #   insights_2 + chunk_3 traces → Step 2+3 → insights_3
        # ... until all traces are consumed
        # ==============================================================
        remaining = list(self.insight_store.items())  # [(name, desc), ...]
        total_traces = len(remaining)

        accumulated_insights: Dict[str, str] = {}
        accumulated_tokens = 0
        chunk_results = []
        existing_encyclopedia = self._load_existing_encyclopedia(output_dir)
        chunk_idx = 0

        print(f"\nStarting dynamic rolling merge ({total_traces} traces to consume)")

        while remaining:
            # Budget for new traces = total budget - space used by accumulated insights
            trace_budget = self.budget - accumulated_tokens
            if trace_budget < 1:
                # Accumulated insights alone fill the budget; force a minimal
                # chunk of 1 trace so we always make progress.
                trace_budget = self._count_tokens(
                    f"- {remaining[0][0]}: {remaining[0][1]}\n"
                )

            chunk, num_consumed = self._take_next_chunk(remaining, trace_budget)
            if not chunk:
                # Safety: should not happen, but avoid infinite loop
                break

            chunk_idx += 1
            num_prev = len(accumulated_insights)
            chunk_text = "\n".join(f"- {n}: {d}" for n, d in chunk.items())
            chunk_tok = self._count_tokens(chunk_text)

            # Combine accumulated insights + new traces
            combined_store: Dict[str, str] = {}
            combined_store.update(accumulated_insights)
            combined_store.update(chunk)

            consumed_so_far = total_traces - len(remaining)
            print(f"\n{'=' * 80}")
            print(f"ROUND {chunk_idx} "
                  f"({consumed_so_far}/{total_traces} traces consumed): "
                  f"{len(chunk)} new traces ({chunk_tok} tok) + "
                  f"{num_prev} accumulated insights ({accumulated_tokens} tok) "
                  f"= {len(combined_store)} entries")
            print(f"{'=' * 80}")

            # Step 2: Profiling on combined store
            print(f"\n--- Round {chunk_idx} Step 2: Text-Based Profiling ---")
            profiling_result = self._step_text_profiling(combined_store)
            time.sleep(1)

            # Step 3: Knowledge extraction on combined store
            # Pass existing encyclopedia only on the first round.
            enc_for_step3 = existing_encyclopedia if chunk_idx == 1 else ""
            print(f"\n--- Round {chunk_idx} Step 3: Knowledge Extraction ---")
            extraction_result = self._step_knowledge_extraction(
                combined_store, self.insight_relationships, enc_for_step3
            )

            # Parse the produced encyclopedia
            chunk_encyclopedia = extraction_result.get("encyclopedia_dict")
            if chunk_encyclopedia is None:
                json_content = self._extract_json_only(
                    extraction_result.get("encyclopedia", "")
                )
                chunk_encyclopedia = self._try_parse_json(json_content)

            if chunk_encyclopedia and isinstance(chunk_encyclopedia, dict):
                accumulated_insights = chunk_encyclopedia
                # Measure token cost of the new accumulated insights
                acc_text = "\n".join(
                    f"- {n}: {d}" for n, d in accumulated_insights.items()
                )
                accumulated_tokens = self._count_tokens(acc_text)
                print(f"  Round {chunk_idx} produced {len(accumulated_insights)} insights "
                      f"({accumulated_tokens} tokens)")
            else:
                print(f"  Warning: Round {chunk_idx} produced no parseable insights; "
                      f"keeping previous {num_prev} accumulated insights")

            chunk_results.append({
                "round": chunk_idx,
                "num_new_traces": len(chunk),
                "new_traces_tokens": chunk_tok,
                "num_accumulated_insights_in": num_prev,
                "accumulated_insights_tokens_in": accumulated_tokens - chunk_tok if chunk_idx > 1 else 0,
                "num_combined_entries": len(combined_store),
                "trace_budget": trace_budget,
                "profiling": profiling_result,
                "extraction": extraction_result,
                "num_insights_produced": (
                    len(chunk_encyclopedia) if chunk_encyclopedia else 0
                ),
                "accumulated_insights_tokens_out": accumulated_tokens,
            })
            time.sleep(1)

        print(f"\n{'=' * 80}")
        print(f"ROLLING MERGE COMPLETE: {len(accumulated_insights)} final insights "
              f"({accumulated_tokens} tokens) after {chunk_idx} rounds")
        print(f"{'=' * 80}")

        if not accumulated_insights:
            print("ERROR: No insights produced from any round!")
            return {
                "error": "No insights produced from chunks",
                "collection_result": collection_result,
                "chunk_results": chunk_results,
                "aggregation_steps": self.aggregation_steps,
            }

        # The last round's extraction already wrote self.encyclopedia;
        # just build the result dict.
        total_output_tokens = sum(
            r["profiling"].get("output_tokens", 0) + r["extraction"].get("output_tokens", 0)
            for r in chunk_results
        )
        result = {
            "collection": collection_result,
            "profiling": chunk_results[-1]["profiling"],
            "extraction": chunk_results[-1]["extraction"],
            "aggregation_steps": self.aggregation_steps,
            "encyclopedia": self.encyclopedia,
            "insight_store": self.insight_store,
            "insight_relationships": self.insight_relationships,
            "insight_datastore_tokens": total_tokens,
            "total_output_tokens": total_output_tokens,
            "chunking": {
                "num_rounds": chunk_idx,
                "budget": self.budget,
                "token_limit": self.token_limit,
                "total_tokens": total_tokens,
                "final_insights_count": len(accumulated_insights),
                "final_insights_tokens": accumulated_tokens,
                "chunk_results": chunk_results,
            },
        }

        return result

    # ------------------------------------------------------------------
    # Single-pass (non-chunked) helper
    # ------------------------------------------------------------------
    def _run_single_pass(self, output_dir: str) -> Dict:
        """Run the standard Step 2 + Step 3 without chunking."""
        # Step 2: Profiling
        print("\n" + "=" * 80)
        print("STEP 2: Text-Based Profiling of Insight Relationships")
        print("=" * 80)
        profiling_result = self._step_text_profiling(self.insight_store)
        time.sleep(1)

        # Step 3: Knowledge Extraction
        print("\n" + "=" * 80)
        print("STEP 3: Extracting General, Fundamental Knowledge")
        print("=" * 80)
        existing_encyclopedia = self._load_existing_encyclopedia(output_dir)
        extraction_result = self._step_knowledge_extraction(
            self.insight_store,
            self.insight_relationships,
            existing_encyclopedia,
        )

        total_output_tokens = (
            profiling_result.get("output_tokens", 0)
            + extraction_result.get("output_tokens", 0)
        )
        return {
            "collection": self.aggregation_steps[0] if self.aggregation_steps else {},
            "profiling": profiling_result,
            "extraction": extraction_result,
            "aggregation_steps": self.aggregation_steps,
            "encyclopedia": self.encyclopedia,
            "insight_store": self.insight_store,
            "insight_relationships": self.insight_relationships,
            "total_output_tokens": total_output_tokens,
            "chunking": None,
        }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Chunked Text-Based Insight Aggregation Server — "
            "automatically splits traces into chunks when they exceed "
            "the model's context window."
        )
    )
    parser.add_argument(
        "-i", "--input-dir", type=str, nargs="+", default=["math_output"],
        help="Input directory/directories containing insight JSON files.",
    )
    parser.add_argument(
        "-m", "--model", type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="HuggingFace model name.",
    )
    parser.add_argument(
        "-d", "--device", type=str, default=None,
        help="Device to use: 'cuda' or 'cpu' (default: auto-detect).",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="math_output",
        help="Output directory for encyclopedia.",
    )
    parser.add_argument(
        "--use-api", action="store_true",
        help="Use an API provider instead of HuggingFace model.",
    )
    parser.add_argument(
        "--api-provider", type=str, default="gemini",
        choices=["gemini", "openrouter"],
        help="Which API provider to use (default: gemini).",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key for the chosen provider.",
    )
    parser.add_argument(
        "--num-insights", type=int, default=None,
        help="Number of insights to extract.",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Maximum number of JSON files to use.",
    )
    parser.add_argument(
        "--token-limit", type=int, default=16384,
        help=(
            "Total token budget for the insight-store portion of prompts. "
            "Usable budget is 0.75 * token_limit; each round's new traces "
            "fill the remaining space after accumulated insights. Default: 16384."
        ),
    )

    args = parser.parse_args()

    server = ChunkedTextInsightAggregationServer(
        model_name=args.model,
        device=args.device,
        input_dirs=args.input_dir,
        use_api=args.use_api,
        api_key=args.api_key,
        api_provider=args.api_provider,
        num_insights=args.num_insights,
        max_files=args.max_files,
        token_limit=args.token_limit,
    )

    try:
        result = server.aggregate_and_build_encyclopedia(output_dir=args.output_dir)
        server.save_results(result, output_dir=args.output_dir)

        print("\n" + "=" * 80)
        print("AGGREGATION COMPLETE")
        print("=" * 80)
        print(f"Total insights collected: {len(server.insight_store)}")
        print(f"Encyclopedia length: {len(server.encyclopedia)} characters")
        print(f"Insight library output tokens: {result.get('total_output_tokens', 0)}")
        if result.get("chunking"):
            info = result["chunking"]
            print(f"Rolling merge: {info['num_rounds']} rounds "
                  f"(budget: {info['budget']} = 0.75 * {info['token_limit']})")
            print(f"Final insights: {info['final_insights_count']} "
                  f"({info['final_insights_tokens']} tokens)")
        else:
            print("Chunking: not needed (traces fit in single pass)")
        print(f"Output directory: {args.output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nExamples:")
        print("  python server_text_chunk.py -i math_output -o enc_output -d cuda "
              "-m deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        print("  python server_text_chunk.py -i math_output -o enc_output "
              "--use-api --api-provider gemini --token-limit 32768")
