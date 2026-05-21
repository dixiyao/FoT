"""Multimodal extension of task_benchmark_domain.

This keeps the same iterative pipeline/argparse behavior as task_benchmark_domain.py,
with one key difference: supports image input for multimodal datasets (currently HLE).
"""

import argparse
import json
import os
import random
import re
import time
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from hle_datasets.hle import (
    HLE_DATASET_ID,
    _call_gemini_multimodal,
    extract_hle_final_answer,
    extract_hle_image,
    format_hle_question,
    get_hle_answer,
    is_hle_answer_correct,
    load_hle_dataset,
)
from task_benchmark_domain import BenchmarkDomainPipeline, _parse_list_arg


class _LazyDatasetView:
    """Memory-safe wrapper for Hugging Face Dataset.

    - Avoids materializing the full dataset into a Python list.
    - Supports int indexing, slicing, iteration, and len() like a list.
    """

    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [dict(self._dataset[i]) for i in range(start, stop, step)]
        return dict(self._dataset[index])

    def __iter__(self) -> Iterator[Dict]:
        for i in range(len(self)):
            yield dict(self._dataset[i])


class MultimodalBenchmarkDomainPipeline(BenchmarkDomainPipeline):
    """Same pipeline as BenchmarkDomainPipeline, with HLE multimodal solve support."""

    @staticmethod
    def _is_hle(dataset_name: str) -> bool:
        return dataset_name in {"hle", HLE_DATASET_ID}

    @staticmethod
    def _extract_insight_book_from_response(response_text: str) -> Dict[str, str]:
        if not response_text:
            return {}

        json_str = None
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if code_block:
            json_str = code_block.group(1)
        else:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}")
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx + 1]

        if not json_str:
            return {}

        try:
            json_str = re.sub(r",\s*}", "}", json_str)
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return {}

        if not isinstance(parsed, dict):
            return {}

        insight_book: Dict[str, str] = {}
        for name, description in parsed.items():
            key = str(name)
            if not key.startswith("insight_"):
                key = f"insight_{key}"
            value = str(description).strip()
            if len(value) >= 20:
                insight_book[key] = value
        return insight_book

    @staticmethod
    def _build_solution_prompt(
        problem: str,
        insights_section: Optional[str] = None,
    ) -> str:
        insights_text = insights_section or ""
        return f"{insights_text}Problem: {problem}"

    @staticmethod
    def _build_reflection_prompt(problem: str, solution: str) -> str:
        return f"""
Analyze the solution below to extract procedural knowledge that reflect the reasoning traces.

Problem:
{problem}

Step-by-Step Solution:
{solution}

Your task: Extract the fundamental techniques used in reslution that can be packaged as reasoning traces.
"""

    @staticmethod
    def _build_behavior_prompt(problem: str, solution: str, reflection: str) -> str:
        return f"""
Extract reasoning traces from the solution below and output as JSON.

Problem: {problem}

Solution: {solution}

Reflection: {reflection}

Output a JSON object where each key starts with \"insight_\" and each value is a string description.
"""

    def load_math_dataset(self, dataset_name: str) -> List[Dict]:
        if self._is_hle(dataset_name):
            ds = load_hle_dataset(split="test")
            return _LazyDatasetView(ds)
        return super().load_math_dataset(dataset_name)

    def _format_problem(self, problem_data: Dict, dataset_name: str):
        if self._is_hle(dataset_name):
            return format_hle_question(problem_data), None
        return super()._format_problem(problem_data, dataset_name)

    def _get_ground_truth(self, problem_data: Dict, dataset_name: str) -> str:
        if self._is_hle(dataset_name):
            return get_hle_answer(problem_data)
        return super()._get_ground_truth(problem_data, dataset_name)

    def _extract_answer_from_solution(
        self, solution: str, dataset_name: str, problem_data: Dict
    ) -> Optional[str]:
        if self._is_hle(dataset_name):
            return extract_hle_final_answer(solution)
        return super()._extract_answer_from_solution(solution, dataset_name, problem_data)

    def _check_answer_match(
        self,
        predicted: str,
        ground_truth: str,
        dataset_name: Optional[str] = None,
        problem_text: Optional[str] = None,
    ) -> bool:
        if dataset_name and self._is_hle(dataset_name):
            return is_hle_answer_correct(predicted, ground_truth)
        return super()._check_answer_match(predicted, ground_truth, dataset_name, problem_text)

    def _extract_insights_for_dataset(
        self,
        dataset_name: str,
        problems: List[Dict],
        max_problems: Optional[int],
        encyclopedia_paths: Optional[List[str]] = None,
        iteration: int = 0,
    ) -> Tuple[str, List[Dict]]:
        if not self._is_hle(dataset_name):
            return super()._extract_insights_for_dataset(
                dataset_name, problems, max_problems, encyclopedia_paths, iteration
            )

        if not self.use_api:
            raise ValueError(
                "HLE multimodal solving requires Gemini API. Use --use-api for HLE datasets."
            )

        self._ensure_client()
        insights_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(insights_dir, exist_ok=True)

        worklist = problems[:max_problems] if max_problems else problems
        print(
            f"\nIteration {iteration}: Extracting insights for {dataset_name} ({len(worklist)} problems)..."
        )

        insights_section = ""
        if encyclopedia_paths:
            valid_eps = [ep for ep in encyclopedia_paths if ep and os.path.exists(ep)]
            if valid_eps:
                print(f"  Loading {len(valid_eps)} encyclopedias for guidance...")
                self.client.load_encyclopedias(valid_eps, mode=self.mode)
                if self.client.encyclopedia_loaded:
                    if self.client.encyclopedia_dict:
                        insights_list = []
                        for insight_name, insight_desc in self.client.encyclopedia_dict.items():
                            insights_list.append(f"**{insight_name}**:\n{insight_desc}")
                        insights_text = "\n\n".join(insights_list)
                    else:
                        insights_text = self.client.encyclopedia

                    insights_section = f"""Available Insights to Guide Your Solution:

{insights_text}

---
INSTRUCTIONS: Review the insights above and actively apply the relevant techniques from insights to solve this problem. Consider which insights can help you approach the problem more effectively.

"""
            else:
                print("  No valid encyclopedias found; proceeding without guidance")

        results = []
        number_output_tokens_list = []
        total_pipeline_output_tokens_list = []
        total_pipeline_thinking_tokens_list = []
        loop_count_list = []

        for idx, problem_data in enumerate(worklist, 1):
            problem_text = format_hle_question(problem_data)
            image, image_source = extract_hle_image(problem_data)

            if not problem_text:
                print(f"  [skip] Problem {idx} missing text")
                continue

            print(f"  [{idx}/{len(worklist)}] {problem_text[:80]}...")
            if image is None:
                print(f"    No image found (source={image_source}); using text-only Gemini call")

            try:
                solution_prompt = self._build_solution_prompt(
                    problem_text,
                    insights_section=insights_section,
                )
                solution, solution_token_info = _call_gemini_multimodal(
                    gemini_model=self.client.gemini_model,
                    question=solution_prompt,
                    image=image,
                    max_output_tokens=8192,
                    temperature=0.0,
                    return_token_info=True,
                )

                reflection_prompt = self._build_reflection_prompt(
                    problem_text,
                    solution,
                )
                reflection, reflection_token_info = _call_gemini_multimodal(
                    gemini_model=self.client.gemini_model,
                    question=reflection_prompt,
                    image=image,
                    max_output_tokens=4096,
                    temperature=0.0,
                    return_token_info=True,
                )

                behavior_prompt = self._build_behavior_prompt(
                    problem_text,
                    solution,
                    reflection,
                )
                behavior_response, behavior_token_info = _call_gemini_multimodal(
                    gemini_model=self.client.gemini_model,
                    question=behavior_prompt,
                    image=image,
                    max_output_tokens=8192,
                    temperature=0.0,
                    return_token_info=True,
                )

                insight_book = self._extract_insight_book_from_response(behavior_response)
                insight_book = {
                    k: v for k, v in insight_book.items() if not k.startswith("insight_fallback")
                }
                if not insight_book:
                    print("    No insights extracted")
                    continue

                number_output_tokens = int(
                    solution_token_info.get("output_tokens") or max(len(solution) // 4, 0)
                )
                total_pipeline_output_tokens = (
                    int(solution_token_info.get("output_tokens", 0) or 0)
                    + int(reflection_token_info.get("output_tokens", 0) or 0)
                    + int(behavior_token_info.get("output_tokens", 0) or 0)
                )
                total_pipeline_thinking_tokens = (
                    int(solution_token_info.get("thinking_tokens", 0) or 0)
                    + int(reflection_token_info.get("thinking_tokens", 0) or 0)
                    + int(behavior_token_info.get("thinking_tokens", 0) or 0)
                )
                number_output_tokens_list.append(number_output_tokens)
                total_pipeline_output_tokens_list.append(total_pipeline_output_tokens)
                total_pipeline_thinking_tokens_list.append(total_pipeline_thinking_tokens)
                loop_count = self._count_consecutive_sentence_loops(solution)
                loop_count_list.append(loop_count)

                predicted_answer = extract_hle_final_answer(solution)
                ground_truth = get_hle_answer(problem_data)
                is_correct = is_hle_answer_correct(predicted_answer, ground_truth)

                status = "✓" if is_correct else "✗"
                print(
                    f"    {status} Predicted: {predicted_answer if predicted_answer else 'N/A'} | GT: {ground_truth if ground_truth else 'N/A'}"
                )

                output_data = {
                    "problem": problem_text,
                    "problem_id": problem_data.get("id", idx),
                    "insight_book": insight_book,
                    "iteration": iteration,
                    "is_correct": is_correct,
                    "number_output_tokens": number_output_tokens,
                    "solution_token_info": solution_token_info,
                    "reflection_token_info": reflection_token_info,
                    "behavior_token_info": behavior_token_info,
                    "total_pipeline_output_tokens": total_pipeline_output_tokens,
                    "total_pipeline_thinking_tokens": total_pipeline_thinking_tokens,
                    "loop_count": loop_count,
                }

                output_path = os.path.join(insights_dir, f"problem_{idx:04d}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

                results.append(
                    {
                        "is_correct": is_correct,
                        "number_output_tokens": number_output_tokens,
                        "total_pipeline_output_tokens": total_pipeline_output_tokens,
                        "total_pipeline_thinking_tokens": total_pipeline_thinking_tokens,
                        "loop_count": loop_count,
                    }
                )
                time.sleep(0.5)
            except (RuntimeError, ValueError, TypeError, OSError, KeyError) as exc:
                print(f"    Error processing problem {idx}: {exc}")

        if number_output_tokens_list:
            avg_number_output_tokens = sum(number_output_tokens_list) / len(
                number_output_tokens_list
            )
            print(
                f"\n  Dataset '{dataset_name}' - Average Solution Output Tokens: {avg_number_output_tokens:.1f}"
            )
        if total_pipeline_output_tokens_list:
            avg_total_pipeline_output_tokens = sum(total_pipeline_output_tokens_list) / len(
                total_pipeline_output_tokens_list
            )
            print(
                f"  Dataset '{dataset_name}' - Average Pipeline Output Tokens: {avg_total_pipeline_output_tokens:.1f}"
            )
        if total_pipeline_thinking_tokens_list:
            avg_total_pipeline_thinking_tokens = sum(total_pipeline_thinking_tokens_list) / len(
                total_pipeline_thinking_tokens_list
            )
            print(
                f"  Dataset '{dataset_name}' - Average Pipeline Thinking Tokens: {avg_total_pipeline_thinking_tokens:.1f}"
            )

        if loop_count_list:
            total_loop_count = sum(loop_count_list)
            print(f"  Dataset '{dataset_name}' - Total Loop Count: {total_loop_count}")

        return insights_dir, results


def main():
    parser = argparse.ArgumentParser(
        description="Iterative benchmark learning pipeline: solve (if encyclopedia available) + extract insights + aggregate → repeat"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hle"],
        help="Datasets for iterative learning (space- or comma-separated).",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Limit problems per dataset per iteration.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gemini-3-pro-preview",
        help="Model name (HF or API model name when --use-api).",
    )
    parser.add_argument(
        "-d", "--device", type=str, default=None, help="Device to use (cuda or cpu)."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="multimodal_output",
        help="Root output directory.",
    )
    parser.add_argument(
        "--r1", type=float, default=0.95, help="r1 threshold for same insights."
    )
    parser.add_argument(
        "--r2", type=float, default=0.6, help="r2 threshold for linked insights."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--use-api", action="store_true", help="Use an API provider instead of HuggingFace model."
    )
    parser.add_argument(
        "--api-provider", type=str, default="gemini", choices=["gemini", "openrouter"],
        help="Which API provider to use (default: gemini).",
    )
    parser.add_argument(
        "--api-key", type=str, default=None, help="API key for the chosen provider.",
    )
    parser.add_argument(
        "--load-in-8bit",
        type=bool,
        default=False,
        help="Load model with 8-bit quantization (default: False, uses FP16 instead)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="text",
        choices=["normal", "text"],
        help="Aggregation/inference mode (normal=GraphRAG, text=text-based). Default: text",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=3,
        help="Number of iterations (default: 3)",
    )
    parser.add_argument(
        "--start-from-step",
        type=int,
        default=1,
        choices=[1, 2],
        help="Start from step: 1=extract insights (default), 2=aggregate only (assumes insights already exist)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Eval-only mode: solve problems and check accuracy without extracting traces or aggregating.",
    )
    parser.add_argument(
        "--encyclopedia",
        type=str,
        nargs="*",
        default=None,
        help="Path(s) to encyclopedia file(s) to use for eval-only or iterative mode.",
    )

    args = parser.parse_args()

    datasets = _parse_list_arg(args.datasets)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {args.seed}")

    pipeline = MultimodalBenchmarkDomainPipeline(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
        use_api=args.use_api,
        api_key=args.api_key,
        api_provider=args.api_provider,
        mode=args.mode,
        num_iterations=args.num_iterations,
        load_in_8bit=args.load_in_8bit,
    )

    try:
        if not datasets:
            raise ValueError("--datasets is required")

        if args.eval_only:
            pipeline.run_eval_only(
                dataset_list=datasets,
                max_problems=args.max_problems,
                encyclopedia_paths=args.encyclopedia,
            )
        else:
            pipeline.run_iterative_pipeline(
                dataset_list=datasets,
                max_problems=args.max_problems,
                r1=args.r1,
                r2=args.r2,
                start_from_step=args.start_from_step,
            )
    except (RuntimeError, ValueError, FileNotFoundError, ImportError, OSError) as exc:
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
