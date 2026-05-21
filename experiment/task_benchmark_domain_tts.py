"""
True Thinking Score (TTS) Evaluation Pipeline (task_benchmark_domain_tts)

Measures the True Thinking Score for each reasoning step in Chain-of-Thought,
following the methodology from:
  "Can Aha Moments Be Fake? Identifying True and Decorative Thinking Steps
   in Chain-of-Thought" (arxiv 2510.24941)

TTS quantifies the causal contribution of each CoT step via perturbation-based
interventions. For each step s_i with context C = (s_1, ..., s_{i-1}):

  TTS(s_i) = 0.5 * ( |S_1(1) - S_0(1)| + |S_1(0) - S_0(0)| )

where:
  S_X(C) = P(y* | context=C, step=X)   (model confidence via logprobs)
  X=1 -> intact step, X=0 -> perturbed step
  C=1 -> intact context, C=0 -> perturbed context
  y* = reference answer from the original full CoT

P(y*) is measured using the model's logprobs (geometric mean of per-token
probabilities for the reference answer tokens). This requires a HuggingFace
model with accessible logits (e.g. DeepSeek-R1-Distill-Qwen-7B).

Perturbation: small random numerical offsets to quantities in reasoning text.
Early-exit probing cue: "The final result is" (from the paper).
TTS thresholds (from paper): alpha=0.9 (true-thinking), beta=0 (decorative).

Output per-problem: tts_per_step array, tts_mean, correctness, etc.
Output per-dataset in tts_eval_summary.json: tts_per_dataset array with one
  tts_mean per problem (every problem gets a score).

Usage:
    python task_benchmark_domain_tts.py --datasets aime25 --max-problems 5 \
            --eval-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
  python task_benchmark_domain_tts.py --datasets math500 --max-problems 10 \\
            --eval-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
            --encyclopedia path/to/encyclopedia.json
"""

import argparse
import json
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from task_benchmark_domain import BenchmarkDomainPipeline, _parse_list_arg

# ---------------------------------------------------------------------------
# Constants from the paper
# ---------------------------------------------------------------------------
# TTS classification thresholds (from paper Section 4)
TTS_ALPHA = 0.9   # >= alpha => true-thinking step
TTS_BETA = 0.0    # <= beta  => decorative step

# Default probe cue (from paper); overridden per dataset below
_DEFAULT_ANSWER_BRIDGE = "\nThe final result is "


class TTSEvalPipeline(BenchmarkDomainPipeline):
    """Extends BenchmarkDomainPipeline with True Thinking Score computation.

    Uses HuggingFace model logprobs to measure P(y*) for each perturbation
    condition, following the methodology of arxiv 2510.24941.

    Inherits dataset loading, problem formatting, answer extraction/checking
    from the parent class.
    """

    def __init__(self, **kwargs):
        # Force HF mode: TTS requires logprob access, not available via Gemini API
        kwargs["use_api"] = False
        kwargs["api_key"] = None
        super().__init__(**kwargs)
        self._tts_rng = random.Random(42)

    # ------------------------------------------------------------------
    # HF model access for logprob computation
    # ------------------------------------------------------------------
    def _ensure_hf_model(self):
        """Ensure HuggingFace model and tokenizer are loaded."""
        self._ensure_client()
        self.client._load_model()

    # ------------------------------------------------------------------
    # Dataset-specific probe format for TTS
    # ------------------------------------------------------------------
    @staticmethod
    def _get_tts_probe_format(
        dataset_name: str, reference_answer: str
    ) -> Tuple[str, str]:
        """Return (answer_bridge, formatted_reference) for TTS logprob probing.

        Different datasets have different natural answer formats.  The bridge
        text and reference string must match so that the model's logprobs
        reflect genuine answer confidence.

        Math competitions  → \\boxed{answer}
        GPQA (multi-choice)→ (A)/(B)/(C)/(D)
        GSM8K              → #### answer
        Default            → "The final result is answer" (paper cue)
        """
        # Math competition datasets: \boxed{} format
        if dataset_name in (
            "aime25", "aime24", "math500", "math1000",
        ) or dataset_name.startswith("imo") or (
            dataset_name and "livemathbench" in dataset_name
        ):
            return "\nThe final answer is $\\boxed{", reference_answer + "}$"

        # GPQA: multiple choice (A)/(B)/(C)/(D)
        if dataset_name and dataset_name.startswith("gpqa"):
            return "\nThe answer is (", reference_answer + ")"

        # GSM8K: #### format
        if dataset_name in ("gsm8k", "gsm8k_train"):
            return "\n#### ", reference_answer

        # Default fallback: paper's original probe cue
        return _DEFAULT_ANSWER_BRIDGE, reference_answer

    # ------------------------------------------------------------------
    # CoT segmentation
    # ------------------------------------------------------------------
    @staticmethod
    def _segment_cot(cot: str) -> List[str]:
        """Segment a Chain-of-Thought into individual reasoning steps.

        Strategy: split by newline paragraphs first; if too few, fall back
        to sentence-level splitting.
        """
        # Try paragraph-level splitting (common in math reasoning)
        paragraphs = [p.strip() for p in cot.split("\n") if p.strip()]
        paragraphs = [p for p in paragraphs if len(p) > 15]

        if len(paragraphs) >= 3:
            return paragraphs

        # Fall back to sentence-level splitting
        sentences = re.split(r"(?<=[.!?])\s+", cot)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]

        if sentences:
            return sentences

        # Last resort: return the whole CoT as a single step
        return [cot.strip()] if cot.strip() else []

    # ------------------------------------------------------------------
    # Numerical perturbation
    # ------------------------------------------------------------------
    _NUMBER_RE = re.compile(r"(?<![a-zA-Z_])-?\d+(?:\.\d+)?(?![a-zA-Z_])")

    def _has_numbers(self, text: str) -> bool:
        return bool(self._NUMBER_RE.search(text))

    def _perturb_text(self, text: str) -> str:
        """Perturb numbers in *text* by adding small random offsets.

        Follows the paper: "introducing small random numerical offsets to
        quantities appearing in the reasoning text" while maintaining
        grammatical and semantic structure.
        """
        rng = self._tts_rng

        def _replace(match: re.Match) -> str:
            original = match.group(0)
            try:
                num = float(original)
            except ValueError:
                return original

            if num == 0:
                return str(rng.choice([-2, -1, 1, 2]))

            is_int = "." not in original and "e" not in original.lower()
            offset = num * rng.uniform(0.05, 0.15) * rng.choice([-1, 1])

            if is_int:
                perturbed = int(round(num + offset))
                if perturbed == int(num):
                    perturbed += rng.choice([-1, 1])
                return str(perturbed)
            else:
                perturbed = num + offset
                dec_places = len(original.split(".")[-1]) if "." in original else 2
                return f"{perturbed:.{dec_places}f}"

        return self._NUMBER_RE.sub(_replace, text)

    # ------------------------------------------------------------------
    # Logprob-based answer confidence (core TTS measurement)
    # ------------------------------------------------------------------
    def _compute_answer_confidence(
        self,
        cot_prefix: str,
        reference_answer: str,
        answer_bridge: str = _DEFAULT_ANSWER_BRIDGE,
    ) -> float:
        """Compute P(reference_answer | cot_prefix) using HF model logprobs.

        Constructs: cot_prefix + answer_bridge + reference_answer
        Returns: geometric mean of per-token probabilities (float in [0, 1]).
        """
        model = self.client.model
        tokenizer = self.client.tokenizer
        device = self.client.device

        prefix_text = cot_prefix + answer_bridge
        full_text = prefix_text + reference_answer

        # Tokenize: use the model's max context length
        max_length = getattr(tokenizer, "model_max_length", 32768)
        if max_length > 131072:
            max_length = 32768

        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=True)
        full_ids = tokenizer.encode(full_text, add_special_tokens=True)

        prefix_len = len(prefix_ids)
        full_len = len(full_ids)
        target_len = full_len - prefix_len

        if target_len <= 0:
            # Reference answer tokenizes to nothing beyond prefix
            return 0.0

        # Truncate from the LEFT of prefix if too long (keep answer tokens intact)
        if full_len > max_length:
            excess = full_len - max_length
            # Remove tokens from the beginning of the prefix (after BOS)
            if excess < prefix_len - 1:
                full_ids = [full_ids[0]] + full_ids[1 + excess:]
                prefix_len = prefix_len - excess
                full_len = len(full_ids)
            else:
                # Cannot fit; return low confidence
                return 0.0

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # (1, full_len, vocab_size)

        # logits[t] predicts token at position t+1
        # Answer tokens are at positions [prefix_len, prefix_len+1, ..., full_len-1]
        # So we need logits at [prefix_len-1, prefix_len, ..., full_len-2]
        relevant_logits = logits[0, prefix_len - 1: full_len - 1, :]  # (target_len, V)
        log_probs = torch.log_softmax(relevant_logits, dim=-1)

        target_ids = input_ids[0, prefix_len: full_len]  # (target_len,)
        target_log_probs = log_probs[
            torch.arange(target_len, device=device), target_ids
        ]

        # Geometric mean of per-token probabilities
        confidence = torch.exp(target_log_probs.mean()).item()

        return max(0.0, min(1.0, confidence))

    # ------------------------------------------------------------------
    # TTS computation for a single problem
    # ------------------------------------------------------------------
    def _compute_tts(
        self,
        question: str,
        cot: str,
        reference_answer: str,
        insights_section: str = "",
        dataset_name: str = "",
    ) -> Dict:
        """Compute per-step True Thinking Score for one problem.

        For each step s_i, measures 4 conditions via logprob-based P(y*):
          S_1(1): intact context + intact step
          S_0(1): intact context + perturbed step
          S_1(0): perturbed context + intact step
          S_0(0): perturbed context + perturbed step

        TTS(s_i) = 0.5 * (|S_1(1) - S_0(1)| + |S_1(0) - S_0(0)|)

        Returns dict with:
          tts_per_step: List[float]        - TTS for each step
          tts_mean: float                  - average TTS across all steps
          tts_high_ratio: float            - fraction of steps with TTS >= alpha
          tts_decorative_ratio: float      - fraction of steps with TTS <= beta
          num_steps: int
          num_probed_steps: int
          step_details: List[Dict]         - per-step S values and classification
        """
        # Get dataset-specific answer bridge and formatted reference
        answer_bridge, formatted_ref = self._get_tts_probe_format(
            dataset_name, reference_answer
        )

        steps = self._segment_cot(cot)
        if not steps:
            return {
                "tts_per_step": [],
                "tts_mean": 0.0,
                "tts_high_ratio": 0.0,
                "tts_decorative_ratio": 0.0,
                "num_steps": 0,
                "num_probed_steps": 0,
                "step_details": [],
            }

        # Build the question prefix (same format as _get_solution_prompt)
        question_prefix = f"{insights_section}Problem: {question}\n\n"

        tts_per_step = [0.0] * len(steps)
        step_details = []
        num_probed = 0

        for i in range(len(steps)):
            step_text = steps[i]
            context_parts = steps[:i]

            step_has_numbers = self._has_numbers(step_text)
            context_has_numbers = any(self._has_numbers(p) for p in context_parts)

            if not step_has_numbers:
                tts_per_step[i] = 0.0
                step_details.append({
                    "step_index": i,
                    "tts": 0.0,
                    "skipped": True,
                    "reason": "no_numbers_in_step",
                    "classification": "decorative",
                })
                continue

            num_probed += 1
            print(f"      TTS probing step {i + 1}/{len(steps)} ...")

            # Build 4 CoT prefixes (question + context + step)
            intact_cot = "\n".join(context_parts + [step_text])
            perturbed_step = self._perturb_text(step_text)
            cot_with_perturbed_step = "\n".join(context_parts + [perturbed_step])

            if context_has_numbers:
                perturbed_context = [self._perturb_text(p) for p in context_parts]
            else:
                perturbed_context = list(context_parts)

            cot_perturbed_ctx_intact_step = "\n".join(perturbed_context + [step_text])
            cot_both_perturbed = "\n".join(perturbed_context + [perturbed_step])

            # Full prefixes including question
            prefix_11 = question_prefix + intact_cot
            prefix_01 = question_prefix + cot_with_perturbed_step
            prefix_10 = question_prefix + cot_perturbed_ctx_intact_step
            prefix_00 = question_prefix + cot_both_perturbed

            # Measure P(y*) for each condition using logprobs
            s1_1 = self._compute_answer_confidence(prefix_11, formatted_ref, answer_bridge)
            s0_1 = self._compute_answer_confidence(prefix_01, formatted_ref, answer_bridge)
            s1_0 = self._compute_answer_confidence(prefix_10, formatted_ref, answer_bridge)
            s0_0 = self._compute_answer_confidence(prefix_00, formatted_ref, answer_bridge)

            tts = 0.5 * (abs(s1_1 - s0_1) + abs(s1_0 - s0_0))
            tts_per_step[i] = tts

            # Classify step
            if tts >= TTS_ALPHA:
                classification = "true_thinking"
            elif tts <= TTS_BETA:
                classification = "decorative"
            else:
                classification = "intermediate"

            step_details.append({
                "step_index": i,
                "tts": round(tts, 4),
                "s1_1": round(s1_1, 4),
                "s0_1": round(s0_1, 4),
                "s1_0": round(s1_0, 4),
                "s0_0": round(s0_0, 4),
                "classification": classification,
                "skipped": False,
            })

            print(f"        S1(1)={s1_1:.4f} S0(1)={s0_1:.4f} "
                  f"S1(0)={s1_0:.4f} S0(0)={s0_0:.4f} -> TTS={tts:.4f} [{classification}]")

        tts_mean = float(np.mean(tts_per_step)) if tts_per_step else 0.0
        tts_high = sum(1 for t in tts_per_step if t >= TTS_ALPHA)
        tts_decorative = sum(1 for t in tts_per_step if t <= TTS_BETA)
        n = len(tts_per_step) if tts_per_step else 1

        return {
            "tts_per_step": [round(t, 4) for t in tts_per_step],
            "tts_mean": round(tts_mean, 4),
            "tts_high_ratio": round(tts_high / n, 4),
            "tts_decorative_ratio": round(tts_decorative / n, 4),
            "num_steps": len(steps),
            "num_probed_steps": num_probed,
            "step_details": step_details,
        }

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------
    def run_tts_eval(
        self,
        dataset_list: List[str],
        max_problems: Optional[int],
        encyclopedia_paths: Optional[List[str]] = None,
    ) -> Dict:
        """Run TTS evaluation: solve each problem, then measure TTS per step.

        Pipeline per problem:
          1. Solve the problem (single call, like eval-only) -> get CoT + answer
          2. Check answer correctness
          3. Compute TTS for each step via logprob-based perturbation analysis
          4. Save per-problem report with tts_per_step and tts_mean

        Every problem is guaranteed to have a tts_mean score in the output.

        Args:
            dataset_list: Datasets to evaluate
            max_problems: Max problems per dataset
            encyclopedia_paths: Optional encyclopedia files for guidance

        Returns:
            Summary dict with per-dataset accuracy and TTS arrays
        """
        # Load HF model for both generation and logprob probing
        self._ensure_hf_model()

        # Build insights section from encyclopedia (if any)
        insights_section = ""
        if encyclopedia_paths:
            valid_eps = [ep for ep in encyclopedia_paths if ep and os.path.exists(ep)]
            if valid_eps:
                print(f"Loading {len(valid_eps)} encyclopedias for guidance...")
                self.client.load_encyclopedias(valid_eps, mode=self.mode)

                if self.client.encyclopedia_loaded:
                    if self.client.encyclopedia_dict:
                        insights_list = []
                        for name, desc in self.client.encyclopedia_dict.items():
                            insights_list.append(f"**{name}**:\n{desc}")
                        insights_text = "\n\n".join(insights_list)
                    else:
                        insights_text = self.client.encyclopedia

                    insights_section = (
                        f"Available Insights to Guide Your Solution:\n\n"
                        f"{insights_text}\n\n---\n"
                        f"INSTRUCTIONS: Review the insights above and actively apply "
                        f"the relevant techniques from insights to solve this problem. "
                        f"Consider which insights can help you approach the problem "
                        f"more effectively.\n\n"
                    )
            if not insights_section:
                print("No valid encyclopedias found; proceeding without guidance")

        print(f"\n{'=' * 80}")
        print("TTS EVALUATION (logprob-based, arxiv 2510.24941)")
        print(f"Model: {self.model_name}")
        print(f"Datasets: {', '.join(dataset_list)}")
        print(f"Max problems per dataset: {max_problems or 'all'}")
        print(f"TTS thresholds: alpha={TTS_ALPHA}, beta={TTS_BETA}")
        print(f"Probe cue: dataset-specific (default: '{_DEFAULT_ANSWER_BRIDGE.strip()}')")
        print("Pipeline mode: inference + TTS only (no insight extraction/aggregation)")
        print(f"Encyclopedia guidance: {'enabled' if insights_section else 'disabled'}")
        print(f"{'=' * 80}\n")

        accuracy_map: Dict[str, float] = {}
        tts_array_map: Dict[str, List[float]] = {}

        for dataset_name in dataset_list:
            problems = self.load_math_dataset(dataset_name)
            worklist = problems[:max_problems] if max_problems else problems
            print(f"\nEvaluating {dataset_name} ({len(worklist)} problems)...")

            eval_dir = os.path.join(self.output_dir, dataset_name)
            os.makedirs(eval_dir, exist_ok=True)

            results = []
            tts_scores_for_dataset: List[float] = []

            for idx, problem_data in enumerate(worklist, 1):
                problem_text, test_cases_for_eval = self._format_problem(
                    problem_data, dataset_name
                )
                if not problem_text:
                    print(f"  [skip] Problem {idx} missing text")
                    # Still record 0.0 so every problem has a score
                    tts_scores_for_dataset.append(0.0)
                    continue

                print(f"\n  [{idx}/{len(worklist)}] {problem_text[:80]}...")

                try:
                    # Step A: Solve the problem (single call)
                    prompt = self.client._get_solution_prompt(
                        problem_text, insights_section=insights_section
                    )
                    response, token_info = self.client._call_model(
                        prompt, None, max_new_tokens=32768
                    )
                    solution = response
                    number_output_tokens = token_info.get("output_tokens", 0)
                    loop_count = self._count_consecutive_sentence_loops(solution)

                    # Step B: Extract answer and check correctness
                    predicted_answer = self._extract_answer_from_solution(
                        solution, dataset_name, problem_data
                    )
                    is_correct = False
                    ground_truth = ""
                    if test_cases_for_eval:
                        is_correct = self._check_answer_match(
                            solution, test_cases_for_eval, dataset_name, problem_text
                        )
                    else:
                        ground_truth = self._get_ground_truth(problem_data, dataset_name)
                        if predicted_answer:
                            is_correct = self._check_answer_match(
                                predicted_answer, ground_truth, dataset_name, problem_text
                            )

                    status = "+" if is_correct else "x"
                    print(f"    {status} Predicted: {predicted_answer or 'N/A'} "
                          f"| GT: {ground_truth or 'N/A'}")

                    # Step C: Compute TTS
                    # y* = model's own predicted answer (reference for TTS probing)
                    reference_answer = predicted_answer or ""
                    if not reference_answer:
                        print("    [TTS] No answer extracted; assigning tts_mean=0.0")
                        tts_result = {
                            "tts_per_step": [],
                            "tts_mean": 0.0,
                            "tts_high_ratio": 0.0,
                            "tts_decorative_ratio": 0.0,
                            "num_steps": 0,
                            "num_probed_steps": 0,
                            "step_details": [],
                        }
                    else:
                        print(f"    Computing TTS (y*={reference_answer})...")
                        tts_result = self._compute_tts(
                            question=problem_text,
                            cot=solution,
                            reference_answer=reference_answer,
                            insights_section=insights_section,
                            dataset_name=dataset_name,
                        )

                    tts_mean = tts_result["tts_mean"]
                    tts_scores_for_dataset.append(tts_mean)
                    print(f"    TTS: mean={tts_mean:.4f}, "
                          f"true_thinking={tts_result['tts_high_ratio']:.2%}, "
                          f"decorative={tts_result['tts_decorative_ratio']:.2%}, "
                          f"steps={tts_result['num_steps']} "
                          f"(probed={tts_result['num_probed_steps']})")

                    # Save per-problem report
                    output_data = {
                        "problem": problem_text,
                        "problem_id": problem_data.get("id", idx),
                        "solution": solution,
                        "predicted_answer": predicted_answer,
                        "is_correct": is_correct,
                        "number_output_tokens": number_output_tokens,
                        "loop_count": loop_count,
                        "tts_per_step": tts_result["tts_per_step"],
                        "tts_mean": tts_mean,
                        "tts_high_ratio": tts_result["tts_high_ratio"],
                        "tts_decorative_ratio": tts_result["tts_decorative_ratio"],
                        "num_steps": tts_result["num_steps"],
                        "num_probed_steps": tts_result["num_probed_steps"],
                        "step_details": tts_result["step_details"],
                    }

                    output_path = os.path.join(eval_dir, f"problem_{idx:04d}.json")
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, indent=2, ensure_ascii=False)

                    results.append({
                        "is_correct": is_correct,
                        "number_output_tokens": number_output_tokens,
                        "loop_count": loop_count,
                        "tts_mean": tts_mean,
                    })
                    time.sleep(0.3)

                except Exception as exc:
                    print(f"    Error processing problem {idx}: {exc}")
                    import traceback
                    traceback.print_exc()
                    # Ensure every problem has a score
                    tts_scores_for_dataset.append(0.0)

            # Dataset summary
            if results:
                num_correct = sum(1 for r in results if r["is_correct"])
                accuracy = num_correct / len(results)
                avg_tokens = sum(r["number_output_tokens"] for r in results) / len(results)
                avg_tts = float(np.mean([r["tts_mean"] for r in results]))
            else:
                accuracy = 0.0
                avg_tokens = 0.0
                avg_tts = 0.0

            accuracy_map[dataset_name] = accuracy
            tts_array_map[dataset_name] = tts_scores_for_dataset

            print(f"\n  {dataset_name}: Accuracy={accuracy:.2%}, "
                  f"Avg Tokens={avg_tokens:.1f}, Avg TTS={avg_tts:.4f}")

        # Save overall summary
        summary = {
            "mode": "tts_eval",
            "model": self.model_name,
            "eval_model": self.model_name,
            "pipeline_mode": "inference_and_tts_only",
            "datasets": dataset_list,
            "accuracy_per_dataset": accuracy_map,
            "tts_per_dataset": tts_array_map,
            "tts_thresholds": {"alpha": TTS_ALPHA, "beta": TTS_BETA},
            "probe_cue": {
                ds: self._get_tts_probe_format(ds, "")[0].strip()
                for ds in dataset_list
            },
            "encyclopedia_used": [
                ep for ep in (encyclopedia_paths or []) if ep and os.path.exists(ep)
            ],
        }

        summary_path = os.path.join(self.output_dir, "tts_eval_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 80}")
        print("TTS EVALUATION COMPLETE")
        print(f"{'=' * 80}")
        for dataset_name in dataset_list:
            acc = accuracy_map.get(dataset_name, 0.0)
            tts_arr = tts_array_map.get(dataset_name, [])
            avg = float(np.mean(tts_arr)) if tts_arr else 0.0
            print(f"  {dataset_name}: Accuracy={acc:.2%}, "
                  f"Avg TTS={avg:.4f} ({len(tts_arr)} problems)")
        print(f"\nSummary saved: {summary_path}")

        return summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "TTS Evaluation Pipeline: eval-model-only mode for inference and "
            "True Thinking Score measurement per reasoning step (arxiv 2510.24941). "
            "Optionally uses encyclopedia guidance if provided. "
            "Requires HuggingFace model for logprob-based P(y*) measurement."
        )
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["aime25"],
        help="Datasets to evaluate (space- or comma-separated).",
    )
    parser.add_argument(
        "--max-problems", type=int, default=None,
        help="Limit problems per dataset.",
    )
    parser.add_argument(
        "--eval-model", type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Evaluation model used for inference and TTS measurement only.",
    )
    parser.add_argument(
        "-m", "--model", dest="legacy_model", type=str, default=None,
        help="Deprecated alias for --eval-model.",
    )
    parser.add_argument(
        "-d", "--device", type=str, default=None,
        help="Device to use (cuda or cpu).",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="tts_eval_output",
        help="Root output directory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--load-in-8bit", type=bool, default=False,
        help="Load model with 8-bit quantization.",
    )
    parser.add_argument(
        "--mode", type=str, default="text", choices=["normal", "text"],
        help="Encyclopedia loading mode (default: text).",
    )
    parser.add_argument(
        "--encyclopedia", type=str, nargs="*", default=None,
        help="Optional encyclopedia path(s) for guidance. If omitted, runs without encyclopedia.",
    )
    parser.add_argument(
        "--eval-only", "--eval_only", dest="eval_only", action="store_true",
        help="Eval-model-only mode (inference + TTS). This script already defaults to this behavior.",
    )

    args = parser.parse_args()

    datasets = _parse_list_arg(args.datasets)
    eval_model = args.legacy_model or args.eval_model
    if args.legacy_model:
        print("[Deprecated] --model is used as alias of --eval-model")
    if args.eval_only:
        print("Eval-only mode enabled: inference + TTS measurement")

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {args.seed}")

    pipeline = TTSEvalPipeline(
        model_name=eval_model,
        device=args.device,
        output_dir=args.output_dir,
        mode=args.mode,
        num_iterations=1,
        load_in_8bit=args.load_in_8bit,
    )

    try:
        if not datasets:
            raise ValueError("--datasets is required")
        pipeline.run_tts_eval(
            dataset_list=datasets,
            max_problems=args.max_problems,
            encyclopedia_paths=args.encyclopedia,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback
        traceback.print_exc()
        print("\nExamples:")
        print("  python task_benchmark_domain_tts.py --datasets aime25 --max-problems 5")
        print("  python task_benchmark_domain_tts.py --datasets math500 --max-problems 10 "
              "--encyclopedia enc.json")
        print("  python task_benchmark_domain_tts.py --datasets gsm8k --max-problems 20 "
              "-m deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")


if __name__ == "__main__":
    main()
