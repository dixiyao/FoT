"""
Metacognitive Reuse Benchmark Pipeline (task_benchmark_domain_metacognitive)

Implements the evaluation pipeline from:
"Metacognitive Reuse: Turning Recurring LLM Reasoning Into Concise Behaviors"

Pipeline per iteration:
  1. First iteration (iteration 1):
     - Run all questions of selected benchmarks (solve + extract behaviors via 3-phase pipeline)
     - Store behaviors per dataset under the dataset folder
  2. Second iteration onwards (iteration >= 2):
     Three baselines are evaluated for each dataset:
       A) Per-dataset Behavior-Conditioned Inference (BCI):
          Each dataset uses ONLY its own behavior book to solve its questions.
       B) Combined Behavior Book BCI:
          All datasets' behaviors are merged into one combined book;
          every dataset uses this single combined book for BCI.
       C) Encyclopedia/Insight Library:
          Use server_text.py to aggregate behaviors into an encyclopedia
          (insight library), then use it for each dataset.
     If behavior book or encyclopedia is not found in iteration >= 2, an error is raised.

Supported datasets: same as task_benchmark_domain.py (gsm8k, aime24, aime25, math500, etc.)
"""

import argparse
import csv
import json
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from client_metacognitive import MetacognitiveClient
from math_datasets.imo_benchmark import imo_evaluator
from math_datasets.livemathbench import livemathbench_evaluator
from math_datasets.utils import extract_numbers
from server_text import TextBasedInsightAggregationServer

# ---------------------------------------------------------------------------
# Dataset registry (same as task_benchmark_domain.py)
# ---------------------------------------------------------------------------
DATASET_REGISTRY: Dict[str, Tuple[str, str, Optional[str], Optional[str]]] = {
    "gsm8k": ("hf", "openai/gsm8k", "main", "test"),
    "gsm8k_train": ("hf", "openai/gsm8k", "main", "train"),
    "aime25": ("hf", "math-ai/aime25", None, "test"),
    "aime24": ("hf", "math-ai/aime24", None, "test"),
    "math500": ("hf", "HuggingFaceH4/MATH-500", None, "test"),
    "math1000": ("hf", "hendrycks/competition_math", None, "test"),
    "gpqa": ("hf", "Idavidrein/gpqa", "gpqa_main", "train"),
    "gpqa_diamond": ("hf", "Idavidrein/gpqa", "gpqa_diamond", "train"),
    "livecodebench": ("hf", "bzantium/livecodebench", "release_v6", "test"),
    "livecodebench_lite": ("hf", "bzantium/livecodebench", "v6", "test"),
    "livemathbench_amc": ("hf", "opencompass/LiveMathBench", "v202412_AMC_en", "test"),
    "livemathbench_ccee": ("hf", "opencompass/LiveMathBench", "v202412_CCEE_en", "test"),
    "livemathbench_cnmo": ("hf", "opencompass/LiveMathBench", "v202412_CNMO_en", "test"),
    "livemathbench_wlpmc": ("hf", "opencompass/LiveMathBench", "v202412_WLPMC_en", "test"),
    "livemathbench_hard_2024": ("hf", "opencompass/LiveMathBench", "v202412_hard_en", "test"),
    "livemathbench_hard_2025": ("hf", "opencompass/LiveMathBench", "v202505_hard_en", "test"),
    "imo_answerbench": ("csv", "math_datasets/answerbench.csv", None, None),
    "imo_answerbench_algebra": ("csv", "math_datasets/imo_algebra.csv", None, None),
    "imo_answerbench_geometry": ("csv", "math_datasets/imo_geometry.csv", None, None),
    "imo_answerbench_number_theory": ("csv", "math_datasets/imo_number_theory.csv", None, None),
    "imo_proofbench": ("csv", "math_datasets/proofbench.csv", None, None),
    "imo_gradingbench": ("csv", "math_datasets/gradingbench.csv", None, None),
}


class MetacognitiveBenchmarkPipeline:
    """Benchmark pipeline using the Metacognitive Reuse framework.

    Iteration 1: solve + extract behaviors (3-phase pipeline)
    Iteration >= 2: three BCI baselines (per-dataset, combined, encyclopedia)
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device: Optional[str] = None,
        output_dir: str = "metacognitive_output",
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
        num_iterations: int = 2,
        load_in_8bit: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        self.use_api = use_api
        self.api_key = api_key
        self.api_provider = api_provider
        self.num_iterations = num_iterations
        self.load_in_8bit = load_in_8bit
        self._deepseek_token_patch_applied = False

        os.makedirs(output_dir, exist_ok=True)

        self.client: Optional[MetacognitiveClient] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _count_consecutive_sentence_loops(self, text: str) -> int:
        if not text:
            return 0
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        prev = None
        loops = 0
        for sentence in sentences:
            cleaned = sentence.strip()
            if not cleaned:
                continue
            if prev is not None and cleaned == prev:
                loops += 1
            prev = cleaned
        return loops

    def _ensure_client(self):
        if self.client is None:
            self.client = MetacognitiveClient(
                model_name=self.model_name,
                device=self.device,
                use_api=self.use_api,
                api_key=self.api_key,
                api_provider=self.api_provider,
                load_in_8bit=self.load_in_8bit,
            )

    # ------------------------------------------------------------------
    # Dataset loading (same as task_benchmark_domain.py)
    # ------------------------------------------------------------------
    def _load_local_json(self, dataset_name: str, explicit_path: Optional[str]) -> List[Dict]:
        candidate_path = explicit_path or os.path.join("math_datasets", f"{dataset_name}.json")
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' not found. Provide {candidate_path} or update DATASET_REGISTRY."
            )
        with open(candidate_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _load_csv_file(self, dataset_name: str, explicit_path: Optional[str]) -> List[Dict]:
        candidate_path = explicit_path or os.path.join("math_datasets", f"{dataset_name}.csv")
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(f"CSV file for dataset '{dataset_name}' not found at {candidate_path}")

        print(f"Loading CSV file from {candidate_path}...")
        with open(candidate_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file {candidate_path} is empty or has no header")

            fieldnames_lower = [fn.lower() for fn in reader.fieldnames]
            print(f"CSV columns: {reader.fieldnames}")

            problem_cols = ["problem", "question", "problem_text", "task", "statement", "text"]
            answer_cols = ["answer", "solution", "final_answer", "answer_text", "short answer", "short_answer", "response"]
            id_cols = ["id", "problem_id", "problem id", "grading_id", "grading id", "num", "number", "idx"]

            problem_col = None
            answer_col = None
            id_col = None

            for col in problem_cols:
                if col in fieldnames_lower:
                    problem_col = reader.fieldnames[fieldnames_lower.index(col)]
                    break
            for col in answer_cols:
                if col in fieldnames_lower:
                    answer_col = reader.fieldnames[fieldnames_lower.index(col)]
                    break
            for col in id_cols:
                if col in fieldnames_lower:
                    id_col = reader.fieldnames[fieldnames_lower.index(col)]
                    break

            if not problem_col:
                raise ValueError(
                    f"Could not find problem column in CSV. Available: {reader.fieldnames}. Expected one of: {problem_cols}"
                )
            if not answer_col:
                raise ValueError(
                    f"Could not find answer column in CSV. Available: {reader.fieldnames}. Expected one of: {answer_cols}"
                )

            data = list(reader)
            print(f"Loaded {len(data)} rows from CSV")
            return data

    def _normalize_problems(self, raw_problems: List[Dict], dataset_name: str) -> List[Dict]:
        def get_field(obj: Dict, candidates: List[str], default: str = "") -> str:
            for candidate in candidates:
                for key in obj.keys():
                    if key.lower() == candidate.lower():
                        val = obj[key]
                        return str(val) if val is not None else default
            return default

        normalized = []
        for idx, problem in enumerate(raw_problems):
            problem_text = get_field(problem, ["problem", "question", "problem_text", "task", "statement", "text"])
            answer_text = get_field(problem, ["answer", "solution", "final_answer", "answer_text"])
            id_val = get_field(problem, ["id", "problem_id", "num", "number", "idx"], str(idx + 1))

            normalized_problem = {
                "id": int(id_val) if id_val.isdigit() else id_val,
                "problem": problem_text,
                "question": problem_text,
                "solution": get_field(problem, ["solution", "step_by_step"]),
                "answer": answer_text,
            }

            if dataset_name.startswith("gsm8k"):
                if "####" in answer_text:
                    parts = answer_text.split("####")
                    normalized_problem["solution"] = parts[0].strip()
                    normalized_problem["answer"] = parts[-1].strip()

            for key, value in problem.items():
                if key not in normalized_problem:
                    normalized_problem[key] = value
            normalized.append(normalized_problem)
        return normalized

    def load_dataset(self, dataset_name: str) -> List[Dict]:
        entry = DATASET_REGISTRY.get(dataset_name)
        if not entry:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {', '.join(DATASET_REGISTRY.keys())}"
            )

        source_type, path_or_hf_name, data_dir, split = entry

        if source_type == "hf":
            if load_dataset is None:
                raise ImportError("datasets library is required. Install with: pip install datasets")
            print(f"Loading dataset '{dataset_name}' from Hugging Face ({path_or_hf_name}, split={split})...")
            if data_dir and split:
                ds = load_dataset(path_or_hf_name, name=data_dir, split=split)
            else:
                ds = load_dataset(path_or_hf_name, split=split)

            raw = []
            for i, item in enumerate(ds):
                if dataset_name == "math1000" and i >= 1000:
                    break
                if dataset_name and dataset_name.startswith("gpqa"):
                    raw_item = dict(item)
                    raw_item["id"] = item.get("id", i + 1)
                    raw.append(raw_item)
                elif dataset_name and dataset_name.startswith("livecodebench"):
                    raw_item = dict(item)
                    raw_item["id"] = item.get("question_id", item.get("id", i + 1))
                    raw.append(raw_item)
                else:
                    problem_text = item.get("problem") or item.get("question", "")
                    solution = item.get("solution", "")
                    answer = item.get("answer", "")
                    if dataset_name == "math1000" and "####" in solution:
                        answer = solution.split("####")[-1].strip()
                    raw.append({
                        "id": item.get("id", i + 1),
                        "problem": problem_text,
                        "question": problem_text,
                        "solution": solution or answer or "",
                        "answer": answer,
                    })
            print(f"Loaded {len(raw)} problems from Hugging Face")
            if dataset_name and (dataset_name.startswith("gpqa") or dataset_name.startswith("livecodebench")):
                return raw
            return self._normalize_problems(raw, dataset_name)

        if source_type == "csv":
            raw = self._load_csv_file(dataset_name, path_or_hf_name)
            print(f"Loaded {len(raw)} problems from CSV for '{dataset_name}'")
            return self._normalize_problems(raw, dataset_name)

        if source_type == "json":
            raw = self._load_local_json(dataset_name, path_or_hf_name)
            print(f"Loaded {len(raw)} problems from JSON for '{dataset_name}'")
            return self._normalize_problems(raw, dataset_name)

        raise ValueError(f"Unknown source type: {source_type}")

    # ------------------------------------------------------------------
    # Problem formatting (dataset-specific)
    # ------------------------------------------------------------------
    def _format_problem(self, problem_data: Dict, dataset_name: str) -> Tuple[Optional[str], Optional[object]]:
        """Format a problem using dataset-specific formatter.

        Returns:
            (problem_text, test_cases_or_none)
        """
        problem_text = None
        test_cases_for_eval = None

        if dataset_name == "aime25":
            from math_datasets.aime25 import aime25_formatter
            problem_text, _ = aime25_formatter(problem_data)
        elif dataset_name == "aime24":
            from math_datasets.aime25 import aime25_formatter
            problem_text, _ = aime25_formatter(problem_data)
        elif dataset_name and "livemathbench" in dataset_name:
            from math_datasets.livemathbench import livemathbench_formatter
            problem_text, _ = livemathbench_formatter(problem_data, dataset_name)
        elif dataset_name and dataset_name.startswith("imo"):
            from math_datasets.imo_benchmark import imo_formatter
            problem_text, _ = imo_formatter(problem_data, dataset_name)
        elif dataset_name == "math500":
            from math_datasets.math500 import math500_formatter
            problem_text, _ = math500_formatter(problem_data)
        elif dataset_name == "gsm8k":
            from math_datasets.gsm8k import gsm8k_formatter
            problem_text, _ = gsm8k_formatter(problem_data)
        elif dataset_name and dataset_name.startswith("gpqa"):
            from science_datasets.gpqa import gpqa_formatter
            problem_text, _ = gpqa_formatter(problem_data)
        elif dataset_name and "livecodebench" in dataset_name:
            from code_datasets.livecodebench import livecodebench_formatter
            problem_text, test_cases_for_eval = livecodebench_formatter(problem_data)
        else:
            problem_text = problem_data.get("problem") or problem_data.get("question", "")

        return problem_text, test_cases_for_eval

    def _get_ground_truth(self, problem_data: Dict, dataset_name: str) -> str:
        """Get ground truth answer for a problem."""
        if dataset_name and dataset_name.startswith("gpqa"):
            from science_datasets.gpqa import gpqa_formatter
            _, ground_truth = gpqa_formatter(problem_data)
            return ground_truth
        return problem_data.get("answer") or problem_data.get("solution", "")

    # ------------------------------------------------------------------
    # Answer extraction and checking (same as task_benchmark_domain.py)
    # ------------------------------------------------------------------
    def _extract_answer_from_solution(self, solution: str, dataset_name: str, problem_data: Dict) -> Optional[str]:
        if not solution:
            return None

        def _extract_boxed_balanced(text: str) -> Optional[str]:
            marker = "\\boxed{"
            start = text.rfind(marker)
            if start == -1:
                return None

            i = start + len(marker)
            depth = 1
            while i < len(text) and depth > 0:
                char = text[i]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                i += 1

            if depth != 0:
                return None

            return text[start + len(marker) : i - 1].strip()

        boxed_answer = _extract_boxed_balanced(solution)
        if boxed_answer:
            boxed_answer = boxed_answer.replace("\\,", "").replace("\\:", "").replace("\\;", "")
            boxed_answer = boxed_answer.replace("\\text{", "").replace("}", "")
            return boxed_answer

        boxed_patterns = [
            r"\\\(\\boxed\{([^}]+)\}\\\)",
            r"\\\[\\boxed\{([^}]+)\}\\\]",
            r"\\boxed\{([^}]+)\}",
        ]
        for pattern in boxed_patterns:
            match = re.search(pattern, solution)
            if match:
                answer = match.group(1).strip()
                answer = answer.replace("\\,", "").replace("\\:", "").replace("\\;", "")
                answer = answer.replace("\\text{", "").replace("}", "")
                return answer

        if "## Answer:" in solution:
            start_idx = solution.find("## Answer:") + len("## Answer:")
            end_idx = solution.find("## End of Answer:")
            if end_idx == -1:
                next_heading = solution.find("##", start_idx)
                end_idx = next_heading if next_heading != -1 else len(solution)
            answer = solution[start_idx:end_idx].strip()

            boxed_answer = _extract_boxed_balanced(answer)
            if boxed_answer:
                return boxed_answer

            for pattern in boxed_patterns:
                match = re.search(pattern, answer)
                if match:
                    return match.group(1).strip()
            return answer

        if dataset_name:
            if dataset_name in ["aime25", "aime24"] or dataset_name.startswith("imo"):
                answer_patterns = [
                    r"(?:the answer is|answer:|final answer:?)\s*\$?([^.$\n]+)\$?",
                    r"(?:therefore|thus|so),?\s+(?:the answer is)?\s*\$?([^.$\n]+)\$?",
                ]
                search_text = solution[-1000:] if len(solution) > 1000 else solution
                for pattern in answer_patterns:
                    matches = re.finditer(pattern, search_text, re.IGNORECASE)
                    last_match = None
                    for match in matches:
                        last_match = match
                    if last_match:
                        answer = last_match.group(1).strip()
                        answer = answer.replace("\\,", "").replace("$", "")
                        numbers = extract_numbers(answer)
                        if numbers:
                            num = numbers[-1]
                            return str(int(num)) if num == int(num) else str(num)
            elif dataset_name == "gsm8k":
                numbers = extract_numbers(solution)
                if numbers:
                    num = numbers[-1]
                    return str(int(num)) if num == int(num) else str(num)
            elif "livemathbench" in dataset_name:
                numbers = extract_numbers(solution)
                if numbers:
                    num = numbers[-1]
                    return str(int(num)) if num == int(num) else str(num)

        numbers = extract_numbers(solution)
        if numbers:
            num = numbers[-1]
            return str(int(num)) if num == int(num) else str(num)

        lines = [line.strip() for line in solution.split("\n") if line.strip()]
        if lines:
            last_line = lines[-1]
            last_line = re.sub(r"\.$", "", last_line)
            last_line = last_line.replace("\\)", "").replace("\\(", "")
            return last_line[:100]

        return None

    def _normalize_answer_for_comparison(self, answer: str) -> str:
        if not answer:
            return ""
        answer = answer.replace("$", "").replace("\\(", "").replace("\\)", "")
        answer = answer.replace("\\[", "").replace("\\]", "")
        answer = answer.replace("\\left", "").replace("\\right", "")
        answer = answer.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
        answer = re.sub(r"\\text\{([^}]+)\}", r"\1", answer)
        answer = re.sub(r"\\displaystyle\s*", "", answer)
        answer = re.sub(r"^\s*[a-zA-Z]\w*\s*=\s*", "", answer)
        answer = re.sub(r"^\s*(?:answer|finalanswer|ans)\s*[:=]\s*", "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", answer)
        answer = re.sub(r"\s+", "", answer)
        answer = answer.lower()
        return answer.strip()

    def _check_answer_match(
        self, predicted: str, ground_truth: str, dataset_name: Optional[str] = None, problem_text: Optional[str] = None,
    ) -> bool:
        if not predicted or not ground_truth:
            return False

        if dataset_name and "livemathbench" in dataset_name:
            return livemathbench_evaluator(predicted, ground_truth, dataset_name=dataset_name, problem_text=problem_text)

        if dataset_name and dataset_name.startswith("imo"):
            return imo_evaluator(
                prediction=predicted, ground_truth=ground_truth, benchmark=dataset_name,
                problem_text=problem_text, use_gemini=self.use_api and self.client is not None, client=self.client,
            )

        if dataset_name and dataset_name.startswith("gpqa"):
            from science_datasets.gpqa import gpqa_evaluator
            return gpqa_evaluator(predicted, ground_truth, dataset_name=dataset_name, problem_text=problem_text)

        if dataset_name and "livecodebench" in dataset_name:
            from code_datasets.livecodebench import livecodebench_evaluator
            return livecodebench_evaluator(predicted, ground_truth, dataset_name=dataset_name, problem_text=problem_text)

        pred_normalized = self._normalize_answer_for_comparison(predicted)
        gt_normalized = self._normalize_answer_for_comparison(ground_truth)
        if pred_normalized and gt_normalized and pred_normalized == gt_normalized:
            return True

        pred_nums = extract_numbers(predicted)
        gt_nums = extract_numbers(ground_truth)
        if pred_nums and gt_nums:
            return any(abs(p - g) < 1e-6 for p in pred_nums for g in gt_nums)

        return predicted.strip().lower() == ground_truth.strip().lower()

    # ------------------------------------------------------------------
    # Phase A: Solve + Extract Behaviors (Iteration 1)
    # ------------------------------------------------------------------
    def _solve_and_extract_behaviors_for_dataset(
        self,
        dataset_name: str,
        problems: List[Dict],
        max_problems: Optional[int],
        iteration: int = 1,
    ) -> Tuple[str, List[Dict]]:
        """Iteration 1: Solve all questions and extract behaviors using 3-phase pipeline.

        Behaviors are stored per dataset under output_dir/dataset_name/.

        Returns:
            (behaviors_dir, results_list)
        """
        self._ensure_client()
        behaviors_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(behaviors_dir, exist_ok=True)

        worklist = problems[:max_problems] if max_problems else problems
        print(f"\nIteration {iteration}: Solving + extracting behaviors for {dataset_name} ({len(worklist)} problems)...")

        results = []
        number_output_tokens_list = []
        loop_count_list = []

        for idx, problem_data in enumerate(worklist, 1):
            problem_text, test_cases_for_eval = self._format_problem(problem_data, dataset_name)

            if not problem_text:
                print(f"  [skip] Problem {idx} missing text")
                continue

            print(f"  [{idx}/{len(worklist)}] {problem_text[:80]}...")

            predicted_answer = None
            is_correct = False
            try:
                # Run 3-phase pipeline: Solution → Self-Reflection → Behavior Distillation
                result = self.client.solve_and_extract_behaviors(task=problem_text)

                solution = result.get("solution", "")

                # Token tracking
                token_info = result.get("token_info", {})
                number_output_tokens = token_info.get("output_tokens", 0)
                number_output_tokens_list.append(number_output_tokens)

                # Loop detection
                loop_count = self._count_consecutive_sentence_loops(solution)
                loop_count_list.append(loop_count)

                # Extract answer
                predicted_answer = self._extract_answer_from_solution(solution, dataset_name, problem_data)

                # Get behavior book
                behavior_book = result.get("behavior_book", {})
                if not behavior_book:
                    print("    No behaviors extracted")

                # Evaluate correctness
                if test_cases_for_eval:
                    is_correct = self._check_answer_match(solution, test_cases_for_eval, dataset_name, problem_text)
                    status = "+" if is_correct else "x"
                    print(f"    {status} Code execution test results")
                else:
                    ground_truth = self._get_ground_truth(problem_data, dataset_name)
                    if predicted_answer:
                        is_correct = self._check_answer_match(predicted_answer, ground_truth, dataset_name, problem_text)
                    status = "+" if is_correct else "x"
                    print(f"    {status} Predicted: {predicted_answer if predicted_answer else 'N/A'} | GT: {ground_truth if ground_truth else 'N/A'}")

                # Save per-problem output (including behavior_book)
                output_data = {
                    "problem": problem_text,
                    "problem_id": problem_data.get("id", idx),
                    "behavior_book": behavior_book,
                    "iteration": iteration,
                    "is_correct": is_correct,
                    "number_output_tokens": number_output_tokens,
                    "loop_count": loop_count,
                }

                output_path = os.path.join(behaviors_dir, f"problem_{idx:04d}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

                results.append({
                    "is_correct": is_correct,
                    "number_output_tokens": number_output_tokens,
                    "loop_count": loop_count,
                })
                time.sleep(0.5)

            except Exception as exc:
                print(f"    Error processing problem {idx}: {exc}")

        if number_output_tokens_list:
            avg_tokens = sum(number_output_tokens_list) / len(number_output_tokens_list)
            print(f"\n  Dataset '{dataset_name}' - Average Output Tokens: {avg_tokens:.1f}")
        if loop_count_list:
            print(f"  Dataset '{dataset_name}' - Total Loop Count: {sum(loop_count_list)}")

        return behaviors_dir, results

    # ------------------------------------------------------------------
    # Phase B: Behavior-Conditioned Inference (BCI)
    # ------------------------------------------------------------------
    def _run_bci_for_dataset(
        self,
        dataset_name: str,
        problems: List[Dict],
        max_problems: Optional[int],
        behaviors: Dict[str, str],
        iteration: int,
        baseline_name: str,
    ) -> List[Dict]:
        """Run Behavior-Conditioned Inference for a dataset.

        Args:
            dataset_name: Name of dataset
            problems: List of problem dicts
            max_problems: Max problems to process
            behaviors: Behavior book {name: instruction} to use
            iteration: Current iteration number
            baseline_name: Name of the baseline (per_dataset_bci, combined_bci, encyclopedia)

        Returns:
            List of result dicts
        """
        self._ensure_client()

        bci_dir = os.path.join(self.output_dir, dataset_name, f"bci_{baseline_name}_iter{iteration}")
        os.makedirs(bci_dir, exist_ok=True)

        worklist = problems[:max_problems] if max_problems else problems
        print(f"\n  [{baseline_name}] BCI for {dataset_name} ({len(worklist)} problems, {len(behaviors)} behaviors)...")

        results = []
        number_output_tokens_list = []
        loop_count_list = []

        for idx, problem_data in enumerate(worklist, 1):
            problem_text, test_cases_for_eval = self._format_problem(problem_data, dataset_name)

            if not problem_text:
                print(f"    [skip] Problem {idx} missing text")
                continue

            print(f"    [{idx}/{len(worklist)}] {problem_text[:80]}...")

            predicted_answer = None
            is_correct = False
            try:
                # Behavior-Conditioned Inference
                bci_result = self.client.behavior_conditioned_inference(
                    problem=problem_text,
                    behaviors=behaviors,
                )

                solution = bci_result.get("solution", "")

                token_info = bci_result.get("token_info", {})
                number_output_tokens = token_info.get("output_tokens", 0)
                number_output_tokens_list.append(number_output_tokens)

                loop_count = self._count_consecutive_sentence_loops(solution)
                loop_count_list.append(loop_count)

                predicted_answer = self._extract_answer_from_solution(solution, dataset_name, problem_data)

                if test_cases_for_eval:
                    is_correct = self._check_answer_match(solution, test_cases_for_eval, dataset_name, problem_text)
                    status = "+" if is_correct else "x"
                    print(f"      {status} Code execution test results")
                else:
                    ground_truth = self._get_ground_truth(problem_data, dataset_name)
                    if predicted_answer:
                        is_correct = self._check_answer_match(predicted_answer, ground_truth, dataset_name, problem_text)
                    status = "+" if is_correct else "x"
                    print(f"      {status} Predicted: {predicted_answer if predicted_answer else 'N/A'} | GT: {ground_truth if ground_truth else 'N/A'}")

                output_data = {
                    "problem": problem_text,
                    "problem_id": problem_data.get("id", idx),
                    "solution": solution,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "baseline": baseline_name,
                    "iteration": iteration,
                    "num_behaviors_used": len(behaviors),
                    "number_output_tokens": number_output_tokens,
                    "loop_count": loop_count,
                }

                output_path = os.path.join(bci_dir, f"problem_{idx:04d}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

                results.append({
                    "is_correct": is_correct,
                    "number_output_tokens": number_output_tokens,
                    "loop_count": loop_count,
                })
                time.sleep(0.5)

            except Exception as exc:
                print(f"      Error processing problem {idx}: {exc}")

        if number_output_tokens_list:
            avg_tokens = sum(number_output_tokens_list) / len(number_output_tokens_list)
            print(f"\n    [{baseline_name}] '{dataset_name}' - Avg Output Tokens: {avg_tokens:.1f}")
        if loop_count_list:
            print(f"    [{baseline_name}] '{dataset_name}' - Total Loop Count: {sum(loop_count_list)}")

        return results

    # ------------------------------------------------------------------
    # Encyclopedia generation via server_text.py
    # ------------------------------------------------------------------
    def _generate_encyclopedia_from_behaviors(
        self, dataset_list: List[str]
    ) -> str:
        """Aggregate behaviors from all datasets into an encyclopedia using server_text.py.

        Returns:
            Path to generated encyclopedia file
        """
        print("\n" + "=" * 80)
        print("Generating Encyclopedia (Insight Library) from All Behaviors")
        print("=" * 80)

        # Collect all problem_*.json files across datasets
        all_json_files = []
        for name in dataset_list:
            dataset_dir = os.path.join(self.output_dir, name)
            if not os.path.isdir(dataset_dir):
                print(f"  Warning: Directory not found for {name}: {dataset_dir}")
                continue
            problem_files = sorted(
                os.path.join(dataset_dir, f)
                for f in os.listdir(dataset_dir)
                if f.startswith("problem_") and f.endswith(".json")
            )
            if problem_files:
                all_json_files.extend(problem_files)
                print(f"  - {name}: {len(problem_files)} problem files")

        if not all_json_files:
            raise FileNotFoundError("No problem files found for encyclopedia generation!")

        print(f"\nTotal problem files to aggregate: {len(all_json_files)}")

        # Use TextBasedInsightAggregationServer to build encyclopedia
        server_text = TextBasedInsightAggregationServer(
            model_name=self.model_name,
            device=self.device,
            input_dirs=[self.output_dir],
            use_api=self.use_api,
            api_key=self.api_key,
            api_provider=self.api_provider,
            custom_prompt_section=(
                "Customization: In current context behavior has the same meaning "
                "of reasoning traces."
            ),
        )

        result = server_text.aggregate_and_build_encyclopedia(
            json_files=all_json_files, output_dir=self.output_dir
        )

        # Save as encyclopedia.json
        encyclopedia_path = os.path.join(self.output_dir, "encyclopedia.json")
        encyclopedia_dict = server_text._try_parse_json(server_text.encyclopedia)
        if encyclopedia_dict is None:
            json_content = server_text._extract_json_only(server_text.encyclopedia)
            encyclopedia_dict = server_text._try_parse_json(json_content)
        if encyclopedia_dict is None:
            raise ValueError(
                f"Could not parse encyclopedia as JSON. Content preview: {server_text.encyclopedia[:500]}"
            )

        with open(encyclopedia_path, "w", encoding="utf-8") as f:
            json.dump(encyclopedia_dict, f, indent=2, ensure_ascii=False)

        print(f"\nEncyclopedia saved to: {encyclopedia_path}")
        return encyclopedia_path

    def _load_encyclopedia(self, encyclopedia_path: str):
        """Load encyclopedia from file.

        Returns either a dict {insight_name: description} or a plain text string
        if the file cannot be parsed as a flat insight dict.
        """
        with open(encyclopedia_path, "r", encoding="utf-8") as f:
            raw = f.read()

        # Try JSON parse
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Not valid JSON — return as plain text
            print(f"  Warning: Encyclopedia is not valid JSON, using as plain text")
            return raw

        # Handle nested structure with general_insights list
        if isinstance(data, dict) and "general_insights" in data:
            flat = {}
            for insight in data["general_insights"]:
                if isinstance(insight, dict):
                    name = insight.get("insight_name", "")
                    desc = insight.get("description", "")
                    if name and desc:
                        flat[name] = desc
            if flat:
                return flat
            # If general_insights didn't yield anything useful, fall through

        # Flat dict — return as-is
        if isinstance(data, dict):
            return data

        # Unexpected structure — return raw text
        print(f"  Warning: Encyclopedia has unexpected structure, using as plain text")
        return raw

    def _format_insights_section(self, encyclopedia) -> str:
        """Format encyclopedia into an insights section for the solution prompt.

        Uses the same format as client.py / task_benchmark_domain.py for encyclopedia-based
        inference (NOT behavior-conditioned inference).

        Args:
            encyclopedia: Either a dict {insight_name: description} or a plain text string.
        """
        if isinstance(encyclopedia, dict):
            insights_list = []
            for insight_name, insight_desc in encyclopedia.items():
                insights_list.append(f"**{insight_name}**:\n{insight_desc}")
            insights_text = "\n\n".join(insights_list)
        else:
            # Plain text encyclopedia
            insights_text = str(encyclopedia)

        insights_section = f"""Available Insights to Guide Your Solution:

{insights_text}

---
INSTRUCTIONS: Review the insights above and actively apply the relevant techniques from insights to solve this problem. Consider which insights can help you approach the problem more effectively.

"""
        return insights_section

    def _run_encyclopedia_for_dataset(
        self,
        dataset_name: str,
        problems: List[Dict],
        max_problems: Optional[int],
        encyclopedia,
        iteration: int,
    ) -> List[Dict]:
        """Run encyclopedia-based inference for a dataset.

        Unlike BCI (which uses the behavior-conditioned prompt), this uses the
        insights_section prompt from client.py / task_benchmark_domain.py to prepend
        the encyclopedia as guidance to the solution prompt.

        Args:
            dataset_name: Name of dataset
            problems: List of problem dicts
            max_problems: Max problems to process
            encyclopedia: Encyclopedia dict {insight_name: description} or plain text string
            iteration: Current iteration number

        Returns:
            List of result dicts
        """
        self._ensure_client()

        enc_dir = os.path.join(self.output_dir, dataset_name, f"bci_encyclopedia_bci_iter{iteration}")
        os.makedirs(enc_dir, exist_ok=True)

        worklist = problems[:max_problems] if max_problems else problems
        insights_section = self._format_insights_section(encyclopedia)
        num_insights = len(encyclopedia) if isinstance(encyclopedia, dict) else len(encyclopedia.split('\n'))
        print(f"\n  [encyclopedia_bci] Encyclopedia inference for {dataset_name} ({len(worklist)} problems, {num_insights} insights)...")

        results = []
        number_output_tokens_list = []
        loop_count_list = []

        for idx, problem_data in enumerate(worklist, 1):
            problem_text, test_cases_for_eval = self._format_problem(problem_data, dataset_name)

            if not problem_text:
                print(f"    [skip] Problem {idx} missing text")
                continue

            print(f"    [{idx}/{len(worklist)}] {problem_text[:80]}...")

            predicted_answer = None
            is_correct = False
            try:
                # Build prompt matching client.py's _get_solution_prompt with insights_section
                # (NOT using MetacognitiveClient which has behavior_section instead)
                prompt = f"""{insights_section}Problem: {problem_text}"""

                response, token_info = self.client._call_model(prompt, None, max_new_tokens=32768)
                print(f"      Encyclopedia Solution Response: {response}")

                solution = response
                number_output_tokens = token_info.get("output_tokens", 0)
                number_output_tokens_list.append(number_output_tokens)

                loop_count = self._count_consecutive_sentence_loops(solution)
                loop_count_list.append(loop_count)

                predicted_answer = self._extract_answer_from_solution(solution, dataset_name, problem_data)

                if test_cases_for_eval:
                    is_correct = self._check_answer_match(solution, test_cases_for_eval, dataset_name, problem_text)
                    status = "+" if is_correct else "x"
                    print(f"      {status} Code execution test results")
                else:
                    ground_truth = self._get_ground_truth(problem_data, dataset_name)
                    if predicted_answer:
                        is_correct = self._check_answer_match(predicted_answer, ground_truth, dataset_name, problem_text)
                    status = "+" if is_correct else "x"
                    print(f"      {status} Predicted: {predicted_answer if predicted_answer else 'N/A'} | GT: {ground_truth if ground_truth else 'N/A'}")

                output_data = {
                    "problem": problem_text,
                    "problem_id": problem_data.get("id", idx),
                    "solution": solution,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "baseline": "encyclopedia_bci",
                    "iteration": iteration,
                    "num_insights_used": num_insights,
                    "number_output_tokens": number_output_tokens,
                    "loop_count": loop_count,
                }

                output_path = os.path.join(enc_dir, f"problem_{idx:04d}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

                results.append({
                    "is_correct": is_correct,
                    "number_output_tokens": number_output_tokens,
                    "loop_count": loop_count,
                })
                time.sleep(0.5)

            except Exception as exc:
                print(f"      Error processing problem {idx}: {exc}")

        if number_output_tokens_list:
            avg_tokens = sum(number_output_tokens_list) / len(number_output_tokens_list)
            print(f"\n    [encyclopedia_bci] '{dataset_name}' - Avg Output Tokens: {avg_tokens:.1f}")
        if loop_count_list:
            print(f"    [encyclopedia_bci] '{dataset_name}' - Total Loop Count: {sum(loop_count_list)}")

        return results

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------
    def _compute_accuracy(self, results: List[Dict]) -> float:
        if not results:
            return 0.0
        return sum(1 for r in results if r["is_correct"]) / len(results)

    def _append_summary_entry(self, entry: Dict):
        """Append entry to iterative_summary.json."""
        summary_file = os.path.join(self.output_dir, "iterative_summary.json")
        try:
            if os.path.exists(summary_file):
                with open(summary_file, "r", encoding="utf-8") as f:
                    current = json.load(f)
            else:
                current = []
            if not isinstance(current, list):
                current = []
            current.append(entry)
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(current, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  Warning: failed to append summary: {e}")

    # ------------------------------------------------------------------
    # Main Iterative Pipeline
    # ------------------------------------------------------------------
    def run_iterative_pipeline(
        self,
        dataset_list: List[str],
        max_problems: Optional[int],
        start_iteration: int = 1,
        resume_from_encyclopedia_step: bool = False,
    ) -> Dict:
        """Run the metacognitive reuse iterative pipeline.

        Iteration 1:
          - Solve all questions and extract behaviors (3-phase pipeline)
          - Store behaviors per dataset

        Iteration >= 2:
          Three baselines for each dataset:
            A) Per-dataset BCI: use only this dataset's own behaviors
            B) Combined BCI: use all datasets' behaviors merged into one book
            C) Encyclopedia BCI: use encyclopedia from server_text.py aggregation
          If behavior book or encyclopedia not found, raise error.

        Args:
            dataset_list: List of dataset names
            max_problems: Max problems per dataset
            start_iteration: Iteration index to start from (default: 1)

        Returns:
            Summary dict with iteration history
        """
        if not dataset_list:
            raise ValueError("Provide at least one dataset for the pipeline")
        if start_iteration < 1 or start_iteration > self.num_iterations:
            raise ValueError(
                f"start_iteration must be between 1 and {self.num_iterations}, got {start_iteration}"
            )

        start_time = time.time()
        iteration_history = []

        # Load all datasets once
        all_datasets: Dict[str, List[Dict]] = {}
        for name in dataset_list:
            all_datasets[name] = self.load_dataset(name)

        print(f"\n{'=' * 80}")
        print(f"Metacognitive Reuse Pipeline: {self.num_iterations} iterations")
        print(f"Datasets: {', '.join(dataset_list)}")
        print(f"Max problems per dataset: {max_problems or 'all'}")
        print(f"{'=' * 80}\n")

        if start_iteration > 1:
            print(
                f"Starting from iteration {start_iteration}; skipping 1 to {start_iteration - 1}"
            )

        for iteration in range(start_iteration, self.num_iterations + 1):
            print(f"\n{'=' * 80}")
            print(f"ITERATION {iteration}/{self.num_iterations}")
            print(f"{'=' * 80}")

            iteration_summary = {
                "iteration": iteration,
                "datasets": dataset_list,
                "accuracy_per_dataset": {},
            }

            if iteration == 1:
                # ======================================================
                # ITERATION 1: Solve + Extract Behaviors
                # ======================================================
                print("\n--- Iteration 1: Solve + Extract Behaviors (3-Phase Pipeline) ---")

                for name in dataset_list:
                    if resume_from_encyclopedia_step:
                        print(f"\n  [resume] Loading existing iteration-1 outputs for {name}...")
                    else:
                        problems = all_datasets[name]
                        _, results = self._solve_and_extract_behaviors_for_dataset(
                            name, problems, max_problems, iteration=1
                        )

                        accuracy = self._compute_accuracy(results)
                        iteration_summary["accuracy_per_dataset"][name] = {
                            "solve_extract": accuracy,
                        }

                        question_correctness = [1 if r["is_correct"] else 0 for r in results] if results else []
                        self._append_summary_entry({
                            "iteration": 1,
                            "dataset": name,
                            "baseline": "solve_extract",
                            "accuracy": accuracy,
                            "model": "gemini-3-pro-preview" if self.use_api else self.model_name,
                            "question_correctness": question_correctness,
                        })

                        print(f"\n  {name}: Solve+Extract accuracy = {accuracy:.2%}")

            else:
                # ======================================================
                # ITERATION >= 2: Three BCI Baselines
                # ======================================================
                print(f"\n--- Iteration {iteration}: Three BCI Baselines ---")

                # ----- Collect per-dataset behavior books -----
                per_dataset_behaviors: Dict[str, Dict[str, str]] = {}
                for name in dataset_list:
                    dataset_dir = os.path.join(self.output_dir, name)
                    behaviors = MetacognitiveClient.collect_behaviors_from_dir(dataset_dir)
                    if not behaviors:
                        raise FileNotFoundError(
                            f"No behavior book found for dataset '{name}' in {dataset_dir}. "
                            f"Iteration >= 2 requires behaviors from iteration 1."
                        )
                    per_dataset_behaviors[name] = behaviors
                    print(f"  Loaded {len(behaviors)} behaviors for {name}")

                # ----- Build combined behavior book -----
                combined_behaviors = MetacognitiveClient.merge_behavior_books(
                    list(per_dataset_behaviors.values())
                )
                print(f"  Combined behavior book: {len(combined_behaviors)} behaviors")

                # Save combined behavior book
                combined_path = os.path.join(self.output_dir, "combined_behavior_book.json")
                with open(combined_path, "w", encoding="utf-8") as f:
                    json.dump(combined_behaviors, f, indent=2, ensure_ascii=False)

                # ----- Build encyclopedia -----
                encyclopedia_path = os.path.join(self.output_dir, "encyclopedia.json")
                if resume_from_encyclopedia_step:
                    print("  [resume] Regenerating encyclopedia from existing behaviors...")
                    encyclopedia_path = self._generate_encyclopedia_from_behaviors(dataset_list)
                elif not os.path.exists(encyclopedia_path):
                    # Generate encyclopedia from behaviors
                    encyclopedia_path = self._generate_encyclopedia_from_behaviors(dataset_list)
                else:
                    print(f"  Using existing encyclopedia: {encyclopedia_path}")

                # Verify encyclopedia exists
                if not os.path.exists(encyclopedia_path):
                    raise FileNotFoundError(
                        f"Encyclopedia not found at {encyclopedia_path}. "
                        f"Iteration >= 2 requires an encyclopedia."
                    )

                encyclopedia_dict = self._load_encyclopedia(encyclopedia_path)
                enc_count = len(encyclopedia_dict) if isinstance(encyclopedia_dict, dict) else len(encyclopedia_dict.split('\n'))
                print(f"  Encyclopedia loaded: {enc_count} insights")

                # ----- Run three baselines for each dataset -----
                for name in dataset_list:
                    problems = all_datasets[name]
                    dataset_accuracies = {}

                    # Baseline A: Per-dataset BCI
                    print(f"\n  === Baseline A: Per-Dataset BCI for {name} ===")
                    results_a = self._run_bci_for_dataset(
                        name, problems, max_problems,
                        behaviors=per_dataset_behaviors[name],
                        iteration=iteration,
                        baseline_name="per_dataset_bci",
                    )
                    acc_a = self._compute_accuracy(results_a)
                    dataset_accuracies["per_dataset_bci"] = acc_a
                    correctness_a = [1 if r["is_correct"] else 0 for r in results_a] if results_a else []
                    self._append_summary_entry({
                        "iteration": iteration,
                        "dataset": name,
                        "baseline": "per_dataset_bci",
                        "accuracy": acc_a,
                        "num_behaviors": len(per_dataset_behaviors[name]),
                        "model": "gemini-3-pro-preview" if self.use_api else self.model_name,
                        "question_correctness": correctness_a,
                    })
                    print(f"    Per-dataset BCI accuracy: {acc_a:.2%}")

                    # Baseline B: Combined BCI
                    print(f"\n  === Baseline B: Combined BCI for {name} ===")
                    results_b = self._run_bci_for_dataset(
                        name, problems, max_problems,
                        behaviors=combined_behaviors,
                        iteration=iteration,
                        baseline_name="combined_bci",
                    )
                    acc_b = self._compute_accuracy(results_b)
                    dataset_accuracies["combined_bci"] = acc_b
                    correctness_b = [1 if r["is_correct"] else 0 for r in results_b] if results_b else []
                    self._append_summary_entry({
                        "iteration": iteration,
                        "dataset": name,
                        "baseline": "combined_bci",
                        "accuracy": acc_b,
                        "num_behaviors": len(combined_behaviors),
                        "model": "gemini-3-pro-preview" if self.use_api else self.model_name,
                        "question_correctness": correctness_b,
                    })
                    print(f"    Combined BCI accuracy: {acc_b:.2%}")

                    # Baseline C: Encyclopedia (insight library) inference
                    print(f"\n  === Baseline C: Encyclopedia for {name} ===")
                    results_c = self._run_encyclopedia_for_dataset(
                        name, problems, max_problems,
                        encyclopedia=encyclopedia_dict,
                        iteration=iteration,
                    )
                    acc_c = self._compute_accuracy(results_c)
                    dataset_accuracies["encyclopedia_bci"] = acc_c
                    correctness_c = [1 if r["is_correct"] else 0 for r in results_c] if results_c else []
                    self._append_summary_entry({
                        "iteration": iteration,
                        "dataset": name,
                        "baseline": "encyclopedia_bci",
                        "accuracy": acc_c,
                        "num_insights": enc_count,
                        "model": (
                            "gemini-3-pro-preview"
                            if self.use_api
                            else self.model_name
                        ),
                        "question_correctness": correctness_c,
                    })
                    print(f"    Encyclopedia accuracy: {acc_c:.2%}")

                    iteration_summary["accuracy_per_dataset"][name] = dataset_accuracies

            iteration_history.append(iteration_summary)

            # Print iteration summary
            print(f"\nIteration {iteration} Summary:")
            for dataset, accs in iteration_summary["accuracy_per_dataset"].items():
                if isinstance(accs, dict):
                    for baseline, acc in accs.items():
                        print(f"  - {dataset} [{baseline}]: {acc:.2%}")
                else:
                    print(f"  - {dataset}: {accs:.2%}")

        # ======================================================
        # Final summary
        # ======================================================
        final_summary = {
            "mode": "metacognitive_iterative",
            "num_iterations": self.num_iterations,
            "datasets": dataset_list,
            "iteration_history": iteration_history,
            "total_time_seconds": time.time() - start_time,
        }

        # Save final summary
        summary_path = os.path.join(self.output_dir, "iterative_summary.json")
        try:
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            else:
                existing = []
            if not isinstance(existing, list):
                existing = []
            existing.append({"final": final_summary})
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: failed to write final summary: {e}")

        print(f"\n{'=' * 80}")
        print("METACOGNITIVE REUSE PIPELINE COMPLETE")
        print(f"{'=' * 80}")
        print(f"\nResults per iteration:")
        for iter_sum in iteration_history:
            print(f"  Iteration {iter_sum['iteration']}:")
            for dataset, accs in iter_sum["accuracy_per_dataset"].items():
                if isinstance(accs, dict):
                    for baseline, acc in accs.items():
                        print(f"    - {dataset} [{baseline}]: {acc:.2%}")
                else:
                    print(f"    - {dataset}: {accs:.2%}")
        print(f"\nFinal summary saved: {summary_path}")

        return final_summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_list_arg(raw: Optional[List[str]]) -> Optional[List[str]]:
    if raw is None:
        return None
    normalized: List[str] = []
    for item in raw:
        parts = [p.strip() for p in item.split(",") if p.strip()]
        normalized.extend(parts)
    return normalized or None


def main():
    parser = argparse.ArgumentParser(
        description="Metacognitive Reuse Benchmark Pipeline: 3-phase behavior extraction + 3 BCI baselines"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["aime25"],
        help="Datasets for the pipeline (space- or comma-separated).",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Limit problems per dataset per iteration.",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name (HF).",
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="metacognitive_output",
        help="Root output directory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use-api", action="store_true", help="Use an API provider instead of HuggingFace model.")
    parser.add_argument("--api-provider", type=str, default="gemini", choices=["gemini", "openrouter"], help="Which API provider to use (default: gemini).")
    parser.add_argument("--api-key", type=str, default=None, help="API key for the chosen provider.")
    parser.add_argument(
        "--load-in-8bit",
        type=bool,
        default=False,
        help="Load model with 8-bit quantization (default: False, uses FP16 instead)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=2,
        help="Number of iterations (default: 2, minimum 2 for BCI baselines)",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=1,
        help="Iteration index to start from (default: 1)",
    )
    parser.add_argument(
        "--resume-from-encyclopedia-step",
        action="store_true",
        help=(
            "Do not recompute iteration-1 solving; load existing iteration-1 outputs, "
            "then resume from encyclopedia build in iteration >=2."
        ),
    )

    args = parser.parse_args()

    datasets = _parse_list_arg(args.datasets)

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

    pipeline = MetacognitiveBenchmarkPipeline(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
        use_api=args.use_api,
        api_key=args.api_key,
        api_provider=args.api_provider,
        num_iterations=args.num_iterations,
        load_in_8bit=args.load_in_8bit,
    )

    try:
        if not datasets:
            raise ValueError("--datasets is required")
        pipeline.run_iterative_pipeline(
            dataset_list=datasets,
            max_problems=args.max_problems,
            start_iteration=args.start_iteration,
            resume_from_encyclopedia_step=args.resume_from_encyclopedia_step,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback
        traceback.print_exc()
        print("\nExamples:")
        print("  python task_benchmark_domain_metacognitive.py --datasets aime25 --max-problems 10 --num-iterations 2")
        print("  python task_benchmark_domain_metacognitive.py --datasets gsm8k math500 --max-problems 20 --use-api")
        print("  python task_benchmark_domain_metacognitive.py --datasets aime25 aime24 --max-problems 30 --num-iterations 3")
        print("  python task_benchmark_domain_metacognitive.py --datasets aime25 --num-iterations 3 --start-iteration 2")
        print("  python task_benchmark_domain_metacognitive.py --datasets aime25 --num-iterations 2 --resume-from-encyclopedia-step")


if __name__ == "__main__":
    main()
