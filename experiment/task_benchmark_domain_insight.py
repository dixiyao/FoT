"""
Domain Insight Evaluation Pipeline (task_benchmark_domain_insight)

Evaluates how effectively a model uses encyclopedia insights when solving problems.
Only runs the first step (solution generation) from client.py with a custom instruction
that asks the model to analyze which insights it used and how they helped.

For each problem, the report captures:
  1. The predicted answer and correctness
  2. Which insights from the encyclopedia the model used
  3. How each insight helped the model's reasoning

Usage:
  python task_benchmark_domain_insight.py --datasets aime25 --max-problems 10 \
      --encyclopedia path/to/encyclopedia.json --use-api --api-provider gemini
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

from client import ChainOfThoughtReader
from math_datasets.imo_benchmark import imo_evaluator
from math_datasets.livemathbench import livemathbench_evaluator
from math_datasets.utils import extract_numbers

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

# ---------------------------------------------------------------------------
# Custom instruction for insight analysis
# ---------------------------------------------------------------------------
INSIGHT_ANALYSIS_INSTRUCTION = """\
After solving the problem, provide the following analysis sections.
IMPORTANT: You MUST identify and use at least one insight from the encyclopedia above. Do NOT say "no insights are relevant" — every problem can benefit from at least one insight. Find the most relevant insight and apply it.

## Answer:
State your final answer clearly.
## End of Answer:

## Insights Used:
List the names of insights from the provided encyclopedia that you applied to solve this problem. Use the exact insight names as they appear in the encyclopedia.
You MUST list at least one insight. Even if no insight is a perfect match, identify the most relevant one and explain how it connects to this problem.
## End of Insights Used:

## Insight Analysis:
For each insight you listed above, explain:
1. How you applied it to this specific problem (concrete steps)
2. How it helped guide your reasoning toward the answer (what would be different without it)
3. Whether it was critical or supplementary to your solution
For the most relevant insight, provide a detailed explanation of the connection between the insight's technique and this problem's solution approach.
## End of Insight Analysis:
"""


class InsightEvalPipeline:
    """Evaluate how well a model leverages encyclopedia insights during problem solving."""

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device: Optional[str] = None,
        output_dir: str = "insight_eval_output",
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
        load_in_8bit: bool = False,
        mode: str = "text",
    ):
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        self.use_api = use_api
        self.api_key = api_key
        self.api_provider = api_provider
        self.load_in_8bit = load_in_8bit
        self.mode = mode

        os.makedirs(output_dir, exist_ok=True)

        self.client: Optional[ChainOfThoughtReader] = None

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
            self.client = ChainOfThoughtReader(
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
    def _load_csv_file(self, dataset_name: str, explicit_path: Optional[str]) -> List[Dict]:
        candidate_path = explicit_path or os.path.join("math_datasets", f"{dataset_name}.csv")
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(f"CSV file for '{dataset_name}' not found at {candidate_path}")

        print(f"Loading CSV file from {candidate_path}...")
        with open(candidate_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file {candidate_path} is empty or has no header")

            fieldnames_lower = [fn.lower() for fn in reader.fieldnames]
            print(f"CSV columns: {reader.fieldnames}")

            problem_cols = ["problem", "question", "problem_text", "task", "statement", "text"]
            answer_cols = ["answer", "solution", "final_answer", "answer_text", "short answer", "short_answer", "response"]

            problem_col = None
            answer_col = None
            for col in problem_cols:
                if col in fieldnames_lower:
                    problem_col = reader.fieldnames[fieldnames_lower.index(col)]
                    break
            for col in answer_cols:
                if col in fieldnames_lower:
                    answer_col = reader.fieldnames[fieldnames_lower.index(col)]
                    break

            if not problem_col:
                raise ValueError(f"Could not find problem column in CSV. Available: {reader.fieldnames}")
            if not answer_col:
                raise ValueError(f"Could not find answer column in CSV. Available: {reader.fieldnames}")

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

            if dataset_name.startswith("gsm8k") and "####" in answer_text:
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
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {', '.join(DATASET_REGISTRY.keys())}")

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
            return self._normalize_problems(raw, dataset_name)

        if source_type == "json":
            candidate_path = path_or_hf_name or os.path.join("math_datasets", f"{dataset_name}.json")
            if not os.path.exists(candidate_path):
                raise FileNotFoundError(f"Dataset '{dataset_name}' not found at {candidate_path}")
            with open(candidate_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._normalize_problems(data, dataset_name)

        raise ValueError(f"Unknown source type: {source_type}")

    # ------------------------------------------------------------------
    # Problem formatting
    # ------------------------------------------------------------------
    def _format_problem(self, problem_data: Dict, dataset_name: str) -> Tuple[Optional[str], Optional[object]]:
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
        if dataset_name and dataset_name.startswith("gpqa"):
            from science_datasets.gpqa import gpqa_formatter
            _, ground_truth = gpqa_formatter(problem_data)
            return ground_truth
        return problem_data.get("answer") or problem_data.get("solution", "")

    # ------------------------------------------------------------------
    # Answer extraction (same as task_benchmark_domain.py)
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
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            if depth != 0:
                return None
            return text[start + len(marker): i - 1].strip()

        # Try boxed first
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

        # Try ## Answer: section
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

        # Dataset-specific extraction
        if dataset_name:
            if dataset_name in ["aime25", "aime24"] or dataset_name.startswith("imo"):
                answer_patterns = [
                    r"(?:the answer is|answer:|final answer:?)\s*\$?([^.$\n]+)\$?",
                    r"(?:therefore|thus|so),?\s+(?:the answer is)?\s*\$?([^.$\n]+)\$?",
                ]
                search_text = solution[-1000:] if len(solution) > 1000 else solution
                for pattern in answer_patterns:
                    matches = list(re.finditer(pattern, search_text, re.IGNORECASE))
                    if matches:
                        answer = matches[-1].group(1).strip()
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

        # Generic fallback
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

    # ------------------------------------------------------------------
    # Answer checking (same as task_benchmark_domain.py)
    # ------------------------------------------------------------------
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
        self, predicted: str, ground_truth: str,
        dataset_name: Optional[str] = None, problem_text: Optional[str] = None,
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
    # Insight analysis extraction from model response
    # ------------------------------------------------------------------
    def _extract_insights_used(self, response: str) -> List[str]:
        """Extract insight names from the '## Insights Used:' section."""
        match = re.search(
            r"## Insights Used:(.*?)(?:## End of Insights Used:|## Insight Analysis:)",
            response, re.DOTALL,
        )
        if not match:
            return []

        section = match.group(1).strip()
        if not section or section.lower() == "none":
            return []

        # Extract insight names: lines starting with - or *, or bold **name**
        insights = []
        for line in section.split("\n"):
            line = line.strip()
            if not line or line.lower() == "none":
                continue
            # Remove list markers
            line = re.sub(r"^[-*•]\s*", "", line).strip()
            # Extract bold names like **insight_name**
            bold_match = re.search(r"\*\*([^*]+)\*\*", line)
            if bold_match:
                insights.append(bold_match.group(1).strip())
            elif line:
                # Take the first meaningful segment (before any colon or dash explanation)
                name = re.split(r"[:\-–—]", line, maxsplit=1)[0].strip()
                if name:
                    insights.append(name)

        return insights

    def _extract_insight_analysis(self, response: str) -> str:
        """Extract the insight analysis section from the response."""
        match = re.search(
            r"## Insight Analysis:(.*?)(?:## End of Insight Analysis:|$)",
            response, re.DOTALL,
        )
        if not match:
            return ""
        return match.group(1).strip()

    # ------------------------------------------------------------------
    # Build insights section from encyclopedia
    # ------------------------------------------------------------------
    def _build_insights_section(self, encyclopedia_paths: List[str]) -> str:
        """Load encyclopedias and build the insights section for the prompt."""
        self._ensure_client()
        valid_eps = [ep for ep in encyclopedia_paths if ep and os.path.exists(ep)]
        if not valid_eps:
            return ""

        print(f"Loading {len(valid_eps)} encyclopedias for guidance...")
        self.client.load_encyclopedias(valid_eps, mode=self.mode)

        if not self.client.encyclopedia_loaded:
            return ""

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
        return insights_section

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------
    def run_eval(
        self,
        dataset_list: List[str],
        max_problems: Optional[int],
        encyclopedia_paths: Optional[List[str]] = None,
    ) -> Dict:
        """Run insight evaluation: solve problems with encyclopedia and analyze insight usage.

        For each problem:
          1. Call client._step_solution with custom instruction for insight analysis
          2. Extract answer, check correctness
          3. Extract insights used and insight analysis from the response
          4. Save per-problem report

        Args:
            dataset_list: List of datasets to evaluate
            max_problems: Max problems per dataset
            encyclopedia_paths: Paths to encyclopedia files

        Returns:
            Summary dict with accuracy and insight usage stats per dataset
        """
        self._ensure_client()

        # Build insights section
        insights_section = ""
        if encyclopedia_paths:
            insights_section = self._build_insights_section(encyclopedia_paths)
            if not insights_section:
                print("No valid encyclopedias found; proceeding without guidance")

        print(f"\n{'=' * 80}")
        print("INSIGHT EVALUATION: Solve + Analyze Insight Usage (single iteration)")
        print(f"Datasets: {', '.join(dataset_list)}")
        print(f"Max problems per dataset: {max_problems or 'all'}")
        print(f"Encyclopedia: {'yes' if insights_section else 'none'}")
        print(f"{'=' * 80}\n")

        accuracy_map = {}
        token_map = {}
        loop_map = {}
        insight_usage_map = {}

        for dataset_name in dataset_list:
            problems = self.load_dataset(dataset_name)
            worklist = problems[:max_problems] if max_problems else problems
            print(f"\nEvaluating {dataset_name} ({len(worklist)} problems)...")

            eval_dir = os.path.join(self.output_dir, dataset_name)
            os.makedirs(eval_dir, exist_ok=True)

            results = []
            number_output_tokens_list = []
            loop_count_list = []
            all_insights_used = []

            for idx, problem_data in enumerate(worklist, 1):
                problem_text, test_cases_for_eval = self._format_problem(problem_data, dataset_name)

                if not problem_text:
                    print(f"  [skip] Problem {idx} missing text")
                    continue

                print(f"  [{idx}/{len(worklist)}] {problem_text[:80]}...")

                predicted_answer = None
                is_correct = False
                insights_used = []
                insight_analysis = ""
                try:
                    # Use client._step_solution with custom instruction
                    step_result = self.client._step_solution(
                        problem_text,
                        custom_instruction=INSIGHT_ANALYSIS_INSTRUCTION,
                        insights_section=insights_section,
                    )

                    solution = step_result["response"]
                    token_info = step_result.get("token_info", {})
                    number_output_tokens = token_info.get("output_tokens", 0)
                    number_output_tokens_list.append(number_output_tokens)

                    loop_count = self._count_consecutive_sentence_loops(solution)
                    loop_count_list.append(loop_count)

                    # Extract answer
                    predicted_answer = self._extract_answer_from_solution(
                        solution, dataset_name, problem_data
                    )

                    # Extract insight usage from the structured response
                    insights_used = self._extract_insights_used(solution)
                    insight_analysis = self._extract_insight_analysis(solution)
                    all_insights_used.append(insights_used)

                    # Check correctness
                    if test_cases_for_eval:
                        is_correct = self._check_answer_match(
                            solution, test_cases_for_eval, dataset_name, problem_text
                        )
                        status = "+" if is_correct else "x"
                        print(f"    {status} Code execution test results")
                    else:
                        ground_truth = self._get_ground_truth(problem_data, dataset_name)
                        if predicted_answer:
                            is_correct = self._check_answer_match(
                                predicted_answer, ground_truth, dataset_name, problem_text
                            )
                        status = "+" if is_correct else "x"
                        print(f"    {status} Predicted: {predicted_answer or 'N/A'} | GT: {ground_truth or 'N/A'}")

                    print(f"    Insights used: {insights_used if insights_used else 'None'}")

                    # Save per-problem report
                    output_data = {
                        "problem": problem_text,
                        "problem_id": problem_data.get("id", idx),
                        "solution": solution,
                        "predicted_answer": predicted_answer,
                        "is_correct": is_correct,
                        "insights_used": insights_used,
                        "insight_analysis": insight_analysis,
                        "number_output_tokens": number_output_tokens,
                        "loop_count": loop_count,
                    }

                    output_path = os.path.join(eval_dir, f"problem_{idx:04d}.json")
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, indent=2, ensure_ascii=False)

                    results.append({
                        "is_correct": is_correct,
                        "number_output_tokens": number_output_tokens,
                        "loop_count": loop_count,
                        "num_insights_used": len(insights_used),
                    })
                    time.sleep(0.5)

                    # Reset client reasoning_steps for next problem
                    self.client.reasoning_steps = []

                except Exception as exc:
                    print(f"    Error processing problem {idx}: {exc}")

            # Summarize dataset results
            if results:
                num_correct = sum(1 for r in results if r["is_correct"])
                accuracy = num_correct / len(results)
                avg_tokens = sum(r["number_output_tokens"] for r in results) / len(results)
                total_loops = sum(r["loop_count"] for r in results)
                avg_insights = sum(r["num_insights_used"] for r in results) / len(results)
                problems_using_insights = sum(1 for r in results if r["num_insights_used"] > 0)
            else:
                accuracy = 0.0
                avg_tokens = 0.0
                total_loops = 0
                avg_insights = 0.0
                problems_using_insights = 0

            accuracy_map[dataset_name] = accuracy
            token_map[dataset_name] = avg_tokens
            loop_map[dataset_name] = total_loops
            insight_usage_map[dataset_name] = {
                "avg_insights_per_problem": avg_insights,
                "problems_using_insights": problems_using_insights,
                "total_problems": len(results),
                "insight_usage_rate": problems_using_insights / len(results) if results else 0.0,
            }

            # Count frequency of each insight
            insight_frequency: Dict[str, int] = {}
            for used_list in all_insights_used:
                for name in used_list:
                    insight_frequency[name] = insight_frequency.get(name, 0) + 1

            insight_usage_map[dataset_name]["insight_frequency"] = dict(
                sorted(insight_frequency.items(), key=lambda x: x[1], reverse=True)
            )

            print(f"\n  {dataset_name}: Accuracy={accuracy:.2%}, Avg Tokens={avg_tokens:.1f}, "
                  f"Loops={total_loops}, Avg Insights Used={avg_insights:.1f}, "
                  f"Insight Usage Rate={problems_using_insights}/{len(results)}")

        # Save overall summary
        summary = {
            "mode": "insight_eval",
            "datasets": dataset_list,
            "accuracy_per_dataset": accuracy_map,
            "avg_tokens_per_dataset": token_map,
            "loop_count_per_dataset": loop_map,
            "insight_usage_per_dataset": insight_usage_map,
            "encyclopedia_used": [ep for ep in (encyclopedia_paths or []) if ep and os.path.exists(ep)],
            "model": "gemini-3-pro-preview" if self.use_api else self.model_name,
        }

        summary_path = os.path.join(self.output_dir, "insight_eval_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 80}")
        print("INSIGHT EVALUATION COMPLETE")
        print(f"{'=' * 80}")
        for dataset, acc in accuracy_map.items():
            usage = insight_usage_map[dataset]
            print(f"  {dataset}: Accuracy={acc:.2%}, "
                  f"Insight Usage={usage['problems_using_insights']}/{usage['total_problems']} "
                  f"({usage['insight_usage_rate']:.0%}), "
                  f"Avg Insights/Problem={usage['avg_insights_per_problem']:.1f}")
            if usage.get("insight_frequency"):
                top_insights = list(usage["insight_frequency"].items())[:5]
                print(f"    Top insights: {', '.join(f'{n}({c})' for n, c in top_insights)}")
        print(f"\nSummary saved: {summary_path}")

        return summary


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
        description="Insight Evaluation Pipeline: solve problems with encyclopedia and analyze insight usage"
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
        "-m", "--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name (HF).",
    )
    parser.add_argument(
        "-d", "--device", type=str, default=None,
        help="Device to use (cuda or cpu).",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="insight_eval_output",
        help="Root output directory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use-api", action="store_true", help="Use an API provider instead of HuggingFace model.")
    parser.add_argument("--api-provider", type=str, default="gemini", choices=["gemini", "openrouter"], help="Which API provider to use (default: gemini).")
    parser.add_argument("--api-key", type=str, default=None, help="API key for the chosen provider.")
    parser.add_argument(
        "--load-in-8bit", type=bool, default=False,
        help="Load model with 8-bit quantization (default: False)",
    )
    parser.add_argument(
        "--mode", type=str, default="text", choices=["normal", "text"],
        help="Encyclopedia loading mode (default: text)",
    )
    parser.add_argument(
        "--encyclopedia", type=str, nargs="*", default=None,
        help="Path(s) to encyclopedia file(s) to use for guidance.",
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

    pipeline = InsightEvalPipeline(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
        use_api=args.use_api,
        api_key=args.api_key,
        api_provider=args.api_provider,
        load_in_8bit=args.load_in_8bit,
        mode=args.mode,
    )

    try:
        if not datasets:
            raise ValueError("--datasets is required")
        pipeline.run_eval(
            dataset_list=datasets,
            max_problems=args.max_problems,
            encyclopedia_paths=args.encyclopedia,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback
        traceback.print_exc()
        print("\nExamples:")
        print("  python task_benchmark_domain_insight.py --datasets aime25 --max-problems 10 "
              "--encyclopedia path/to/encyclopedia.json --use-api")
        print("  python task_benchmark_domain_insight.py --datasets math500 gsm8k --max-problems 20 "
              "--encyclopedia enc1.json enc2.json --use-api")


if __name__ == "__main__":
    main()
