"""
Multi-dataset Benchmark Problem Solving Pipeline (task_benchmark_domain)
- Supports running STEP 1 (insight extraction) across multiple datasets.
- Allows choosing which insight sets to aggregate in STEP 2.
- Evaluates multiple target datasets sequentially in STEP 3 using the shared encyclopedia.

Usage is similar to math_pipeline.py but adds list-style arguments.

## Supported Datasets

### Hugging Face Datasets (via 🤗 datasets library):
- gsm8k, gsm8k_train: Grade School Math (8K examples)
- aime24, aime25: American Invitational Mathematics Examination
- math500, math1000: Competition math problems
- gpqa, gpqa_diamond: Graduate-level science problems (GPQA benchmark)

### Local Datasets:
- CSV or JSON files in math_datasets/ directory

### IMOBench (International Mathematical Olympiad Benchmark)
From: https://github.com/google-deepmind/superhuman/tree/main/imobench
See: https://imobench.github.io

IMOBench consists of three specialized benchmarks:

1. **IMO-AnswerBench** (400 problems)
   - Short-answer problems with verifiable final answers
   - Categories: Algebra, Combinatorics, Geometry, Number Theory
   - Difficulty: pre-IMO, IMO-Easy, IMO-Medium, IMO-Hard
   - CSV columns: problem/question, answer/solution, id, difficulty
   - Evaluation: Symbolic comparison with algebraic normalization

2. **IMO-ProofBench** (60 problems)
   - Proof-writing evaluation (not just final answers)
   - Requires human expert grading (0-7 scale)
   - Can use ProofAutoGrader with Gemini 2.5 Pro for automatic evaluation
   - Correlation with human grading: 0.96 (basic), 0.93 (advanced)
   - Not automatically evaluated in this script - use external graders

3. **IMO-GradingBench** (1000 examples)
   - Dataset for evaluating grading capability
   - Problem + proposed solution + human grade (0-7)
   - Classification labels: Correct (7), Almost (6), Partial (1), Incorrect (0)
   - CSV columns: problem, solution, grade, grade_label

### Answer Verification for IMOBench:
- Numeric: Direct numeric comparison with tolerance 1e-6
- Symbolic: Algebraic equivalence checking (normalized forms)
- String: Case-insensitive exact matching
- Partial: Substring matching for multi-answer problems
- Unit handling: Removes common units (degrees, radians, cm, m, etc.)

For proof-based evaluation on IMO-ProofBench, consider:
- Using Gemini 2.5 Pro's ProofAutoGrader (available in superhuman repo)
- Implementing LLM-based grading with reference solutions
- Human expert evaluation for rigorous assessment
"""

import argparse
import csv
import json
import math
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
from client_metacognitive import MetacognitiveClient
from client_trt import TRTClient
from client_hyperagents import HyperAgentsClient
from client_evolveprompt import EvolvePromptClient
from client_ace import ACEClient
from math_datasets.imo_benchmark import imo_evaluator
from math_datasets.livemathbench import livemathbench_evaluator
from math_datasets.utils import extract_numbers
from server import InsightAggregationServer
from server_text import TextBasedInsightAggregationServer

# Dataset registry: (source, path_or_hf_name, data_dir_or_col_map, split_or_none)
# - source="hf": use Hugging Face with (hf_name, data_dir, split)
# - source="json": load JSON from path
# - source="csv": load CSV from path with optional column mapping dict
#
# IMOBench Benchmarks (https://imobench.github.io/):
# - IMO-AnswerBench: 400 short-answer problems (CSV)
#   Columns: problem/question, answer/solution, id, difficulty
#   Evaluation: Symbolic comparison with unit normalization
#
# - IMO-ProofBench: 60 proof-based problems (not directly scored here)
#   Requires: ProofAutoGrader (Gemini 2.5 Pro) or human evaluation
#   Grading: 0-7 scale, ~high correlation with human experts (0.96)
#
# - IMO-GradingBench: 1000 grading examples (CSV)
#   Columns: problem, solution, grade (0-7), grade_label (Correct/Almost/Partial/Incorrect)
#   Use for training automatic graders
# LiveMathBench (Live Mathematical Reasoning Benchmark) — From OpenCompass
#   Configs: v202412_AMC_en, v202412_CCEE_en, v202412_CNMO_en, v202412_WLPMC_en, v202412_hard_en, v202505_hard_en
#   Reference: https://huggingface.co/datasets/opencompass/LiveMathBench
#   Columns: question, answer, question_type
# GPQA (Graduate-Level Google-Proof Q&A Benchmark) — Graduate-level science problems
#   Reference: https://huggingface.co/datasets/Idavidrein/gpqa
#   Diamond variant: https://huggingface.co/datasets/fingertap/GPQA-Diamond
#   Contains graduate-level questions in physics, chemistry, and biology
DATASET_REGISTRY: Dict[str, Tuple[str, str, Optional[str], Optional[str]]] = {
    "gsm8k": ("hf", "openai/gsm8k", "main", "test"),
    "gsm8k_train": ("hf", "openai/gsm8k", "main", "train"),
    "aime25": ("hf", "math-ai/aime25", None, "test"),
    "aime24": ("hf", "math-ai/aime24", None, "test"),
    "math500": ("hf", "HuggingFaceH4/MATH-500", None, "test"),
    "math1000": ("hf", "hendrycks/competition_math", None, "test"),
    # GPQA datasets (Graduate-level science problems)
    "gpqa": ("hf", "Idavidrein/gpqa", "gpqa_main", "train"),
    "gpqa_diamond": ("hf", "Idavidrein/gpqa", "gpqa_diamond", "train"),
    # LiveCodeBench datasets (Code generation)
    # Using bzantium/livecodebench (Parquet-based clone, works with datasets 3.0+)
    "livecodebench": ("hf", "bzantium/livecodebench", "release_v6", "test"),
    "livecodebench_lite": ("hf", "bzantium/livecodebench", "v6", "test"),  # Latest version increment
    # LiveMathBench datasets (OpenCompass)
    "livemathbench_amc": ("hf", "opencompass/LiveMathBench", "v202412_AMC_en", "test"),
    "livemathbench_ccee": (
        "hf",
        "opencompass/LiveMathBench",
        "v202412_CCEE_en",
        "test",
    ),
    "livemathbench_cnmo": (
        "hf",
        "opencompass/LiveMathBench",
        "v202412_CNMO_en",
        "test",
    ),
    "livemathbench_wlpmc": (
        "hf",
        "opencompass/LiveMathBench",
        "v202412_WLPMC_en",
        "test",
    ),
    "livemathbench_hard_2024": (
        "hf",
        "opencompass/LiveMathBench",
        "v202412_hard_en",
        "test",
    ),
    "livemathbench_hard_2025": (
        "hf",
        "opencompass/LiveMathBench",
        "v202505_hard_en",
        "test",
    ),
    # IMO benchmark (IMOBench) — CSV files from https://github.com/google-deepmind/superhuman/tree/main/imobench
    # Download from: https://github.com/google-deepmind/superhuman/tree/main/imobench
    "imo_answerbench": ("csv", "math_datasets/answerbench.csv", None, None),
    "imo_answerbench_algebra": ("csv", "math_datasets/imo_algebra.csv", None, None),
    "imo_answerbench_geometry": ("csv", "math_datasets/imo_geometry.csv", None, None),
    "imo_answerbench_number_theory": (
        "csv",
        "math_datasets/imo_number_theory.csv",
        None,
        None,
    ),
    # IMO-ProofBench: Requires external evaluation (ProofAutoGrader or human experts)
    "imo_proofbench": ("csv", "math_datasets/proofbench.csv", None, None),
    # IMO-GradingBench: For training/evaluating automatic graders
    "imo_gradingbench": ("csv", "math_datasets/gradingbench.csv", None, None),
}


_CLIENT_CHOICES = ["default", "metacognitive", "trt", "hyperagents", "evolveprompt", "ace"]


def _build_client(
    client_type: str,
    model_name: str,
    device: Optional[str],
    use_api: bool,
    api_key: Optional[str],
    api_provider: str,
    output_dir: str,
    load_in_8bit: bool,
):
    """Factory: return the requested client instance."""
    common = dict(
        model_name=model_name,
        device=device,
        use_api=use_api,
        api_key=api_key,
        api_provider=api_provider,
        output_dir=output_dir,
        load_in_8bit=load_in_8bit,
    )
    if client_type == "metacognitive":
        return MetacognitiveClient(**common)
    elif client_type == "trt":
        return TRTClient(**common)
    elif client_type == "hyperagents":
        return HyperAgentsClient(**common)
    elif client_type == "evolveprompt":
        return EvolvePromptClient(**common)
    elif client_type == "ace":
        return ACEClient(**common)
    else:  # "default"
        return ChainOfThoughtReader(
            model_name=model_name,
            device=device,
            use_api=use_api,
            api_key=api_key,
            api_provider=api_provider,
            load_in_8bit=load_in_8bit,
        )


class BenchmarkDomainPipeline:
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device: Optional[str] = None,
        output_dir: str = "math_output",
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
        mode: str = "text",
        num_iterations: int = 3,
        load_in_8bit: bool = False,
        client_type: str = "default",
    ):
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        self.use_api = use_api
        self.api_key = api_key
        self.api_provider = api_provider
        self.mode = mode  # "normal" uses server.py, "text" uses server_text.py
        self.iterative = True  # Always True
        self.num_iterations = num_iterations
        self.load_in_8bit = load_in_8bit
        self.client_type = client_type

        os.makedirs(output_dir, exist_ok=True)

        self.client = None
        self.server: Optional[InsightAggregationServer] = None
        self.server_text: Optional[TextBasedInsightAggregationServer] = None
        self.encyclopedia_loaded = False

    def _count_consecutive_sentence_loops(self, text: str) -> int:
        """Count repeated consecutive sentences in the generated text.

        A loop is counted when a sentence is identical to the immediately
        preceding sentence. Multiple consecutive repeats are counted
        individually (e.g., A A A B → loops=2).
        """
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

    # ------------------------------------------------------------------
    # Dataset loading helpers
    # ------------------------------------------------------------------
    def _load_local_json(
        self, dataset_name: str, explicit_path: Optional[str]
    ) -> List[Dict]:
        """Load a dataset from a local JSON file."""
        candidate_path = explicit_path or os.path.join(
            "math_datasets", f"{dataset_name}.json"
        )
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' not found. Provide {candidate_path} or update DATASET_REGISTRY."
            )
        with open(candidate_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _load_csv_file(
        self, dataset_name: str, explicit_path: Optional[str]
    ) -> List[Dict]:
        """Load a dataset from a CSV file.

        Tries to infer column names from common patterns:
        - Problem: problem, question, problem_text, task, statement
        - Answer: answer, solution, final_answer, answer_text
        - ID: id, problem_id, num, number
        """
        candidate_path = explicit_path or os.path.join(
            "math_datasets", f"{dataset_name}.csv"
        )
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(
                f"CSV file for dataset '{dataset_name}' not found at {candidate_path}"
            )

        print(f"Loading CSV file from {candidate_path}...")
        with open(candidate_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file {candidate_path} is empty or has no header")

            fieldnames_lower = [fn.lower() for fn in reader.fieldnames]
            print(f"CSV columns: {reader.fieldnames}")

            # Map CSV columns to standard schema
            problem_cols = [
                "problem",
                "question",
                "problem_text",
                "task",
                "statement",
                "text",
            ]
            # Note: For gradingbench, "response" is the student answer being graded
            # For answerbench/proofbench, "short answer" or "solution" is the correct answer
            answer_cols = [
                "answer",
                "solution",
                "final_answer",
                "answer_text",
                "short answer",
                "short_answer",
                "response",  # For gradingbench
            ]
            id_cols = [
                "id",
                "problem_id",
                "problem id",
                "grading_id",
                "grading id",
                "num",
                "number",
                "idx",
            ]

            problem_col = None
            answer_col = None
            id_col = None  # Reserved for future ID column mapping

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
                    f"Could not find problem column in CSV. Available columns: {reader.fieldnames}. "
                    f"Expected one of: {problem_cols}"
                )
            if not answer_col:
                raise ValueError(
                    f"Could not find answer column in CSV. Available columns: {reader.fieldnames}. "
                    f"Expected one of: {answer_cols}"
                )

            data = []
            for row_idx, row in enumerate(reader, 1):
                data.append(row)

            print(f"Loaded {len(data)} rows from CSV")
            return data

    def _normalize_problems(
        self, raw_problems: List[Dict], dataset_name: str
    ) -> List[Dict]:
        """Ensure a consistent schema across sources."""

        # Helper to find field from problem dict with multiple possible names
        def get_field(obj: Dict, candidates: List[str], default: str = "") -> str:
            for candidate in candidates:
                for key in obj.keys():
                    if key.lower() == candidate.lower():
                        val = obj[key]
                        return str(val) if val is not None else default
            return default

        normalized = []
        for idx, problem in enumerate(raw_problems):
            # Try various column name combinations (case-insensitive)
            problem_text = get_field(
                problem,
                ["problem", "question", "problem_text", "task", "statement", "text"],
            )
            answer_text = get_field(
                problem, ["answer", "solution", "final_answer", "answer_text"]
            )
            id_val = get_field(
                problem, ["id", "problem_id", "num", "number", "idx"], str(idx + 1)
            )

            normalized_problem = {
                "id": int(id_val) if id_val.isdigit() else id_val,
                "problem": problem_text,
                "question": problem_text,  # Keep both for compatibility
                "solution": get_field(problem, ["solution", "step_by_step"]),
                "answer": answer_text,
            }

            # GSM8K stores answer as "solution #### answer"; split when present.
            if dataset_name.startswith("gsm8k"):
                if "####" in answer_text:
                    parts = answer_text.split("####")
                    normalized_problem["solution"] = parts[0].strip()
                    normalized_problem["answer"] = parts[-1].strip()

            # Preserve all other fields from original
            for key, value in problem.items():
                if key not in normalized_problem:
                    normalized_problem[key] = value
            normalized.append(normalized_problem)
        return normalized

    def load_math_dataset(self, dataset_name: str) -> List[Dict]:
        """Load a dataset from Hugging Face, CSV, or JSON file."""
        # Registry lookup
        entry = DATASET_REGISTRY.get(dataset_name)
        if not entry:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available datasets: {', '.join(DATASET_REGISTRY.keys())}"
            )

        source_type, path_or_hf_name, data_dir, split = entry

        # Attempt Hugging Face
        if source_type == "hf":
            if load_dataset is None:
                raise ImportError(
                    "datasets library is required. Install with: pip install datasets"
                )
            print(
                f"Loading dataset '{dataset_name}' from Hugging Face ({path_or_hf_name}, split={split})..."
            )
            if data_dir and split:
                ds = load_dataset(path_or_hf_name, name=data_dir, split=split)
            else:
                ds = load_dataset(path_or_hf_name, split=split)

            raw = []
            for i, item in enumerate(ds):
                if dataset_name == "math1000" and i >= 1000:
                    break

                # For GPQA datasets, preserve all original fields
                if dataset_name and dataset_name.startswith("gpqa"):
                    # Keep all original fields for GPQA (Question, Correct Answer, Incorrect Answer 1/2/3)
                    raw_item = dict(item)
                    raw_item["id"] = item.get("id", i + 1)
                    raw.append(raw_item)
                # For LiveCodeBench datasets, preserve all original fields
                elif dataset_name and dataset_name.startswith("livecodebench"):
                    # Keep all original fields for LiveCodeBench (question_content, test_cases, etc.)
                    raw_item = dict(item)
                    raw_item["id"] = item.get("question_id", item.get("id", i + 1))
                    raw.append(raw_item)
                else:
                    # Extract fields with fallbacks for different dataset formats
                    # Most datasets (AIME, MATH500, GSM8K): use "problem" field
                    # LiveMathBench: uses "question" field
                    problem_text = item.get("problem") or item.get("question", "")
                    solution = item.get("solution", "")
                    answer = item.get("answer", "")

                    # Special handling for GSM8K format (solution contains "#### answer")
                    if dataset_name == "math1000" and "####" in solution:
                        answer = solution.split("####")[-1].strip()

                    raw.append(
                        {
                            "id": item.get("id", i + 1),
                            "problem": problem_text,
                            "question": problem_text,
                            "solution": solution or answer or "",
                            "answer": answer,
                        }
                    )
            print(f"Loaded {len(raw)} problems from Hugging Face")
            # Skip normalization for GPQA and LiveCodeBench datasets to preserve original field structure
            if dataset_name and (dataset_name.startswith("gpqa") or dataset_name.startswith("livecodebench")):
                return raw
            return self._normalize_problems(raw, dataset_name)

        # Attempt CSV file
        if source_type == "csv":
            raw = self._load_csv_file(dataset_name, path_or_hf_name)
            print(f"Loaded {len(raw)} problems from CSV for '{dataset_name}'")
            return self._normalize_problems(raw, dataset_name)

        # Attempt JSON file
        if source_type == "json":
            raw = self._load_local_json(dataset_name, path_or_hf_name)
            print(f"Loaded {len(raw)} problems from JSON for '{dataset_name}'")
            return self._normalize_problems(raw, dataset_name)

        raise ValueError(f"Unknown source type: {source_type}")

    # ------------------------------------------------------------------
    # STEP 1: Insight extraction across multiple datasets
    # ------------------------------------------------------------------
    def _ensure_client(self):
        if self.client is None:
            self.client = _build_client(
                client_type=self.client_type,
                model_name=self.model_name,
                device=self.device,
                use_api=self.use_api,
                api_key=self.api_key,
                api_provider=self.api_provider,
                output_dir=self.output_dir,
                load_in_8bit=self.load_in_8bit,
            )
            print(f"[Pipeline] Using client: {self.client_type} ({type(self.client).__name__})")

    def _extract_insights_for_dataset(
        self,
        dataset_name: str,
        problems: List[Dict],
        max_problems: Optional[int],
        encyclopedia_paths: Optional[List[str]] = None,
        iteration: int = 0,
    ) -> Tuple[str, List[Dict]]:
        """Extract insights from dataset, optionally solving with encyclopedia first.

        Args:
            dataset_name: Name of dataset
            problems: List of problems
            max_problems: Max problems to process
            encyclopedia_path: If provided, solve with encyclopedia before extracting insights
            iteration: Current iteration number (for logging)

        Returns:
            Tuple of (insights_dir, results_list)
        """
        self._ensure_client()
        insights_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(insights_dir, exist_ok=True)

        worklist = problems[:max_problems] if max_problems else problems
        print(
            f"\nIteration {iteration}: Extracting insights for {dataset_name} ({len(worklist)} problems)..."
        )

        # Load encyclopedias once at the start of this dataset iteration
        insights_section = ""
        if encyclopedia_paths:
            valid_eps = [ep for ep in encyclopedia_paths if ep and os.path.exists(ep)]
            if valid_eps:
                print(f"  Loading {len(valid_eps)} encyclopedias for guidance...")
                self.client.load_encyclopedias(valid_eps, mode=self.mode)

                # Generate insights section once to reuse for all problems
                if self.client.encyclopedia_loaded:
                    if self.client.encyclopedia_dict:
                        # Text mode: Format from dictionary
                        insights_list = []
                        for (
                            insight_name,
                            insight_desc,
                        ) in self.client.encyclopedia_dict.items():
                            insights_list.append(f"**{insight_name}**:\n{insight_desc}")
                        insights_text = "\n\n".join(insights_list)
                    else:
                        # Normal mode: Use raw encyclopedia text
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
        loop_count_list = []

        for idx, problem_data in enumerate(worklist, 1):
            # Use dataset-specific formatter if available
            problem_text = None
            test_cases_for_eval = None  # Special for code generation datasets
            if dataset_name == "aime25":
                from math_datasets.aime25 import aime25_formatter

                problem_text, _ = aime25_formatter(problem_data)
            elif dataset_name == "aime24":
                # AIME24 uses same format as AIME25
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
                # Fallback: extract raw problem text
                problem_text = problem_data.get("problem") or problem_data.get(
                    "question", ""
                )

            if not problem_text:
                print(f"  [skip] Problem {idx} missing text")
                continue

            print(f"  [{idx}/{len(worklist)}] {problem_text[:80]}...")

            # Extract solution, reflection, and insights in one call
            predicted_answer = None
            is_correct = False
            try:
                result = self.client.solve_problem(
                    task=problem_text,
                    insights_section=insights_section,
                )

                # Extract solution first
                solution = result.get("solution", "")

                # Extract output tokens from Step 1 (Solution Generation)
                number_output_tokens = 0
                token_info = result.get("token_info", {})
                number_output_tokens = token_info.get("output_tokens", 0)
                number_output_tokens_list.append(number_output_tokens)

                # Loop detection: count repeated consecutive sentences in Step 1 solution
                loop_count = self._count_consecutive_sentence_loops(solution)
                loop_count_list.append(loop_count)

                # Extract answer from solution using dataset-specific extractors
                predicted_answer = self._extract_answer_from_solution(
                    solution, dataset_name, problem_data
                )

                # Get extracted insights
                insight_book = result.get("insight_book", {})
                if not insight_book:
                    print("    No insights extracted")
                    continue

                # Filter out fallback insights
                insight_book = {
                    k: v
                    for k, v in insight_book.items()
                    if not k.startswith("insight_fallback")
                }
                if not insight_book:
                    print("    No insights extracted")
                    continue

                # Check answer correctness (pass problem_text for Gemini grading)
                # For code generation datasets, use test_cases; for others, use ground_truth
                if test_cases_for_eval:
                    # Code generation dataset (e.g., LiveCodeBench)
                    is_correct = self._check_answer_match(
                        solution, test_cases_for_eval, dataset_name, problem_text
                    )
                    status = "✓" if is_correct else "✗"
                    print(f"    {status} Code execution test results")
                else:
                    # Standard dataset
                    # For GPQA datasets, get ground truth from formatter
                    if dataset_name and dataset_name.startswith("gpqa"):
                        from science_datasets.gpqa import gpqa_formatter
                        _, ground_truth = gpqa_formatter(problem_data)
                    else:
                        ground_truth = problem_data.get("answer") or problem_data.get(
                            "solution", ""
                        )

                    if predicted_answer:
                        is_correct = self._check_answer_match(
                            predicted_answer, ground_truth, dataset_name, problem_text
                        )

                    status = "✓" if is_correct else "✗"
                    print(
                        f"    {status} Predicted: {predicted_answer if predicted_answer else 'N/A'} | GT: {ground_truth if ground_truth else 'N/A'}"
                    )

                # Save insights only
                output_data = {
                    "problem": problem_text,
                    "problem_id": problem_data.get("id", idx),
                    "insight_book": insight_book,
                    "iteration": iteration,
                    "is_correct": is_correct,
                    "number_output_tokens": number_output_tokens,
                    "loop_count": loop_count,
                }

                output_path = os.path.join(insights_dir, f"problem_{idx:04d}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

                # Track for accuracy calculation
                results.append(
                    {
                        "is_correct": is_correct,
                        "number_output_tokens": number_output_tokens,
                        "loop_count": loop_count,
                    }
                )
                time.sleep(0.5)
            except Exception as exc:  # noqa: BLE001
                print(f"    Error processing problem {idx}: {exc}")

        # Calculate and log average output tokens
        if number_output_tokens_list:
            avg_number_output_tokens = sum(number_output_tokens_list) / len(
                number_output_tokens_list
            )
            print(
                f"\n  Dataset '{dataset_name}' - Average Output Tokens: {avg_number_output_tokens:.1f}"
            )

        if loop_count_list:
            total_loop_count = sum(loop_count_list)
            print(f"  Dataset '{dataset_name}' - Total Loop Count: {total_loop_count}")

        return insights_dir, results

    def learn_insights_from_datasets(
        self,
        dataset_names: List[str],
        max_problems: Optional[int],
        encyclopedia_paths: Optional[List[str]] = None,
        iteration: int = 0,
    ) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Learn insights from datasets, optionally solving with encyclopedia first.

        Returns:
            Tuple of (insights_map, accuracy_map) where accuracy_map has dataset -> accuracy
        """
        if not dataset_names:
            raise ValueError("Provide at least one dataset for STEP 1.")

        insights_map: Dict[str, str] = {}
        accuracy_map: Dict[str, float] = {}
        token_map: Dict[str, float] = {}
        loop_map: Dict[str, float] = {}

        # Helper to append per-dataset entry to iterative_summary.json immediately
        summary_file = os.path.join(self.output_dir, "iterative_summary.json")

        for name in dataset_names:
            problems = self.load_math_dataset(name)
            insights_dir, results = self._extract_insights_for_dataset(
                name, problems, max_problems, encyclopedia_paths, iteration
            )
            insights_map[name] = insights_dir

            # Calculate accuracy and average output tokens for this dataset
            if results:
                num_correct = sum(1 for r in results if r["is_correct"])
                accuracy = num_correct / len(results)
                accuracy_map[name] = accuracy

                # Calculate average output tokens
                number_output_tokens_list = [
                    r.get("number_output_tokens", 0) for r in results
                ]
                if number_output_tokens_list:
                    avg_tokens = sum(number_output_tokens_list) / len(
                        number_output_tokens_list
                    )
                    token_map[name] = avg_tokens
                else:
                    token_map[name] = 0.0

                # Calculate total loop count
                loop_counts = [r.get("loop_count", 0) for r in results]
                loop_map[name] = sum(loop_counts) if loop_counts else 0.0
            else:
                accuracy_map[name] = 0.0
                token_map[name] = 0.0
                loop_map[name] = 0.0

            # Build per-question correctness list
            question_correctness = [1 if r["is_correct"] else 0 for r in results] if results else []

            # Append per-dataset summary entry immediately
            entry = {
                "iteration": iteration,
                "dataset": name,
                "accuracy": accuracy_map[name],
                "model": (
                    "gemini-3-pro-preview" if self.use_api else self.model_name
                ),
                "encyclopedia_used": [
                    ep for ep in (encyclopedia_paths or []) if ep and os.path.exists(ep)
                ],
                "average_output_tokens": token_map.get(name, 0.0),
                "total_loop_count": loop_map.get(name, 0.0),
                "question_correctness": question_correctness,
            }
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
                print(f"  Warning: failed to append iterative summary: {e}")

        print("\nFinished STEP 1 across datasets:")
        for name, path in insights_map.items():
            print(
                f"  - {name}: {path} (Accuracy: {accuracy_map[name]:.2%}, Avg Output Tokens: {token_map[name]:.1f})"
            )

        return insights_map, accuracy_map

    # ------------------------------------------------------------------
    # STEP 2: Aggregate chosen insights into one encyclopedia
    # ------------------------------------------------------------------
    def aggregate_insights(
        self, insight_sets: List[str], r1: float, r2: float
    ) -> Dict[str, str]:
        if not insight_sets:
            raise ValueError("Provide at least one dataset to aggregate in STEP 2.")
        # Build an encyclopedia per dataset folder
        print("\nAggregating insights per dataset:")
        per_dataset_encyclopedias: Dict[str, str] = {}
        for name in insight_sets:
            insights_dir = os.path.join(self.output_dir, name)
            if not os.path.isdir(insights_dir):
                raise FileNotFoundError(
                    f"Insights directory not found for {name}: {insights_dir}"
                )
            dataset_files = [
                os.path.join(insights_dir, f)
                for f in os.listdir(insights_dir)
                if f.endswith(".json") and f.startswith("problem_")
            ]
            dataset_files = sorted(dataset_files)
            if not dataset_files:
                print(f"  - {name}: no insight JSON files found; skipping")
                continue

            print(f"  - {name} ({len(dataset_files)} files)")
            if self.mode == "text":
                self.server_text = TextBasedInsightAggregationServer(
                    model_name=self.model_name,
                    device=self.device,
                    input_dirs=[self.output_dir],
                    use_api=self.use_api,
                    api_key=self.api_key,
                    api_provider=self.api_provider,
                )
                result = self.server_text.aggregate_and_build_encyclopedia(
                    json_files=dataset_files, output_dir=insights_dir
                )
                self.server_text.save_results(result, output_dir=insights_dir)
                encyclopedia_path = os.path.join(insights_dir, "encyclopedia.json")
            else:
                self.server = InsightAggregationServer(
                    model_name=self.model_name,
                    device=self.device,
                    input_dir=self.output_dir,
                    use_api=self.use_api,
                    api_key=self.api_key,
                    api_provider=self.api_provider,
                )
                result = self.server.aggregate_and_build_encyclopedia(
                    json_files=dataset_files, r1=r1, r2=r2, output_dir=insights_dir
                )
                self.server.save_results(result, output_dir=insights_dir)
                encyclopedia_path = os.path.join(insights_dir, "encyclopedia.txt")

            print(f"    Encyclopedia saved to {encyclopedia_path}")
            per_dataset_encyclopedias[name] = encyclopedia_path

        return per_dataset_encyclopedias

    def generate_combined_encyclopedia(
        self, dataset_list: List[str], r1: float = 0.95, r2: float = 0.4
    ) -> Optional[str]:
        """Generate a combined encyclopedia from all problem_*.json files across all datasets.

        This method collects all skills from all datasets' problem_*.json files and
        generates a single encyclopedia_all.json saved under output_dir.

        Args:
            dataset_list: List of dataset names to collect skills from
            r1: First threshold for insight aggregation (default: 0.95)
            r2: Second threshold for insight aggregation (default: 0.4)

        Returns:
            Path to the combined encyclopedia (encyclopedia_all.json) or None if failed
        """
        print("\n" + "=" * 80)
        print("Generating Combined Encyclopedia from All Datasets")
        print("=" * 80)

        # Collect all problem_*.json files from all datasets
        all_json_files = []
        for name in dataset_list:
            insights_dir = os.path.join(self.output_dir, name)
            if not os.path.isdir(insights_dir):
                print(f"  Warning: Directory not found for {name}: {insights_dir}")
                continue
            dataset_files = [
                os.path.join(insights_dir, f)
                for f in os.listdir(insights_dir)
                if f.endswith(".json") and f.startswith("problem_")
            ]
            dataset_files = sorted(dataset_files)
            if dataset_files:
                all_json_files.extend(dataset_files)
                print(f"  - {name}: {len(dataset_files)} problem files")

        if not all_json_files:
            print("  No problem files found across all datasets!")
            return None

        print(f"\nTotal problem files to aggregate: {len(all_json_files)}")

        # Generate combined encyclopedia using server_text.py
        if self.mode == "text":
            self.server_text = TextBasedInsightAggregationServer(
                model_name=self.model_name,
                device=self.device,
                input_dirs=[self.output_dir],
                use_api=self.use_api,
                api_key=self.api_key,
                api_provider=self.api_provider,
            )
            result = self.server_text.aggregate_and_build_encyclopedia(
                json_files=all_json_files, output_dir=self.output_dir
            )
            # Save as encyclopedia_all.json under output_dir
            encyclopedia_all_path = os.path.join(self.output_dir, "encyclopedia_all.json")
            encyclopedia_dict = self.server_text._try_parse_json(self.server_text.encyclopedia)
            if encyclopedia_dict is None:
                json_content = self.server_text._extract_json_only(self.server_text.encyclopedia)
                encyclopedia_dict = self.server_text._try_parse_json(json_content)
            if encyclopedia_dict is None:
                error_msg = f"ERROR: Could not parse combined encyclopedia as JSON. Encyclopedia content: {self.server_text.encyclopedia[:500]}"
                print(error_msg)
                raise ValueError(error_msg)
            with open(encyclopedia_all_path, "w", encoding="utf-8") as f:
                json.dump(encyclopedia_dict, f, indent=2, ensure_ascii=False)
        else:
            self.server = InsightAggregationServer(
                model_name=self.model_name,
                device=self.device,
                input_dir=self.output_dir,
                use_api=self.use_api,
                api_key=self.api_key,
                api_provider=self.api_provider,
            )
            result = self.server.aggregate_and_build_encyclopedia(
                json_files=all_json_files, r1=r1, r2=r2, output_dir=self.output_dir
            )
            # Save as encyclopedia_all.txt under output_dir
            encyclopedia_all_path = os.path.join(self.output_dir, "encyclopedia_all.txt")
            with open(encyclopedia_all_path, "w", encoding="utf-8") as f:
                f.write(self.server.encyclopedia)

        print(f"\nCombined encyclopedia saved to: {encyclopedia_all_path}")
        return encyclopedia_all_path

    # ------------------------------------------------------------------
    # Eval-only mode: solve + check accuracy, no trace extraction or aggregation
    # ------------------------------------------------------------------
    def _format_problem(self, problem_data: Dict, dataset_name: str):
        """Format a problem for the given dataset. Returns (problem_text, test_cases_for_eval)."""
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

    def run_eval_only(
        self,
        dataset_list: List[str],
        max_problems: Optional[int],
        encyclopedia_paths: Optional[List[str]] = None,
        problem_overrides: Optional[Dict[str, List[Dict]]] = None,
        output_subdir: str = "eval_only",
        summary_name: str = "eval_only_summary.json",
    ) -> Dict:
        """Eval-only mode: solve problems and check accuracy.

        Uses existing encyclopedia (if provided) to guide solutions,
        but does NOT extract traces or aggregate insights.

        Args:
            dataset_list: List of datasets to evaluate
            max_problems: Max problems per dataset
            encyclopedia_paths: Paths to encyclopedia files to use for guidance

        Returns:
            Summary dict with accuracy per dataset
        """
        self._ensure_client()

        # Build insights_section from encyclopedia
        insights_section = ""
        if encyclopedia_paths:
            valid_eps = [ep for ep in encyclopedia_paths if ep and os.path.exists(ep)]
            if valid_eps:
                print(f"Loading {len(valid_eps)} encyclopedias for guidance...")
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
            print("No valid encyclopedias found; proceeding without guidance")

        print(f"\n{'='*80}")
        print("EVAL-ONLY MODE: Solve + Accuracy Check (no trace extraction / aggregation)")
        print(f"Datasets: {', '.join(dataset_list)}")
        print(f"Max problems per dataset: {max_problems or 'all'}")
        print(f"Encyclopedia: {'yes' if insights_section else 'none'}")
        print(f"{'='*80}\n")

        accuracy_map = {}
        token_map = {}
        loop_map = {}

        for dataset_name in dataset_list:
            if problem_overrides and dataset_name in problem_overrides:
                worklist = problem_overrides[dataset_name]
            else:
                problems = self.load_math_dataset(dataset_name)
                worklist = problems[:max_problems] if max_problems else problems
            print(f"\nEvaluating {dataset_name} ({len(worklist)} problems)...")

            eval_dir = os.path.join(self.output_dir, dataset_name, output_subdir)
            os.makedirs(eval_dir, exist_ok=True)

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
                    # Only solve — no reflection or trace extraction
                    prompt = self.client._get_solution_prompt(
                        problem_text, insights_section=insights_section
                    )
                    response, token_info = self.client._call_model(prompt, None, max_new_tokens=32768)

                    solution = response
                    number_output_tokens = token_info.get("output_tokens", 0)
                    number_output_tokens_list.append(number_output_tokens)

                    loop_count = self._count_consecutive_sentence_loops(solution)
                    loop_count_list.append(loop_count)

                    predicted_answer = self._extract_answer_from_solution(
                        solution, dataset_name, problem_data
                    )

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
                        print(f"    {status} Predicted: {predicted_answer if predicted_answer else 'N/A'} | GT: {ground_truth if ground_truth else 'N/A'}")

                    output_data = {
                        "problem": problem_text,
                        "problem_id": problem_data.get("id", idx),
                        "solution": solution,
                        "predicted_answer": predicted_answer,
                        "is_correct": is_correct,
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
                    })
                    time.sleep(0.5)

                except Exception as exc:
                    print(f"    Error processing problem {idx}: {exc}")

            # Summarize
            if results:
                num_correct = sum(1 for r in results if r["is_correct"])
                accuracy = num_correct / len(results)
                avg_tokens = sum(r["number_output_tokens"] for r in results) / len(results)
                total_loops = sum(r["loop_count"] for r in results)
            else:
                accuracy = 0.0
                avg_tokens = 0.0
                total_loops = 0

            accuracy_map[dataset_name] = accuracy
            token_map[dataset_name] = avg_tokens
            loop_map[dataset_name] = total_loops

            print(f"\n  {dataset_name}: Accuracy={accuracy:.2%}, Avg Tokens={avg_tokens:.1f}, Loops={total_loops}")

        # Save summary
        summary = {
            "mode": "eval_only",
            "datasets": dataset_list,
            "accuracy_per_dataset": accuracy_map,
            "avg_tokens_per_dataset": token_map,
            "loop_count_per_dataset": loop_map,
            "encyclopedia_used": [ep for ep in (encyclopedia_paths or []) if ep and os.path.exists(ep)],
        }

        summary_path = os.path.join(self.output_dir, summary_name)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print("EVAL-ONLY COMPLETE")
        print(f"{'='*80}")
        for dataset, acc in accuracy_map.items():
            print(f"  {dataset}: {acc:.2%}")
        print(f"\nSummary saved: {summary_path}")

        return summary

    def run_split_pipeline(
        self,
        dataset_list: List[str],
        max_problems: Optional[int],
        split: float,
        seed: int,
        r1: float = 0.95,
        r2: float = 0.4,
    ) -> Dict:
        """Train an insight library on a random split, then eval on held-out problems.

        The training split runs the normal Step 1/2/3 flow. The evaluation split
        only solves with the generated library and computes accuracy/performance.
        """
        if not 0.0 < split < 1.0:
            raise ValueError("--split must be a float strictly between 0 and 1")
        if not dataset_list:
            raise ValueError("Provide at least one dataset for split mode")

        base_output_dir = self.output_dir
        train_output_dir = os.path.join(base_output_dir, "split_train")
        eval_output_dir = os.path.join(base_output_dir, "split_eval")

        split_manifest = {
            "mode": "split",
            "split": split,
            "seed": seed,
            "max_problems": max_problems,
            "train_output_dir": train_output_dir,
            "eval_output_dir": eval_output_dir,
            "datasets": {},
        }
        train_problem_map: Dict[str, List[Dict]] = {}
        eval_problem_map: Dict[str, List[Dict]] = {}

        for dataset_index, dataset_name in enumerate(dataset_list):
            problems = self.load_math_dataset(dataset_name)
            worklist = problems[:max_problems] if max_problems else problems
            indices = list(range(len(worklist)))
            rng = random.Random(seed + dataset_index)
            rng.shuffle(indices)
            train_size = int(len(indices) * split)
            if len(indices) > 1:
                train_size = max(1, min(train_size, len(indices) - 1))
            train_indices = indices[:train_size]
            eval_indices = indices[train_size:]
            train_problem_map[dataset_name] = [worklist[i] for i in train_indices]
            eval_problem_map[dataset_name] = [worklist[i] for i in eval_indices]
            split_manifest["datasets"][dataset_name] = {
                "total": len(worklist),
                "train": len(train_indices),
                "eval": len(eval_indices),
                "train_indices": train_indices,
                "eval_indices": eval_indices,
                "train_ids": [worklist[i].get("id", i + 1) for i in train_indices],
                "eval_ids": [worklist[i].get("id", i + 1) for i in eval_indices],
            }

        os.makedirs(base_output_dir, exist_ok=True)
        split_path = os.path.join(base_output_dir, "split_manifest.json")
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_manifest, f, indent=2, ensure_ascii=False)
        print(f"Split manifest saved: {split_path}")

        orig_output_dir = self.output_dir
        iteration_history = []
        encyclopedia_paths: Optional[List[str]] = None
        combined_ency_path = None
        try:
            for iteration in range(1, self.num_iterations + 1):
                iter_train_dir = os.path.join(train_output_dir, f"iter_{iteration:02d}")
                iter_eval_dir = os.path.join(eval_output_dir, f"iter_{iteration:02d}")

                # --- TRAIN: extract insights on training split ---
                self.output_dir = iter_train_dir
                os.makedirs(self.output_dir, exist_ok=True)
                print("\n" + "=" * 80)
                print(
                    f"SPLIT ITERATION {iteration}/{self.num_iterations} "
                    f"TRAIN: Step 1/2/3 on {split:.0%} split"
                )
                print("=" * 80)
                train_results_map: Dict[str, List[Dict]] = {}
                for dataset_name in dataset_list:
                    _, train_results = self._extract_insights_for_dataset(
                        dataset_name=dataset_name,
                        problems=train_problem_map[dataset_name],
                        max_problems=None,
                        encyclopedia_paths=encyclopedia_paths,
                        iteration=iteration,
                    )
                    train_results_map[dataset_name] = train_results

                train_accuracy_map = {
                    ds: (sum(1 for r in res if r["is_correct"]) / len(res) if res else 0.0)
                    for ds, res in train_results_map.items()
                }
                print(f"\nIteration {iteration} TRAIN accuracy:")
                for ds, acc in train_accuracy_map.items():
                    print(f"  - {ds}: {acc:.2%}")

                print(f"\nIteration {iteration}: Generating combined encyclopedia from training split...")
                combined_ency_path = self.generate_combined_encyclopedia(dataset_list, r1=r1, r2=r2)
                encyclopedia_paths = [combined_ency_path] if combined_ency_path else []

                # --- EVAL: eval-only on held-out split ---
                self.output_dir = iter_eval_dir
                os.makedirs(self.output_dir, exist_ok=True)
                print("\n" + "=" * 80)
                print(f"SPLIT ITERATION {iteration}/{self.num_iterations} EVAL: held-out split")
                print("=" * 80)
                eval_summary = self.run_eval_only(
                    dataset_list=dataset_list,
                    max_problems=None,
                    encyclopedia_paths=encyclopedia_paths,
                    problem_overrides=eval_problem_map,
                    output_subdir="problems",
                    summary_name="split_eval_summary.json",
                )

                iteration_history.append({
                    "iteration": iteration,
                    "train_accuracy": train_accuracy_map,
                    "eval_summary": eval_summary,
                    "combined_encyclopedia": combined_ency_path,
                })

                print(f"\nIteration {iteration} EVAL accuracy:")
                for ds, acc in eval_summary.get("accuracy_per_dataset", {}).items():
                    print(f"  - {ds}: {acc:.2%}")
        finally:
            self.output_dir = orig_output_dir

        summary = {
            "mode": "split",
            "split": split,
            "seed": seed,
            "num_iterations": self.num_iterations,
            "datasets": dataset_list,
            "split_manifest": split_manifest,
            "combined_encyclopedia": combined_ency_path,
            "iteration_history": iteration_history,
        }
        summary_path = os.path.join(base_output_dir, "split_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSplit summary saved: {summary_path}")
        return summary

    # ------------------------------------------------------------------
    # Iterative Learning Pipeline
    # ------------------------------------------------------------------
    def run_iterative_pipeline(
        self,
        dataset_list: List[str],
        max_problems: Optional[int],
        r1: float = 0.95,
        r2: float = 0.4,
        start_from_step: int = 1,
    ) -> Dict:
        """Run iterative learning pipeline.

        Each iteration:
        1. Use encyclopedia (from previous iteration) to solve problems and log accuracy
        2. Extract insights from the same problems
        3. Aggregate insights into new encyclopedia
        4. Repeat

        Args:
            dataset_list: List of datasets to train on
            max_problems: Max problems per dataset
            r1: Similarity threshold for aggregation
            r2: Similarity threshold for aggregation
            start_from_step: Start from step 1 (extract) or 2 (aggregate only)

        Returns:
            Summary dict with iteration history
        """
        if not dataset_list:
            raise ValueError("Provide at least one dataset for iterative learning")

        start_time = time.time()
        iteration_history = []

        # Check if combined encyclopedia already exists (from previous runs)
        # Use encyclopedia_all.json/txt instead of per-dataset encyclopedias
        encyclopedia_paths: Optional[List[str]] = None
        if self.mode == "text":
            combined_ency_path = os.path.join(self.output_dir, "encyclopedia_all.json")
        else:
            combined_ency_path = os.path.join(self.output_dir, "encyclopedia_all.txt")

        if os.path.exists(combined_ency_path):
            encyclopedia_paths = [combined_ency_path]
            print(f"Found existing combined encyclopedia: {combined_ency_path}")
        else:
            # Fallback: check for per-dataset encyclopedias
            per_dataset_encyclopedias = []
            for dataset_name in dataset_list:
                if self.mode == "text":
                    dataset_ency = os.path.join(self.output_dir, dataset_name, "encyclopedia.json")
                else:
                    dataset_ency = os.path.join(self.output_dir, dataset_name, "encyclopedia.txt")
                if os.path.exists(dataset_ency):
                    per_dataset_encyclopedias.append(dataset_ency)

            if per_dataset_encyclopedias:
                encyclopedia_paths = per_dataset_encyclopedias
                print(f"Found {len(per_dataset_encyclopedias)} per-dataset encyclopedias:")
                for ep in per_dataset_encyclopedias:
                    print(f"  - {ep}")

        print(f"\n{'='*80}")
        print(f"Starting Iterative Learning Pipeline: {self.num_iterations} iterations")
        print(f"Datasets: {', '.join(dataset_list)}")
        print(f"Max problems per dataset: {max_problems or 'all'}")
        if encyclopedia_paths:
            print(
                f"Using {len(encyclopedia_paths)} existing encyclopedias for iteration 1"
            )
        print(f"{'='*80}\n")

        for iteration in range(1, self.num_iterations + 1):
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration}/{self.num_iterations}")
            print(f"{'='*80}")

            # STEP 1: Extract insights (and solve if encyclopedia exists)
            if start_from_step == 1:
                insights_map, accuracy_map = self.learn_insights_from_datasets(
                    dataset_list, max_problems, encyclopedia_paths, iteration
                )
            else:
                # Starting from step 2: check if insights exist from previous run
                print(
                    f"\nSkipping Step 1 (insight extraction) - assuming insights already exist"
                )
                insights_exist = all(
                    os.path.isdir(os.path.join(self.output_dir, name))
                    for name in dataset_list
                )
                if not insights_exist:
                    raise FileNotFoundError(
                        f"Cannot start from step 2: Insight directories not found in {self.output_dir}. "
                        "Run with --start-from-step 1 first to extract insights."
                    )
                accuracy_map = {name: 0.0 for name in dataset_list}

            # STEP 2: Generate combined encyclopedia from all datasets at once
            # Collect all skills from all available datasets instead of processing each individually
            print(f"\nIteration {iteration}: Generating combined encyclopedia from all datasets...")
            combined_ency_path = self.generate_combined_encyclopedia(dataset_list, r1=r1, r2=r2)
            # Use only the combined encyclopedia for next iteration's Step 1
            encyclopedia_paths = [combined_ency_path] if combined_ency_path else []

            # Note: Skipping per-dataset aggregation - we now do "generate all" approach
            # to collect all skills from all datasets at once instead of one-by-one
            # Old fallback approach was: per_dataset_ency = self.aggregate_insights(dataset_list, r1=r1, r2=r2)

            # Save iteration results
            iteration_summary = {
                "iteration": iteration,
                "datasets": dataset_list,
                "accuracy_per_dataset": accuracy_map,
                "combined_encyclopedia": combined_ency_path,
            }
            iteration_history.append(iteration_summary)

            print(f"\nIteration {iteration} Summary:")
            for dataset, acc in accuracy_map.items():
                print(f"  - {dataset}: {acc:.2%}")
            if encyclopedia_paths:
                print(f"  Combined encyclopedia: {encyclopedia_paths[0]}")

        # Final summary
        final_summary = {
            "mode": "iterative",
            "num_iterations": self.num_iterations,
            "datasets": dataset_list,
            "iteration_history": iteration_history,
            "total_time_seconds": time.time() - start_time,
        }

        summary_path = os.path.join(self.output_dir, "iterative_summary.json")
        # Persist final summary, merging with existing per-dataset entries
        try:
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            else:
                existing = []
            payload = {
                "final": final_summary,
            }
            # Store both the final aggregate and previously appended per-dataset entries
            combined = existing if isinstance(existing, list) else []
            combined.append(payload)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: failed to write final iterative summary: {e}")

        print(f"\n{'='*80}")
        print("ITERATIVE LEARNING COMPLETE")
        print(f"{'='*80}")
        print(f"\nAccuracy per iteration:")
        for iter_sum in iteration_history:
            print(f"  Iteration {iter_sum['iteration']}:")
            for dataset, acc in iter_sum["accuracy_per_dataset"].items():
                print(f"    - {dataset}: {acc:.2%}")
        print(f"\nFinal summary saved: {summary_path}")

        return final_summary

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def _extract_answer_from_solution(
        self, solution: str, dataset_name: str, problem_data: Dict
    ) -> Optional[str]:
        """Extract answer from solution text using dataset-specific strategies.

        Args:
            solution: The model's generated solution text
            dataset_name: Name of the dataset (e.g., 'aime25', 'livemathbench', 'gsm8k')
            problem_data: Original problem data dictionary

        Returns:
            Extracted answer string or None if extraction failed
        """
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

        # Strategy 1: Extract from \boxed{} format (with or without LaTeX math mode)
        # Handles: \boxed{204}, \(\boxed{204}\), \[\boxed{204}\]
        boxed_answer = _extract_boxed_balanced(solution)
        if boxed_answer:
            boxed_answer = boxed_answer.replace("\\,", "").replace("\\:", "").replace("\\;", "")
            boxed_answer = boxed_answer.replace("\\text{", "").replace("}", "")
            return boxed_answer

        boxed_patterns = [
            r"\\\(\\boxed\{([^}]+)\}\\\)",  # \(\boxed{answer}\)
            r"\\\[\\boxed\{([^}]+)\}\\\]",  # \[\boxed{answer}\]
            r"\\boxed\{([^}]+)\}",  # \boxed{answer}
        ]

        for pattern in boxed_patterns:
            match = re.search(pattern, solution)
            if match:
                answer = match.group(1).strip()
                # Clean up LaTeX formatting from inside boxed
                answer = answer.replace("\\,", "").replace("\\:", "").replace("\\;", "")
                answer = answer.replace("\\text{", "").replace("}", "")
                return answer

        # Strategy 2: Extract from ## Answer: section (structured format)
        if "## Answer:" in solution:
            start_idx = solution.find("## Answer:") + len("## Answer:")
            end_idx = solution.find("## End of Answer:")
            if end_idx == -1:
                # Try to find next ## heading or use rest of text
                next_heading = solution.find("##", start_idx)
                end_idx = next_heading if next_heading != -1 else len(solution)
            answer = solution[start_idx:end_idx].strip()

            boxed_answer = _extract_boxed_balanced(answer)
            if boxed_answer:
                return boxed_answer

            # Try to extract boxed from this section
            for pattern in boxed_patterns:
                match = re.search(pattern, answer)
                if match:
                    return match.group(1).strip()
            return answer

        # Strategy 3: Dataset-specific extraction strategies
        if dataset_name:
            # AIME/Math competition formats: look for "the answer is" patterns
            if dataset_name in ["aime25", "aime24"] or dataset_name.startswith("imo"):
                # Look for common answer phrases near the end
                answer_patterns = [
                    r"(?:the answer is|answer:|final answer:?)\s*\$?([^.$\n]+)\$?",
                    r"(?:therefore|thus|so),?\s+(?:the answer is)?\s*\$?([^.$\n]+)\$?",
                ]
                # Search in last 1000 characters for efficiency
                search_text = solution[-1000:] if len(solution) > 1000 else solution
                for pattern in answer_patterns:
                    matches = re.finditer(pattern, search_text, re.IGNORECASE)
                    # Get the last match
                    last_match = None
                    for match in matches:
                        last_match = match
                    if last_match:
                        answer = last_match.group(1).strip()
                        # Clean LaTeX and extract number
                        answer = answer.replace("\\,", "").replace("$", "")
                        numbers = extract_numbers(answer)
                        if numbers:
                            num = numbers[-1]
                            return str(int(num)) if math.isfinite(num) and num == int(num) else str(num)

            # GSM8K: typically ends with #### answer format in ground truth,
            # but model should use boxed or numeric answer
            elif dataset_name == "gsm8k":
                # GSM8K answers are typically simple numbers
                # Look for last number in solution
                numbers = extract_numbers(solution)
                if numbers:
                    num = numbers[-1]
                    return str(int(num)) if math.isfinite(num) and num == int(num) else str(num)

            # LiveMathBench: various formats depending on sub-benchmark
            elif "livemathbench" in dataset_name:
                # Try numeric extraction first
                numbers = extract_numbers(solution)
                if numbers:
                    num = numbers[-1]
                    return str(int(num)) if math.isfinite(num) and num == int(num) else str(num)

        # Strategy 4: Generic fallback - extract last number from solution
        numbers = extract_numbers(solution)
        if numbers:
            num = numbers[-1]
            return str(int(num)) if math.isfinite(num) and num == int(num) else str(num)

        # Strategy 5: Last resort - return last non-empty line (cleaned)
        lines = [l.strip() for l in solution.split("\n") if l.strip()]
        if lines:
            last_line = lines[-1]
            # Remove common suffixes
            last_line = re.sub(r"\.$", "", last_line)
            last_line = last_line.replace("\\)", "").replace("\\(", "")
            return last_line[:100]  # Limit length

        return None

    def _normalize_answer_for_comparison(self, answer: str) -> str:
        """Normalize answer text for comparison.

        Removes LaTeX delimiters, whitespace, and other formatting that doesn't
        affect mathematical equivalence.
        """
        if not answer:
            return ""

        # Remove LaTeX delimiters
        answer = answer.replace("$", "").replace("\\(", "").replace("\\)", "")
        answer = answer.replace("\\[", "").replace("\\]", "")
        answer = answer.replace("\\left", "").replace("\\right", "")
        answer = answer.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

        # Remove common LaTeX commands that don't affect meaning
        answer = re.sub(r"\\text\{([^}]+)\}", r"\1", answer)  # \text{...} -> ...
        answer = re.sub(r"\\displaystyle\s*", "", answer)
        answer = re.sub(r"^\s*[a-zA-Z]\w*\s*=\s*", "", answer)
        answer = re.sub(r"^\s*(?:answer|finalanswer|ans)\s*[:=]\s*", "", answer, flags=re.IGNORECASE)
        answer = re.sub(
            r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", answer
        )  # \frac{a}{b} -> (a)/(b)

        # Remove all whitespace (spaces, tabs, newlines)
        answer = re.sub(r"\s+", "", answer)

        # Normalize case
        answer = answer.lower()

        return answer.strip()

    def _check_answer_match(
        self,
        predicted: str,
        ground_truth: str,
        dataset_name: Optional[str] = None,
        problem_text: Optional[str] = None,
    ) -> bool:
        """Check if predicted answer matches ground truth using dataset-specific evaluators.

        Routes evaluation based on dataset:
        - LiveMathBench: livemathbench_evaluator
        - IMOBench: imo_evaluator (optionally uses Gemini if enabled)
        - Others: numeric/string/symbolic comparison

        Returns True if answers match.
        """
        if not predicted or not ground_truth:
            return False

        # LiveMathBench datasets
        if dataset_name and "livemathbench" in dataset_name:
            return livemathbench_evaluator(
                predicted,
                ground_truth,
                dataset_name=dataset_name,
                problem_text=problem_text,
            )

        # IMOBench datasets
        if dataset_name and dataset_name.startswith("imo"):
            return imo_evaluator(
                prediction=predicted,
                ground_truth=ground_truth,
                benchmark=dataset_name,
                problem_text=problem_text,
                use_gemini=self.use_api and self.client is not None,
                client=self.client,
            )

        # GPQA datasets (multiple choice)
        if dataset_name and dataset_name.startswith("gpqa"):
            from science_datasets.gpqa import gpqa_evaluator

            return gpqa_evaluator(
                predicted,
                ground_truth,
                dataset_name=dataset_name,
                problem_text=problem_text,
            )

        # LiveCodeBench datasets (code generation)
        if dataset_name and "livecodebench" in dataset_name:
            from code_datasets.livecodebench import livecodebench_evaluator

            return livecodebench_evaluator(
                predicted,
                ground_truth,  # This will be test_cases dict
                dataset_name=dataset_name,
                problem_text=problem_text,
            )

        # Standard datasets: try multiple comparison strategies

        # Strategy 1: Normalized symbolic comparison (for algebraic expressions like 10^{2^n-n-1})
        pred_normalized = self._normalize_answer_for_comparison(predicted)
        gt_normalized = self._normalize_answer_for_comparison(ground_truth)

        if pred_normalized and gt_normalized and pred_normalized == gt_normalized:
            return True

        # Strategy 2: Numeric comparison (for numeric answers)
        pred_nums = extract_numbers(predicted)
        gt_nums = extract_numbers(ground_truth)

        if pred_nums and gt_nums:
            # Check if any predicted number matches any ground truth number
            return any(abs(p - g) < 1e-6 for p in pred_nums for g in gt_nums)

        # Strategy 3: Fallback to case-insensitive string comparison
        return predicted.strip().lower() == ground_truth.strip().lower()

    # (Legacy IMO evaluation helpers removed; using math_datasets.imo_benchmark)


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
        description="Iterative benchmark learning pipeline: solve (if encyclopedia available) + extract insights + aggregate → repeat"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["aime25"],
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
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name (HF).",
    )
    parser.add_argument(
        "-d", "--device", type=str, default=None, help="Device to use (cuda or cpu)."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="math_output",
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
        "--split",
        type=float,
        default=None,
        help=(
            "Optional train/eval split fraction. If set, randomly use this "
            "fraction of each dataset for Step 1/2/3 insight generation and "
            "evaluate the remaining problems with Step 1 only."
        ),
    )
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
    parser.add_argument(
        "--client",
        type=str,
        default="default",
        choices=_CLIENT_CHOICES,
        help=(
            "Client algorithm to use for insight extraction. "
            "default=ChainOfThoughtReader (client.py), "
            "metacognitive=MetacognitiveClient, "
            "trt=TRTClient, "
            "hyperagents=HyperAgentsClient, "
            "evolveprompt=EvolvePromptClient, "
            "ace=ACEClient."
        ),
    )

    args = parser.parse_args()

    # Normalize dataset lists
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

    pipeline = BenchmarkDomainPipeline(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
        use_api=args.use_api,
        api_key=args.api_key,
        api_provider=args.api_provider,
        mode=args.mode,
        num_iterations=args.num_iterations,
        load_in_8bit=args.load_in_8bit,
        client_type=args.client,
    )

    try:
        if not datasets:
            raise ValueError("--datasets is required")

        if args.split is not None and args.eval_only:
            # Eval-only on the held-out (eval) portion of a deterministic split.
            # Uses the same seed+split fraction as run_split_pipeline so the eval
            # subset is identical to what that pipeline would have evaluated.
            if not 0.0 < args.split < 1.0:
                raise ValueError("--split must be a float strictly between 0 and 1")
            eval_problem_map: Dict[str, List[Dict]] = {}
            for dataset_index, dataset_name in enumerate(datasets):
                problems = pipeline.load_math_dataset(dataset_name)
                worklist = problems[: args.max_problems] if args.max_problems else problems
                indices = list(range(len(worklist)))
                rng = random.Random(args.seed + dataset_index)
                rng.shuffle(indices)
                train_size = int(len(indices) * args.split)
                if len(indices) > 1:
                    train_size = max(1, min(train_size, len(indices) - 1))
                eval_indices = indices[train_size:]
                eval_problem_map[dataset_name] = [worklist[i] for i in eval_indices]
                print(
                    f"[split-eval] {dataset_name}: {len(eval_indices)}/{len(worklist)} "
                    f"problems selected as eval set (split={args.split}, seed={args.seed})"
                )
            pipeline.run_eval_only(
                dataset_list=datasets,
                max_problems=None,
                encyclopedia_paths=args.encyclopedia,
                problem_overrides=eval_problem_map,
            )
        elif args.split is not None:
            pipeline.run_split_pipeline(
                dataset_list=datasets,
                max_problems=args.max_problems,
                split=args.split,
                seed=args.seed,
                r1=args.r1,
                r2=args.r2,
            )
        elif args.eval_only:
            # Eval-only mode: solve + accuracy, no trace extraction or aggregation
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
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()
        print("\nExamples:")
        print(
            "  python task_benchmark_domain.py --datasets aime25 --max-problems 10 --num-iterations 3"
        )
        print(
            "  python task_benchmark_domain.py --datasets gsm8k math500 --max-problems 20 --mode text"
        )
        print(
            "  python task_benchmark_domain.py --datasets imo_answerbench --max-problems 30 --num-iterations 5"
        )
        print(
            "  python task_benchmark_domain.py --datasets gpqa_diamond --max-problems 50 --use-api"
        )


if __name__ == "__main__":
    main()
