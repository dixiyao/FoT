"""
Task Benchmark Domain with Trace Appending (RAG-based)
- Based on task_benchmark_domain.py but only performs Step 1 inference
- Takes a folder containing problem/paper*.json files with reasoning traces
- Appends all reasoning traces together to create a comprehensive RAG datastore
- Uses the RAG datastore to solve tasks the same way as task_benchmark_domain.py

Usage example:
  python task_benchmark_domain_traceappending.py \
      --trace-folder mix3_output_gemini/ \
      --datasets aime24 aime25 gpqa_diamond \
      --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
      --output-dir traceappending_output \
      --device cuda
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
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

# Import genai for file search (RAG mode)
try:
    import google.genai as genai_new  # type: ignore
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    genai_new = None
    types = None

# Dataset registry: (source, path_or_hf_name, data_dir_or_col_map, split_or_none)
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
    "livecodebench": ("hf", "bzantium/livecodebench", "release_v6", "test"),
    "livecodebench_lite": ("hf", "bzantium/livecodebench", "v6", "test"),
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
    # IMOBench datasets
    "answerbench": ("csv", "math_datasets/answerbench.csv", None, None),
    "proofbench": ("csv", "math_datasets/proofbench.csv", None, None),
    "gradingbench": ("csv", "math_datasets/gradingbench.csv", None, None),
}


def _get_client(
    client_type: str,
    model_name: str,
    device: Optional[str],
    use_api: bool,
    api_key: Optional[str],
    api_provider: str,
    load_in_8bit: bool,
) -> object:
    """Factory function to create the appropriate client."""
    common = {
        "model_name": model_name,
        "device": device,
        "use_api": use_api,
        "api_key": api_key,
        "api_provider": api_provider,
        "load_in_8bit": load_in_8bit,
    }

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


class TraceAppendingPipeline:
    def __init__(
        self,
        trace_folder: str,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device: Optional[str] = None,
        output_dir: str = "traceappending_output",
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
        mode: str = "text",
        load_in_8bit: bool = False,
        client_type: str = "default",
    ):
        self.trace_folder = trace_folder
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        self.use_api = use_api
        self.api_key = api_key
        self.api_provider = api_provider
        self.mode = mode  # "normal" uses server.py, "text" uses server_text.py
        self.load_in_8bit = load_in_8bit
        self.client_type = client_type

        os.makedirs(output_dir, exist_ok=True)

        self.client = None
        self.rag_store = None

    def _ensure_client(self):
        """Lazy initialization of client."""
        if self.client is None:
            self.client = _get_client(
                self.client_type,
                self.model_name,
                self.device,
                self.use_api,
                self.api_key,
                self.api_provider,
                self.load_in_8bit,
            )

    def create_rag_datastore(self) -> str:
        """Create RAG datastore by appending all reasoning traces from problem/paper*.json files."""
        print(f"\n{'='*80}")
        print("CREATING RAG DATASTORE FROM REASONING TRACES")
        print(f"{'='*80}")

        if not HAS_GENAI:
            raise ImportError(
                "google-genai package is required for RAG datastore creation. "
                "Install with: pip install google-genai"
            )

        # Initialize client using the new API (required for file search)
        genai_client = genai_new.Client(api_key=self.api_key)

        # Find all problem/paper*.json files in the trace folder
        trace_path = Path(self.trace_folder)
        if not trace_path.exists():
            raise ValueError(f"Trace folder does not exist: {self.trace_folder}")

        # Find all matching trace files recursively under the trace folder
        trace_files = list(trace_path.rglob("problem*.json")) + list(trace_path.rglob("paper*.json"))
        if not trace_files:
            raise ValueError(
                f"No problem or paper trace files found under {self.trace_folder}"
            )

        print(f"Found {len(trace_files)} trace files under {self.trace_folder}")

        # Collect all reasoning traces
        all_traces = []
        total_files = 0

        for trace_file in trace_files:
            rel_path = trace_file.relative_to(trace_path)
            print(f"  Processing trace file: {rel_path}")
            try:
                with open(trace_file, 'r', encoding='utf-8') as f:
                    problem_data = json.load(f)

                # Extract reasoning trace (solution field typically contains the trace)
                solution = problem_data.get('solution', '')
                if solution and len(solution.strip()) > 0:
                    trace_doc = f"""
Problem: {problem_data.get('problem', '')}

Reasoning Trace:
{solution}

Source: {rel_path}
---
"""
                    all_traces.append(trace_doc)
                    total_files += 1

            except Exception as e:
                print(f"  Warning: Failed to process {trace_file}: {e}")
                continue

        if not all_traces:
            raise ValueError("No reasoning traces found in the trace folder")

        print(f"\nCollected {total_files} reasoning traces from {len(trace_files)} trace files")

        # Create a single large text file with all traces
        combined_traces = "\n\n".join(all_traces)

        # Create stable display name based on folder name
        folder_name = Path(self.trace_folder).name
        store_display_name = f"traceappending-{folder_name}"

        # Check if file search store with this name already exists
        print(f"\nChecking for existing file search store: {store_display_name}")
        file_search_store = None
        store_is_reused = False

        try:
            # List existing stores and find matching one
            stores = genai_client.file_search_stores.list()
            for store in stores:
                if hasattr(store, 'display_name') and store.display_name == store_display_name:
                    file_search_store = store
                    store_is_reused = True
                    print(f"✓ Found existing file search store: {store.name}")
                    print(f"  Reusing existing store to avoid duplicates")
                    break
        except Exception as e:
            print(f"Warning: Could not list existing stores: {e}")

        # Create new store if not found
        if file_search_store is None:
            print(f"Creating new file search store: {store_display_name}")
            file_search_store = genai_client.file_search_stores.create(
                config={'display_name': store_display_name}
            )
            print(f"✓ Created file search store: {file_search_store.name}")

        # Upload the combined traces as a text file
        if not store_is_reused:
            print(f"\nUploading combined reasoning traces ({len(combined_traces)} characters)...")

            # Save combined traces to a temporary file
            temp_file_path = os.path.join(self.output_dir, "combined_reasoning_traces.txt")
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                f.write(combined_traces)

            # Upload to file search store
            try:
                uploaded_file = genai_client.files.upload(
                    file=temp_file_path,
                    config={'display_name': 'combined_reasoning_traces.txt'}
                )
                print(f"✓ Uploaded traces file: {uploaded_file.name}")

                # Add to file search store
                genai_client.files.link_to_file_search_store(
                    file_search_store_name=file_search_store.name,
                    file_name=uploaded_file.name
                )
                print("✓ Linked file to search store")

            except Exception as e:
                print(f"Error uploading traces: {e}")
                raise
            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        self.rag_store = file_search_store
        print(f"\n✓ RAG datastore ready: {file_search_store.name}")
        return file_search_store.name

    def load_problems(self, dataset_name: str, max_problems: Optional[int] = None) -> List[Dict]:
        """Load problems for a dataset."""
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        source_type, path_or_hf_name, data_dir, split = DATASET_REGISTRY[dataset_name]

        # Attempt Hugging Face
        if source_type == "hf":
            if load_dataset is None:
                raise ImportError(
                    "datasets library is required. Install with: pip install datasets"
                )
            print(f"Loading dataset '{dataset_name}' from Hugging Face...")
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
                    raw_item = dict(item)
                    raw_item["id"] = item.get("id", i + 1)
                    raw.append(raw_item)
                # For LiveCodeBench datasets, preserve all original fields
                elif dataset_name and dataset_name.startswith("livecodebench"):
                    raw_item = dict(item)
                    raw_item["id"] = item.get("question_id", item.get("id", i + 1))
                    raw.append(raw_item)
                else:
                    # Extract fields with fallbacks for different dataset formats
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
                        }
                    )

            if max_problems:
                raw = raw[:max_problems]
            return raw

        # CSV loading
        elif source_type == "csv":
            candidate_path = path_or_hf_name
            if not os.path.exists(candidate_path):
                raise FileNotFoundError(
                    f"CSV file for dataset '{dataset_name}' not found at {candidate_path}"
                )

            print(f"Loading CSV file from {candidate_path}...")
            with open(candidate_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV file {candidate_path} is empty or has no header")

                raw = []
                for i, row in enumerate(reader):
                    if max_problems and i >= max_problems:
                        break

                    # Map CSV columns to standard schema
                    problem_cols = [
                        "problem",
                        "question",
                        "problem_text",
                        "task",
                        "statement",
                        "text",
                    ]
                    answer_cols = [
                        "answer",
                        "solution",
                        "final_answer",
                        "answer_text",
                        "short answer",
                        "short_answer",
                        "response",
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

                    problem_text = None
                    for col in problem_cols:
                        if col in row and row[col]:
                            problem_text = row[col]
                            break

                    answer_text = None
                    for col in answer_cols:
                        if col in row and row[col]:
                            answer_text = row[col]
                            break

                    problem_id = None
                    for col in id_cols:
                        if col in row and row[col]:
                            problem_id = row[col]
                            break

                    if problem_text:
                        raw.append(
                            {
                                "id": problem_id or i + 1,
                                "problem": problem_text,
                                "question": problem_text,
                                "solution": answer_text or "",
                            }
                        )

            return raw

        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    def _extract_answer_from_solution(
        self, solution: str, dataset_name: str, problem_data: Dict
    ) -> Optional[str]:
        """Extract answer from solution text."""
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

        # Strategy 1: Extract from \boxed{} format
        boxed_answer = _extract_boxed_balanced(solution)
        if boxed_answer:
            boxed_answer = boxed_answer.replace("\\,", "").replace("\\:", "").replace("\\;", "")
            return boxed_answer

        # Strategy 2: Numeric comparison
        pred_nums = extract_numbers(solution)
        if pred_nums:
            return str(pred_nums[-1])  # Return last number found

        # Strategy 3: Fallback to last line
        lines = solution.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if last_line:
                return last_line

        return None

    def _check_answer_match(
        self,
        predicted: str,
        ground_truth: str,
        dataset_name: Optional[str] = None,
        problem_text: Optional[str] = None,
    ) -> bool:
        """Check if predicted answer matches ground truth."""
        if not predicted or not ground_truth:
            return False

        # Normalize answers for comparison
        def normalize_answer(ans: str) -> str:
            if not ans:
                return ""
            # Remove LaTeX delimiters
            ans = ans.replace("$", "").replace("\\(", "").replace("\\)", "")
            ans = ans.replace("\\[", "").replace("\\]", "")
            # Remove whitespace
            ans = re.sub(r"\s+", "", ans)
            return ans.lower()

        pred_normalized = normalize_answer(predicted)
        gt_normalized = normalize_answer(ground_truth)

        if pred_normalized and gt_normalized and pred_normalized == gt_normalized:
            return True

        # Numeric comparison
        pred_nums = extract_numbers(predicted)
        gt_nums = extract_numbers(ground_truth)

        if pred_nums and gt_nums:
            return any(abs(p - g) < 1e-6 for p in pred_nums for g in gt_nums)

        return False

    def run_eval_only(
        self,
        dataset_list: List[str],
        max_problems: Optional[int],
    ) -> Dict:
        """Eval-only mode: solve problems using the RAG datastore."""
        print(f"\n{'='*80}")
        print("STEP 1: SOLVING PROBLEMS WITH RAG DATASTORE")
        print(f"{'='*80}")

        self._ensure_client()

        # Load RAG datastore
        if not self.rag_store:
            self.create_rag_datastore()

        # Configure client to use RAG
        self.client.load_rag_store(self.rag_store, self.api_key)

        summary = {}

        for dataset_name in dataset_list:
            print(f"\nEvaluating dataset: {dataset_name}")
            problems = self.load_problems(dataset_name, max_problems)

            correct = 0
            total = len(problems)

            for i, problem_data in enumerate(problems, 1):
                problem_text = problem_data.get("problem") or problem_data.get("question", "")
                ground_truth = problem_data.get("answer") or problem_data.get("solution", "")

                print(f"  [{i}/{total}] {problem_text[:60]}...")

                try:
                    result = self.client.solve_problem(problem_text)
                    solution = result.get("solution", "")
                    predicted_answer = self._extract_answer_from_solution(solution, dataset_name, problem_data)

                    is_correct = self._check_answer_match(predicted_answer, ground_truth, dataset_name, problem_text)

                    if is_correct:
                        correct += 1
                        print(f"    ✓ Correct: {predicted_answer}")
                    else:
                        print(f"    ✗ Wrong: predicted='{predicted_answer}', expected='{ground_truth}'")

                except Exception as e:
                    print(f"    Error: {e}")

            accuracy = correct / total if total > 0 else 0
            summary[dataset_name] = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy
            }
            print(".2%")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Task Benchmark Domain with Trace Appending: Create RAG datastore from reasoning traces and solve problems"
    )

    parser.add_argument(
        "--trace-folder",
        type=str,
        required=True,
        help="Folder containing problem/paper*.json files with reasoning traces",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["aime25"],
        help="Datasets to evaluate (space- or comma-separated).",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Limit problems per dataset.",
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
        default="traceappending_output",
        help="Root output directory.",
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
        help="Load model with 8-bit quantization (default: False)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="text",
        choices=["normal", "text"],
        help="Aggregation/inference mode (normal=GraphRAG, text=text-based). Default: text",
    )
    parser.add_argument(
        "--client",
        type=str,
        default="default",
        choices=["default", "metacognitive", "trt", "hyperagents", "evolveprompt", "ace"],
        help="Client algorithm to use for problem solving.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    # Normalize dataset lists
    datasets = []
    for item in args.datasets:
        parts = [p.strip() for p in item.split(",") if p.strip()]
        datasets.extend(parts)

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

    # Create pipeline
    pipeline = TraceAppendingPipeline(
        trace_folder=args.trace_folder,
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
        use_api=args.use_api,
        api_key=args.api_key,
        api_provider=args.api_provider,
        mode=args.mode,
        load_in_8bit=args.load_in_8bit,
        client_type=args.client,
    )

    # Run evaluation
    summary = pipeline.run_eval_only(datasets, args.max_problems)

    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    for dataset, stats in summary.items():
        print(f"{dataset}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
