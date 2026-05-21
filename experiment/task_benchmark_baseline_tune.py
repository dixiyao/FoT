"""
Baseline Fine-Tuning Pipeline (task_benchmark_baseline_tune)

Two-step pipeline for fine-tuning DeepSeek / Qwen reasoning models with LoRA.

  STEP 1 — Generate solutions and collect reasoning traces
    Runs the same ChainOfThoughtReader pipeline as task_benchmark_domain.py
    (solution → reflection → insight extraction) and saves each problem's
    full solution text plus reflection alongside the insight book.  Skippable
    via --skip-step when traces already exist on disk.

  STEP 2 — Simplified LoRA alignment (all-trace SFT)
    Follows the alignment framework of RPAM (arxiv 2601.03506v1), Step 2,
    simplified for the single-model setting: all collected reasoning traces
    (regardless of correctness) are used as supervision targets.

      Original paper — optimise per-layer merge coefficients λ so that the
      merged model's hidden representations z_M match those of the positive
      model z_pos:
          L = ||z_M - z_pos||² + ω · contrastive_loss(z_M, z_pos, z_neg)

      Simplified here — directly updating the base model's weights via LoRA
      cross-entropy SFT on all reasoning traces drives its output distribution
      toward the demonstrated reasoning style, achieving the same alignment
      effect without requiring a second model or contrastive term.

    Best hyper-parameters follow the paper's calibration grid (lr ∈
    {0.1, 0.01, 0.001}, epochs ∈ {50, 100}) mapped to the LoRA SFT scale
    for 7-14B DeepSeek / Qwen models (lr=2e-4, 3 epochs, r=16, α=32,
    effective batch 32, cosine schedule, warmup 0.03).

Only HuggingFace-accessible DeepSeek / Qwen models are supported; Gemini is
excluded because LoRA requires direct weight access.

Usage:
    # Full pipeline — generate traces then fine-tune
    python task_benchmark_baseline_tune.py \\
        --datasets aime24 aime25 gpqa_diamond \\
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
        --device cuda --trace-root tune_traces --output-dir tune_lora

    # Skip trace generation; collect existing traces and fine-tune
    python task_benchmark_baseline_tune.py \\
        --datasets aime24 aime25 gpqa_diamond \\
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
        --device cuda --trace-root tune_traces --output-dir tune_lora \\
        --skip-step

    # Trace generation only (no fine-tuning)
    python task_benchmark_baseline_tune.py \\
        --datasets aime24 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
        --device cuda --trace-root tune_traces --trace-only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from task_benchmark_domain import BenchmarkDomainPipeline, _parse_list_arg


# ---------------------------------------------------------------------------
# Model guard
# ---------------------------------------------------------------------------

def _check_is_deepseek_qwen(model_name: str) -> None:
    """Reject Gemini and non-DeepSeek/Qwen identifiers."""
    lower = model_name.lower()
    if "gemini" in lower:
        raise ValueError(
            "Gemini models are not supported in this pipeline. "
            "Use a DeepSeek or Qwen HuggingFace model."
        )

    # Allow local LoRA adapter directories by validating their base model.
    if os.path.isdir(model_name):
        adapter_config = os.path.join(model_name, "adapter_config.json")
        if os.path.exists(adapter_config):
            with open(adapter_config, "r", encoding="utf-8") as fp:
                adapter_cfg = json.load(fp)
            base_model = str(adapter_cfg.get("base_model_name_or_path", "")).strip()
            if not base_model:
                raise ValueError(
                    f"adapter_config.json at {model_name} is missing base_model_name_or_path"
                )
            if base_model == model_name:
                raise ValueError(
                    f"Invalid adapter base model reference in {adapter_config}: {base_model}"
                )
            _check_is_deepseek_qwen(base_model)
            return

        # For local model directories without adapter metadata, allow and defer
        # compatibility validation to runtime model loading.
        return

    if "deepseek" not in lower and "qwen" not in lower:
        raise ValueError(
            f"This pipeline only supports DeepSeek / Qwen models. Got: {model_name}"
        )


def _find_latest_lora_checkpoint(output_dir: str) -> str:
    """Find the latest LoRA checkpoint directory under output_dir.

    Preference order:
      1) Highest iter_NN directory, then highest checkpoint-STEP inside it.
      2) Iter directory itself if it contains adapter_config.json.
      3) output_dir itself if it contains adapter_config.json.
    """
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    iter_dirs = [
        path
        for path in glob.glob(os.path.join(output_dir, "iter_*"))
        if os.path.isdir(path)
    ]

    def _iter_index(path: str) -> int:
        name = os.path.basename(path)
        try:
            return int(name.split("iter_")[-1])
        except Exception:
            return -1

    candidate_root = max(iter_dirs, key=_iter_index) if iter_dirs else output_dir

    checkpoint_dirs = [
        path
        for path in glob.glob(os.path.join(candidate_root, "checkpoint-*"))
        if os.path.isdir(path)
    ]

    def _checkpoint_step(path: str) -> int:
        name = os.path.basename(path)
        try:
            return int(name.split("checkpoint-")[-1])
        except Exception:
            return -1

    if checkpoint_dirs:
        return max(checkpoint_dirs, key=_checkpoint_step)

    if os.path.exists(os.path.join(candidate_root, "adapter_config.json")):
        return candidate_root
    if os.path.exists(os.path.join(output_dir, "adapter_config.json")):
        return output_dir

    raise FileNotFoundError(
        "No LoRA checkpoint found. Expected checkpoint-* or adapter_config.json under "
        f"{candidate_root} (or {output_dir})."
    )


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class TraceExample:
    problem: str
    solution: str
    reflection: str = ""
    insight_book: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class BaselineTunePipeline(BenchmarkDomainPipeline):
    """Reuses benchmark dataset loading and the ChainOfThoughtReader for
    trace generation; adds LoRA fine-tuning in step 2."""

    def __init__(self, **kwargs):
        # Force HuggingFace path — no Gemini
        kwargs["use_api"] = False
        kwargs["api_key"] = None
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Problem formatting (dataset-specific)
    # ------------------------------------------------------------------

    def _format_problem(
        self, problem_data: Dict, dataset_name: str
    ) -> Tuple[str, str]:
        """Return (problem_text, ground_truth) using dataset-specific formatters.

        Mirrors the inline formatter dispatch in
        BenchmarkDomainPipeline._extract_insights_for_dataset.
        """
        problem_text: Optional[str] = None
        ground_truth: str = ""

        try:
            if dataset_name in ("aime25", "aime24"):
                from math_datasets.aime25 import aime25_formatter
                problem_text, ground_truth = aime25_formatter(problem_data)
            elif "livemathbench" in dataset_name:
                from math_datasets.livemathbench import livemathbench_formatter
                problem_text, ground_truth = livemathbench_formatter(problem_data, dataset_name)
            elif dataset_name.startswith("imo"):
                from math_datasets.imo_benchmark import imo_formatter
                problem_text, ground_truth = imo_formatter(problem_data, dataset_name)
            elif dataset_name == "math500":
                from math_datasets.math500 import math500_formatter
                problem_text, ground_truth = math500_formatter(problem_data)
            elif dataset_name.startswith("gsm8k"):
                from math_datasets.gsm8k import gsm8k_formatter
                problem_text, ground_truth = gsm8k_formatter(problem_data)
            elif dataset_name.startswith("gpqa"):
                from science_datasets.gpqa import gpqa_formatter
                problem_text, ground_truth = gpqa_formatter(problem_data)
            elif "livecodebench" in dataset_name:
                from code_datasets.livecodebench import livecodebench_formatter
                problem_text, _ = livecodebench_formatter(problem_data)
                ground_truth = ""
        except Exception:
            pass

        if not problem_text:
            problem_text = (
                problem_data.get("problem")
                or problem_data.get("question", "")
            )
        if not ground_truth:
            ground_truth = (
                problem_data.get("answer")
                or problem_data.get("solution", "")
            )

        return problem_text, ground_truth

    # ------------------------------------------------------------------
    # STEP 1 — Trace generation
    # ------------------------------------------------------------------

    def _generate_traces_for_dataset(
        self,
        dataset_name: str,
        max_problems: Optional[int],
        trace_root: str,
    ) -> str:
        """Generate and save reasoning traces for one dataset."""
        self._ensure_client()

        problems = self.load_math_dataset(dataset_name)
        worklist = problems[:max_problems] if max_problems else problems

        dataset_dir = os.path.join(trace_root, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        progress_path = os.path.join(dataset_dir, "step1_progress.json")

        total_problems = len(worklist)
        solved_count = 0
        correct_count = 0
        failed_count = 0
        skipped_count = 0

        def _write_step1_progress(last_problem_idx: int, status: str) -> None:
            progress_payload = {
                "dataset": dataset_name,
                "trace_dir": dataset_dir,
                "total_problems": total_problems,
                "processed": last_problem_idx,
                "solved": solved_count,
                "correct": correct_count,
                "incorrect": max(solved_count - correct_count, 0),
                "failed": failed_count,
                "skipped": skipped_count,
                "accuracy_on_solved": (
                    float(correct_count / solved_count) if solved_count > 0 else None
                ),
                "accuracy_on_processed": (
                    float(correct_count / last_problem_idx)
                    if last_problem_idx > 0
                    else None
                ),
                "last_problem_idx": last_problem_idx,
                "last_status": status,
            }
            with open(progress_path, "w", encoding="utf-8") as fp:
                json.dump(progress_payload, fp, indent=2, ensure_ascii=False)

        print(
            f"[Step 1] Generating traces for '{dataset_name}': "
            f"{len(worklist)} problems -> {dataset_dir}"
        )

        for idx, problem_data in enumerate(worklist, 1):
            problem_text, ground_truth = self._format_problem(problem_data, dataset_name)
            if not problem_text:
                print(f"  [skip] {dataset_name} #{idx}: missing problem text")
                skipped_count += 1
                _write_step1_progress(last_problem_idx=idx, status="skipped")
                continue

            print(f"  [{idx}/{len(worklist)}] {problem_text[:80]}...")
            try:
                result = self.client.solve_problem(task=problem_text)
            except Exception as exc:
                print(f"  [error] {dataset_name} #{idx}: {exc}")
                failed_count += 1
                _write_step1_progress(last_problem_idx=idx, status="error")
                continue

            solution = result.get("solution", "")
            predicted = self._extract_answer_from_solution(
                solution, dataset_name, problem_data
            )
            is_correct = False
            if predicted and ground_truth:
                is_correct = self._check_answer_match(
                    predicted, ground_truth, dataset_name, problem_text
                )

            solved_count += 1
            if is_correct:
                correct_count += 1

            trace_payload = {
                "dataset": dataset_name,
                "problem_id": problem_data.get("id", idx),
                "problem": problem_text,
                "solution": solution,
                "reflection": result.get("reflection", ""),
                "insight_book": result.get("insight_book", {}),
                "predicted_answer": predicted,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
            }

            out_path = os.path.join(dataset_dir, f"problem_{idx:04d}.json")
            with open(out_path, "w", encoding="utf-8") as fp:
                json.dump(trace_payload, fp, indent=2, ensure_ascii=False)

            _write_step1_progress(last_problem_idx=idx, status="saved")
            running_acc = (correct_count / solved_count) if solved_count > 0 else 0.0
            print(
                f"  [progress] {dataset_name}: solved={solved_count}/{total_problems}, "
                f"correct={correct_count}, running_acc={running_acc:.4f}"
            )

        final_acc = (correct_count / solved_count) if solved_count > 0 else 0.0
        print(
            f"[Step 1] Completed '{dataset_name}': solved={solved_count}, "
            f"correct={correct_count}, failed={failed_count}, skipped={skipped_count}, "
            f"accuracy={final_acc:.4f}"
        )

        return dataset_dir

    def generate_traces(
        self,
        datasets: List[str],
        max_problems: Optional[int],
        trace_root: str,
    ) -> Dict[str, str]:
        """Run Step 1 for all datasets.  Returns dataset -> trace_dir map."""
        trace_dirs: Dict[str, str] = {}
        overall_progress_path = os.path.join(trace_root, "step1_overall_progress.json")
        overall_progress = {
            "trace_root": trace_root,
            "datasets_total": len(datasets),
            "datasets_completed": 0,
            "datasets": {},
        }
        for dataset in datasets:
            trace_dirs[dataset] = self._generate_traces_for_dataset(
                dataset_name=dataset,
                max_problems=max_problems,
                trace_root=trace_root,
            )

            progress_path = os.path.join(trace_dirs[dataset], "step1_progress.json")
            if os.path.exists(progress_path):
                with open(progress_path, "r", encoding="utf-8") as fp:
                    overall_progress["datasets"][dataset] = json.load(fp)
            overall_progress["datasets_completed"] += 1

            with open(overall_progress_path, "w", encoding="utf-8") as fp:
                json.dump(overall_progress, fp, indent=2, ensure_ascii=False)

            print(
                f"[Step 1] Overall progress: "
                f"{overall_progress['datasets_completed']}/{overall_progress['datasets_total']} datasets completed"
            )

        return trace_dirs


# ---------------------------------------------------------------------------
# Trace collection (shared by both run modes)
# ---------------------------------------------------------------------------

def collect_trace_examples(
    trace_root: str,
    datasets: List[str],
) -> List[TraceExample]:
    """Scan <trace_root>/<dataset>/problem_*.json and return TraceExample list.

    Accepts files written by both this pipeline and by
    task_benchmark_domain.py (which also saves problem_NNNN.json files;
    those may lack the 'solution' key, in which case the insight_book
    descriptions are joined as a fallback).

    All traces are collected regardless of correctness.
    """
    examples: List[TraceExample] = []

    for dataset in datasets:
        dataset_dir = os.path.join(trace_root, dataset)
        files = sorted(glob.glob(os.path.join(dataset_dir, "problem_*.json")))

        if not files:
            print(f"  [collect] No traces found under {dataset_dir} — skipping")
            continue

        loaded = 0
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as fp:
                    payload = json.load(fp)
            except Exception as exc:
                print(f"  Warning: failed to read {file_path}: {exc}")
                continue

            problem = str(payload.get("problem", "")).strip()
            solution = str(payload.get("solution", "")).strip()
            reflection = str(payload.get("reflection", "")).strip()
            insight_book = payload.get("insight_book", {})

            # Fallback: build solution text from insight_book when solution absent
            if not solution and isinstance(insight_book, dict) and insight_book:
                solution = "\n\n".join(
                    f"{k}: {v}" for k, v in insight_book.items()
                )

            if not problem or not solution:
                continue
            if not isinstance(insight_book, dict):
                insight_book = {}

            examples.append(
                TraceExample(
                    problem=problem,
                    solution=solution,
                    reflection=reflection,
                    insight_book=insight_book,
                )
            )
            loaded += 1

        print(f"  [collect] '{dataset}': {loaded} traces loaded from {dataset_dir}")

    print(f"  Total trace examples: {len(examples)}")
    return examples


# ---------------------------------------------------------------------------
# SFT data preparation
# ---------------------------------------------------------------------------

def build_sft_jsonl(
    examples: List[TraceExample],
    output_jsonl: str,
) -> int:
    """Write examples to a JSONL file with 'prompt' / 'completion' fields."""
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)

    count = 0
    with open(output_jsonl, "w", encoding="utf-8") as fp:
        for ex in examples:
            prompt = (
                "Problem: " + ex.problem + "\n\n"
                "Please solve the problem step by step, showing your full reasoning."
            )
            completion = ex.solution

            fp.write(
                json.dumps(
                    {"prompt": prompt, "completion": completion},
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    return count


# ---------------------------------------------------------------------------
# LoRA fine-tuning (Step 2)
# ---------------------------------------------------------------------------

def run_lora_sft(
    model_name: str,
    train_jsonl: str,
    output_dir: str,
    *,
    learning_rate: float = 2e-4,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 16,
    max_seq_length: int = 4096,
    warmup_ratio: float = 0.03,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    seed: int = 42,
) -> None:
    """Fine-tune model with LoRA on collected reasoning traces.

    Implements the simplified alignment from RPAM Step 2 (arxiv 2601.03506v1):

      Paper objective:
        L = ||z_merged - z_pos||² + ω · contrastive(z_merged, z_pos, z_neg)

      Here (no model merging, all traces used):
        L = cross_entropy(model(prompt), trace)   [via LoRA SFT]

    The cross-entropy objective on all reasoning traces directly drives the
    model's hidden representations toward the demonstrated reasoning style.

    LoRA hyper-parameters (best defaults for DeepSeek/Qwen 7-14B):
      r=16, alpha=32 (2x rank), dropout=0.05 — target all projection layers.
      lr=2e-4, 3 epochs, effective batch 32, cosine decay, warmup 3%.
      These correspond to the best region of the paper's calibration grid
      ({lr: 0.001, epochs: 100} for coefficient optimisation), scaled to
      the gradient-based weight update regime of LoRA SFT.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            "Missing training dependencies. "
            "Install with: pip install peft transformers datasets"
        ) from exc

    _check_is_deepseek_qwen(model_name)

    print(f"\n[Step 2] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Step 2] Loading base model: {model_name}")
    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
    )

    # LoRA configuration — target all linear projection layers common to
    # DeepSeek-R1-Distill-Qwen and DeepSeek-R1-Distill-Llama architectures.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # required when gradient_checkpointing=True
    model.print_trainable_parameters()

    # ---- Dataset ----
    print(f"[Step 2] Loading training data from {train_jsonl}")
    raw_ds = hf_load_dataset("json", data_files=train_jsonl, split="train")

    def _preprocess(row):
        prompt = row["prompt"]
        completion = row["completion"]

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

        # Concatenate prompt + completion + EOS
        input_ids = prompt_ids + completion_ids + [tokenizer.eos_token_id]
        # Mask prompt tokens from the loss so the model learns to generate the trace
        labels = [-100] * len(prompt_ids) + completion_ids + [tokenizer.eos_token_id]

        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            labels = labels[:max_seq_length]

        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_ds = raw_ds.map(_preprocess, remove_columns=raw_ds.column_names)
    print(f"  Training examples: {len(train_ds)}")

    def _collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch: Dict[str, List] = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [tokenizer.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

    # ---- Training arguments ----
    os.makedirs(output_dir, exist_ok=True)
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        seed=seed,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=_collate,
    )

    print(
        f"\n[Step 2] Starting LoRA fine-tuning:\n"
        f"  r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}\n"
        f"  lr={learning_rate}, epochs={num_train_epochs}, "
        f"effective batch={per_device_train_batch_size * gradient_accumulation_steps}\n"
        f"  max_seq_length={max_seq_length}, warmup_ratio={warmup_ratio}"
    )
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save a config record for reproducibility
    tune_config = {
        "base_model": model_name,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": per_device_train_batch_size * gradient_accumulation_steps,
        "max_seq_length": max_seq_length,
        "warmup_ratio": warmup_ratio,
        "n_training_examples": len(train_ds),
        "alignment_ref": "arxiv 2601.03506v1 (RPAM) Step 2, simplified to all-trace SFT",
    }
    with open(os.path.join(output_dir, "tune_config.json"), "w", encoding="utf-8") as fp:
        json.dump(tune_config, fp, indent=2)

    print(f"\n  LoRA adapter saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Baseline fine-tuning: Step 1 — generate reasoning traces, "
            "Step 2 — LoRA SFT on all collected traces.  Runs for "
            "--num-iterations rounds.  --skip-step skips Step 1 of the "
            "first iteration only; all subsequent iterations run both steps."
        )
    )

    # Dataset / generation args
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["aime25"],
        help="Datasets for trace generation / collection (space- or comma-separated).",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Limit problems processed per dataset in Step 1.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help=(
            "Number of iterative rounds of (Step 1 + Step 2).  "
            "Iteration 1 with --skip-step runs Step 2 only; "
            "all subsequent iterations always run both steps (default: 1)."
        ),
    )
    parser.add_argument(
        "--skip-step",
        action="store_true",
        help=(
            "Skip Step 1 (trace generation) in the first iteration only.  "
            "Collects existing traces from <trace-root>/<dataset>/ directories.  "
            "From iteration 2 onward both steps always run."
        ),
    )
    parser.add_argument(
        "--trace-only",
        action="store_true",
        help="Run Step 1 only for all iterations; skip fine-tuning.",
    )
    parser.add_argument(
        "--trace-root",
        type=str,
        default="baseline_tune_traces",
        help="Root directory for per-dataset trace JSON files (default: baseline_tune_traces).",
    )
    parser.add_argument(
        "--train-jsonl",
        type=str,
        default=None,
        help=(
            "Path for the SFT JSONL file.  "
            "Defaults to <trace-root>/train_sft.jsonl."
        ),
    )

    # Model / device args
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="HuggingFace model identifier (DeepSeek / Qwen only).",
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default=None,
        help="Device for Step 1 generation (cuda / cpu).  Auto-detected if omitted.",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        default=False,
        help="Load the HuggingFace model with 8-bit quantisation for Step 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_tune_lora",
        help="Directory where the LoRA adapter will be saved (default: baseline_tune_lora).",
    )
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help=(
            "Evaluation-only mode: skip Step 1/2 and run benchmark eval by loading "
            "the latest LoRA checkpoint under --output-dir."
        ),
    )
    parser.add_argument(
        "--eval-checkpoint",
        type=str,
        default=None,
        help=(
            "Explicit LoRA adapter/checkpoint directory for evaluation. If omitted, "
            "the latest checkpoint under --output-dir is used."
        ),
    )
    # LoRA / training hyper-parameters
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4,
        help="Peak learning rate (default: 2e-4).",
    )
    parser.add_argument(
        "--num-train-epochs", type=int, default=3,
        help="Number of fine-tuning epochs (default: 3).",
    )
    parser.add_argument(
        "--per-device-train-batch-size", type=int, default=2,
        help="Per-device batch size (default: 2).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=16,
        help="Gradient accumulation steps (default: 16; effective batch = 2 x 16 = 32).",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=4096,
        help="Maximum token sequence length (default: 4096).",
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.03,
        help="Fraction of steps used for linear warmup (default: 0.03).",
    )
    parser.add_argument(
        "--lora-r", type=int, default=16,
        help="LoRA rank (default: 16).",
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=32,
        help="LoRA alpha scaling (default: 32 = 2 x rank).",
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.05,
        help="LoRA dropout probability (default: 0.05).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    datasets = _parse_list_arg(args.datasets)
    if not datasets:
        parser.error("--datasets is required")

    _check_is_deepseek_qwen(args.model)

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.eval_mode:
        eval_model = args.eval_checkpoint or _find_latest_lora_checkpoint(args.output_dir)
        _check_is_deepseek_qwen(eval_model)

        eval_output_dir = args.output_dir
        os.makedirs(eval_output_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("EVAL MODE")
        print("=" * 60)
        print(f"Model checkpoint: {eval_model}")
        print(f"Eval output dir: {eval_output_dir}")

        eval_pipeline = BaselineTunePipeline(
            model_name=eval_model,
            device=args.device,
            output_dir=eval_output_dir,
            mode="text",
            num_iterations=1,
            load_in_8bit=args.load_in_8bit,
        )

        eval_pipeline.run_eval_only(
            dataset_list=datasets,
            max_problems=args.max_problems,
        )
        print(f"\nDone. Eval summary saved under: {eval_output_dir}")
        return

    base_train_jsonl = args.train_jsonl or os.path.join(args.trace_root, "train_sft.jsonl")

    pipeline = BaselineTunePipeline(
        model_name=args.model,
        device=args.device,
        output_dir=args.trace_root,
        mode="text",
        num_iterations=1,
        load_in_8bit=args.load_in_8bit,
    )

    for iteration in range(1, args.num_iterations + 1):
        print("\n" + "=" * 60)
        print(f"ITERATION {iteration} / {args.num_iterations}")
        print("=" * 60)

        # Iteration-specific paths so each round's data and adapter are kept
        iter_trace_root = os.path.join(args.trace_root, f"iter_{iteration:02d}")
        iter_train_jsonl = base_train_jsonl.replace(
            ".jsonl", f"_iter{iteration:02d}.jsonl"
        )
        iter_output_dir = os.path.join(args.output_dir, f"iter_{iteration:02d}")

        # ---- Step 1 ----
        # Skip only on first iteration when --skip-step is set
        run_step1 = not (args.skip_step and iteration == 1)

        if run_step1:
            print(f"\n[Iter {iteration}] STEP 1: Generating reasoning traces")
            pipeline.output_dir = iter_trace_root
            pipeline.generate_traces(
                datasets=datasets,
                max_problems=args.max_problems,
                trace_root=iter_trace_root,
            )
        else:
            print(
                f"\n[Iter {iteration}] STEP 1 skipped (--skip-step on first iteration). "
                f"Collecting existing traces from {args.trace_root}."
            )
            # Collect from the original trace_root (not the iter sub-dir)
            iter_trace_root = args.trace_root

        if args.trace_only:
            continue

        # ---- Step 2 ----
        print(f"\n[Iter {iteration}] STEP 2: LoRA fine-tuning")

        examples = collect_trace_examples(
            trace_root=iter_trace_root,
            datasets=datasets,
        )
        if not examples:
            raise ValueError(
                f"No usable traces found for iteration {iteration}. "
                f"Check {iter_trace_root} for non-empty 'problem' and 'solution' fields."
            )

        num_rows = build_sft_jsonl(
            examples=examples,
            output_jsonl=iter_train_jsonl,
        )
        print(f"SFT dataset prepared: {num_rows} rows -> {iter_train_jsonl}")

        run_lora_sft(
            model_name=args.model,
            train_jsonl=iter_train_jsonl,
            output_dir=iter_output_dir,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        warmup_ratio=args.warmup_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
    )

    print(f"\nDone. LoRA adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
