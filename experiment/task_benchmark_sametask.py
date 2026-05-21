"""
Same-benchmark, multi-agent pipeline.

This variant keeps the benchmark fixed and runs an array of agent models on the
same dataset(s).  Each agent saves its own problem traces, then the server
aggregates all traces from all agents into a shared encyclopedia.

Example:
    python task_benchmark_sametask.py --datasets livemathbench_hard_2025 --agent-models gemini-3.1-pro-preview gemini-2.5-flash gemini-2.5-flash-lite gemini-2.5-pro --use-api --api-provider gemini --api-key "$GEMINI_API_KEY" --num-iterations 5 -o sametask_agents_output
"""

import argparse
import glob
import json
import os
import random
import re
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from server import InsightAggregationServer
from server_text import TextBasedInsightAggregationServer
from task_benchmark_domain import (
    BenchmarkDomainPipeline,
    _CLIENT_CHOICES,
    _parse_list_arg,
)
from utils import setup_gemini


def _slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.strip())
    slug = slug.strip("._-")
    return slug or "model"


class SameTaskBenchmarkPipeline(BenchmarkDomainPipeline):
    """Run the same benchmark with multiple agent models before aggregation."""

    def __init__(
        self,
        agent_models: List[str],
        server_model_name: Optional[str] = None,
        **kwargs,
    ):
        if not agent_models:
            raise ValueError("Provide at least one agent model")
        super().__init__(**kwargs)
        self.agent_models = agent_models
        self.root_output_dir = self.output_dir
        self.server_model_name = server_model_name or self.model_name

    def _configure_client_for_agent(self, agent_model: str) -> None:
        """Make the inherited client use this agent model."""
        self.model_name = agent_model
        self.client = None
        self._ensure_client()

        if self.use_api and self.api_provider == "gemini":
            self.client.gemini_model = setup_gemini(
                api_key=self.api_key,
                model_name=agent_model,
            )
        elif self.use_api and self.api_provider == "openrouter":
            self.client.api_model_name = agent_model

    def _build_agent_output_dir(self, iteration: int, agent_model: str) -> str:
        return os.path.join(
            self.root_output_dir,
            f"iter_{iteration:02d}",
            _slugify_model_name(agent_model),
        )

    def _collect_iteration_json_files(self, iteration_dir: str) -> List[str]:
        files = glob.glob(os.path.join(iteration_dir, "*", "*", "problem_*.json"))
        return sorted(files)

    def _aggregate_iteration_json_files(
        self,
        json_files: List[str],
        output_dir: str,
        r1: float,
        r2: float,
    ) -> Optional[str]:
        if not json_files:
            print("No problem trace files found for aggregation")
            return None

        os.makedirs(output_dir, exist_ok=True)
        print(f"\nAggregating {len(json_files)} trace files from all agent models...")

        if self.mode == "text":
            server = TextBasedInsightAggregationServer(
                model_name=self.server_model_name,
                device=self.device,
                input_dirs=[output_dir],
                use_api=self.use_api,
                api_key=self.api_key,
                api_provider=self.api_provider,
            )
            if self.use_api and self.api_provider == "gemini":
                server.gemini_model = setup_gemini(
                    api_key=self.api_key,
                    model_name=self.server_model_name,
                )
            elif self.use_api and self.api_provider == "openrouter":
                server.api_model_name = self.server_model_name

            result = server.aggregate_and_build_encyclopedia(
                json_files=json_files,
                output_dir=output_dir,
            )
            server.save_results(result, output_dir=output_dir)

            encyclopedia_dict = server._try_parse_json(server.encyclopedia)
            if encyclopedia_dict is None:
                json_content = server._extract_json_only(server.encyclopedia)
                encyclopedia_dict = server._try_parse_json(json_content)
            if encyclopedia_dict is None:
                raise ValueError("Could not parse aggregated encyclopedia as JSON")

            encyclopedia_path = os.path.join(output_dir, "encyclopedia_all.json")
            with open(encyclopedia_path, "w", encoding="utf-8") as f:
                json.dump(encyclopedia_dict, f, indent=2, ensure_ascii=False)
            return encyclopedia_path

        server = InsightAggregationServer(
            model_name=self.server_model_name,
            device=self.device,
            input_dir=output_dir,
            use_api=self.use_api,
            api_key=self.api_key,
            api_provider=self.api_provider,
        )
        if self.use_api and self.api_provider == "gemini":
            server.gemini_model = setup_gemini(
                api_key=self.api_key,
                model_name=self.server_model_name,
            )
        elif self.use_api and self.api_provider == "openrouter":
            server.api_model_name = self.server_model_name

        result = server.aggregate_and_build_encyclopedia(
            json_files=json_files,
            r1=r1,
            r2=r2,
            output_dir=output_dir,
        )
        server.save_results(result, output_dir=output_dir)
        encyclopedia_path = os.path.join(output_dir, "encyclopedia_all.txt")
        with open(encyclopedia_path, "w", encoding="utf-8") as f:
            f.write(server.encyclopedia)
        return encyclopedia_path

    def run_multi_agent_pipeline(
        self,
        dataset_list: List[str],
        max_problems: Optional[int],
        r1: float = 0.95,
        r2: float = 0.4,
        start_from_step: int = 1,
        initial_encyclopedia_paths: Optional[List[str]] = None,
    ) -> Dict:
        if not dataset_list:
            raise ValueError("Provide at least one benchmark dataset")

        os.makedirs(self.root_output_dir, exist_ok=True)
        start_time = time.time()
        iteration_history = []
        encyclopedia_paths = initial_encyclopedia_paths

        print(f"\n{'=' * 80}")
        print("Starting Same-Benchmark Multi-Agent Pipeline")
        print(f"Benchmarks: {', '.join(dataset_list)}")
        print(f"Agent models: {', '.join(self.agent_models)}")
        print(f"Server model: {self.server_model_name}")
        print(f"Iterations: {self.num_iterations}")
        print(f"Max problems per benchmark: {max_problems or 'all'}")
        print(f"{'=' * 80}\n")

        for iteration in range(1, self.num_iterations + 1):
            iteration_dir = os.path.join(self.root_output_dir, f"iter_{iteration:02d}")
            os.makedirs(iteration_dir, exist_ok=True)

            print(f"\n{'=' * 80}")
            print(f"ITERATION {iteration}/{self.num_iterations}")
            print(f"{'=' * 80}")

            model_accuracy: Dict[str, Dict[str, float]] = {}

            if start_from_step == 1:
                for agent_model in self.agent_models:
                    agent_output_dir = self._build_agent_output_dir(iteration, agent_model)
                    os.makedirs(agent_output_dir, exist_ok=True)

                    print(f"\n{'-' * 80}")
                    print(f"Agent model: {agent_model}")
                    print(f"Output dir: {agent_output_dir}")
                    print(f"{'-' * 80}")

                    self.output_dir = agent_output_dir
                    self._configure_client_for_agent(agent_model)
                    _, accuracy_map = self.learn_insights_from_datasets(
                        dataset_list,
                        max_problems,
                        encyclopedia_paths,
                        iteration,
                    )
                    model_accuracy[agent_model] = accuracy_map
            else:
                print("Skipping agent trace generation; aggregating existing traces")

            json_files = self._collect_iteration_json_files(iteration_dir)
            if not json_files:
                raise FileNotFoundError(
                    f"No problem_*.json trace files found under {iteration_dir}"
                )

            aggregation_dir = os.path.join(iteration_dir, "aggregation")
            encyclopedia_path = self._aggregate_iteration_json_files(
                json_files=json_files,
                output_dir=aggregation_dir,
                r1=r1,
                r2=r2,
            )
            encyclopedia_paths = [encyclopedia_path] if encyclopedia_path else None

            iteration_summary = {
                "iteration": iteration,
                "agent_models": self.agent_models,
                "datasets": dataset_list,
                "accuracy_by_agent": model_accuracy,
                "trace_files": len(json_files),
                "encyclopedia": encyclopedia_path,
            }
            iteration_history.append(iteration_summary)

            print(f"\nIteration {iteration} Summary:")
            for agent_model, accuracy_map in model_accuracy.items():
                pretty = ", ".join(
                    f"{dataset}={accuracy:.2%}"
                    for dataset, accuracy in accuracy_map.items()
                )
                print(f"  - {agent_model}: {pretty}")
            if encyclopedia_path:
                print(f"  Encyclopedia: {encyclopedia_path}")

        self.output_dir = self.root_output_dir
        final_summary = {
            "mode": "same_benchmark_multi_agent",
            "num_iterations": self.num_iterations,
            "datasets": dataset_list,
            "agent_models": self.agent_models,
            "server_model": self.server_model_name,
            "iteration_history": iteration_history,
            "total_time_seconds": time.time() - start_time,
        }

        summary_path = os.path.join(self.root_output_dir, "sametask_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 80}")
        print("SAME-BENCHMARK MULTI-AGENT PIPELINE COMPLETE")
        print(f"{'=' * 80}")
        print(f"Summary saved: {summary_path}")
        return final_summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the same benchmark with multiple agent models, then aggregate "
            "all model traces into one encyclopedia."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["aime25"],
        help="Benchmark datasets to run for every agent model.",
    )
    parser.add_argument(
        "--agent-models",
        nargs="+",
        default=[
            "gemini-3.1-pro-preview",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
        ],
        help="Agent model array. Each model runs the same benchmark.",
    )
    parser.add_argument(
        "--server-model",
        type=str,
        default=None,
        help="Model used by the aggregation server. Defaults to --model.",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Limit problems per dataset per agent.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gemini-3.1-pro-preview",
        help="Default model and fallback server model.",
    )
    parser.add_argument(
        "-d", "--device", type=str, default=None, help="Device to use for HF models."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="sametask_agents_output",
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
        "--use-api",
        action="store_true",
        help="Use API provider instead of HuggingFace models.",
    )
    parser.add_argument(
        "--api-provider",
        type=str,
        default="gemini",
        choices=["gemini", "openrouter"],
        help="API provider for agent and server models.",
    )
    parser.add_argument(
        "--api-key", type=str, default=None, help="API key for the provider."
    )
    parser.add_argument(
        "--load-in-8bit",
        type=bool,
        default=False,
        help="Load HuggingFace models with 8-bit quantization.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="text",
        choices=["normal", "text"],
        help="Aggregation mode.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Rounds of multi-agent solve + aggregate.",
    )
    parser.add_argument(
        "--start-from-step",
        type=int,
        default=1,
        choices=[1, 2],
        help="1=run agents then aggregate; 2=aggregate existing traces only.",
    )
    parser.add_argument(
        "--encyclopedia",
        type=str,
        nargs="*",
        default=None,
        help="Optional initial encyclopedia guidance for the first agent round.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Eval-only mode inherited from task_benchmark_domain.py.",
    )
    parser.add_argument(
        "--client",
        type=str,
        default="default",
        choices=_CLIENT_CHOICES,
        help="Client algorithm to use for trace extraction.",
    )

    args = parser.parse_args()

    datasets = _parse_list_arg(args.datasets)
    agent_models = _parse_list_arg(args.agent_models)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {args.seed}")

    pipeline = SameTaskBenchmarkPipeline(
        agent_models=agent_models or [],
        server_model_name=args.server_model or args.model,
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

        if args.eval_only:
            pipeline.run_eval_only(
                dataset_list=datasets,
                max_problems=args.max_problems,
                encyclopedia_paths=args.encyclopedia,
            )
        else:
            pipeline.run_multi_agent_pipeline(
                dataset_list=datasets,
                max_problems=args.max_problems,
                r1=args.r1,
                r2=args.r2,
                start_from_step=args.start_from_step,
                initial_encyclopedia_paths=args.encyclopedia,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
