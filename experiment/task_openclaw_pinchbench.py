"""
OpenClaw PinchBench Pipeline
Runs PinchBench tasks with OpenClaw agents, extracts reasoning traces from
transcripts, and aggregates them into an insight library via server_text.py.

Structure mirrors task_paper_insight_reading.py / task_benchmark_domain.py:

  Iteration N:
    Step 1: Run each pinchbench task through the OpenClaw agent
            (calls pinchbench scripts/lib_agent.py execute_openclaw_task)
    Step 2: Reflection — extract procedural knowledge from transcript
    Step 3: Insight extraction — package as reusable traces (JSON)
    Save:   problem_XXXX.json  (same format as other pipelines)

  Aggregation:
    Call server_text.py TextBasedInsightAggregationServer to build an
    encyclopedia from all extracted insights.

  Next Iteration:
    Write the encyclopedia as INSIGHTS.md into the agent workspace.
    prepare_task_workspace() in lib_agent.py preserves it across cleanups
    and injects a mandatory "read INSIGHTS.md first" instruction into
    BOOTSTRAP.md, so the agent is hardcoded to read and apply the insights.

Usage:
    python task_openclaw_pinchbench.py \\
        --model anthropic/claude-sonnet-4 \\
        --output-dir pinchbench_output \\
        --suite automated-only \\
        --use-api --api-provider gemini --api-key YOUR_KEY \\
        --iterations 2

    # Start from aggregation step (skip task execution):
    python task_openclaw_pinchbench.py --start-from-step2 \\
        --output-dir pinchbench_output --use-api --api-provider gemini --api-key YOUR_KEY

    # Start from evaluation with existing encyclopedia:
    python task_openclaw_pinchbench.py \\
        --model anthropic/claude-sonnet-4 \\
        --encyclopedia pinchbench_output/encyclopedia.json \\
        --output-dir pinchbench_output_iter2 \\
        --use-api --api-provider gemini --api-key YOUR_KEY
"""

import argparse
import collections
import datetime
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Import pinchbench internals (must be on sys.path)
# ---------------------------------------------------------------------------
_PINCHBENCH_SCRIPTS = Path(__file__).parent / "pinchbench" / "scripts"
if str(_PINCHBENCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PINCHBENCH_SCRIPTS))

from lib_agent import (
    cleanup_agent_sessions,
    configure_bench_models,
    ensure_agent_exists,
    execute_openclaw_task,
    slugify_model,
    validate_openrouter_model,
    ModelValidationError,
    _get_agent_workspace,
)
from lib_grading import grade_task
from lib_tasks import TaskLoader

# ---------------------------------------------------------------------------
# Import our local pipeline pieces
# ---------------------------------------------------------------------------
from client import ChainOfThoughtReader
from server_text import TextBasedInsightAggregationServer
from utils import call_gemini_thinking


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Predefined OpenClaw skills activated by --openclaw-skill
# ---------------------------------------------------------------------------
_PREDEFINED_OPENCLAW_SKILLS: List[str] = [
    "nano-pdf",
    "summarize",
    "github",
    "notion",
    "slack",
    "weather",
    "coding-agent",
]

# V1 tasks from pinchbench/skill at 417bfce28ab55aad09386ec08caf75d7d3b5827a
# (Apr 9 tree), mapped to the current renamed task IDs.
_PINCHBENCH_V1_TASK_IDS = {
    "task_sanity",
    "task_calendar",
    "task_stock",
    "task_blog",
    "task_weather",
    "task_summary",
    "task_events",
    "task_email",
    "task_memory",
    "task_files",
    "task_workflow",
    "task_clawdhub",
    "task_skill_search",
    "task_image_gen",
    "task_humanizer",
    "task_daily_summary",
    "task_email_triage",
    "task_email_search",
    "task_market_research",
    "task_spreadsheet_summary",
    "task_eli5_pdf_summary",
    "task_openclaw_comprehension",
    "task_second_brain",
}


def _install_openclaw_skills(skills: List[str]) -> None:
    """
    Install OpenClaw skills into the main workspace so that
    prepare_task_workspace() copies them to every task workspace.

    Runs `openclaw install <skill>` for each name. Failures are logged
    as warnings rather than aborting the benchmark.
    """
    import subprocess as _sp
    openclaw_bin = shutil.which("openclaw") or os.environ.get("OPENCLAW_PATH", "openclaw")
    for skill in skills:
        print(f"  [skill] Installing: {skill}")
        try:
            result = _sp.run(
                [openclaw_bin, "skills", "install", skill],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                print(f"  [skill] Installed: {skill}")
            else:
                print(f"  [skill] Warning: install failed for '{skill}' (rc={result.returncode}): {result.stderr.strip()[:200]}")
        except Exception as exc:
            print(f"  [skill] Warning: could not install '{skill}': {exc}")


def _check_openclaw() -> str:
    """
    Verify OpenClaw is available and return the command used to invoke it.
    Raises SystemExit with install instructions if not found.
    """
    path = shutil.which("openclaw")
    if path:
        return path

    override_path = os.environ.get("OPENCLAW_PATH")
    if override_path and Path(override_path).exists():
        return override_path

    print(
        "\n" + "=" * 70 + "\n"
        "ERROR: OpenClaw CLI not found.\n\n"
        "PinchBench requires the OpenClaw binary to be installed and available.\n"
        "Install it from: https://github.com/openclaw/openclaw\n\n"
        "Typical install:\n"
        "  npm install -g openclaw        # Node.js / npm\n"
        "  # or follow the instructions at https://openclaw.dev\n\n"
        "After installing, make sure 'openclaw' is on your PATH:\n"
        "  which openclaw   # should print a path\n"
        "Or set OPENCLAW_PATH to an absolute openclaw binary path.\n"
        + "=" * 70 + "\n"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_transcript_text(transcript: List[Dict[str, Any]]) -> str:
    """
    Extract human-readable text from an OpenClaw transcript.

    Transcript entries look like:
        {"type": "message", "message": {"role": "assistant", "content": "..."}}

    We concatenate all assistant messages (and optionally tool results)
    to form the "solution" that will be reflected on.
    """
    parts = []
    for entry in transcript:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if isinstance(content, list):
            # Content can be a list of blocks
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "") or block.get("content", "")
                    if text:
                        parts.append(f"[{role}]: {text}")
        elif role == "assistant":
            parts.append(f"[assistant]: {content}")
    return "\n\n".join(parts)


def _write_insights_to_workspace(
    agent_id: str,
    encyclopedia_path: str,
    workspace: Optional[Path] = None,
) -> bool:
    """
    Write the encyclopedia as INSIGHTS.md.

    Writes directly to the OpenClaw agent workspace.
    Pass *workspace* to skip the openclaw-agents-list query (preferred).

    Returns True if the workspace write succeeded.
    """
    try:
        with open(encyclopedia_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  Warning: failed to load encyclopedia: {exc}")
        return False

    # Format into readable markdown
    if isinstance(data, dict) and set(data.keys()) == {"insight"}:
        body = data["insight"]
    elif isinstance(data, dict):
        lines = []
        for name, desc in data.items():
            lines.append(f"### {name}\n{desc}\n")
        body = "\n".join(lines)
    else:
        body = str(data)

    content = (
        "# Insight Library\n\n"
        "This file contains reasoning traces and techniques extracted from solving "
        "tasks similar to yours. Read this file carefully before starting each task "
        "and apply the relevant insights.\n\n"
        f"{body}\n"
    )

    # Use provided workspace or fall back to querying openclaw
    if workspace is None:
        workspace = _get_agent_workspace(agent_id)
    if workspace is None:
        print("  Warning: failed to resolve OpenClaw workspace")
        return False

    workspace.mkdir(parents=True, exist_ok=True)
    insights_path = workspace / "INSIGHTS.md"
    insights_path.write_text(content, encoding="utf-8")
    print(f"  Written INSIGHTS.md to agent workspace: {insights_path}")

    return insights_path.exists()


def _iter_dict_nodes(node: Any):
    """Yield all dict nodes recursively from arbitrary JSON-like structures."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_dict_nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_dict_nodes(value)


def _analyze_tool_calls(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Best-effort analysis of tool call usage from an OpenClaw transcript.

    Tracks:
      - tool names used
      - number of calls
      - per-call success/error status (error = explicit tool-return error)
    """
    call_type_markers = {
        "tool_call",
        "tool_use",
        "function_call",
        "tool_invocation",
        "tool-request",
    }
    result_type_markers = {
        "tool_result",
        "tool_return",
        "function_result",
        "tool_response",
        "tool-output",
    }

    def _extract_name(node: Dict[str, Any]) -> Optional[str]:
        for key in ("name", "tool_name", "tool", "function", "function_name"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        nested_fn = node.get("function")
        if isinstance(nested_fn, dict):
            fn_name = nested_fn.get("name")
            if isinstance(fn_name, str) and fn_name.strip():
                return fn_name.strip()
        return None

    def _extract_call_id(node: Dict[str, Any]) -> Optional[str]:
        for key in ("call_id", "tool_call_id", "toolUseId", "id", "tool_use_id"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _looks_like_error(node: Dict[str, Any]) -> bool:
        if node.get("is_error") is True or node.get("isError") is True:
            return True
        status = node.get("status")
        if isinstance(status, str) and status.lower() in {"error", "failed", "failure"}:
            return True
        if node.get("error"):
            return True
        return False

    calls: List[Dict[str, Any]] = []
    calls_by_id: Dict[str, Dict[str, Any]] = {}

    for entry in transcript:
        for node in _iter_dict_nodes(entry):
            node_type = node.get("type")
            node_type_lower = node_type.lower() if isinstance(node_type, str) else ""

            if node_type_lower in call_type_markers:
                name = _extract_name(node) or "unknown_tool"
                call = {
                    "name": name,
                    "call_id": _extract_call_id(node),
                    "status": "unknown",
                    "error": None,
                }
                calls.append(call)
                if call["call_id"]:
                    calls_by_id[call["call_id"]] = call
                continue

            if node_type_lower in result_type_markers:
                call_id = _extract_call_id(node)
                has_error = _looks_like_error(node)
                error_msg = node.get("error")
                if not isinstance(error_msg, str) and error_msg is not None:
                    error_msg = str(error_msg)

                linked = calls_by_id.get(call_id) if call_id else None
                if linked is not None:
                    linked["status"] = "error" if has_error else "ok"
                    linked["error"] = error_msg if has_error else None
                else:
                    name = _extract_name(node) or "unknown_tool"
                    calls.append(
                        {
                            "name": name,
                            "call_id": call_id,
                            "status": "error" if has_error else "ok",
                            "error": error_msg if has_error else None,
                        }
                    )

    # Fallback for transcripts that don't expose explicit call/result typing:
    # infer from message content blocks that include tool names.
    if not calls:
        for entry in transcript:
            if entry.get("type") != "message":
                continue
            message = entry.get("message", {})
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = str(block.get("type", "")).lower()
                if "tool" not in block_type and block_type not in {"function_call", "function_result"}:
                    continue
                name = _extract_name(block) or "unknown_tool"
                status = "error" if _looks_like_error(block) else "ok"
                error_msg = block.get("error")
                if not isinstance(error_msg, str) and error_msg is not None:
                    error_msg = str(error_msg)
                calls.append(
                    {
                        "name": name,
                        "call_id": _extract_call_id(block),
                        "status": status,
                        "error": error_msg if status == "error" else None,
                    }
                )

    tool_names = sorted({call["name"] for call in calls if call.get("name")})
    tool_counter = collections.Counter(call["name"] for call in calls if call.get("name"))
    ok_calls = sum(1 for call in calls if call.get("status") == "ok")
    error_calls = sum(1 for call in calls if call.get("status") == "error")
    unknown_calls = sum(1 for call in calls if call.get("status") == "unknown")

    return {
        "tool_names": tool_names,
        "tool_name_counts": dict(sorted(tool_counter.items())),
        "total_tool_calls": len(calls),
        "successful_tool_calls": ok_calls,
        "error_tool_calls": error_calls,
        "unknown_status_tool_calls": unknown_calls,
        "calls": calls,
    }


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class OpenClawPinchBenchPipeline:
    """
    Pipeline for running PinchBench tasks, extracting reasoning traces,
    and aggregating them into an insight library.
    """

    def __init__(
        self,
        model_id: str,
        output_dir: str = "pinchbench_output",
        suite: str = "all",
        pinchbench_dir: Optional[str] = None,
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
        api_model: str = "gemini-3-pro-preview",
        base_url: Optional[str] = None,
        openclaw_api_key: Optional[str] = None,
        timeout_multiplier: float = 1.0,
        encyclopedia_path: Optional[str] = None,
        trace_folder: Optional[str] = None,
        judge_model: Optional[str] = None,
        thinking_level: Optional[str] = "high",
        openclaw_skill: bool = False,
        exclude_v1: bool = False,
    ):
        self.model_id = model_id
        self.output_dir = output_dir
        self.suite = suite
        self.use_api = use_api
        self.api_provider = api_provider
        self.api_key = api_key or (
            os.getenv("GEMINI_API_KEY") if api_provider == "gemini" else os.getenv("OPENROUTER_API_KEY")
        )
        self.api_model = api_model
        self.base_url = base_url
        self.openclaw_api_key = openclaw_api_key
        self.timeout_multiplier = timeout_multiplier
        self.encyclopedia_path = encyclopedia_path
        self.trace_folder = trace_folder
        # Judge model for LLM-judge tasks; defaults to the same model as the agent
        self.judge_model = judge_model or model_id
        self.thinking_level = thinking_level  # "low" / "medium" / "high" / None
        self.openclaw_skill = openclaw_skill
        self.exclude_v1 = exclude_v1

        # Ensure downstream judge/API helpers that read GEMINI_API_KEY from
        # environment can use the CLI-provided key.
        if self.api_provider == "gemini" and self.api_key:
            os.environ["GEMINI_API_KEY"] = self.api_key

        # Pinchbench skill root
        if pinchbench_dir:
            self.skill_dir = Path(pinchbench_dir)
        else:
            self.skill_dir = Path(__file__).parent / "pinchbench"

        self.tasks_dir = self.skill_dir / "tasks"

        # Agent identifier — includes a short hash of output_dir so each run
        # gets its own workspace and INSIGHTS.md does not bleed across runs
        # that share the same model but differ in encyclopedia / output path.
        self.model_slug = slugify_model(model_id)
        import hashlib as _hashlib
        _dir_hash = _hashlib.md5(os.path.abspath(output_dir).encode()).hexdigest()[:8]
        self.agent_id = f"bench-{self.model_slug}-{_dir_hash}"

        os.makedirs(self.output_dir, exist_ok=True)

        # Lazy-loaded insight extractor client
        self._client: Optional[ChainOfThoughtReader] = None
        self._metrics_cache_by_output_dir: Dict[str, Dict[str, Any]] = {}

    def _metrics_output_key(self) -> str:
        return str(Path(self.output_dir).resolve())

    def _metrics_log_path(self) -> Path:
        return Path(self.output_dir) / "metrics_log.json"

    def _build_metrics_summary(
        self,
        task_metrics: List[Dict[str, Any]],
        library_output_tokens: Optional[int],
    ) -> Dict[str, Any]:
        graded_tasks = [
            task for task in task_metrics if task.get("grade", {}).get("score") is not None
        ]
        graded_score_sum = sum(float(task["grade"].get("score", 0.0)) for task in graded_tasks)
        graded_max_score_sum = sum(float(task["grade"].get("max_score", 0.0)) for task in graded_tasks)
        overall_accuracy_pct = (
            (graded_score_sum / graded_max_score_sum * 100.0)
            if graded_max_score_sum > 0
            else None
        )

        tool_name_counts: Dict[str, int] = {}
        total_tool_calls = 0
        successful_tool_calls = 0
        error_tool_calls = 0
        unknown_tool_calls = 0
        for task in task_metrics:
            tools = task.get("tools", {})
            total_tool_calls += int(tools.get("total_calls", 0) or 0)
            successful_tool_calls += int(tools.get("successful_calls", 0) or 0)
            error_tool_calls += int(tools.get("error_calls", 0) or 0)
            unknown_tool_calls += int(tools.get("unknown_status_calls", 0) or 0)
            for name, count in (tools.get("name_counts", {}) or {}).items():
                tool_name_counts[name] = tool_name_counts.get(name, 0) + int(count)

        total_execution_time = sum(
            float(task.get("execution_time_seconds", 0.0) or 0.0) for task in task_metrics
        )
        total_agent_output_tokens = sum(
            int((task.get("output_tokens") or {}).get("agent", 0) or 0) for task in task_metrics
        )
        total_extraction_output_tokens = sum(
            int((task.get("output_tokens") or {}).get("extraction", 0) or 0)
            for task in task_metrics
        )

        return {
            "tasks_total": len(task_metrics),
            "tasks_graded": len(graded_tasks),
            "graded_score_sum": graded_score_sum,
            "graded_max_score_sum": graded_max_score_sum,
            "overall_accuracy_pct": overall_accuracy_pct,
            "output_tokens_agent_total": total_agent_output_tokens,
            "output_tokens_extraction_total": total_extraction_output_tokens,
            "output_tokens_total": total_agent_output_tokens + total_extraction_output_tokens,
            "tool_names": sorted(tool_name_counts.keys()),
            "tool_name_counts": dict(sorted(tool_name_counts.items())),
            "total_tool_calls": total_tool_calls,
            "successful_tool_calls": successful_tool_calls,
            "error_tool_calls": error_tool_calls,
            "unknown_status_tool_calls": unknown_tool_calls,
            "execution_time_total_seconds": total_execution_time,
            "library_output_tokens": int(library_output_tokens) if library_output_tokens is not None else None,
        }

    def _write_metrics_log(
        self,
        task_metrics: List[Dict[str, Any]],
        *,
        library_output_tokens: Optional[int] = None,
    ) -> None:
        log_path = self._metrics_log_path()
        if library_output_tokens is None:
            cached = self._metrics_cache_by_output_dir.get(self._metrics_output_key())
            if cached:
                library_output_tokens = cached.get("summary", {}).get("library_output_tokens")

        payload = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "output_dir": str(Path(self.output_dir).resolve()),
            "model": self.model_id,
            "judge_model": self.judge_model,
            "suite": self.suite,
            "exclude_v1": self.exclude_v1,
            "use_api": self.use_api,
            "summary": self._build_metrics_summary(task_metrics, library_output_tokens),
            "tasks": task_metrics,
        }
        log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self._metrics_cache_by_output_dir[self._metrics_output_key()] = payload

    def _write_overall_metrics_log(self, root_output_dir: str, elapsed_seconds: float) -> None:
        root = Path(root_output_dir)
        iter_logs = sorted(root.glob("iter_*/metrics_log.json"))
        if not iter_logs:
            return

        iteration_summaries = []
        for log_path in iter_logs:
            try:
                payload = json.loads(log_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            summary = payload.get("summary", {})
            iteration_summaries.append(
                {
                    "iteration": log_path.parent.name,
                    "metrics_log": str(log_path),
                    "summary": summary,
                }
            )

        if not iteration_summaries:
            return

        total_tasks = sum(int(item["summary"].get("tasks_total", 0) or 0) for item in iteration_summaries)
        total_graded = sum(int(item["summary"].get("tasks_graded", 0) or 0) for item in iteration_summaries)
        graded_score_sum = sum(
            float(item["summary"].get("graded_score_sum", 0.0) or 0.0)
            for item in iteration_summaries
        )
        graded_max_score_sum = sum(
            float(item["summary"].get("graded_max_score_sum", 0.0) or 0.0)
            for item in iteration_summaries
        )
        overall_accuracy_pct = (
            (graded_score_sum / graded_max_score_sum * 100.0)
            if graded_max_score_sum > 0
            else None
        )

        overall_tool_counts: Dict[str, int] = {}
        for item in iteration_summaries:
            for name, count in (item["summary"].get("tool_name_counts", {}) or {}).items():
                overall_tool_counts[name] = overall_tool_counts.get(name, 0) + int(count)

        overall_payload = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "output_dir": str(root.resolve()),
            "iterations": iteration_summaries,
            "summary": {
                "iterations": len(iteration_summaries),
                "tasks_total": total_tasks,
                "tasks_graded": total_graded,
                "graded_score_sum": graded_score_sum,
                "graded_max_score_sum": graded_max_score_sum,
                "overall_accuracy_pct": overall_accuracy_pct,
                "output_tokens_agent_total": sum(
                    int(item["summary"].get("output_tokens_agent_total", 0) or 0)
                    for item in iteration_summaries
                ),
                "output_tokens_extraction_total": sum(
                    int(item["summary"].get("output_tokens_extraction_total", 0) or 0)
                    for item in iteration_summaries
                ),
                "output_tokens_total": sum(
                    int(item["summary"].get("output_tokens_total", 0) or 0)
                    for item in iteration_summaries
                ),
                "tool_names": sorted(overall_tool_counts.keys()),
                "tool_name_counts": dict(sorted(overall_tool_counts.items())),
                "total_tool_calls": sum(
                    int(item["summary"].get("total_tool_calls", 0) or 0)
                    for item in iteration_summaries
                ),
                "successful_tool_calls": sum(
                    int(item["summary"].get("successful_tool_calls", 0) or 0)
                    for item in iteration_summaries
                ),
                "error_tool_calls": sum(
                    int(item["summary"].get("error_tool_calls", 0) or 0)
                    for item in iteration_summaries
                ),
                "unknown_status_tool_calls": sum(
                    int(item["summary"].get("unknown_status_tool_calls", 0) or 0)
                    for item in iteration_summaries
                ),
                "execution_time_total_seconds": sum(
                    float(item["summary"].get("execution_time_total_seconds", 0.0) or 0.0)
                    for item in iteration_summaries
                ),
                "library_output_tokens_total": sum(
                    int(item["summary"].get("library_output_tokens", 0) or 0)
                    for item in iteration_summaries
                ),
                "pipeline_elapsed_seconds": elapsed_seconds,
            },
        }

        (root / "metrics_log_overall.json").write_text(
            json.dumps(overall_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _ensure_client(self) -> ChainOfThoughtReader:
        if self._client is None:
            self._client = ChainOfThoughtReader(
                use_api=self.use_api,
                api_key=self.api_key,
                api_provider=self.api_provider,
            )
            if self.use_api and self.api_provider == "gemini" and self.api_model:
                from utils import setup_gemini
                self._client.gemini_model = setup_gemini(
                    api_key=self.api_key,
                    model_name=self.api_model,
                )
        return self._client

    def _setup_agent(self) -> None:
        """Validate openclaw install, validate model, and ensure agent exists."""
        # Fail fast if openclaw is not installed — don't silently run 25 tasks
        _check_openclaw()

        if self.openclaw_skill:
            print(f"\nInstalling predefined OpenClaw skills ({len(_PREDEFINED_OPENCLAW_SKILLS)} skills)...")
            _install_openclaw_skills(_PREDEFINED_OPENCLAW_SKILLS)
            print("Skill installation complete.\n")
        # Determine effective model ID and connection config.
        # For google/gemini models, use OpenClaw's native Google provider
        # (reads GEMINI_API_KEY from env) rather than a custom OpenAI-compat
        # endpoint — the custom provider triggers gateway/pairing mode.
        effective_base_url = self.base_url
        effective_api_key = self.openclaw_api_key
        effective_model_id = self.model_id
        if self.use_api and self.api_provider == "gemini" and not self.base_url:
            # Normalise to google/ prefix so OpenClaw uses its native Gemini provider
            if self.model_id.startswith("gemini/"):
                effective_model_id = "google/" + self.model_id.split("/", 1)[1]
            elif not self.model_id.startswith("google/"):
                effective_model_id = "google/" + self.model_id
            else:
                effective_model_id = self.model_id
            # Ensure GEMINI_API_KEY is in env so OpenClaw can authenticate
            gemini_key = self.api_key or os.getenv("GEMINI_API_KEY")
            if gemini_key:
                os.environ["GEMINI_API_KEY"] = gemini_key
                os.environ.setdefault("GOOGLE_AI_STUDIO_KEY", gemini_key)
            else:
                print("Warning: GEMINI_API_KEY is not set; OpenClaw Gemini calls may fail")
            print(f"Using OpenClaw native Google provider for model '{effective_model_id}'")

        if not effective_base_url:
            if self.api_provider != "gemini":
                print("No custom OpenClaw base URL provided, using default openrouter.ai endpoints")
                try:
                    print(f"Validating model: {self.model_id}")
                    validate_openrouter_model(self.model_id)
                except ModelValidationError as exc:
                    print(f"Warning: {exc}")
        else:
            print(f"Using custom OpenClaw base URL: {effective_base_url}")

        print(f"Ensuring OpenClaw agent exists for model '{self.model_id}' with ID '{self.agent_id}'")
        agent_workspace = _get_agent_workspace(self.agent_id)
        if agent_workspace is None:
            # Deterministic fallback aligned with OpenClaw convention
            normalized_id = self.agent_id.replace(":", "-").lower()
            agent_workspace = (
                Path.home() / ".openclaw" / "agents" / normalized_id / "workspace"
            )
        # Retry once on ConfigMutationConflictError (openclaw config race)
        agent_ready = False
        for _attempt in range(2):
            ok = ensure_agent_exists(
                self.agent_id,
                effective_model_id,
                agent_workspace,
                base_url=effective_base_url,
                api_key=effective_api_key,
            )
            if ok is not False:
                agent_ready = True
                break
            # Re-query in case the agent was created by a concurrent process
            resolved = _get_agent_workspace(self.agent_id)
            if resolved is not None:
                agent_workspace = resolved
                agent_ready = True
                break
            import time as _time; _time.sleep(1)
        if not agent_ready:
            raise RuntimeError(
                f"Failed to create or find OpenClaw agent '{self.agent_id}'. "
                "Run `openclaw agents list` and retry after any concurrent OpenClaw command finishes."
            )
        # Store resolved workspace so downstream calls don't need to re-query
        self._agent_workspace: Path = agent_workspace
        # For truly custom (non-native) endpoints, re-write models.json after a
        # brief delay to overwrite any async re-initialisation openclaw may do.
        if effective_base_url:
            import time as _time
            _time.sleep(2)
            configure_bench_models(
                self.agent_id, effective_model_id, effective_base_url, effective_api_key
            )
        cleanup_agent_sessions(self.agent_id)

    def _load_tasks(self) -> list:
        loader = TaskLoader(self.tasks_dir)
        tasks = loader.load_all_tasks()

        if self.suite == "all":
            selected = tasks
        elif self.suite == "automated-only":
            selected = [t for t in tasks if t.grading_type == "automated"]
        else:
            # Comma-separated list of task IDs
            ids = {tid.strip() for tid in self.suite.split(",") if tid.strip()}
            selected = [t for t in tasks if t.task_id in ids]

        if self.exclude_v1:
            before = len(selected)
            selected = [t for t in selected if t.task_id not in _PINCHBENCH_V1_TASK_IDS]
            excluded = before - len(selected)
            print(
                f"Excluded {excluded} PinchBench V1 tasks "
                f"({len(_PINCHBENCH_V1_TASK_IDS)} configured)"
            )

        return selected

    def _call_for_extraction(self, prompt: str, max_new_tokens: int) -> Tuple[str, Dict]:
        """Call model for reflection/extraction, using ThinkingConfig when set."""
        if self.use_api and self.api_provider == "gemini" and self.thinking_level:
            return call_gemini_thinking(
                api_key=self.api_key,
                model_name=self.api_model,
                prompt=prompt,
                thinking_level=self.thinking_level,
                max_new_tokens=max_new_tokens,
            )
        client = self._ensure_client()
        return client._call_model(prompt, max_new_tokens=max_new_tokens)

    def _apply_reflection_and_extraction(
        self, task_prompt: str, agent_response: str
    ) -> Dict[str, Any]:
        """
        Apply Steps 2 & 3 from client.py (reflection + insight extraction)
        to the agent's response transcript.

        Returns the insight_book dict extracted from the agent's solution.
        """
        client = self._ensure_client()

        # Step 2: Reflection
        print("  Step 2: Reflecting on agent solution...")
        reflection_prompt = client._get_reflection_prompt(task_prompt, agent_response)
        reflection, _ = self._call_for_extraction(reflection_prompt, max_new_tokens=4096)
        print(f"  Reflection length: {len(reflection)} chars")

        # Step 3: Insight extraction
        print("  Step 3: Extracting reasoning traces...")
        behavior_prompt = client._get_behavior_prompt(task_prompt, agent_response, reflection)
        extraction_response, token_info = self._call_for_extraction(
            behavior_prompt, max_new_tokens=8192
        )

        # Parse insight JSON (same logic as client._step_behavior_extraction)
        insights = {}
        try:
            json_code_block = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", extraction_response, re.DOTALL
            )
            if json_code_block:
                json_str = json_code_block.group(1)
            else:
                start = extraction_response.find("{")
                if start != -1:
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    json_str = None
                    for i in range(start, len(extraction_response)):
                        ch = extraction_response[i]
                        if escape_next:
                            escape_next = False
                            continue
                        if ch == "\\":
                            escape_next = True
                            continue
                        if ch == '"':
                            in_string = not in_string
                            continue
                        if not in_string:
                            if ch == "{":
                                brace_count += 1
                            elif ch == "}":
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = extraction_response[start: i + 1]
                                    break
                else:
                    json_str = None

            if json_str:
                json_str = re.sub(r",\s*}", "}", json_str)
                json_str = re.sub(r",\s*]", "]", json_str)
                raw = json.loads(json_str)
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        name = k if k.startswith("insight_") else f"insight_{k}"
                        desc = v if isinstance(v, str) else str(v)
                        desc = re.sub(r"\s+", " ", desc).strip()
                        if len(desc) >= 20:
                            insights[name] = desc
        except Exception as exc:
            print(f"  Warning: insight extraction parse error: {exc}")

        print(f"  Extracted {len(insights)} insights")
        return {
            "insight_book": insights,
            "output_tokens": token_info.get("output_tokens", 0),
        }

    def _create_trace_insight_library(self) -> Optional[str]:
        """Create an INSIGHTS-style JSON file from trace folder reasoning traces."""
        if not self.trace_folder:
            return None

        trace_path = Path(self.trace_folder)
        if not trace_path.exists():
            print(f"Trace folder not found: {trace_path}")
            return None

        trace_files = sorted(trace_path.rglob("problem*.json")) + sorted(trace_path.rglob("paper*.json"))
        if not trace_files:
            print(f"No trace files found under {trace_path}")
            return None

        insight_items: Dict[str, str] = {}
        for idx, trace_file in enumerate(trace_files, start=1):
            try:
                with open(trace_file, "r", encoding="utf-8") as f:
                    trace_data = json.load(f)
            except Exception as exc:
                print(f"  Warning: failed to read trace file {trace_file}: {exc}")
                continue

            problem_text = trace_data.get("problem") or trace_data.get("task_prompt") or trace_data.get("question") or ""
            solution_text = trace_data.get("solution") or trace_data.get("reasoning") or trace_data.get("agent_response") or ""
            if not solution_text:
                continue

            key = f"trace_{idx:04d}"
            desc = (
                f"Source: {trace_file.relative_to(trace_path)}\n"
                f"Problem: {problem_text}\n\n"
                f"Reasoning Trace:\n{solution_text}"
            )
            insight_items[key] = desc

        if not insight_items:
            print(f"No usable trace content found in {trace_path}")
            return None

        trace_insights_path = Path(self.output_dir) / "trace_insights.json"
        with open(trace_insights_path, "w", encoding="utf-8") as f:
            json.dump(insight_items, f, indent=2, ensure_ascii=False)

        print(f"Created trace insight library: {trace_insights_path} ({len(insight_items)} entries)")
        return str(trace_insights_path)

    def run_tasks_and_extract(
        self,
        run_id: Optional[str] = None,
        tasks: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Step 1 + Steps 2/3: For each pinchbench task, run the OpenClaw agent
        and extract reasoning traces from the transcript.

        Saves each task's insights as problem_XXXX.json in output_dir.
        Returns a list of per-task result dicts.
        """
        self._setup_agent()

        # Write trace folder content or encyclopedia as INSIGHTS.md into the agent workspace.
        # prepare_task_workspace() preserves this file and injects a
        # mandatory read instruction into BOOTSTRAP.md for every task.
        trace_insights_path = self._create_trace_insight_library()
        if trace_insights_path:
            print(f"\nWriting trace-folder insights to agent workspace as INSIGHTS.md")
            _write_insights_to_workspace(self.agent_id, trace_insights_path, self._agent_workspace)
        elif self.encyclopedia_path and Path(self.encyclopedia_path).exists():
            print(f"\nWriting encyclopedia to agent workspace as INSIGHTS.md")
            _write_insights_to_workspace(self.agent_id, self.encyclopedia_path, self._agent_workspace)
        else:
            if self.trace_folder:
                print(f"\nTrace folder configured but no usable trace insights were created from {self.trace_folder}")
            print("\nNo encyclopedia or trace-folder insights — running without prior insights")

        tasks = tasks if tasks is not None else self._load_tasks()
        if not tasks:
            print(f"No tasks found in {self.tasks_dir}")
            return []

        print(f"\nLoaded {len(tasks)} tasks (suite='{self.suite}')")
        print("=" * 80)

        if run_id is None:
            run_id = f"fot_{int(time.time())}"

        results = []
        task_metrics: List[Dict[str, Any]] = []
        task_counter = 0

        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] Task: {task.task_id} — {task.name}")
            print("-" * 60)

            # -----------------------------------------------------------------
            # Step 1: Execute the task with the OpenClaw agent
            # -----------------------------------------------------------------
            try:
                exec_result = execute_openclaw_task(
                    task=task,
                    agent_id=self.agent_id,
                    model_id=self.model_id,
                    run_id=f"{run_id}-{i}",
                    timeout_multiplier=self.timeout_multiplier,
                    skill_dir=self.skill_dir,
                    output_dir=Path(self.output_dir) / "transcripts",
                    verbose=False,
                )
            except Exception as exc:
                print(f"  Error executing task: {exc}")
                task_metrics.append(
                    {
                        "task_id": task.task_id,
                        "task_name": task.name,
                        "execution_status": "error",
                        "execution_time_seconds": None,
                        "grade": {
                            "score": None,
                            "max_score": None,
                            "accuracy_pct": None,
                            "grading_type": task.grading_type,
                        },
                        "output_tokens": {"agent": 0, "extraction": 0, "total": 0},
                        "tools": {
                            "names": [],
                            "name_counts": {},
                            "total_calls": 0,
                            "successful_calls": 0,
                            "error_calls": 0,
                            "unknown_status_calls": 0,
                            "calls": [],
                        },
                        "insights_extracted": 0,
                        "error": str(exc),
                    }
                )
                self._write_metrics_log(task_metrics)
                continue

            status = exec_result.get("status", "error")
            transcript = exec_result.get("transcript", [])
            print(f"  Execution status: {status} | transcript entries: {len(transcript)}")
            if status != "success":
                command = exec_result.get("command")
                if command:
                    print(f"  OpenClaw command: {command}")
                print(f"  Exit code: {exec_result.get('exit_code')} | timed_out: {exec_result.get('timed_out')}")
                stdout = (exec_result.get("stdout") or "").strip()
                stderr = (exec_result.get("stderr") or "").strip()
                if stdout:
                    print(f"  Stdout (first 500 chars): {stdout[:500]}")
                if stderr:
                    print(f"  Stderr (first 500 chars): {stderr[:500]}")

            # -----------------------------------------------------------------
            # Grade the task (automated or LLM-judge using the same model)
            # -----------------------------------------------------------------
            try:
                grade = grade_task(
                    task=task,
                    execution_result=exec_result,
                    skill_dir=self.skill_dir,
                    judge_model=self.judge_model,
                    judge_backend="api",
                    judge_api_key=self.api_key,
                )
                score_pct = grade.score / grade.max_score * 100 if grade.max_score > 0 else 0
                print(f"  Grade: {grade.score:.2f}/{grade.max_score:.2f} ({score_pct:.0f}%)"
                      f" [{grade.grading_type}]")
                if grade.breakdown:
                    for criterion, val in grade.breakdown.items():
                        print(f"    {criterion}: {val}")
                if grade.notes:
                    print(f"  Notes: {grade.notes}")
            except Exception as exc:
                print(f"  Warning: grading failed: {exc}")
                grade = None

            usage = exec_result.get("usage", {}) or {}
            agent_output_tokens = int(usage.get("output_tokens", 0) or 0)
            extraction_output_tokens = 0
            execution_time_seconds = exec_result.get("execution_time")
            if execution_time_seconds is not None:
                execution_time_seconds = float(execution_time_seconds)

            tool_stats = _analyze_tool_calls(transcript)
            task_metric = {
                "task_id": task.task_id,
                "task_name": task.name,
                "execution_status": status,
                "execution_time_seconds": execution_time_seconds,
                "grade": {
                    "score": grade.score if grade else None,
                    "max_score": grade.max_score if grade else None,
                    "accuracy_pct": (
                        (grade.score / grade.max_score * 100.0)
                        if grade and grade.max_score > 0
                        else None
                    ),
                    "grading_type": grade.grading_type if grade else task.grading_type,
                    "notes": grade.notes if grade else "",
                },
                "output_tokens": {
                    "agent": agent_output_tokens,
                    "extraction": extraction_output_tokens,
                    "total": agent_output_tokens,
                },
                "tools": {
                    "names": tool_stats.get("tool_names", []),
                    "name_counts": tool_stats.get("tool_name_counts", {}),
                    "total_calls": tool_stats.get("total_tool_calls", 0),
                    "successful_calls": tool_stats.get("successful_tool_calls", 0),
                    "error_calls": tool_stats.get("error_tool_calls", 0),
                    "unknown_status_calls": tool_stats.get("unknown_status_tool_calls", 0),
                    "calls": tool_stats.get("calls", []),
                },
                "insights_extracted": 0,
            }

            if not transcript:
                print("  Skipping insight extraction — empty transcript")
                task_metrics.append(task_metric)
                self._write_metrics_log(task_metrics)
                continue

            # -----------------------------------------------------------------
            # Extract readable solution text from transcript
            # -----------------------------------------------------------------
            agent_response = _extract_transcript_text(transcript)
            if not agent_response.strip():
                print("  Skipping insight extraction — no extractable text")
                task_metrics.append(task_metric)
                self._write_metrics_log(task_metrics)
                continue

            # -----------------------------------------------------------------
            # Steps 2 & 3: Reflection + insight extraction
            # -----------------------------------------------------------------
            try:
                extraction = self._apply_reflection_and_extraction(
                    task_prompt=task.prompt,
                    agent_response=agent_response,
                )
            except Exception as exc:
                print(f"  Error during insight extraction: {exc}")
                task_metric["extraction_error"] = str(exc)
                task_metrics.append(task_metric)
                self._write_metrics_log(task_metrics)
                continue

            insight_book = extraction.get("insight_book", {})
            if not insight_book:
                print("  Warning: no insights extracted for this task")
                # Still save an empty file so the counter is consistent
                insight_book = {}

            extraction_output_tokens = int(extraction.get("output_tokens", 0) or 0)
            task_metric["output_tokens"] = {
                "agent": agent_output_tokens,
                "extraction": extraction_output_tokens,
                "total": agent_output_tokens + extraction_output_tokens,
            }
            task_metric["insights_extracted"] = len(insight_book)

            # -----------------------------------------------------------------
            # Save as problem_XXXX.json (matches server_text expected format)
            # -----------------------------------------------------------------
            task_counter += 1
            output_file = os.path.join(
                self.output_dir, f"problem_{task_counter:04d}.json"
            )
            grade_info = {}
            if grade is not None:
                grade_info = {
                    "score": grade.score,
                    "max_score": grade.max_score,
                    "grading_type": grade.grading_type,
                    "notes": grade.notes,
                }

            save_data = {
                "task_id": task.task_id,
                "task_name": task.name,
                "task_prompt": task.prompt,
                "execution_status": status,
                "output_tokens": extraction.get("output_tokens", 0),
                "grade": grade_info,
                "insight_book": insight_book,
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            print(f"  Saved {len(insight_book)} insights → {output_file}")
            results.append({
                "task_id": task.task_id,
                "task_name": task.name,
                "status": status,
                "score": grade.score if grade else None,
                "max_score": grade.max_score if grade else None,
                "insights_extracted": len(insight_book),
                "output_file": output_file,
            })
            task_metrics.append(task_metric)
            self._write_metrics_log(task_metrics)

        print("\n" + "=" * 80)
        total_insights = sum(int(t.get("insights_extracted", 0) or 0) for t in task_metrics)
        graded = [t for t in task_metrics if t.get("grade", {}).get("score") is not None]
        print(f"Tasks processed: {len(task_metrics)}/{len(tasks)}")
        if graded:
            score_sum = sum(float(t["grade"].get("score", 0.0) or 0.0) for t in graded)
            max_sum = sum(float(t["grade"].get("max_score", 0.0) or 0.0) for t in graded)
            overall = score_sum / max_sum * 100 if max_sum > 0 else 0.0
            print(f"Overall score:   {overall:.1f}%  ({len(graded)} tasks graded, judge={self.judge_model})")
        print(f"Total insights extracted: {total_insights}")
        self._write_metrics_log(task_metrics)
        return results

    def aggregate_insights(self) -> Optional[str]:
        """
        Aggregate all problem_XXXX.json files using TextBasedInsightAggregationServer
        and save the resulting encyclopedia.json.

        Returns the path to the encyclopedia file, or None on failure.
        """
        print("\n" + "=" * 80)
        print("Aggregating Insights via server_text")
        print("=" * 80)

        json_files = sorted(Path(self.output_dir).glob("problem_*.json"))
        if not json_files:
            print("No problem_*.json files found — nothing to aggregate")
            return None

        print(f"Found {len(json_files)} insight files")

        server = TextBasedInsightAggregationServer(
            use_api=self.use_api,
            api_key=self.api_key,
            api_provider=self.api_provider,
            input_dirs=[self.output_dir],
        )

        result = server.aggregate_and_build_encyclopedia(
            json_files=[str(f) for f in json_files],
            output_dir=self.output_dir,
        )

        # Save encyclopedia
        encyclopedia_path = os.path.join(self.output_dir, "encyclopedia.json")
        enc_dict = server._try_parse_json(server.encyclopedia)
        if enc_dict is None:
            enc_dict = server._try_parse_json(server._extract_json_only(server.encyclopedia))

        if enc_dict is None:
            # Fallback: save as plain text
            encyclopedia_path = os.path.join(self.output_dir, "encyclopedia.txt")
            with open(encyclopedia_path, "w", encoding="utf-8") as f:
                f.write(server.encyclopedia)
            print(f"Warning: encyclopedia saved as plain text: {encyclopedia_path}")
        else:
            with open(encyclopedia_path, "w", encoding="utf-8") as f:
                json.dump(enc_dict, f, indent=2, ensure_ascii=False)
            print(f"Encyclopedia saved: {encyclopedia_path}")

        total_output_tokens = result.get("total_output_tokens", 0)
        print(f"Insight library output tokens: {total_output_tokens}")
        metrics_key = self._metrics_output_key()
        existing_payload = self._metrics_cache_by_output_dir.get(metrics_key)
        existing_tasks = existing_payload.get("tasks", []) if existing_payload else []
        self._write_metrics_log(existing_tasks, library_output_tokens=int(total_output_tokens or 0))
        return encyclopedia_path

    def run_tasks_eval_only(
        self,
        tasks: List[Any],
        *,
        run_id: Optional[str] = None,
        encyclopedia_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run PinchBench tasks with an existing insight library, but do not extract or aggregate."""
        self._setup_agent()

        if encyclopedia_path and Path(encyclopedia_path).exists():
            print(f"\nWriting split-train encyclopedia to agent workspace as INSIGHTS.md")
            _write_insights_to_workspace(self.agent_id, encyclopedia_path, self._agent_workspace)
        else:
            # Remove any stale INSIGHTS.md so this run sees no prior insights
            stale = self._agent_workspace / "INSIGHTS.md"
            if stale.exists():
                stale.unlink()
                print("\nRemoved stale INSIGHTS.md — evaluating without prior insights")
            else:
                print("\nNo split-train encyclopedia found — evaluating without prior insights")

        if not tasks:
            print("No eval tasks found")
            return []

        print(f"\nLoaded {len(tasks)} held-out eval tasks")
        print("=" * 80)

        if run_id is None:
            run_id = f"fot_split_eval_{int(time.time())}"

        results = []
        task_metrics: List[Dict[str, Any]] = []

        for i, task in enumerate(tasks, 1):
            print(f"\n[eval {i}/{len(tasks)}] Task: {task.task_id} — {task.name}")
            print("-" * 60)

            try:
                exec_result = execute_openclaw_task(
                    task=task,
                    agent_id=self.agent_id,
                    model_id=self.model_id,
                    run_id=f"{run_id}-{i}",
                    timeout_multiplier=self.timeout_multiplier,
                    skill_dir=self.skill_dir,
                    output_dir=Path(self.output_dir) / "transcripts",
                    verbose=False,
                )
            except Exception as exc:
                print(f"  Error executing task: {exc}")
                task_metrics.append(
                    {
                        "task_id": task.task_id,
                        "task_name": task.name,
                        "execution_status": "error",
                        "execution_time_seconds": None,
                        "grade": {
                            "score": None,
                            "max_score": None,
                            "accuracy_pct": None,
                            "grading_type": task.grading_type,
                        },
                        "output_tokens": {"agent": 0, "extraction": 0, "total": 0},
                        "tools": {
                            "names": [],
                            "name_counts": {},
                            "total_calls": 0,
                            "successful_calls": 0,
                            "error_calls": 0,
                            "unknown_status_calls": 0,
                            "calls": [],
                        },
                        "insights_extracted": 0,
                        "error": str(exc),
                    }
                )
                self._write_metrics_log(task_metrics)
                continue

            status = exec_result.get("status", "error")
            transcript = exec_result.get("transcript", [])
            print(f"  Execution status: {status} | transcript entries: {len(transcript)}")
            if status != "success":
                command = exec_result.get("command")
                if command:
                    print(f"  OpenClaw command: {command}")
                print(f"  Exit code: {exec_result.get('exit_code')} | timed_out: {exec_result.get('timed_out')}")
                stdout = (exec_result.get("stdout") or "").strip()
                stderr = (exec_result.get("stderr") or "").strip()
                if stdout:
                    print(f"  Stdout (first 500 chars): {stdout[:500]}")
                if stderr:
                    print(f"  Stderr (first 500 chars): {stderr[:500]}")

            try:
                grade = grade_task(
                    task=task,
                    execution_result=exec_result,
                    skill_dir=self.skill_dir,
                    judge_model=self.judge_model,
                    judge_backend="api",
                )
                score_pct = grade.score / grade.max_score * 100 if grade.max_score > 0 else 0
                print(f"  Grade: {grade.score:.2f}/{grade.max_score:.2f} ({score_pct:.0f}%)"
                      f" [{grade.grading_type}]")
                if grade.breakdown:
                    for criterion, val in grade.breakdown.items():
                        print(f"    {criterion}: {val}")
                if grade.notes:
                    print(f"  Notes: {grade.notes}")
            except Exception as exc:
                print(f"  Warning: grading failed: {exc}")
                grade = None

            usage = exec_result.get("usage", {}) or {}
            agent_output_tokens = int(usage.get("output_tokens", 0) or 0)
            execution_time_seconds = exec_result.get("execution_time")
            if execution_time_seconds is not None:
                execution_time_seconds = float(execution_time_seconds)

            tool_stats = _analyze_tool_calls(transcript)
            task_metric = {
                "task_id": task.task_id,
                "task_name": task.name,
                "execution_status": status,
                "execution_time_seconds": execution_time_seconds,
                "grade": {
                    "score": grade.score if grade else None,
                    "max_score": grade.max_score if grade else None,
                    "accuracy_pct": (
                        (grade.score / grade.max_score * 100.0)
                        if grade and grade.max_score > 0
                        else None
                    ),
                    "grading_type": grade.grading_type if grade else task.grading_type,
                    "notes": grade.notes if grade else "",
                },
                "output_tokens": {
                    "agent": agent_output_tokens,
                    "extraction": 0,
                    "total": agent_output_tokens,
                },
                "tools": {
                    "names": tool_stats.get("tool_names", []),
                    "name_counts": tool_stats.get("tool_name_counts", {}),
                    "total_calls": tool_stats.get("total_tool_calls", 0),
                    "successful_calls": tool_stats.get("successful_tool_calls", 0),
                    "error_calls": tool_stats.get("error_tool_calls", 0),
                    "unknown_status_calls": tool_stats.get("unknown_status_tool_calls", 0),
                    "calls": tool_stats.get("calls", []),
                },
                "insights_extracted": 0,
            }

            output_file = os.path.join(self.output_dir, f"problem_{i:04d}.json")
            grade_info = {}
            if grade is not None:
                grade_info = {
                    "score": grade.score,
                    "max_score": grade.max_score,
                    "grading_type": grade.grading_type,
                    "notes": grade.notes,
                }
            save_data = {
                "task_id": task.task_id,
                "task_name": task.name,
                "task_prompt": task.prompt,
                "execution_status": status,
                "output_tokens": agent_output_tokens,
                "grade": grade_info,
                "eval_only": True,
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            results.append(
                {
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "status": status,
                    "score": grade.score if grade else None,
                    "max_score": grade.max_score if grade else None,
                    "output_file": output_file,
                }
            )
            task_metrics.append(task_metric)
            self._write_metrics_log(task_metrics)

        graded = [t for t in task_metrics if t.get("grade", {}).get("score") is not None]
        print("\n" + "=" * 80)
        print(f"Held-out eval tasks processed: {len(task_metrics)}/{len(tasks)}")
        if graded:
            score_sum = sum(float(t["grade"].get("score", 0.0) or 0.0) for t in graded)
            max_sum = sum(float(t["grade"].get("max_score", 0.0) or 0.0) for t in graded)
            overall = score_sum / max_sum * 100 if max_sum > 0 else 0.0
            print(f"Held-out eval score: {overall:.1f}% ({len(graded)} tasks graded)")
        self._write_metrics_log(task_metrics)
        return results

    def run_split_pipeline(self, split: float, seed: int) -> None:
        """Run train split through Step 1/2/3, then eval held-out split with Step 1 only."""
        if not 0.0 < split < 1.0:
            raise ValueError("--split must be a float strictly between 0 and 1")

        start_time = time.time()
        base_output_dir = self.output_dir
        tasks = self._load_tasks()
        if len(tasks) < 2:
            raise ValueError("Split mode requires at least two tasks")

        indices = list(range(len(tasks)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        train_size = int(len(indices) * split)
        train_size = max(1, min(train_size, len(indices) - 1))
        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]
        train_tasks = [tasks[i] for i in train_indices]
        eval_tasks = [tasks[i] for i in eval_indices]

        split_manifest = {
            "mode": "split",
            "split": split,
            "seed": seed,
            "suite": self.suite,
            "exclude_v1": self.exclude_v1,
            "total_tasks": len(tasks),
            "train": len(train_tasks),
            "eval": len(eval_tasks),
            "train_indices": train_indices,
            "eval_indices": eval_indices,
            "train_tasks": [{"task_id": t.task_id, "name": t.name} for t in train_tasks],
            "eval_tasks": [{"task_id": t.task_id, "name": t.name} for t in eval_tasks],
        }
        os.makedirs(base_output_dir, exist_ok=True)
        split_manifest_path = Path(base_output_dir) / "split_manifest.json"
        split_manifest_path.write_text(
            json.dumps(split_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Split manifest saved: {split_manifest_path}")

        train_dir = os.path.join(base_output_dir, "split_train")
        eval_dir = os.path.join(base_output_dir, "split_eval")

        orig_output_dir = self.output_dir
        orig_encyclopedia = self.encyclopedia_path
        try:
            print("\n" + "=" * 80)
            print(f"SPLIT TRAIN: Step 1/2/3 on {len(train_tasks)} tasks ({split:.0%})")
            print("=" * 80)
            self.output_dir = train_dir
            os.makedirs(self.output_dir, exist_ok=True)
            self.encyclopedia_path = orig_encyclopedia
            self.run_tasks_and_extract(run_id="split-train", tasks=train_tasks)
            encyclopedia_path = self.aggregate_insights()

            print("\n" + "=" * 80)
            print(f"SPLIT EVAL: Step 1 only on {len(eval_tasks)} held-out tasks")
            print("=" * 80)
            self.output_dir = eval_dir
            os.makedirs(self.output_dir, exist_ok=True)
            self.encyclopedia_path = encyclopedia_path
            self.run_tasks_eval_only(
                eval_tasks,
                run_id="split-eval",
                encyclopedia_path=encyclopedia_path,
            )
        finally:
            self.output_dir = orig_output_dir
            self.encyclopedia_path = orig_encyclopedia

        elapsed = time.time() - start_time
        summary = {
            "mode": "split",
            "split": split,
            "seed": seed,
            "train_dir": train_dir,
            "eval_dir": eval_dir,
            "train_encyclopedia": encyclopedia_path if "encyclopedia_path" in locals() else None,
            "elapsed_seconds": elapsed,
        }
        summary_path = Path(base_output_dir) / "split_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n" + "=" * 80)
        print("Split Pipeline Complete")
        print("=" * 80)
        print(f"Summary saved: {summary_path}")

    def run_pipeline(
        self,
        iterations: int = 1,
        start_from_step2: bool = False,
    ) -> None:
        """
        Run the full pipeline for N iterations.

        Iteration 1: no encyclopedia, collect insights, aggregate.
        Iteration 2+: inject encyclopedia, collect insights, aggregate.
        """
        start_time = time.time()
        base_output_dir = self.output_dir

        current_encyclopedia: Optional[str] = self.encyclopedia_path

        for iteration in range(1, iterations + 1):
            iter_label = f"Iteration {iteration}/{iterations}"
            print("\n" + "=" * 80)
            print(iter_label)
            print("=" * 80)

            # Use separate sub-directory per iteration when doing multi-iteration
            if iterations > 1:
                iter_dir = os.path.join(self.output_dir, f"iter_{iteration:02d}")
                os.makedirs(iter_dir, exist_ok=True)
                orig_output_dir = self.output_dir
                self.output_dir = iter_dir
            else:
                orig_output_dir = None

            # Set current encyclopedia for this iteration
            if current_encyclopedia:
                self.encyclopedia_path = current_encyclopedia

            if not start_from_step2 or iteration > 1:
                print(f"\n--- Step 1/2/3: Task Execution + Insight Extraction ---")
                self.run_tasks_and_extract()
            else:
                print("Skipping task execution (start_from_step2=True)")

            print(f"\n--- Aggregation ---")
            encyclopedia_path = self.aggregate_insights()

            if encyclopedia_path:
                current_encyclopedia = encyclopedia_path

            if orig_output_dir is not None:
                self.output_dir = orig_output_dir

        elapsed = time.time() - start_time
        if iterations > 1:
            self._write_overall_metrics_log(base_output_dir, elapsed)
        print("\n" + "=" * 80)
        print("Pipeline Complete")
        print("=" * 80)
        print(f"Total time: {elapsed:.1f}s")
        if current_encyclopedia:
            print(f"Final encyclopedia: {current_encyclopedia}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OpenClaw PinchBench Pipeline — extract reasoning traces from benchmark tasks"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help="OpenClaw model identifier (e.g., anthropic/claude-sonnet-4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pinchbench_output",
        help="Directory to save insights and encyclopedia (default: pinchbench_output)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        help='Tasks to run: "all", "automated-only", or comma-separated task IDs (default: all)',
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=None,
        help=(
            "Judge model for LLM-judge tasks. Defaults to the same model as --model. "
            "Use OpenRouter format (e.g. google/gemini-2.5-pro-preview) or an "
            "Anthropic model ID."
        ),
    )
    parser.add_argument(
        "--pinchbench-dir",
        type=str,
        default=None,
        help="Path to pinchbench skill root (default: ./pinchbench)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of pipeline iterations (default: 1). Iteration 2+ uses the encyclopedia from iteration 1.",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=None,
        help=(
            "Optional train/eval split fraction. If set, randomly use this "
            "fraction of PinchBench tasks for Step 1/2/3 insight generation "
            "and evaluate the remaining held-out tasks with Step 1 only."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --split partitioning (default: 42).",
    )
    parser.add_argument(
        "--encyclopedia",
        type=str,
        default=None,
        help="Path to existing encyclopedia.json to inject into agent for the first iteration",
    )
    parser.add_argument(
        "--trace-folder",
        type=str,
        default=None,
        help="Path to a trace folder containing problem*.json or paper*.json for RAG-style agent insights",
    )
    parser.add_argument(
        "--start-from-step2",
        action="store_true",
        help="Skip task execution and start from aggregation",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use an API provider for reflection/extraction steps",
    )
    parser.add_argument(
        "--api-provider",
        type=str,
        default="gemini",
        choices=["gemini", "openrouter"],
        help="Which API provider to use for extraction (default: gemini)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the chosen extraction provider (or set GEMINI_API_KEY / OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--api-model",
        type=str,
        default="gemini-3-pro-preview",
        help="Model name for extraction (default: gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--thinking-level",
        type=str,
        default="high",
        choices=["low", "medium", "high"],
        help=(
            "Thinking level for Gemini models that support ThinkingConfig "
            "(e.g., gemini-3.1-pro-preview). Choices: low, medium, high (default: high). "
            "Uses the new google-genai SDK with types.ThinkingConfig."
        ),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom OpenAI-compatible API base URL for the OpenClaw agent",
    )
    parser.add_argument(
        "--openclaw-api-key",
        type=str,
        default=None,
        help="API key for custom OpenClaw agent endpoint (default: $OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.0,
        help="Scale all task timeouts (default: 1.0)",
    )
    parser.add_argument(
        "--openclaw-skill",
        action="store_true",
        help=(
            "Pre-install all predefined OpenClaw skills before running tasks. "
            f"Skills installed: {', '.join(_PREDEFINED_OPENCLAW_SKILLS)}. "
            "Each skill is installed into ~/.openclaw/workspace/skills/ via "
            "'openclaw install <skill>' and is then available in every task workspace."
        ),
    )
    parser.add_argument(
        "--exclude-V1",
        dest="exclude_v1",
        action="store_true",
        help=(
            "Exclude the 23 V1 PinchBench tasks from the Apr 9 "
            "pinchbench/skill tree, using their current renamed task IDs."
        ),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Eval-only mode: run tasks and grade them without extracting or aggregating insights. "
            "Use --encyclopedia to inject an existing insight library into the agent workspace."
        ),
    )

    args = parser.parse_args()

    if not args.model and not args.start_from_step2 and not args.eval_only:
        parser.error("--model is required unless using --start-from-step2 or --eval-only")

    # Default model for aggregation-only runs
    model_id = args.model or "anthropic/claude-sonnet-4"

    # Judge defaults to the same model as the agent
    judge_model = args.judge or model_id

    pipeline = OpenClawPinchBenchPipeline(
        model_id=model_id,
        output_dir=args.output_dir,
        suite=args.suite,
        pinchbench_dir=args.pinchbench_dir,
        use_api=args.use_api,
        api_key=args.api_key,
        api_provider=args.api_provider,
        api_model=args.api_model,
        base_url=args.base_url,
        openclaw_api_key=args.openclaw_api_key,
        timeout_multiplier=args.timeout_multiplier,
        encyclopedia_path=args.encyclopedia,
        trace_folder=args.trace_folder,
        judge_model=judge_model,
        thinking_level=args.thinking_level if args.use_api else None,
        openclaw_skill=args.openclaw_skill,
        exclude_v1=args.exclude_v1,
    )

    if args.eval_only:
        tasks = pipeline._load_tasks()
        pipeline.run_tasks_eval_only(tasks, encyclopedia_path=args.encyclopedia)
    elif args.split is not None:
        pipeline.run_split_pipeline(split=args.split, seed=args.seed)
    else:
        pipeline.run_pipeline(
            iterations=args.iterations,
            start_from_step2=args.start_from_step2,
        )


if __name__ == "__main__":
    main()
