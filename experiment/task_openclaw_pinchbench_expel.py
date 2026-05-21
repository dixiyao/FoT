"""
Standalone ExpeL-style runner for PinchBench with OpenClaw.

This script intentionally does not use the FoT insight pipeline:
it does not import client.py, server.py, server_text.py, or write INSIGHTS.md.

It reuses only the PinchBench task loader, OpenClaw execution harness, and
grading harness, then implements an ExpeL-style train/eval workflow:

  1. Load PinchBench tasks and split them in deterministic loader order.
  2. Train on the first half with repeated reflection trials.
  3. Extract ExpeL rules from successful and failed trajectories.
  4. Index successful trajectories in Gemini File Search.
  5. Evaluate on the second half with rules and retrieved trajectories
     prepended directly to the task prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


_PINCHBENCH_SCRIPTS = Path(__file__).parent / "pinchbench" / "scripts"
if str(_PINCHBENCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PINCHBENCH_SCRIPTS))

from lib_agent import (  # type: ignore
    ModelValidationError,
    _get_agent_workspace,
    cleanup_agent_sessions,
    ensure_agent_exists,
    execute_openclaw_task,
    slugify_model,
    validate_openrouter_model,
)
from lib_grading import GradeResult, grade_task  # type: ignore
from lib_tasks import Task, TaskLoader  # type: ignore


# ---------------------------------------------------------------------------
# ExpeL prompt/rule machinery adapted from LeapLabTHU/ExpeL
# ---------------------------------------------------------------------------

FORMAT_RULES_OPERATION_TEMPLATE = """<OPERATION> <RULE NUMBER>: <RULE>

The available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD <NEW RULE NUMBER>: <NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation."""

CRITIQUE_SUMMARY_SUFFIX = {
    "full": "Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:\n",
    "not_full": "Below are the operations you do to the above list of EXISTING RULES:\n",
}

PINCHBENCH_COMPARE_INSTRUCTION = (
    "You will be given two previous PinchBench task trials completed by an "
    "autonomous browser/coding agent: one successful and one unsuccessful. "
    "The failed trial may have produced the wrong artifact, missed a required "
    "tool/file operation, stopped early, or failed the grader."
)

PINCHBENCH_ALL_SUCCESS_INSTRUCTION = (
    "You will be given successful PinchBench task trials completed by an "
    "autonomous browser/coding agent. Extract reusable high-level rules that "
    "would help the same agent solve different PinchBench tasks."
)


def parse_rules(llm_text: str) -> List[Tuple[str, str]]:
    """Parse ExpeL ADD/EDIT/REMOVE/AGREE operations from model output."""
    pattern = r"((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)"
    matches = re.findall(pattern, llm_text)

    res: List[Tuple[str, str]] = []
    banned_words = ["ADD", "AGREE", "EDIT"]
    for operation, text in matches:
        text = text.strip()
        if text and not any(w in text for w in banned_words) and text.endswith("."):
            if "ADD" in operation:
                res.append(("ADD", text))
            else:
                res.append((operation.strip(), text))
    return res


def _retrieve_rule_index(rules: List[Tuple[str, int]], operation: Tuple[str, str]) -> Optional[int]:
    operation_rule_text = operation[1]
    for i, rule in enumerate(rules):
        if rule[0] in operation_rule_text:
            return i
    return None


def _is_existing_rule(rules: List[Tuple[str, int]], operation_rule_text: str) -> bool:
    return any(rule[0] in operation_rule_text for rule in rules)


def update_rules(
    rules: List[Tuple[str, int]],
    operations: List[Tuple[str, str]],
    list_full: bool = False,
) -> List[Tuple[str, int]]:
    """Update ExpeL counted rule list using parsed operations."""
    delete_indices: List[int] = []
    operations = list(operations)

    for i, (operation, operation_rule_text) in enumerate(operations):
        operation_type = operation.split(" ")[0]
        rule_num: Optional[int] = None
        if " " in operation:
            try:
                rule_num = int(operation.split(" ")[1])
            except ValueError:
                rule_num = None

        if operation_type == "ADD":
            if _is_existing_rule(rules, operation_rule_text):
                delete_indices.append(i)
        elif operation_type == "EDIT":
            if _is_existing_rule(rules, operation_rule_text):
                idx = _retrieve_rule_index(rules, (operation, operation_rule_text))
                if idx is not None:
                    operations[i] = (f"AGREE {idx + 1}", rules[idx][0])
                else:
                    delete_indices.append(i)
            elif rule_num is None or rule_num > len(rules):
                delete_indices.append(i)
        elif operation_type in {"REMOVE", "AGREE"}:
            if not _is_existing_rule(rules, operation_rule_text):
                delete_indices.append(i)

    operations = [operations[i] for i in range(len(operations)) if i not in delete_indices]

    for op in ["REMOVE", "AGREE", "EDIT", "ADD"]:
        for operation, operation_rule_text in operations:
            operation_type = operation.split(" ")[0]
            if operation_type != op:
                continue
            if operation_type == "REMOVE":
                idx = _retrieve_rule_index(rules, (operation, operation_rule_text))
                if idx is not None:
                    remove_strength = 3 if list_full else 1
                    rules[idx] = (rules[idx][0], rules[idx][1] - remove_strength)
            elif operation_type == "AGREE":
                idx = _retrieve_rule_index(rules, (operation, operation_rule_text))
                if idx is not None:
                    rules[idx] = (rules[idx][0], rules[idx][1] + 1)
            elif operation_type == "EDIT":
                try:
                    idx = int(operation.split(" ")[1]) - 1
                except (IndexError, ValueError):
                    continue
                if 0 <= idx < len(rules):
                    rules[idx] = (operation_rule_text, rules[idx][1] + 1)
            elif operation_type == "ADD":
                rules.append((operation_rule_text, 2))

    rules = [rule for rule in rules if rule[1] > 0]
    rules.sort(key=lambda x: x[1], reverse=True)
    return rules


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ExpelTrajectory:
    task_id: str
    task_name: str
    task_prompt: str
    attempt: int
    split: str
    status: str
    score: Optional[float]
    max_score: Optional[float]
    is_success: bool
    transcript_text: str
    reflections: List[str]
    execution_time_seconds: Optional[float]
    output_tokens: int
    input_tokens: int
    total_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    request_count: int
    token_usage: Dict[str, Any]
    workspace: str
    transcript_file: Optional[str]
    grade: Dict[str, Any]

    def trajectory_text(self, max_chars: int = 12000) -> str:
        grade_line = "unknown"
        if self.score is not None and self.max_score is not None:
            grade_line = f"{self.score:.3f}/{self.max_score:.3f}"
        reflections = "\n".join(f"- {r}" for r in self.reflections) or "(none)"
        text = (
            f"TASK ID: {self.task_id}\n"
            f"TASK NAME: {self.task_name}\n"
            f"TASK PROMPT:\n{self.task_prompt}\n\n"
            f"ATTEMPT: {self.attempt}\n"
            f"STATUS: {self.status}\n"
            f"GRADE: {grade_line}\n"
            f"REFLECTIONS BEFORE ATTEMPT:\n{reflections}\n\n"
            f"TRAJECTORY:\n{self.transcript_text}\n"
        )
        if len(text) > max_chars:
            return text[:max_chars] + "\n...[truncated]..."
        return text


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------


class GeminiCaller:
    """Small direct Gemini wrapper. No client.py/server.py dependency."""

    THINKING_MODELS = {"gemini-3.1-pro-preview"}

    def __init__(self, api_key: Optional[str], model_name: str, thinking_level: str = "high") -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError("google-genai is required: pip install google-genai") from exc

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key is required via --api-key or GEMINI_API_KEY")

        self.api_key = api_key
        self.model_name = model_name
        self.thinking_level = thinking_level
        self.genai = genai
        self.types = types
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, max_output_tokens: int = 2048, temperature: float = 0.0) -> Tuple[str, Dict[str, Any]]:
        kwargs: Dict[str, Any] = {
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        }
        if self.model_name in self.THINKING_MODELS:
            kwargs["thinking_config"] = self.types.ThinkingConfig(
                thinking_level=self.thinking_level
            )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.types.GenerateContentConfig(**kwargs),
        )
        text = extract_response_text(response)
        usage = getattr(response, "usage_metadata", None)
        token_info: Dict[str, Any] = {
            "model": self.model_name,
            "prompt_chars": len(prompt),
            "max_output_tokens": max_output_tokens,
            "output_tokens": len(text) // 4,
        }
        if usage:
            token_info["input_tokens"] = getattr(usage, "prompt_token_count", 0)
            token_info["output_tokens"] = getattr(usage, "candidates_token_count", len(text) // 4)
            token_info["thinking_tokens"] = getattr(usage, "thoughts_token_count", 0)
            token_info["total_tokens"] = getattr(usage, "total_token_count", None)
        return text, token_info


def extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    candidates = getattr(response, "candidates", None) or []
    parts = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text and not getattr(part, "thought", False):
                parts.append(part_text)
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# File Search RAG
# ---------------------------------------------------------------------------


class GeminiFileSearch:
    def __init__(
        self,
        api_key: Optional[str],
        model_name: str,
        output_dir: Path,
        display_name: Optional[str] = None,
        existing_store_name: Optional[str] = None,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError("google-genai is required for Gemini File Search") from exc

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key is required via --api-key or GEMINI_API_KEY")

        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model_name = model_name
        self.output_dir = output_dir
        self.display_name = display_name or f"pinchbench-expel-{int(time.time())}"
        self.store_name = existing_store_name
        self.store_display_name = None

    def create_or_reuse_store(self, corpus_text: str) -> Optional[str]:
        if self.store_name:
            self.store_display_name = self.store_name
            return self.store_name

        store = None
        try:
            for candidate in self.client.file_search_stores.list():
                if getattr(candidate, "display_name", None) == self.display_name:
                    store = candidate
                    break
        except Exception as exc:
            print(f"  Warning: could not list File Search stores: {exc}")

        if store is None:
            store = self.client.file_search_stores.create(
                config={
                    "display_name": self.display_name,
                    "embedding_model": "models/gemini-embedding-2",
                }
            )
            temp_path = self.output_dir / "expel_success_trajectory_corpus.txt"
            temp_path.write_text(corpus_text, encoding="utf-8")
            operation = self.client.file_search_stores.upload_to_file_search_store(
                file=str(temp_path),
                file_search_store_name=store.name,
                config={
                    "display_name": "expel_success_trajectory_corpus.txt",
                    "chunking_config": {
                        "white_space_config": {
                            "max_tokens_per_chunk": 600,
                            "max_overlap_tokens": 80,
                        }
                    },
                },
            )
            while not getattr(operation, "done", False):
                time.sleep(5)
                operation = self.client.operations.get(operation)

        self.store_name = store.name
        self.store_display_name = getattr(store, "display_name", self.display_name)
        return self.store_name

    def retrieve(self, task_prompt: str, top_k: int) -> Tuple[str, Dict[str, Any]]:
        if not self.store_name:
            return "", {}
        query = (
            "Find the most relevant successful PinchBench trajectories for the task below. "
            f"Return at most {top_k} concise excerpts. Each excerpt should include the source task id, "
            "the reusable action pattern, and any concrete file/tool behavior that transfers. "
            "Do not invent details.\n\n"
            f"TASK:\n{task_prompt}"
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=query,
            config=self.types.GenerateContentConfig(
                tools=[
                    self.types.Tool(
                        file_search=self.types.FileSearch(
                            file_search_store_names=[self.store_name]
                        )
                    )
                ],
                max_output_tokens=2048,
                temperature=0.0,
            ),
        )
        text = extract_response_text(response)
        usage = getattr(response, "usage_metadata", None)
        token_info: Dict[str, Any] = {
            "model": self.model_name,
            "prompt_chars": len(query),
            "output_tokens": len(text) // 4,
            "top_k": top_k,
            "store_name": self.store_name,
        }
        if usage:
            token_info["input_tokens"] = getattr(usage, "prompt_token_count", 0)
            token_info["output_tokens"] = getattr(usage, "candidates_token_count", len(text) // 4)
            token_info["thinking_tokens"] = getattr(usage, "thoughts_token_count", 0)
            token_info["total_tokens"] = getattr(usage, "total_token_count", None)
        return text, token_info


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def _check_openclaw() -> str:
    path = shutil.which("openclaw")
    if path:
        return path
    override_path = os.environ.get("OPENCLAW_PATH")
    if override_path and Path(override_path).exists():
        return override_path
    raise SystemExit(
        "ERROR: OpenClaw CLI not found. Install OpenClaw or set OPENCLAW_PATH."
    )


def _extract_transcript_text(transcript: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for entry in transcript:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "") or block.get("content", "")
                    if text:
                        parts.append(f"[{role}]: {text}")
        elif isinstance(content, str):
            parts.append(f"[{role}]: {content}")
    return "\n\n".join(parts)


def _clear_fot_insights(agent_id: str) -> None:
    """Prevent this standalone runner from inheriting FoT INSIGHTS.md state."""
    staging = Path("/tmp/pinchbench_fot/INSIGHTS.md")
    if staging.exists():
        try:
            staging.unlink()
        except OSError:
            pass
    workspace = _get_agent_workspace(agent_id)
    if workspace:
        insights = workspace / "INSIGHTS.md"
        if insights.exists():
            try:
                insights.unlink()
            except OSError:
                pass


def _clone_task_with_prompt(task: Task, prompt: str) -> Task:
    """Clone a Task while preserving metadata and replacing the executed prompt.

    For multi-session tasks, OpenClaw ignores task.prompt and uses frontmatter
    sessions, so inject the context into the first session as well.
    """
    frontmatter = dict(task.frontmatter or {})
    sessions = frontmatter.get("sessions")
    if isinstance(sessions, list) and sessions:
        new_sessions = list(sessions)
        first = new_sessions[0]
        if isinstance(first, str):
            new_sessions[0] = prompt + "\n\n" + first
        elif isinstance(first, dict):
            first_dict = dict(first)
            key = "prompt" if "prompt" in first_dict else "message"
            first_dict[key] = prompt + "\n\n" + str(first_dict.get(key, ""))
            new_sessions[0] = first_dict
        frontmatter["sessions"] = new_sessions

    return Task(
        task_id=task.task_id,
        name=task.name,
        category=task.category,
        grading_type=task.grading_type,
        timeout_seconds=task.timeout_seconds,
        workspace_files=task.workspace_files,
        prompt=prompt,
        expected_behavior=task.expected_behavior,
        grading_criteria=task.grading_criteria,
        automated_checks=task.automated_checks,
        llm_judge_rubric=task.llm_judge_rubric,
        grading_weights=task.grading_weights,
        file_path=task.file_path,
        frontmatter=frontmatter,
    )


def _task_full_prompt(task: Task) -> str:
    sessions = (task.frontmatter or {}).get("sessions")
    if isinstance(sessions, list) and sessions:
        parts = [task.prompt.strip()]
        for idx, session in enumerate(sessions, 1):
            if isinstance(session, str):
                session_prompt = session
            elif isinstance(session, dict):
                session_prompt = str(session.get("prompt") or session.get("message") or "")
            else:
                session_prompt = str(session)
            parts.append(f"Session {idx}:\n{session_prompt.strip()}")
        return "\n\n".join(part for part in parts if part)
    return task.prompt


def _full_credit(grade: Optional[GradeResult]) -> bool:
    if grade is None or grade.max_score <= 0:
        return False
    return grade.score >= grade.max_score


def _grade_to_dict(grade: Optional[GradeResult]) -> Dict[str, Any]:
    return grade.to_dict() if grade else {}


def _safe_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _format_existing_rules(rule_items_with_count: Sequence[Tuple[str, int]]) -> str:
    if not rule_items_with_count:
        return "(none yet)"
    return "\n".join(
        f"{idx}. {rule} {{{count}}}"
        for idx, (rule, count) in enumerate(rule_items_with_count, 1)
    )


def _split_chunks(items: Sequence[Any], chunk_size: int) -> List[List[Any]]:
    if chunk_size <= 0:
        chunk_size = 1
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]


def _trim(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]..."


# ---------------------------------------------------------------------------
# Main ExpeL pipeline
# ---------------------------------------------------------------------------


class OpenClawPinchBenchExpel:
    def __init__(
        self,
        *,
        model_id: str,
        output_dir: str,
        suite: str,
        pinchbench_dir: Optional[str],
        judge_model: Optional[str],
        api_key: Optional[str],
        api_model: str,
        thinking_level: str,
        max_reflection_depth: int,
        max_rules: int,
        success_critique_num: int,
        rag_top_k: int,
        rag_store: Optional[str],
        rag_display_name: Optional[str],
        timeout_multiplier: float,
        base_url: Optional[str],
        openclaw_api_key: Optional[str],
    ) -> None:
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.suite = suite
        self.pinchbench_dir = Path(pinchbench_dir) if pinchbench_dir else Path(__file__).parent / "pinchbench"
        self.tasks_dir = self.pinchbench_dir / "tasks"
        self.skill_dir = self.pinchbench_dir
        self.judge_model = judge_model or model_id
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.api_model = api_model
        self.thinking_level = thinking_level
        self.max_reflection_depth = max_reflection_depth
        self.max_rules = max_rules
        self.success_critique_num = success_critique_num
        self.rag_top_k = rag_top_k
        self.rag_store_name = rag_store or None  # treat empty string (bare flag) same as None
        self.rag_display_name = rag_display_name
        self.timeout_multiplier = timeout_multiplier
        self.base_url = base_url
        self.openclaw_api_key = openclaw_api_key
        self.agent_id = f"pinchbench-expel-{slugify_model(model_id)}"

        self.gemini = GeminiCaller(self.api_key, self.api_model, self.thinking_level)
        self.rule_items_with_count: List[Tuple[str, int]] = []
        self.critique_log = ""
        self.success_trajectories: List[ExpelTrajectory] = []
        self.failed_trajectories: List[ExpelTrajectory] = []
        self.reflection_records: List[Dict[str, Any]] = []
        self.token_records: List[Dict[str, Any]] = []
        self.attempt_records: List[Dict[str, Any]] = []
        self.rag: Optional[GeminiFileSearch] = None

    def _run_config(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "suite": self.suite,
            "judge_model": self.judge_model,
            "api_model": self.api_model,
            "thinking_level": self.thinking_level,
            "max_reflection_depth": self.max_reflection_depth,
            "max_attempts_per_train_task": max(1, self.max_reflection_depth + 1),
            "max_rules": self.max_rules,
            "success_critique_num": self.success_critique_num,
            "rag_top_k": self.rag_top_k,
            "rag_store": self.rag_store_name,
            "rag_display_name": self.rag_display_name,
            "timeout_multiplier": self.timeout_multiplier,
        }

    def _record_token_usage(
        self,
        *,
        stage: str,
        token_info: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "stage": stage,
            "metadata": metadata or {},
            "token_info": token_info or {},
            "timestamp": time.time(),
        }
        self.token_records.append(record)
        self._write_token_usage()

    def _token_summary(self) -> Dict[str, Any]:
        by_stage: Dict[str, Dict[str, float]] = {}
        totals = {
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "thinking_tokens": 0.0,
            "total_tokens": 0.0,
            "cache_read_tokens": 0.0,
            "cache_write_tokens": 0.0,
            "cost_usd": 0.0,
            "request_count": 0.0,
        }
        numeric_keys = set(totals)
        for record in self.token_records:
            stage = record.get("stage", "unknown")
            stage_totals = by_stage.setdefault(stage, {k: 0.0 for k in numeric_keys})
            info = record.get("token_info", {}) or {}
            for key in numeric_keys:
                value = info.get(key, 0) or 0
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    numeric = 0.0
                stage_totals[key] += numeric
                totals[key] += numeric
        return {
            "totals": totals,
            "by_stage": by_stage,
            "records": len(self.token_records),
        }

    def _write_token_usage(self) -> None:
        _safe_write_json(
            self.output_dir / "token_usage.json",
            {
                "config": self._run_config(),
                "summary": self._token_summary(),
                "records": self.token_records,
            },
        )

    def _write_attempt_log(self) -> None:
        _safe_write_json(
            self.output_dir / "train_attempts.json",
            {
                "config": self._run_config(),
                "attempts": self.attempt_records,
            },
        )

    def _setup_agent(self) -> None:
        _check_openclaw()
        effective_base_url = self.base_url
        effective_api_key = self.openclaw_api_key
        effective_model_id = self.model_id

        if self.model_id.startswith("google/") or self.model_id.startswith("gemini/"):
            effective_base_url = effective_base_url or "https://generativelanguage.googleapis.com/v1beta/openai"
            effective_api_key = effective_api_key or self.api_key or os.getenv("GEMINI_API_KEY")
            effective_model_id = self.model_id.split("/", 1)[1]
            print("Using Gemini OpenAI-compatible endpoint for OpenClaw agent")

        if effective_base_url:
            print(f"Using custom OpenClaw base URL: {effective_base_url}")
        else:
            try:
                print(f"Validating OpenRouter model: {self.model_id}")
                validate_openrouter_model(self.model_id)
            except ModelValidationError as exc:
                print(f"Warning: {exc}")

        workspace = _get_agent_workspace(self.agent_id)
        if workspace is None:
            workspace = Path.home() / ".openclaw" / "agents" / self.agent_id.lower() / "workspace"
        ensure_agent_exists(
            self.agent_id,
            effective_model_id,
            workspace,
            base_url=effective_base_url,
            api_key=effective_api_key,
        )
        _clear_fot_insights(self.agent_id)
        cleanup_agent_sessions(self.agent_id)

    def _load_tasks(self) -> List[Task]:
        tasks = TaskLoader(self.tasks_dir).load_all_tasks()
        if self.suite == "all":
            return tasks
        if self.suite == "automated-only":
            return [t for t in tasks if t.grading_type == "automated"]
        ids = {tid.strip() for tid in self.suite.split(",") if tid.strip()}
        return [t for t in tasks if t.task_id in ids]

    def _execute_and_grade(
        self,
        *,
        task: Task,
        execution_task: Task,
        run_id: str,
        output_dir: Path,
    ) -> Tuple[Dict[str, Any], Optional[GradeResult]]:
        _clear_fot_insights(self.agent_id)
        exec_result = execute_openclaw_task(
            task=execution_task,
            agent_id=self.agent_id,
            model_id=self.model_id,
            run_id=run_id,
            timeout_multiplier=self.timeout_multiplier,
            skill_dir=self.skill_dir,
            output_dir=output_dir,
            verbose=False,
        )
        try:
            grade = grade_task(
                task=task,
                execution_result=exec_result,
                skill_dir=self.skill_dir,
                judge_model=self.judge_model,
                judge_backend="api",
                judge_api_key=self.api_key,
            )
        except Exception as exc:
            print(f"    Warning: grading failed for {task.task_id}: {exc}")
            grade = None
        return exec_result, grade

    def _make_trajectory(
        self,
        *,
        task: Task,
        attempt: int,
        split: str,
        exec_result: Dict[str, Any],
        grade: Optional[GradeResult],
        reflections: List[str],
        transcript_file: Optional[Path],
    ) -> ExpelTrajectory:
        transcript = exec_result.get("transcript", []) or []
        usage = exec_result.get("usage", {}) or {}
        token_usage = {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "cache_read_tokens": int(usage.get("cache_read_tokens", 0) or 0),
            "cache_write_tokens": int(usage.get("cache_write_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "cost_usd": float(usage.get("cost_usd", 0.0) or 0.0),
            "request_count": int(usage.get("request_count", 0) or 0),
        }
        return ExpelTrajectory(
            task_id=task.task_id,
            task_name=task.name,
            task_prompt=_task_full_prompt(task),
            attempt=attempt,
            split=split,
            status=str(exec_result.get("status", "unknown")),
            score=grade.score if grade else None,
            max_score=grade.max_score if grade else None,
            is_success=_full_credit(grade),
            transcript_text=_extract_transcript_text(transcript),
            reflections=list(reflections),
            execution_time_seconds=(
                float(exec_result["execution_time"])
                if exec_result.get("execution_time") is not None
                else None
            ),
            output_tokens=token_usage["output_tokens"],
            input_tokens=token_usage["input_tokens"],
            total_tokens=token_usage["total_tokens"],
            cache_read_tokens=token_usage["cache_read_tokens"],
            cache_write_tokens=token_usage["cache_write_tokens"],
            cost_usd=token_usage["cost_usd"],
            request_count=token_usage["request_count"],
            token_usage=token_usage,
            workspace=str(exec_result.get("workspace", "")),
            transcript_file=str(transcript_file) if transcript_file else None,
            grade=_grade_to_dict(grade),
        )

    def _reflection_prompt(self, task: Task, trajectory: ExpelTrajectory) -> str:
        return f"""
You are helping an autonomous PinchBench agent improve after a failed trial.
Reflect on why the trial failed and what should be changed on the next attempt.

Return 2-5 concise bullet points. Focus on concrete actions, file/tool use,
format requirements, and validation steps. Do not mention this as a benchmark.

TASK:
{_task_full_prompt(task)}

FAILED TRIAL:
{trajectory.trajectory_text(max_chars=16000)}
"""

    def _generate_reflection(self, task: Task, trajectory: ExpelTrajectory) -> Tuple[str, Dict[str, Any]]:
        text, token_info = self.gemini.generate(
            self._reflection_prompt(task, trajectory),
            max_output_tokens=1024,
            temperature=0.0,
        )
        return text.strip(), token_info

    def _prompt_with_reflections(self, original_prompt: str, reflections: List[str]) -> str:
        if not reflections:
            return original_prompt
        reflection_block = "\n\n".join(
            f"Reflection from failed attempt {idx}:\n{reflection}"
            for idx, reflection in enumerate(reflections, 1)
        )
        return (
            "Before doing the task, use these reflections from previous failed attempts "
            "on the same task to avoid repeating mistakes.\n\n"
            f"{reflection_block}\n\n"
            "Now complete the original task exactly as requested:\n\n"
            f"{original_prompt}"
        )

    def run_training(self, train_tasks: List[Task]) -> None:
        print("\n" + "=" * 80)
        print(f"ExpeL Training Phase ({len(train_tasks)} tasks)")
        print("=" * 80)

        train_dir = self.output_dir / "train"
        transcript_dir = train_dir / "transcripts"

        for task_index, task in enumerate(train_tasks, 1):
            print(f"\n[train {task_index}/{len(train_tasks)}] {task.task_id} — {task.name}")
            reflections: List[str] = []
            attempts_allowed = max(1, self.max_reflection_depth + 1)

            for attempt in range(1, attempts_allowed + 1):
                print(f"  Attempt {attempt}/{attempts_allowed}")
                prompt = self._prompt_with_reflections(task.prompt, reflections)
                execution_task = _clone_task_with_prompt(task, prompt)
                transcript_archive = transcript_dir / f"{task.task_id}_attempt_{attempt:02d}.jsonl"
                exec_result, grade = self._execute_and_grade(
                    task=task,
                    execution_task=execution_task,
                    run_id=f"expel-train-{task_index}-{attempt}",
                    output_dir=transcript_dir,
                )
                default_archive = transcript_dir / f"{task.task_id}.jsonl"
                if default_archive.exists():
                    default_archive.replace(transcript_archive)
                trajectory = self._make_trajectory(
                    task=task,
                    attempt=attempt,
                    split="train",
                    exec_result=exec_result,
                    grade=grade,
                    reflections=reflections,
                    transcript_file=transcript_archive if transcript_archive.exists() else None,
                )
                self._record_token_usage(
                    stage="openclaw_train_attempt",
                    token_info=trajectory.token_usage,
                    metadata={
                        "task_id": task.task_id,
                        "task_name": task.name,
                        "attempt": attempt,
                        "max_attempts": attempts_allowed,
                        "max_reflection_depth": self.max_reflection_depth,
                        "success": trajectory.is_success,
                    },
                )
                save_path = train_dir / f"problem_{task_index:04d}_attempt_{attempt:02d}.json"
                _safe_write_json(save_path, asdict(trajectory))

                score_str = "not graded"
                if grade:
                    score_str = f"{grade.score:.2f}/{grade.max_score:.2f}"
                print(f"    status={trajectory.status} grade={score_str} success={trajectory.is_success}")

                if trajectory.is_success:
                    self.success_trajectories.append(trajectory)
                    self.attempt_records.append(
                        {
                            "task_id": task.task_id,
                            "task_name": task.name,
                            "attempts_used": attempt,
                            "max_attempts": attempts_allowed,
                            "max_reflection_depth": self.max_reflection_depth,
                            "success": True,
                            "final_score": trajectory.score,
                            "final_max_score": trajectory.max_score,
                        }
                    )
                    self._write_attempt_log()
                    break

                self.failed_trajectories.append(trajectory)
                if attempt < attempts_allowed:
                    reflection, reflection_tokens = self._generate_reflection(task, trajectory)
                    self._record_token_usage(
                        stage="gemini_reflection",
                        token_info=reflection_tokens,
                        metadata={
                            "task_id": task.task_id,
                            "task_name": task.name,
                            "failed_attempt": attempt,
                            "max_reflection_depth": self.max_reflection_depth,
                        },
                    )
                    reflections.append(reflection)
                    self.reflection_records.append(
                        {
                            "task_id": task.task_id,
                            "attempt": attempt,
                            "reflection": reflection,
                        }
                    )
                    print(f"    reflection chars={len(reflection)}")
                else:
                    self.attempt_records.append(
                        {
                            "task_id": task.task_id,
                            "task_name": task.name,
                            "attempts_used": attempt,
                            "max_attempts": attempts_allowed,
                            "max_reflection_depth": self.max_reflection_depth,
                            "success": False,
                            "final_score": trajectory.score,
                            "final_max_score": trajectory.max_score,
                        }
                    )
                    self._write_attempt_log()

            self._write_training_artifacts()

    def _write_training_artifacts(self) -> None:
        _safe_write_json(
            self.output_dir / "expel_train_success_trajectories.json",
            [asdict(t) for t in self.success_trajectories],
        )
        _safe_write_json(
            self.output_dir / "expel_train_failed_trajectories.json",
            [asdict(t) for t in self.failed_trajectories],
        )
        _safe_write_json(self.output_dir / "expel_reflections.json", self.reflection_records)

    def _critique_prompt(
        self,
        *,
        success_history: Optional[str],
        fail_history: Optional[str],
        task_prompt: Optional[str],
    ) -> str:
        existing_rules = _format_existing_rules(self.rule_items_with_count)
        suffix = (
            CRITIQUE_SUMMARY_SUFFIX["full"]
            if len(self.rule_items_with_count) >= self.max_rules + 5
            else CRITIQUE_SUMMARY_SUFFIX["not_full"]
        )
        if fail_history is not None:
            return f"""
{PINCHBENCH_COMPARE_INSTRUCTION}

Here are the two previous trials to compare and critique:

TRIAL TASK:
{_trim(task_prompt or "", 5000)}

SUCCESSFUL TRIAL:
{_trim(success_history or "", 12000)}

FAILED TRIAL:
{_trim(fail_history, 12000)}

Here are the EXISTING RULES:
{existing_rules}

By examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action.

{FORMAT_RULES_OPERATION_TEMPLATE}

{suffix}
"""
        return f"""
{PINCHBENCH_ALL_SUCCESS_INSTRUCTION}

Here are the trials:
{_trim(success_history or "", 24000)}

Here are the EXISTING RULES:
{existing_rules}

By examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action.

{FORMAT_RULES_OPERATION_TEMPLATE}

{suffix}
"""

    def _apply_rule_operations(self, llm_output: str) -> List[Tuple[str, str]]:
        operations = parse_rules(llm_output)
        self.rule_items_with_count = update_rules(
            self.rule_items_with_count,
            operations,
            list_full=(self.max_rules + 5 <= len(self.rule_items_with_count)),
        )
        if len(self.rule_items_with_count) > self.max_rules:
            self.rule_items_with_count = self.rule_items_with_count[: self.max_rules]
        return operations

    def extract_rules(self, train_tasks: List[Task]) -> None:
        print("\n" + "=" * 80)
        print("ExpeL Rule Extraction")
        print("=" * 80)

        by_task_success: Dict[str, List[ExpelTrajectory]] = {}
        by_task_fail: Dict[str, List[ExpelTrajectory]] = {}
        task_by_id = {task.task_id: task for task in train_tasks}
        for trajectory in self.success_trajectories:
            by_task_success.setdefault(trajectory.task_id, []).append(trajectory)
        for trajectory in self.failed_trajectories:
            by_task_fail.setdefault(trajectory.task_id, []).append(trajectory)

        logs: List[str] = ["################ Compare Critiques ################\n"]
        for task_id, successes in by_task_success.items():
            failures = by_task_fail.get(task_id, [])
            if not failures:
                continue
            for success in successes:
                for failure in failures:
                    prompt = self._critique_prompt(
                        success_history=success.trajectory_text(),
                        fail_history=failure.trajectory_text(),
                        task_prompt=_task_full_prompt(task_by_id[task_id])
                        if task_id in task_by_id
                        else success.task_prompt,
                    )
                    llm_output, token_info = self.gemini.generate(prompt, max_output_tokens=2048, temperature=0.0)
                    self._record_token_usage(
                        stage="gemini_rule_compare",
                        token_info=token_info,
                        metadata={
                            "task_id": task_id,
                            "success_attempt": success.attempt,
                            "failed_attempt": failure.attempt,
                            "rules_before": len(self.rule_items_with_count),
                        },
                    )
                    operations = self._apply_rule_operations(llm_output)
                    logs.append(
                        f"TASK {task_id}\n------- MODEL OUTPUT -------\n{llm_output}\n"
                        f"------- PARSED OPERATIONS -------\n{operations}\n"
                        f"------- RULES -------\n{_format_existing_rules(self.rule_items_with_count)}\n"
                    )

        logs.append("\n################ SUCCESS CRITIQUES ################\n")
        success_items = [
            (trajectory.task_id, trajectory.trajectory_text())
            for trajectory in self.success_trajectories
        ]
        for chunk in _split_chunks(success_items, self.success_critique_num):
            success_history = "\n\n".join(
                f"TASK {task_id}\n{trajectory_text}"
                for task_id, trajectory_text in chunk
            )
            prompt = self._critique_prompt(
                success_history=success_history,
                fail_history=None,
                task_prompt=None,
            )
            llm_output, token_info = self.gemini.generate(prompt, max_output_tokens=2048, temperature=0.0)
            self._record_token_usage(
                stage="gemini_rule_success",
                token_info=token_info,
                metadata={
                    "task_ids": [task_id for task_id, _ in chunk],
                    "chunk_size": len(chunk),
                    "rules_before": len(self.rule_items_with_count),
                },
            )
            operations = self._apply_rule_operations(llm_output)
            logs.append(
                f"SUCCESS CHUNK {[task_id for task_id, _ in chunk]}\n"
                f"------- MODEL OUTPUT -------\n{llm_output}\n"
                f"------- PARSED OPERATIONS -------\n{operations}\n"
                f"------- RULES -------\n{_format_existing_rules(self.rule_items_with_count)}\n"
            )

        self.critique_log = "\n".join(logs)
        rules = self.rules_text()
        _safe_write_json(
            self.output_dir / "expel_rules.json",
            {
                "rules": [
                    {"rule": rule, "count": count}
                    for rule, count in self.rule_items_with_count
                ]
            },
        )
        (self.output_dir / "expel_rules.md").write_text(rules + "\n", encoding="utf-8")
        (self.output_dir / "expel_critique_log.txt").write_text(
            self.critique_log,
            encoding="utf-8",
        )
        print(f"Extracted {len(self.rule_items_with_count)} rules")

    def rules_text(self) -> str:
        if not self.rule_items_with_count:
            return "(No ExpeL rules were extracted.)"
        return "\n".join(
            f"{idx}. {rule}"
            for idx, (rule, _count) in enumerate(self.rule_items_with_count, 1)
        )

    def build_rag_store(self) -> None:
        if not self.success_trajectories:
            print("No successful training trajectories; skipping File Search store creation")
            return

        corpus = "\n\n---\n\n".join(
            f"Source task: {t.task_id}\nTask name: {t.task_name}\n\n{t.trajectory_text(max_chars=16000)}"
            for t in self.success_trajectories
        )
        self.rag = GeminiFileSearch(
            api_key=self.api_key,
            model_name=self.api_model,
            output_dir=self.output_dir,
            display_name=self.rag_display_name,
            existing_store_name=self.rag_store_name,
        )
        store_name = self.rag.create_or_reuse_store(corpus)
        _safe_write_json(
            self.output_dir / "rag_store.json",
            {
                "store_name": store_name,
                "display_name": self.rag.store_display_name,
                "source_success_trajectories": len(self.success_trajectories),
            },
        )
        print(f"Gemini File Search store ready: {store_name}")

    def _eval_prompt(self, task: Task, retrieved: str) -> str:
        retrieved_section = retrieved.strip() or "(No retrieved successful trajectories.)"
        return f"""
The following are experiences gathered from successful and failed PinchBench training trials. Use them as references to perform better on the current task.

EXPEL RULES:
{self.rules_text()}

RELEVANT SUCCESSFUL TRAJECTORIES RETRIEVED FROM GOOGLE FILE SEARCH:
{retrieved_section}

Now complete the original task exactly as requested.

ORIGINAL TASK:
{_task_full_prompt(task)}
"""

    def run_eval(self, eval_tasks: List[Task]) -> None:
        print("\n" + "=" * 80)
        print(f"ExpeL Evaluation Phase ({len(eval_tasks)} tasks)")
        print("=" * 80)

        eval_dir = self.output_dir / "eval"
        transcript_dir = eval_dir / "transcripts"
        metrics: List[Dict[str, Any]] = []

        for task_index, task in enumerate(eval_tasks, 1):
            print(f"\n[eval {task_index}/{len(eval_tasks)}] {task.task_id} — {task.name}")
            retrieved = ""
            if self.rag:
                try:
                    retrieved, rag_tokens = self.rag.retrieve(_task_full_prompt(task), self.rag_top_k)
                    self._record_token_usage(
                        stage="gemini_rag_retrieval",
                        token_info=rag_tokens,
                        metadata={
                            "task_id": task.task_id,
                            "task_name": task.name,
                            "rag_top_k": self.rag_top_k,
                        },
                    )
                except Exception as exc:
                    print(f"  Warning: RAG retrieval failed: {exc}")
            prompt = self._eval_prompt(task, retrieved)
            execution_task = _clone_task_with_prompt(task, prompt)
            exec_result, grade = self._execute_and_grade(
                task=task,
                execution_task=execution_task,
                run_id=f"expel-eval-{task_index}",
                output_dir=transcript_dir,
            )
            transcript_archive = transcript_dir / f"{task.task_id}.jsonl"
            trajectory = self._make_trajectory(
                task=task,
                attempt=1,
                split="eval",
                exec_result=exec_result,
                grade=grade,
                reflections=[],
                transcript_file=transcript_archive if transcript_archive.exists() else None,
            )
            self._record_token_usage(
                stage="openclaw_eval_attempt",
                token_info=trajectory.token_usage,
                metadata={
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "attempt": 1,
                    "success": trajectory.is_success,
                },
            )
            score_str = "not graded"
            if grade:
                score_str = f"{grade.score:.2f}/{grade.max_score:.2f}"
            print(f"  status={trajectory.status} grade={score_str}")
            output = {
                **asdict(trajectory),
                "retrieved_context": retrieved,
                "expel_rules": self.rules_text(),
            }
            _safe_write_json(eval_dir / f"problem_{task_index:04d}.json", output)
            metrics.append(
                {
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "status": trajectory.status,
                    "score": trajectory.score,
                    "max_score": trajectory.max_score,
                    "is_success": trajectory.is_success,
                    "output_tokens": trajectory.output_tokens,
                    "input_tokens": trajectory.input_tokens,
                    "total_tokens": trajectory.total_tokens,
                    "cost_usd": trajectory.cost_usd,
                    "request_count": trajectory.request_count,
                    "execution_time_seconds": trajectory.execution_time_seconds,
                }
            )
            self._write_eval_metrics(metrics)

    def _write_eval_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        graded = [m for m in metrics if m.get("score") is not None and m.get("max_score")]
        score_sum = sum(float(m["score"] or 0.0) for m in graded)
        max_sum = sum(float(m["max_score"] or 0.0) for m in graded)
        payload = {
            "tasks": metrics,
            "summary": {
                "tasks": len(metrics),
                "graded": len(graded),
                "score": score_sum,
                "max_score": max_sum,
                "accuracy_pct": (score_sum / max_sum * 100.0) if max_sum else 0.0,
                "successes": sum(1 for m in metrics if m.get("is_success")),
                "output_tokens": sum(int(m.get("output_tokens", 0) or 0) for m in metrics),
                "input_tokens": sum(int(m.get("input_tokens", 0) or 0) for m in metrics),
                "total_tokens": sum(int(m.get("total_tokens", 0) or 0) for m in metrics),
                "cost_usd": sum(float(m.get("cost_usd", 0.0) or 0.0) for m in metrics),
                "request_count": sum(int(m.get("request_count", 0) or 0) for m in metrics),
            },
        }
        _safe_write_json(self.output_dir / "eval" / "metrics.json", payload)

    def run(self) -> None:
        start = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _safe_write_json(self.output_dir / "run_config.json", self._run_config())
        self._write_token_usage()
        self._write_attempt_log()
        self._setup_agent()
        tasks = self._load_tasks()
        if len(tasks) < 2:
            raise ValueError("ExpeL split requires at least two tasks")

        split_at = len(tasks) // 2
        train_tasks = tasks[:split_at]
        eval_tasks = tasks[split_at:]
        split_payload = {
            "suite": self.suite,
            "total_tasks": len(tasks),
            "train_tasks": [{"task_id": t.task_id, "name": t.name} for t in train_tasks],
            "eval_tasks": [{"task_id": t.task_id, "name": t.name} for t in eval_tasks],
        }
        _safe_write_json(self.output_dir / "split.json", split_payload)

        print(f"Loaded {len(tasks)} tasks; train={len(train_tasks)} eval={len(eval_tasks)}")
        self.run_training(train_tasks)
        self.extract_rules(train_tasks)
        self.build_rag_store()
        self.run_eval(eval_tasks)

        elapsed = time.time() - start
        train_success_rate = (
            len(self.success_trajectories) / len(train_tasks) * 100.0
            if train_tasks
            else 0.0
        )
        eval_metrics_path = self.output_dir / "eval" / "metrics.json"
        eval_summary = {}
        if eval_metrics_path.exists():
            eval_summary = json.loads(eval_metrics_path.read_text(encoding="utf-8")).get("summary", {})

        print("\n" + "=" * 80)
        print("Standalone ExpeL PinchBench Complete")
        print("=" * 80)
        print(f"Train successes: {len(self.success_trajectories)}/{len(train_tasks)} ({train_success_rate:.1f}%)")
        print(f"Rules: {len(self.rule_items_with_count)}")
        print(f"RAG store: {self.rag.store_name if self.rag else None}")
        print(f"Eval accuracy: {float(eval_summary.get('accuracy_pct', 0.0) or 0.0):.1f}%")
        token_totals = self._token_summary().get("totals", {})
        print(
            "Token usage: "
            f"input={int(token_totals.get('input_tokens', 0) or 0)} "
            f"output={int(token_totals.get('output_tokens', 0) or 0)} "
            f"thinking={int(token_totals.get('thinking_tokens', 0) or 0)} "
            f"total={int(token_totals.get('total_tokens', 0) or 0)} "
            f"cost=${float(token_totals.get('cost_usd', 0.0) or 0.0):.4f}"
        )
        print(f"Output dir: {self.output_dir}")
        print(f"Elapsed: {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone ExpeL-style OpenClaw PinchBench runner"
    )
    parser.add_argument("--model", default=None, help="OpenClaw model identifier. Defaults to --judge, then google/--api-model.")
    parser.add_argument("--output-dir", default="pinchbench_expel_output")
    parser.add_argument("--suite", default="all", help='"all", "automated-only", or comma-separated task IDs.')
    parser.add_argument("--pinchbench-dir", default=None)
    parser.add_argument("--judge", default=None, help="Judge model; defaults to --model.")
    parser.add_argument("--api-key", default=None, help="Gemini API key for ExpeL reflection/rule/RAG calls.")
    parser.add_argument("--api-model", default="gemini-3-pro-preview", help="Gemini model for ExpeL reflection/rule/RAG calls.")
    parser.add_argument("--thinking-level", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--max-reflection-depth", type=int, default=3)
    parser.add_argument("--max-rules", type=int, default=20)
    parser.add_argument("--success-critique-num", type=int, default=8)
    parser.add_argument("--rag-top-k", type=int, default=3)
    parser.add_argument("--rag-store", nargs="?", const="", default=None, help="Enable RAG File Search store. Pass a store name to reuse an existing one, or pass the flag alone to create a new store.")
    parser.add_argument("--rag-display-name", default=None, help="Display name for a new/reused File Search store.")
    parser.add_argument("--timeout-multiplier", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=1, help="Compatibility only; ExpeL runner always performs one train/eval split.")
    parser.add_argument("--base-url", default=None, help="Optional custom OpenAI-compatible base URL for OpenClaw.")
    parser.add_argument("--openclaw-api-key", default=None, help="Optional API key for the OpenClaw provider.")
    args = parser.parse_args()

    model_id = args.model or args.judge or f"google/{args.api_model}"

    runner = OpenClawPinchBenchExpel(
        model_id=model_id,
        output_dir=args.output_dir,
        suite=args.suite,
        pinchbench_dir=args.pinchbench_dir,
        judge_model=args.judge,
        api_key=args.api_key,
        api_model=args.api_model,
        thinking_level=args.thinking_level,
        max_reflection_depth=args.max_reflection_depth,
        max_rules=args.max_rules,
        success_critique_num=args.success_critique_num,
        rag_top_k=args.rag_top_k,
        rag_store=args.rag_store,
        rag_display_name=args.rag_display_name,
        timeout_multiplier=args.timeout_multiplier,
        base_url=args.base_url,
        openclaw_api_key=args.openclaw_api_key,
    )
    runner.run()


if __name__ == "__main__":
    main()
