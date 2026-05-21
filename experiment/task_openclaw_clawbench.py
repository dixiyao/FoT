"""
OpenClaw ClawBench Pipeline
Runs ClawBench web-automation tasks with an OpenAI-compatible model, extracts
reasoning traces from agent-messages transcripts, and aggregates them into an
insight library via server_text.py.

Structure mirrors task_openclaw_claweval.py:

  Iteration N:
    Step 1: Run each ClawBench task via `clawbench run` (subprocess),
            which drives a real browser inside a Docker/Podman container
    Step 2: Reflection — extract procedural knowledge from agent transcript
    Step 3: Insight extraction — package as reusable traces (JSON)
    Save:   problem_XXXX.json  (same format as other pipelines)

  Aggregation:
    Call server_text.py TextBasedInsightAggregationServer to build an
    encyclopedia from all extracted insights.

  Next Iteration:
    Inject encyclopedia as context when running tasks in subsequent iterations.

NOTE: ClawBench requires Docker or Podman to be installed and running.
      Each task spins up a container with a full browser environment.

Usage:
    python task_openclaw_clawbench.py \\
        --model google/gemini-3-pro-preview \\
        --output-dir clawbench_output \\
        --category daily-life \\
        --use-api --api-provider gemini --api-key YOUR_KEY \\
        --iterations 2

    # Lite subset (20 curated tasks):
    python task_openclaw_clawbench.py \\
        --model google/gemini-3-pro-preview \\
        --output-dir clawbench_output \\
        --category lite \\
        --use-api --api-provider gemini --api-key YOUR_KEY

    # Start from aggregation step (skip task execution):
    python task_openclaw_clawbench.py --start-from-step2 \\
        --output-dir clawbench_output --use-api --api-provider gemini --api-key YOUR_KEY

    # With existing encyclopedia:
    python task_openclaw_clawbench.py \\
        --model google/gemini-3-pro-preview \\
        --encyclopedia clawbench_output/encyclopedia.json \\
        --output-dir clawbench_output_iter2 \\
        --use-api --api-provider gemini --api-key YOUR_KEY
"""

import argparse
import collections
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Locate the clawbench repo (cloned as clawbench/ next to this script)
# ---------------------------------------------------------------------------
_CLAWBENCH_DIR = Path(__file__).parent / "clawbench"

# ---------------------------------------------------------------------------
# Import the clawbench lib_agent programmatic integration layer
# ---------------------------------------------------------------------------
_CLAWBENCH_SCRIPTS = _CLAWBENCH_DIR / "scripts"
if _CLAWBENCH_SCRIPTS.exists() and str(_CLAWBENCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_CLAWBENCH_SCRIPTS))

try:
    # Import as clawbench_lib to avoid name collision with the clawbench package
    import importlib as _il
    _clawbench_lib = _il.util.spec_from_file_location(
        "clawbench_lib_agent", _CLAWBENCH_SCRIPTS / "lib_agent.py"
    )
    _clawbench_lib_mod = _il.util.module_from_spec(_clawbench_lib)
    _clawbench_lib.loader.exec_module(_clawbench_lib_mod)
    _execute_clawbench_task = _clawbench_lib_mod.execute_clawbench_task
    _write_clawbench_models_yaml = _clawbench_lib_mod.write_models_yaml
    _HAS_CLAWBENCH_LIB = True
except Exception:
    _HAS_CLAWBENCH_LIB = False

# ---------------------------------------------------------------------------
# Import our local pipeline pieces
# ---------------------------------------------------------------------------
from client import ChainOfThoughtReader
from server_text import TextBasedInsightAggregationServer
from utils import call_gemini_thinking


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _check_clawbench():
    """Verify clawbench lib_agent (or package) is importable. Exits with instructions if not."""
    if _HAS_CLAWBENCH_LIB:
        return
    try:
        import importlib
        importlib.import_module("clawbench")
        return
    except ImportError:
        pass

    print(
        "\n" + "=" * 70 + "\n"
        "ERROR: clawbench package not importable.\n\n"
        "Install it from the cloned repo:\n"
        "  cd clawbench\n"
        "  pip install -e .\n"
        "  # or: uv pip install -e .\n\n"
        "NOTE: ClawBench also requires Docker or Podman to be running.\n"
        f"The lib_agent.py integration layer is at:\n"
        f"  {_CLAWBENCH_SCRIPTS / 'lib_agent.py'}\n"
        + "=" * 70 + "\n"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------

def _slugify_model(model_id: str) -> str:
    """Convert a model ID to a valid YAML key / clawbench model name."""
    slug = re.sub(r"[^a-zA-Z0-9._-]", "-", model_id)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def _load_task_cases(
    clawbench_dir: Path,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Scan test-cases/ for task directories, returning task metadata.

    If category == "lite", load only the 20-task curated subset from lite.json.
    Otherwise filter by metadata.metaclass or metadata.class if category is set.
    """
    # Search candidate locations in priority order
    _candidates = [
        clawbench_dir / "test-cases",
        clawbench_dir / "src" / "clawbench" / "data" / "test-cases",
    ]
    # Also check the installed package data via clawbench.paths (if installed)
    try:
        from clawbench import paths as _cb_paths
        _candidates.append(_cb_paths.test_cases_dir())
    except Exception:
        pass

    tasks_dir = next((p for p in _candidates if p.exists()), None)
    if tasks_dir is None:
        tried = ", ".join(str(p) for p in _candidates)
        print(f"Warning: test-cases dir not found. Tried: {tried}")
        return []

    # Lite subset
    if category == "lite":
        lite_path = tasks_dir / "lite.json"
        if lite_path.exists():
            with open(lite_path, "r", encoding="utf-8") as f:
                lite_data = json.load(f)
            lite_cases = set()
            for entry in lite_data if isinstance(lite_data, list) else lite_data.get("cases", []):
                if isinstance(entry, str):
                    lite_cases.add(entry)
                elif isinstance(entry, dict):
                    lite_cases.add(entry.get("id") or entry.get("case") or "")
        else:
            print("Warning: lite.json not found, falling back to all tasks")
            lite_cases = None
    else:
        lite_cases = None

    tasks = []
    for entry in sorted(tasks_dir.iterdir()):
        task_json = entry / "task.json"
        if not entry.is_dir() or not task_json.exists():
            continue

        # Lite filter
        if lite_cases is not None and entry.name not in lite_cases:
            continue

        try:
            with open(task_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            metadata = data.get("metadata") or {}
            metaclass = metadata.get("metaclass") or ""
            cls = metadata.get("class") or ""
            task_id = metadata.get("task_id") or entry.name
            task_name = metadata.get("description") or data.get("instruction", "")[:80]
            instruction = data.get("instruction") or ""

            # Category filter (matches metaclass or class)
            # "all" and "lite" are special: "all" = no filter, "lite" handled above
            if category and category not in ("lite", "all"):
                if category not in (metaclass, cls):
                    continue

            tasks.append({
                "task_id": str(task_id),
                "task_name": task_name,
                "case_name": entry.name,
                "task_dir": entry,
                "metaclass": metaclass,
                "class": cls,
                "instruction": instruction,
                "eval_schema": data.get("eval_schema") or {},
                "time_limit": data.get("time_limit") or 30,
            })
        except Exception as exc:
            print(f"  Warning: could not load {task_json}: {exc}")

    return tasks


# ---------------------------------------------------------------------------
# Model config (models.yaml)
# ---------------------------------------------------------------------------

def _write_models_yaml(
    model_slug: str,
    api_key: str,
    base_url: str,
    dest_path: Path,
    thinking_level: Optional[str] = None,
    api_type: str = "openai-completions",
    reasoning_enabled: Optional[bool] = None,
) -> None:
    """
    Write a models.yaml for clawbench with a single model entry.

    clawbench reads model config from ~/.config/claw-bench/models.yaml
    (or from the file pointed to by CLAWBENCH_MODELS env var).
    """
    entry: Dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
        "api_type": api_type,
    }
    if thinking_level:
        entry["thinking_level"] = thinking_level
    if reasoning_enabled is not None:
        entry["reasoning_enabled"] = reasoning_enabled

    config = {model_slug: entry}
    with open(dest_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _find_task_output_dir(
    claw_output_root: Path,
    case_name: str,
    model_slug: str,
) -> Optional[Path]:
    """
    Find the most recently written output directory for a given task run.

    clawbench writes to:
      <claw_output_root>/<model_slug>/<case_name>-<model_slug>-<timestamp>/
    """
    model_dir = claw_output_root / model_slug
    if not model_dir.exists():
        return None

    prefix = f"{case_name}-"
    candidates = [
        d for d in model_dir.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ]
    if not candidates:
        # Also try without model slug in name
        candidates = list(model_dir.iterdir())
        candidates = [d for d in candidates if d.is_dir()]

    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _extract_transcript_text(task_run_dir: Path) -> str:
    """
    Parse agent-messages.jsonl from a ClawBench run directory.

    agent-messages.jsonl contains LLM reasoning turns and tool calls:
      {"role": "assistant", "content": "..."}
      {"role": "tool", "tool_name": "...", "content": "..."}
    """
    messages_path = task_run_dir / "data" / "agent-messages.jsonl"
    if not messages_path.exists():
        return ""

    parts = []
    try:
        with open(messages_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                role = event.get("role") or event.get("type") or ""
                content = event.get("content") or event.get("text") or ""

                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            text = block.get("text") or block.get("content") or ""
                            if text:
                                parts.append(f"[{role}]: {text}")
                elif content and role in ("assistant", "user", "tool"):
                    parts.append(f"[{role}]: {content}")

    except Exception as exc:
        print(f"  Warning: failed to read agent-messages.jsonl: {exc}")

    return "\n\n".join(parts)


def _parse_interception(task_run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Parse interception.json to get task completion status.

    interception.json:
      {"intercepted": true/false, "stop_reason": "...", "request": {...}}
    """
    interception_path = task_run_dir / "data" / "interception.json"
    if not interception_path.exists():
        return None
    try:
        with open(interception_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _parse_run_meta(task_run_dir: Path) -> Optional[Dict[str, Any]]:
    """Parse run-meta.json for task metadata and duration."""
    meta_path = task_run_dir / "run-meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _print_clawbench_failure_diagnostics(
    task_run_dir: Path,
    exec_result: Dict[str, Any],
    max_chars: int = 4000,
) -> None:
    """Print high-signal diagnostics for no-intercept/agent-exited runs."""
    status = exec_result.get("status")
    if status == "success":
        return

    print(f"  Failure diagnostics: run_dir={task_run_dir}")
    interception = _parse_interception(task_run_dir)
    if interception:
        desc = interception.get("stop_description") or interception.get("stop_reason")
        if desc:
            print(f"  Stop reason: {interception.get('stop_reason')} — {desc}")

    for rel in ("agent.log", "gateway.log", "container.log", "stop-reason.txt"):
        path = task_run_dir / rel
        if not path.exists():
            path = task_run_dir / "data" / rel
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as exc:
            print(f"  Could not read {rel}: {exc}")
            continue
        if text:
            print(f"  ---- {rel} (last {max_chars} chars) ----")
            print(text[-max_chars:])

    messages = task_run_dir / "data" / "agent-messages.jsonl"
    actions = task_run_dir / "data" / "actions.jsonl"
    print(
        "  Artifacts: "
        f"agent_messages_exists={messages.exists()} "
        f"actions_exists={actions.exists()} "
        f"actions_bytes={actions.stat().st_size if actions.exists() else 0}"
    )


def _read_task_run_log(task_run_dir: Path, rel: str) -> str:
    path = task_run_dir / rel
    if not path.exists():
        path = task_run_dir / "data" / rel
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _is_model_not_found_failure(task_run_dir: Path) -> bool:
    haystack = "\n".join(
        _read_task_run_log(task_run_dir, rel)
        for rel in ("agent.log", "gateway.log", "container.log")
    ).lower()
    return "model_not_found" in haystack or '"code":404' in haystack and '"not found"' in haystack


def _preflight_gemini_openai_compat(api_key: str, base_url: str, model: str) -> None:
    """Fail fast if Gemini's OpenAI-compatible endpoint rejects a minimal call."""
    if not api_key:
        raise RuntimeError("Gemini API key is empty; set GEMINI_API_KEY or pass --api-key.")

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Return exactly: ok"}],
        "max_tokens": 8,
        "temperature": 0,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()
        print(f"  [gemini-preflight] OpenAI-compatible endpoint OK for {model}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gemini OpenAI-compatible preflight failed: HTTP {exc.code} at {url}. "
            f"Response body: {body[:1000]}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Gemini OpenAI-compatible preflight failed: {exc}") from exc


def _analyze_tool_calls(task_run_dir: Path) -> Dict[str, Any]:
    """
    Analyze tool call usage from agent-messages.jsonl.
    Also cross-references actions.jsonl for browser action counts.
    """
    calls: List[Dict[str, Any]] = []

    # From agent-messages.jsonl
    messages_path = task_run_dir / "data" / "agent-messages.jsonl"
    if messages_path.exists():
        try:
            with open(messages_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    role = event.get("role") or event.get("type") or ""
                    if role == "tool":
                        name = event.get("tool_name") or event.get("name") or "browser_action"
                        has_error = bool(event.get("error") or event.get("is_error"))
                        calls.append({
                            "name": name,
                            "status": "error" if has_error else "ok",
                            "error": event.get("error") if has_error else None,
                        })
        except Exception:
            pass

    # From actions.jsonl (browser-level actions)
    actions_path = task_run_dir / "data" / "actions.jsonl"
    if actions_path.exists() and not calls:
        try:
            with open(actions_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    action_type = event.get("type") or event.get("action") or "browser"
                    calls.append({"name": action_type, "status": "ok", "error": None})
        except Exception:
            pass

    tool_counter = collections.Counter(c["name"] for c in calls)
    return {
        "tool_names": sorted(tool_counter.keys()),
        "tool_name_counts": dict(sorted(tool_counter.items())),
        "total_calls": len(calls),
        "successful_calls": sum(1 for c in calls if c["status"] == "ok"),
        "error_calls": sum(1 for c in calls if c["status"] == "error"),
        "unknown_status_calls": 0,
        "calls": calls,
    }


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class OpenClawClawBenchPipeline:
    """
    Pipeline for running ClawBench web-automation tasks, extracting reasoning
    traces, and aggregating them into an insight library.

    Mirrors OpenClawClawEvalPipeline but drives the clawbench CLI
    (which orchestrates real browsers in Docker containers) instead of claw-eval.
    """

    def __init__(
        self,
        model_id: str,
        output_dir: str = "clawbench_output",
        category: str = "all",
        clawbench_dir: Optional[str] = None,
        test_cases_dir: Optional[str] = None,
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
        api_model: str = "gemini-3-pro-preview",
        base_url: Optional[str] = None,
        openclaw_api_key: Optional[str] = None,
        timeout: int = 1800,        # 30 min default (tasks are browser-automation)
        max_concurrent: int = 1,
        encyclopedia_path: Optional[str] = None,
        thinking_level: Optional[str] = None,
        gemini_api_type: str = "openai-completions",
    ):
        self.model_id = model_id
        self.output_dir = output_dir
        self.category = category
        self.use_api = use_api
        self.api_provider = api_provider
        self.api_key = api_key or (
            os.getenv("GEMINI_API_KEY") if api_provider == "gemini" else os.getenv("OPENROUTER_API_KEY")
        )
        self.api_model = api_model
        self.base_url = base_url
        self.openclaw_api_key = openclaw_api_key
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.encyclopedia_path = encyclopedia_path
        # Which transport OpenClaw uses for the agent when --api-provider gemini.
        # "openai-completions" -> /v1beta/openai (Gemini's OpenAI-compat shim)
        # "google-generative-ai" -> /v1beta (native Gemini API; bypasses the
        # OpenAI-compat layer that emits body-less HTTP 400s after tool rounds).
        self.gemini_api_type = gemini_api_type
        if not self.encyclopedia_path:
            default_encyclopedia = Path(self.output_dir) / "encyclopedia.json"
            if default_encyclopedia.exists():
                self.encyclopedia_path = str(default_encyclopedia)
                print(f"Auto-loaded encyclopedia: {self.encyclopedia_path}")
        self.thinking_level = thinking_level

        if self.api_provider == "gemini" and self.api_key:
            os.environ["GEMINI_API_KEY"] = self.api_key

        self.clawbench_dir = Path(clawbench_dir) if clawbench_dir else _CLAWBENCH_DIR
        self.test_cases_dir = test_cases_dir  # explicit override for task discovery
        self.model_slug = _slugify_model(model_id)

        os.makedirs(self.output_dir, exist_ok=True)

        # clawbench writes its own output here; we'll scan it after each task
        self._claw_output_root = Path(self.output_dir) / "claw-output"
        self._claw_output_root.mkdir(parents=True, exist_ok=True)

        self._client: Optional[ChainOfThoughtReader] = None
        self._metrics_cache_by_output_dir: Dict[str, Dict[str, Any]] = {}
        self._claw_cmd: Optional[List[str]] = None
        self._clawbench_image_refreshed = False
        self._agent_api_preflight_done = False

    # ------------------------------------------------------------------
    # clawbench CLI
    # ------------------------------------------------------------------

    def _resolve_api_config(self) -> tuple:
        """Return (api_key, base_url, api_type) for the configured provider."""
        if self.api_provider == "gemini":
            key = self.api_key or os.getenv("GEMINI_API_KEY") or ""
            if self.gemini_api_type == "google-generative-ai":
                # Native Gemini API. OpenClaw 2026.4.29+ supports this as a
                # first-class provider (see MODEL_APIS in the upstream
                # types.models.d.ts). The native path avoids the
                # OpenAI-compat shim that has been observed to emit
                # body-less HTTP 400s after tool-call rounds.
                #
                # If a stable preview model id is rejected here as
                # "unknown model" by OpenClaw's bundled Google plugin, fall
                # back to --gemini-api-type openai-completions for that run.
                url = self.base_url or "https://generativelanguage.googleapis.com/v1beta"
                api_type = "google-generative-ai"
            else:
                # Use Gemini's documented OpenAI-compatible endpoint for OpenClaw.
                # The native google-generative-ai provider can lag new preview model
                # ids in older OpenClaw builds; OpenAI compatibility accepts model
                # codes like gemini-3-flash-preview directly.
                #
                # NOTE: do NOT include a trailing slash here. OpenClaw appends
                # "/chat/completions" itself, and a trailing slash on the base
                # produces a double-slash URL ("…/openai//chat/completions") that
                # Google's gateway has been observed to reject with HTTP 400.
                # The pinchbench variant has the same convention.
                url = self.base_url or "https://generativelanguage.googleapis.com/v1beta/openai"
                api_type = "openai-completions"
        else:
            key = self.api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            url = self.base_url or "https://openrouter.ai/api/v1"
            api_type = "openai-completions"
        return key, url, api_type

    def _resolve_agent_model_id(self) -> str:
        """Use the provider-native model id inside the ClawBench container."""
        if self.api_provider == "gemini":
            return self.api_model or self.model_id.split("/", 1)[-1]
        return self.model_id

    def _run_single_task(
        self,
        task: Dict[str, Any],
        models_yaml: Optional[Path] = None,  # kept for API compat, unused when lib available
    ) -> Dict[str, Any]:
        """
        Run a ClawBench task via the lib_agent programmatic API.

        Falls back to subprocess when Docker is not available or lib not loaded.
        Returns a dict with status, task_run_dir, execution_time.
        """
        case_name = task["case_name"]
        task_dir = task["task_dir"]

        api_key, base_url, api_type = self._resolve_api_config()
        agent_model_id = self._resolve_agent_model_id()
        agent_thinking_level = self.thinking_level
        reasoning_enabled = True
        if self.api_provider == "gemini" and api_type == "openai-completions":
            # Let Gemini use its provider default reasoning configuration for
            # browser-agent calls. Passing explicit reasoning controls through
            # OpenClaw's OpenAI-compatible path has produced provider-side 400s.
            agent_thinking_level = None
            reasoning_enabled = None
            if not self._agent_api_preflight_done:
                print(f"  [gemini-preflight] Checking {agent_model_id} via {base_url}")
                _preflight_gemini_openai_compat(api_key, base_url, agent_model_id)
                self._agent_api_preflight_done = True
        force_build = False
        if self.api_provider == "gemini" and not self._clawbench_image_refreshed:
            force_build = True
            self._clawbench_image_refreshed = True

        if _HAS_CLAWBENCH_LIB:
            print(
                f"  [clawbench] Running via lib_agent: {case_name} | "
                f"model={agent_model_id} api_type={api_type} "
                f"agent_thinking={agent_thinking_level} reasoning={reasoning_enabled}"
            )
            result = _execute_clawbench_task(
                task_dir=task_dir,
                model_id=agent_model_id,
                api_key=api_key,
                base_url=base_url,
                api_type=api_type,
                output_root=self._claw_output_root,
                no_build=False,
                force_build=force_build,
                thinking_level=agent_thinking_level,
                reasoning_enabled=reasoning_enabled,
                timeout_seconds=self.timeout,
            )
            task_run_dir = result.get("output_dir")
            return {
                "status": result["status"],
                "task_run_dir": task_run_dir,
                "execution_time": result.get("execution_time", 0.0),
                "intercepted": result.get("intercepted", False),
                "stop_reason": result.get("stop_reason", ""),
                "error": result.get("error"),
            }

        # ---- Fallback: subprocess via clawbench CLI -----------------------
        import subprocess as _sp

        # Write models.yaml for CLI mode
        if models_yaml is None or not models_yaml.exists():
            models_yaml = Path(self.output_dir) / "models.yaml"
            _write_models_yaml(
                model_slug=agent_model_id,
                api_key=api_key,
                base_url=base_url,
                dest_path=models_yaml,
                thinking_level=agent_thinking_level,
                api_type=api_type,
                reasoning_enabled=reasoning_enabled,
            )

        clawbench_cmd = [sys.executable, "-m", "clawbench", "run", case_name, agent_model_id]
        env = os.environ.copy()
        env["CLAWBENCH_MODELS"] = str(models_yaml)
        env["CLAWBENCH_OUTPUT_DIR"] = str(self._claw_output_root)

        print(f"  [clawbench] Running via subprocess: {case_name}")
        t0 = time.time()
        try:
            proc = _sp.run(
                clawbench_cmd, capture_output=True, text=True,
                timeout=self.timeout, cwd=str(self.clawbench_dir), env=env,
            )
        except _sp.TimeoutExpired:
            return {"status": "timeout", "task_run_dir": None, "execution_time": self.timeout}
        except Exception as exc:
            return {"status": "error", "error": str(exc), "task_run_dir": None, "execution_time": None}

        elapsed = time.time() - t0
        status = "success" if proc.returncode == 0 else "failed"
        if proc.stderr:
            print(f"  stderr: {proc.stderr[:300]}")

        task_run_dir = _find_task_output_dir(
            self._claw_output_root,
            case_name,
            re.sub(r"[/:]+", "--", agent_model_id),
        )
        return {
            "status": status,
            "task_run_dir": task_run_dir,
            "execution_time": elapsed,
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _metrics_output_key(self) -> str:
        return str(Path(self.output_dir).resolve())

    def _metrics_log_path(self) -> Path:
        return Path(self.output_dir) / "metrics_log.json"

    def _build_metrics_summary(
        self,
        task_metrics: List[Dict[str, Any]],
        library_output_tokens: Optional[int],
    ) -> Dict[str, Any]:
        graded = [t for t in task_metrics if t.get("grade", {}).get("intercepted") is not None]
        passed_count = sum(1 for t in graded if t["grade"].get("intercepted") is True)

        tool_name_counts: Dict[str, int] = {}
        total_tool_calls = 0
        successful_tool_calls = 0
        error_tool_calls = 0
        for task in task_metrics:
            tools = task.get("tools", {})
            total_tool_calls += int(tools.get("total_calls", 0) or 0)
            successful_tool_calls += int(tools.get("successful_calls", 0) or 0)
            error_tool_calls += int(tools.get("error_calls", 0) or 0)
            for name, count in (tools.get("tool_name_counts", {}) or {}).items():
                tool_name_counts[name] = tool_name_counts.get(name, 0) + int(count)

        total_time = sum(float(t.get("execution_time_seconds") or 0.0) for t in task_metrics)
        total_agent_tokens = sum(int((t.get("output_tokens") or {}).get("agent", 0) or 0) for t in task_metrics)
        total_extraction_tokens = sum(int((t.get("output_tokens") or {}).get("extraction", 0) or 0) for t in task_metrics)

        return {
            "tasks_total": len(task_metrics),
            "tasks_graded": len(graded),
            "tasks_intercepted": passed_count,
            "interception_rate_pct": (passed_count / len(graded) * 100.0) if graded else None,
            "output_tokens_agent_total": total_agent_tokens,
            "output_tokens_extraction_total": total_extraction_tokens,
            "output_tokens_total": total_agent_tokens + total_extraction_tokens,
            "tool_names": sorted(tool_name_counts.keys()),
            "tool_name_counts": dict(sorted(tool_name_counts.items())),
            "total_tool_calls": total_tool_calls,
            "successful_tool_calls": successful_tool_calls,
            "error_tool_calls": error_tool_calls,
            "execution_time_total_seconds": total_time,
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
            "model_slug": self.model_slug,
            "category": self.category,
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
            iteration_summaries.append({
                "iteration": log_path.parent.name,
                "metrics_log": str(log_path),
                "summary": payload.get("summary", {}),
            })

        if not iteration_summaries:
            return

        tasks_total = sum(int(s["summary"].get("tasks_total", 0) or 0) for s in iteration_summaries)
        intercepted = sum(int(s["summary"].get("tasks_intercepted", 0) or 0) for s in iteration_summaries)

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
                "tasks_total": tasks_total,
                "tasks_intercepted": intercepted,
                "output_tokens_total": sum(int(s["summary"].get("output_tokens_total", 0) or 0) for s in iteration_summaries),
                "tool_names": sorted(overall_tool_counts.keys()),
                "tool_name_counts": dict(sorted(overall_tool_counts.items())),
                "total_tool_calls": sum(int(s["summary"].get("total_tool_calls", 0) or 0) for s in iteration_summaries),
                "execution_time_total_seconds": sum(float(s["summary"].get("execution_time_total_seconds", 0.0) or 0.0) for s in iteration_summaries),
                "library_output_tokens_total": sum(int(s["summary"].get("library_output_tokens", 0) or 0) for s in iteration_summaries),
                "pipeline_elapsed_seconds": elapsed_seconds,
            },
        }
        (root / "metrics_log_overall.json").write_text(
            json.dumps(overall_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Insight extraction (same as claweval / pinchbench)
    # ------------------------------------------------------------------

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

    def _call_for_extraction(self, prompt: str, max_new_tokens: int) -> tuple:
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
        """
        client = self._ensure_client()

        print("  Step 2: Reflecting on agent solution...")
        reflection_prompt = client._get_reflection_prompt(task_prompt, agent_response)
        reflection, _ = self._call_for_extraction(reflection_prompt, max_new_tokens=4096)
        print(f"  Reflection length: {len(reflection)} chars")

        print("  Step 3: Extracting reasoning traces...")
        behavior_prompt = client._get_behavior_prompt(task_prompt, agent_response, reflection)
        extraction_response, token_info = self._call_for_extraction(
            behavior_prompt, max_new_tokens=8192
        )

        insights = {}
        try:
            json_code_block = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", extraction_response, re.DOTALL
            )
            if json_code_block:
                json_str = json_code_block.group(1)
            else:
                start = extraction_response.find("{")
                json_str = None
                if start != -1:
                    brace_count = 0
                    in_string = False
                    escape_next = False
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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_tasks_and_extract(self) -> List[Dict[str, Any]]:
        """
        Step 1 + Steps 2/3: For each ClawBench task (filtered by category),
        run the clawbench agent (browser-in-container) and extract reasoning
        traces from the agent-messages transcript.

        Saves each task's insights as problem_XXXX.json in output_dir.
        Returns a list of per-task result dicts.
        """
        _check_clawbench()

        search_dir = Path(self.test_cases_dir) if getattr(self, "test_cases_dir", None) else self.clawbench_dir
        tasks = _load_task_cases(search_dir, category=self.category)
        if not tasks:
            print(f"No tasks found in {search_dir} with category='{self.category}'")
            return []

        print(f"\nLoaded {len(tasks)} tasks (category='{self.category}')")
        print("=" * 80)

        if self.encyclopedia_path and Path(self.encyclopedia_path).exists():
            print(f"\nEncyclopedia loaded: {self.encyclopedia_path}")
        else:
            print("\nNo encyclopedia — running without prior insights")

        results = []
        task_metrics: List[Dict[str, Any]] = []
        task_counter = 0

        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] Task: {task['task_id']} ({task['case_name']}) — {task['task_name'][:60]}")
            print("-" * 60)

            # -----------------------------------------------------------------
            # Step 1: Execute via clawbench run
            # -----------------------------------------------------------------
            try:
                exec_result = self._run_single_task(task)
            except Exception as exc:
                print(f"  Error executing task: {exc}")
                task_metrics.append({
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
                    "case_name": task["case_name"],
                    "metaclass": task["metaclass"],
                    "execution_status": "error",
                    "execution_time_seconds": None,
                    "grade": {},
                    "output_tokens": {"agent": 0, "extraction": 0, "total": 0},
                    "tools": {"tool_name_counts": {}, "total_calls": 0, "successful_calls": 0, "error_calls": 0},
                    "insights_extracted": 0,
                    "error": str(exc),
                })
                self._write_metrics_log(task_metrics)
                continue

            status = exec_result.get("status", "error")
            task_run_dir: Optional[Path] = exec_result.get("task_run_dir")
            print(f"  Execution status: {status}")

            if status not in ("success",) and not task_run_dir:
                stderr = (exec_result.get("stderr") or "").strip()
                if stderr:
                    print(f"  Stderr (first 400): {stderr[:400]}")

            # -----------------------------------------------------------------
            # Parse run metadata and interception (score)
            # -----------------------------------------------------------------
            run_meta = _parse_run_meta(task_run_dir) if task_run_dir else None
            interception = _parse_interception(task_run_dir) if task_run_dir else None
            intercepted = interception.get("intercepted") if interception else None
            execution_time = exec_result.get("execution_time")
            if run_meta and run_meta.get("duration_seconds"):
                execution_time = float(run_meta["duration_seconds"])
            if execution_time is not None:
                execution_time = float(execution_time)

            print(f"  Intercepted: {intercepted} | Duration: {execution_time:.1f}s" if execution_time else f"  Intercepted: {intercepted}")

            if task_run_dir and status != "success":
                _print_clawbench_failure_diagnostics(task_run_dir, exec_result)
                if _is_model_not_found_failure(task_run_dir):
                    raise RuntimeError(
                        "OpenClaw agent model was not found by the provider. "
                        f"Configured model='{self._resolve_agent_model_id()}', "
                        f"provider='{self.api_provider}'. Use a model returned by "
                        "the Gemini API for this key, or switch to OpenRouter with "
                        "an OpenRouter model id."
                    )

            tool_stats = _analyze_tool_calls(task_run_dir) if task_run_dir else {
                "tool_names": [], "tool_name_counts": {}, "total_calls": 0,
                "successful_calls": 0, "error_calls": 0, "unknown_status_calls": 0, "calls": [],
            }
            task_metric = {
                "task_id": task["task_id"],
                "task_name": task["task_name"],
                "case_name": task["case_name"],
                "metaclass": task["metaclass"],
                "class": task["class"],
                "execution_status": status,
                "execution_time_seconds": execution_time,
                "task_run_dir": str(task_run_dir) if task_run_dir else None,
                "grade": interception or {},
                "output_tokens": {"agent": 0, "extraction": 0, "total": 0},
                "tools": tool_stats,
                "insights_extracted": 0,
            }

            if not task_run_dir:
                print("  Skipping insight extraction — no output directory")
                task_metrics.append(task_metric)
                self._write_metrics_log(task_metrics)
                continue

            # -----------------------------------------------------------------
            # Extract readable solution text from agent-messages transcript
            # -----------------------------------------------------------------
            agent_response = _extract_transcript_text(task_run_dir)
            if not agent_response.strip():
                print("  Skipping insight extraction — no extractable text in transcript")
                task_metrics.append(task_metric)
                self._write_metrics_log(task_metrics)
                continue

            # Prepend encyclopedia as context if available
            task_prompt = task["instruction"]
            if self.encyclopedia_path and Path(self.encyclopedia_path).exists():
                try:
                    with open(self.encyclopedia_path, "r", encoding="utf-8") as f:
                        enc = json.load(f)
                    enc_text = json.dumps(enc, indent=2)[:8000]
                    task_prompt = (
                        f"## Insight Library (from prior experience)\n{enc_text}\n\n"
                        f"## Task\n{task_prompt}"
                    )
                except Exception:
                    pass

            # -----------------------------------------------------------------
            # Steps 2 & 3: Reflection + insight extraction
            # -----------------------------------------------------------------
            try:
                extraction = self._apply_reflection_and_extraction(
                    task_prompt=task_prompt,
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
                insight_book = {}

            extraction_output_tokens = int(extraction.get("output_tokens", 0) or 0)
            task_metric["output_tokens"] = {
                "agent": 0,
                "extraction": extraction_output_tokens,
                "total": extraction_output_tokens,
            }
            task_metric["insights_extracted"] = len(insight_book)

            # -----------------------------------------------------------------
            # Save as problem_XXXX.json (matches server_text expected format)
            # -----------------------------------------------------------------
            task_counter += 1
            output_file = os.path.join(self.output_dir, f"problem_{task_counter:04d}.json")

            save_data = {
                "task_id": task["task_id"],
                "task_name": task["task_name"],
                "case_name": task["case_name"],
                "metaclass": task["metaclass"],
                "class": task["class"],
                "task_prompt": task["instruction"],
                "execution_status": status,
                "grade": interception or {},
                "output_tokens": extraction.get("output_tokens", 0),
                "insight_book": insight_book,
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            print(f"  Saved {len(insight_book)} insights → {output_file}")
            results.append({
                "task_id": task["task_id"],
                "case_name": task["case_name"],
                "status": status,
                "intercepted": intercepted,
                "insights_extracted": len(insight_book),
                "output_file": output_file,
            })
            task_metrics.append(task_metric)
            self._write_metrics_log(task_metrics)

        print("\n" + "=" * 80)
        total_insights = sum(int(t.get("insights_extracted", 0) or 0) for t in task_metrics)
        graded = [t for t in task_metrics if t.get("grade", {}).get("intercepted") is not None]
        print(f"Tasks processed: {len(task_metrics)}/{len(tasks)}")
        if graded:
            intercepted_count = sum(1 for t in graded if t["grade"].get("intercepted") is True)
            print(f"Interception rate: {intercepted_count}/{len(graded)} ({intercepted_count/len(graded)*100:.1f}%)")
        print(f"Total insights extracted: {total_insights}")
        self._write_metrics_log(task_metrics)
        return results

    def aggregate_insights(self) -> Optional[str]:
        """
        Aggregate all problem_XXXX.json files using TextBasedInsightAggregationServer
        and save the resulting encyclopedia.json.
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

        encyclopedia_path = os.path.join(self.output_dir, "encyclopedia.json")
        enc_dict = server._try_parse_json(server.encyclopedia)
        if enc_dict is None:
            enc_dict = server._try_parse_json(server._extract_json_only(server.encyclopedia))

        if enc_dict is None:
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

            if iterations > 1:
                iter_dir = os.path.join(self.output_dir, f"iter_{iteration:02d}")
                os.makedirs(iter_dir, exist_ok=True)
                orig_output_dir = self.output_dir
                self.output_dir = iter_dir
                self._claw_output_root = Path(iter_dir) / "claw-output"
                self._claw_output_root.mkdir(parents=True, exist_ok=True)
            else:
                orig_output_dir = None

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
                self._claw_output_root = Path(orig_output_dir) / "claw-output"

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
        description="OpenClaw ClawBench Pipeline — extract reasoning traces from ClawBench web-automation tasks"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help="Model identifier, e.g., google/gemini-3-pro-preview or anthropic/claude-sonnet-4",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="clawbench_output",
        help="Directory to save insights and encyclopedia (default: clawbench_output)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        help=(
            'Filter tasks by metaclass: "all" (default, 153 tasks), "lite" (20-task curated subset), '
            '"daily-life", "travel", "education", "shopping", "finance", etc.'
        ),
    )
    parser.add_argument(
        "--clawbench-dir",
        type=str,
        default=None,
        help="Path to ClawBench repo root (default: ./clawbench)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of pipeline iterations (default: 1). Iteration 2+ uses the encyclopedia from iteration 1.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Max concurrent clawbench runs (default: 1; each run needs a Docker container)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout per task in seconds (default: 1800 = 30 min; ClawBench tasks are browser-automation)",
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
        "--encyclopedia",
        type=str,
        default=None,
        help="Path to existing encyclopedia.json to inject as context for the first iteration",
    )
    parser.add_argument(
        "--start-from-step2",
        action="store_true",
        help="Skip task execution and start from aggregation",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use an API provider for reflection/extraction steps (and as agent endpoint if model is gemini/*)",
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
        "--base-url",
        type=str,
        default=None,
        help="Custom OpenAI-compatible API base URL for the ClawBench agent",
    )
    parser.add_argument(
        "--gemini-api-type",
        type=str,
        default="openai-completions",
        choices=["openai-completions", "google-generative-ai"],
        help=(
            "When --api-provider is gemini, choose how OpenClaw talks to the model. "
            "'openai-completions' (default) uses Gemini's OpenAI-compatible shim at "
            "/v1beta/openai. 'google-generative-ai' uses the native Gemini API at "
            "/v1beta directly — bypasses the OpenAI-compat layer that has been "
            "observed to return body-less HTTP 400s after tool-call rounds."
        ),
    )
    parser.add_argument(
        "--openclaw-api-key",
        type=str,
        default=None,
        help="API key for custom OpenClaw agent endpoint (default: $OPENROUTER_API_KEY or $OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--test-cases-dir",
        type=str,
        default=None,
        help="Explicit path to ClawBench test-cases/ directory (overrides auto-discovery)",
    )

    args = parser.parse_args()

    if not args.model and not args.start_from_step2:
        parser.error("--model is required unless using --start-from-step2")

    model_id = args.model or "google/gemini-3-pro-preview"

    # Resolve openclaw agent API key: CLI > env vars
    openclaw_api_key = args.openclaw_api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    pipeline = OpenClawClawBenchPipeline(
        model_id=model_id,
        output_dir=args.output_dir,
        category=args.category,
        clawbench_dir=args.clawbench_dir,
        test_cases_dir=args.test_cases_dir,
        use_api=args.use_api,
        api_key=args.api_key,
        api_provider=args.api_provider,
        api_model=args.api_model,
        base_url=args.base_url,
        openclaw_api_key=openclaw_api_key,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent,
        encyclopedia_path=args.encyclopedia,
        thinking_level=args.thinking_level if args.use_api else None,
        gemini_api_type=args.gemini_api_type,
    )

    pipeline.run_pipeline(
        iterations=args.iterations,
        start_from_step2=args.start_from_step2,
    )


if __name__ == "__main__":
    main()
