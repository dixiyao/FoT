"""
Programmatic Python integration layer for the ClawBench benchmark.

Mirrors the role of pinchbench/scripts/lib_agent.py so that
task_openclaw_clawbench.py can drive ClawBench tasks directly from Python
without re-implementing the Docker orchestration logic.

Primary entry point
-------------------
    result = execute_clawbench_task(
        task_dir=Path("clawbench/test-cases/001-daily-life-food-uber-eats"),
        model_id="anthropic/claude-opus-4-6",
        api_key="sk-or-...",
        base_url="https://openrouter.ai/api/v1",
        output_root=Path("claw-output"),
        no_build=False,          # set True after first run to skip image build
        thinking_level=None,     # "low"/"medium"/"high" for supported models
        timeout_seconds=1800,
    )
    # result = {
    #   "status": "success" | "error" | "timeout",
    #   "task_id": str,
    #   "task_name": str,
    #   "output_dir": Path | None,
    #   "transcript": str,       # human-readable agent-messages text
    #   "intercepted": bool,
    #   "stop_reason": str,
    #   "interception": dict,    # raw interception.json content
    #   "execution_time": float,
    #   "error": str | None,
    # }
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure the clawbench package is importable from the cloned repo layout.
# Package lives in src/clawbench when installed in editable mode.
# ---------------------------------------------------------------------------
_CLAWBENCH_SRC = Path(__file__).resolve().parent.parent / "src"
if _CLAWBENCH_SRC.exists() and str(_CLAWBENCH_SRC) not in sys.path:
    sys.path.insert(0, str(_CLAWBENCH_SRC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_agent_messages(output_dir: Path) -> str:
    """Parse agent-messages.jsonl and return a human-readable transcript."""
    messages_path = output_dir / "data" / "agent-messages.jsonl"
    if not messages_path.exists():
        return ""
    parts: List[str] = []
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
                            if text and role in ("assistant", "user", "tool"):
                                parts.append(f"[{role}]: {text}")
                elif isinstance(content, str) and role in ("assistant", "user", "tool"):
                    parts.append(f"[{role}]: {content}")
    except Exception as exc:
        print(f"  [lib_agent] Warning: could not read agent-messages.jsonl: {exc}")
    return "\n\n".join(parts)


def _read_interception(output_dir: Path) -> Dict[str, Any]:
    """Read interception.json from the task output directory."""
    interception_path = output_dir / "data" / "interception.json"
    if not interception_path.exists():
        return {"intercepted": False, "stop_reason": "no_interception_file"}
    try:
        with open(interception_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"intercepted": False, "stop_reason": "parse_error"}


def _slugify(model_id: str) -> str:
    """Convert a model ID like 'anthropic/claude-opus-4-6' to a path-safe slug."""
    slug = re.sub(r"[/:]+", "--", model_id)
    slug = re.sub(r"[^a-zA-Z0-9._\-]", "-", slug)
    return slug.strip("-")


def _capture_container_diagnostics(container: str, output_dir: Path) -> Dict[str, str]:
    """Persist container/agent logs before docker_copy removes bulky in-data logs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}
    engine = os.environ.get("CONTAINER_ENGINE") or "podman"

    try:
        logs = subprocess.run(
            [engine, "logs", container],
            capture_output=True,
            text=True,
            timeout=20,
        )
        container_log = (logs.stdout or "") + (logs.stderr or "")
        if container_log:
            path = output_dir / "container.log"
            path.write_text(container_log, encoding="utf-8", errors="replace")
            saved["container_log"] = str(path)
    except Exception as exc:
        (output_dir / "container-log-error.txt").write_text(str(exc), encoding="utf-8")

    for name in ("agent.log", "gateway.log", ".stop-reason"):
        dest_name = "stop-reason.txt" if name == ".stop-reason" else name
        dest = output_dir / dest_name
        try:
            cp = subprocess.run(
                [engine, "cp", f"{container}:/data/{name}", str(dest)],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if cp.returncode == 0 and dest.exists():
                saved[dest_name] = str(dest)
        except Exception:
            pass

    config_dest = output_dir / "openclaw.json"
    try:
        cp = subprocess.run(
            [engine, "cp", f"{container}:/root/.openclaw/openclaw.json", str(config_dest)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if cp.returncode == 0 and config_dest.exists():
            saved["openclaw.json"] = str(config_dest)
    except Exception:
        pass

    return saved


def write_models_yaml(
    model_slug: str,
    api_key: str,
    base_url: str,
    api_type: str = "openai-completions",
    thinking_level: Optional[str] = None,
    reasoning_enabled: Optional[bool] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    dest_path: Optional[Path] = None,
) -> Path:
    """
    Write a models.yaml file containing a single model entry and return its path.

    ClawBench reads model credentials from a YAML file. The key is the model
    name that will be passed to `clawbench run <case> <model_name>`.

    Parameters
    ----------
    model_slug   : The model name key used in the YAML (and passed to docker).
    api_key      : API key for the model provider.
    base_url     : API base URL.
    api_type     : One of "openai-completions" (default) or "anthropic".
    thinking_level : Optional thinking level ("low"/"medium"/"high").
    temperature  : Optional temperature override.
    max_tokens   : Optional max output tokens.
    dest_path    : Where to write models.yaml. Defaults to the user config
                   location that ClawBench reads by default.
    """
    import yaml
    from clawbench import paths as _paths

    entry: Dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
        "api_type": api_type,
    }
    if thinking_level:
        entry["thinking_level"] = thinking_level
    if reasoning_enabled is not None:
        entry["reasoning_enabled"] = reasoning_enabled
    if temperature is not None:
        entry["temperature"] = temperature
    if max_tokens is not None:
        entry["max_tokens"] = max_tokens

    config = {model_slug: entry}

    if dest_path is None:
        dest_path = _paths.user_models_yaml()

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"  [lib_agent] Wrote models.yaml → {dest_path}")
    return dest_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def execute_clawbench_task(
    task_dir: Path,
    model_id: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    output_root: Optional[Path] = None,
    no_build: bool = False,
    force_build: bool = False,
    thinking_level: Optional[str] = None,
    reasoning_enabled: Optional[bool] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout_seconds: int = 1800,
    api_type: str = "openai-completions",
) -> Dict[str, Any]:
    """
    Run a single ClawBench task programmatically using the clawbench Python API.

    This function:
    1. Writes a models.yaml with the given model credentials.
    2. Builds the Docker image if needed (skip with no_build=True).
    3. Creates a disposable email account via PurelyMail.
    4. Prepares personal-info files and runs the container.
    5. Copies output data, parses interception.json, and returns results.

    Parameters
    ----------
    task_dir        : Path to the test-case directory (contains task.json).
    model_id        : Full model ID, e.g. "anthropic/claude-opus-4-6".
                      Also used as the YAML key in models.yaml.
    api_key         : API key for the model provider.
    base_url        : API base URL (default: OpenRouter).
    output_root     : Root directory for run output. Defaults to ./claw-output.
    no_build        : Skip Docker image build (assume image already exists).
    thinking_level  : Optional "low"/"medium"/"high" for supported models.
    temperature     : Optional temperature override.
    max_tokens      : Optional max output tokens.
    timeout_seconds : Wall-clock timeout for waiting on the container.
    api_type        : API type string for clawbench model config.

    Returns
    -------
    dict with keys: status, task_id, task_name, output_dir, transcript,
                    intercepted, stop_reason, interception, execution_time, error.
    """
    start_time = time.time()
    task_dir = Path(task_dir).resolve()
    task_json = task_dir / "task.json"

    if not task_json.exists():
        return {
            "status": "error",
            "task_id": task_dir.name,
            "task_name": task_dir.name,
            "output_dir": None,
            "transcript": "",
            "intercepted": False,
            "stop_reason": "task_json_not_found",
            "interception": {},
            "execution_time": 0.0,
            "error": f"task.json not found in {task_dir}",
        }

    try:
        # Import from the clawbench package
        from clawbench.run import (
            create_email,
            delete_email,
            docker_build,
            docker_copy,
            docker_rm,
            docker_run,
            docker_wait,
            ensure_interception,
            build_instruction,
            copy_extra_info,
            prepare_personal_info,
            _load_runtime_env,
            _pick_free_port,
            _fix_data_ownership,
            _image_exists,
            ENGINE,
        )
        from clawbench import paths as _paths

    except ImportError as e:
        return {
            "status": "error",
            "task_id": task_dir.name,
            "task_name": task_dir.name,
            "output_dir": None,
            "transcript": "",
            "intercepted": False,
            "stop_reason": "import_error",
            "interception": {},
            "execution_time": 0.0,
            "error": f"Cannot import clawbench: {e}. Run: pip install -e clawbench/",
        }

    # ---- load task -------------------------------------------------------- #
    with open(task_json, "r", encoding="utf-8") as f:
        task = json.load(f)

    case_name = task_dir.name
    task_id = str(task.get("metadata", {}).get("task_id", case_name))
    task_name = task.get("metadata", {}).get("description") or task.get("instruction", "")[:80]
    time_limit_s = int(task.get("time_limit", 30)) * 60

    # ---- write models.yaml ----------------------------------------------- #
    model_key = model_id
    model_slug = _slugify(model_id)
    write_models_yaml(
        model_slug=model_key,
        api_key=api_key,
        base_url=base_url,
        api_type=api_type,
        thinking_level=thinking_level,
        reasoning_enabled=reasoning_enabled,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # ---- output directory ------------------------------------------------ #
    if output_root is None:
        output_root = Path.cwd() / "claw-output"
    output_root = Path(output_root)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir = output_root / model_slug / f"{case_name}-{model_slug}-{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- build image ------------------------------------------------------ #
    # Skip build entirely if the image already exists.
    if force_build:
        print(f"  [lib_agent] Rebuilding container image 'clawbench' to pick up local changes...")
        try:
            docker_build()
        except SystemExit as e:
            return {
                "status": "error",
                "task_id": task_id,
                "task_name": task_name,
                "output_dir": output_dir,
                "transcript": "",
                "intercepted": False,
                "stop_reason": "image_build_failed",
                "interception": {},
                "execution_time": time.time() - start_time,
                "error": f"Container image rebuild failed (exit {e}).",
            }
    elif _image_exists():
        print(f"  [lib_agent] Container image 'clawbench' already exists — skipping build.")
    elif no_build:
        print(f"  [lib_agent] Warning: image 'clawbench' not found and --no-build set. Run:")
        print(f"    sudo {ENGINE} build -t clawbench clawbench/")
    else:
        print(f"  [lib_agent] Building container image (this may take a few minutes)...")
        print(f"  [lib_agent] If this fails with a permission error, build it manually with:")
        print(f"    sudo {ENGINE} build -t clawbench clawbench/")
        try:
            docker_build()
        except SystemExit as e:
            return {
                "status": "error",
                "task_id": task_id,
                "task_name": task_name,
                "output_dir": output_dir,
                "transcript": "",
                "intercepted": False,
                "stop_reason": "image_build_failed",
                "interception": {},
                "execution_time": time.time() - start_time,
                "error": (
                    f"Container image build failed (exit {e}). "
                    f"Podman rootless mode may lack permission to build images. "
                    f"Fix with: sudo {ENGINE} build -t clawbench clawbench/ "
                    f"or: sudo {ENGINE} system migrate"
                ),
            }

    # ---- infrastructure env (PurelyMail) --------------------------------- #
    env = _load_runtime_env()
    pm_key = env.get("PURELY_MAIL_API_KEY", "")
    pm_domain = env.get("PURELY_MAIL_DOMAIN", "clawbench.cc")

    email: Optional[str] = None
    personal_info_tmp: Optional[Path] = None

    import time as _time
    container = f"clawbench-{case_name}-{model_slug}-{int(_time.time())}"

    try:
        # ---- disposable email -------------------------------------------- #
        print(f"  [lib_agent] Creating disposable email...")
        email, email_pw = create_email(pm_key, pm_domain)

        # ---- personal info ------------------------------------------------ #
        personal_info_tmp = prepare_personal_info(
            _paths.shared_dir(), email, email_pw, output_dir
        )
        copy_extra_info(task, task_dir, personal_info_tmp)

        # ---- eval schema -------------------------------------------------- #
        schema_path = output_dir / "eval-schema.json"
        schema_path.write_text(json.dumps(task.get("eval_schema", {}), indent=2))

        # ---- instruction -------------------------------------------------- #
        instruction = build_instruction(task)
        print(f"  [lib_agent] Instruction: {instruction[:200]}")

        # ---- start container ---------------------------------------------- #
        host_port = _pick_free_port(6080)
        model_cfg = {
            "model": model_key,
            "base_url": base_url,
            "api_type": api_type,
            "api_key": api_key,
            "api_keys": [api_key],
        }
        if thinking_level:
            model_cfg["thinking_level"] = thinking_level
        if reasoning_enabled is not None:
            model_cfg["reasoning_enabled"] = reasoning_enabled
            model_cfg["reasoning_enabled_env"] = "true" if reasoning_enabled else "false"
        if temperature is not None:
            model_cfg["temperature"] = temperature
        if max_tokens is not None:
            model_cfg["max_tokens"] = max_tokens

        print(f"  [lib_agent] Starting container: {container}")
        docker_run(
            container,
            instruction,
            schema_path,
            personal_info_tmp,
            model_cfg,
            time_limit_s=min(time_limit_s, timeout_seconds),
            host_port=host_port,
        )

        # ---- wait --------------------------------------------------------- #
        print(f"  [lib_agent] Waiting for container (max {timeout_seconds}s)...")
        docker_wait(container)

        # Save logs before docker_copy strips bulky in-container log files.
        diagnostics = _capture_container_diagnostics(container, output_dir)
        if diagnostics:
            print(f"  [lib_agent] Saved diagnostics: {', '.join(diagnostics)}")

        # ---- copy output -------------------------------------------------- #
        print(f"  [lib_agent] Copying output data...")
        docker_copy(container, output_dir)
        _fix_data_ownership(output_dir / "data")

        # ---- ensure interception.json exists ------------------------------ #
        ensure_interception(output_dir)

    except Exception as exc:
        print(f"  [lib_agent] ERROR during task execution: {exc}")
        traceback.print_exc()
        # Try to clean up container
        try:
            docker_rm(container)
        except Exception:
            pass
        return {
            "status": "error",
            "task_id": task_id,
            "task_name": task_name,
            "output_dir": output_dir,
            "transcript": "",
            "intercepted": False,
            "stop_reason": "execution_error",
            "interception": {},
            "execution_time": time.time() - start_time,
            "error": str(exc),
        }
    finally:
        # Always remove the container
        try:
            docker_rm(container)
        except Exception:
            pass
        # Clean up disposable email
        if email and pm_key:
            try:
                delete_email(pm_key, email)
            except Exception:
                pass

    # ---- parse results ---------------------------------------------------- #
    interception = _read_interception(output_dir)
    transcript = _read_agent_messages(output_dir)
    intercepted = interception.get("intercepted", False)
    stop_reason = interception.get("stop_reason", "unknown")
    execution_time = time.time() - start_time

    status = "success" if intercepted else "completed_no_intercept"
    print(
        f"  [lib_agent] task={task_id} intercepted={intercepted} "
        f"stop={stop_reason} time={execution_time:.1f}s"
    )

    return {
        "status": status,
        "task_id": task_id,
        "task_name": task_name,
        "output_dir": output_dir,
        "transcript": transcript,
        "intercepted": intercepted,
        "stop_reason": stop_reason,
        "interception": interception,
        "execution_time": execution_time,
        "error": None,
    }
