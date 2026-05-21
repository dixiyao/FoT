"""
vLLM Qwen PinchBench Pipeline

Runs PinchBench tasks with a locally-hosted Qwen3 model via vLLM, using a
Python agent loop with tool calling — no OpenClaw CLI required.

Same task loading, grading, insight extraction, and encyclopedia aggregation
as task_openclaw_pinchbench.py.  The agent has four tools: read_file,
write_file, list_dir, bash.

Setup
-----
1. Start a vLLM server (tensor-parallel across available GPUs):

    vllm serve Qwen/Qwen3-30B-A3B-Instruct \\
        --tensor-parallel-size 4 \\
        --enable-auto-tool-choice \\
        --tool-call-parser hermes \\
        --trust-remote-code \\
        --host 0.0.0.0 --port 8000

2.  pip install openai

3. Run:

    python task_vllm_qwen_pinchbench.py \\
        --model Qwen/Qwen3-30B-A3B-Instruct \\
        --vllm-base-url http://localhost:8000/v1 \\
        --output-dir qwen_pinchbench_output \\
        --suite automated-only \\
        --use-api --api-provider gemini --api-key YOUR_KEY \\
        --iterations 2

    # Use local vLLM as judge (same endpoint):
        --judge vllm/Qwen/Qwen3-30B-A3B-Instruct

    # Or use an external judge:
        --judge google/gemini-2.5-pro-preview
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

# ---------------------------------------------------------------------------
# Pinchbench internals
# ---------------------------------------------------------------------------
_PINCHBENCH_SCRIPTS = Path(__file__).parent / "pinchbench" / "scripts"
if str(_PINCHBENCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PINCHBENCH_SCRIPTS))

from lib_grading import grade_task
from lib_tasks import Task, TaskLoader

# ---------------------------------------------------------------------------
# Local pipeline pieces (insight extraction, aggregation)
# ---------------------------------------------------------------------------
from client import ChainOfThoughtReader
from server_text import TextBasedInsightAggregationServer
from utils import call_gemini_thinking

# ---------------------------------------------------------------------------
# Optional openai package (required at runtime for agent loop)
# ---------------------------------------------------------------------------
try:
    import openai as _openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Agent tool definitions (OpenAI function-call schema)
# ---------------------------------------------------------------------------

_AGENT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to the workspace root or absolute.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (or overwrite) a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Destination path, relative to the workspace root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and subdirectories at a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to workspace root (default: workspace root).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the workspace directory and return stdout+stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to run.",
                    }
                },
                "required": ["command"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _resolve_path(path_str: str, workspace: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = workspace / p
    p = p.resolve()
    # Keep within workspace to avoid escaping
    try:
        p.relative_to(workspace.resolve())
    except ValueError:
        p = workspace / Path(path_str).name
    return p


def _execute_tool(name: str, args: Dict[str, Any], workspace: Path) -> str:
    """Execute a single tool call and return the result as a string."""
    try:
        if name == "read_file":
            path = _resolve_path(args.get("path", ""), workspace)
            if not path.exists():
                return f"Error: file not found: {path.relative_to(workspace) if path.is_relative_to(workspace) else path}"
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                return f"Error reading file: {exc}"

        elif name == "write_file":
            path = _resolve_path(args.get("path", ""), workspace)
            content = args.get("content", "")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return f"Written {len(content)} chars to {path.relative_to(workspace) if path.is_relative_to(workspace) else path}"

        elif name == "list_dir":
            dir_path = workspace
            if args.get("path"):
                dir_path = _resolve_path(args["path"], workspace)
            if not dir_path.exists():
                return f"Error: directory not found: {args.get('path', '.')}"
            entries = sorted(dir_path.iterdir(), key=lambda x: (x.is_file(), x.name))
            lines = []
            for entry in entries:
                suffix = "/" if entry.is_dir() else ""
                try:
                    size = f"  ({entry.stat().st_size} bytes)" if entry.is_file() else ""
                except OSError:
                    size = ""
                lines.append(f"{entry.name}{suffix}{size}")
            return "\n".join(lines) if lines else "(empty directory)"

        elif name == "bash":
            command = args.get("command", "")
            if not command.strip():
                return "Error: empty command"
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=str(workspace),
                    timeout=60,
                )
                out = result.stdout or ""
                err = result.stderr or ""
                combined = out
                if err:
                    combined += f"\n[stderr]:\n{err}"
                if result.returncode != 0:
                    combined += f"\n[exit code: {result.returncode}]"
                return combined or "(no output)"
            except subprocess.TimeoutExpired:
                return "Error: command timed out (60s)"
            except Exception as exc:
                return f"Error: {exc}"

        else:
            return f"Error: unknown tool '{name}'"

    except Exception as exc:
        return f"Error executing tool {name}: {exc}"


# ---------------------------------------------------------------------------
# Transcript helpers (OpenClaw-compatible format)
# ---------------------------------------------------------------------------

def _build_assistant_entry(message: Dict[str, Any], usage_info: Optional[Dict] = None) -> Dict[str, Any]:
    """Convert an OpenAI-format assistant message to OpenClaw transcript format."""
    content_blocks: List[Dict[str, Any]] = []

    # Text content (may contain <think>...</think> for Qwen3 thinking)
    raw_content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    if reasoning:
        raw_content = f"<think>\n{reasoning}\n</think>\n\n{raw_content}"
    if raw_content:
        content_blocks.append({"type": "text", "text": raw_content})

    # Tool calls
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        raw_args = fn.get("arguments", "{}")
        try:
            parsed_args = json.loads(raw_args)
        except json.JSONDecodeError:
            parsed_args = {"raw": raw_args}
        content_blocks.append({
            "type": "toolCall",
            "name": fn.get("name", "unknown"),
            "id": tc.get("id", ""),
            "arguments": parsed_args,
        })

    usage_block: Dict[str, Any] = {}
    if usage_info:
        usage_block = {
            "input": usage_info.get("prompt_tokens", 0),
            "output": usage_info.get("completion_tokens", 0),
            "totalTokens": usage_info.get("total_tokens", 0),
        }

    return {
        "type": "message",
        "message": {
            "role": "assistant",
            "content": content_blocks,
            "usage": usage_block,
        },
    }


def _build_tool_result_entry(tool_call_id: str, tool_name: str, result_text: str) -> Dict[str, Any]:
    return {
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "content": [{"type": "text", "text": result_text}],
        },
    }


# ---------------------------------------------------------------------------
# vLLM agent loop
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a capable AI assistant with access to file system and shell tools.
You are working in workspace directory: {workspace}

## Available Tools
- read_file(path): Read a file (path relative to workspace or absolute)
- write_file(path, content): Write content to a file
- list_dir(path=""): List directory contents (default: workspace root)
- bash(command): Execute a shell command in the workspace directory

## Guidelines
- Explore the workspace first if needed (use list_dir, read_file)
- Work step by step; verify your output
- Use relative paths — they resolve to the workspace directory
- When the task is fully complete, give a concise summary of what you did
- Do NOT call any tools after you have finished the task
{insights_section}"""

_INSIGHTS_SECTION_TEMPLATE = """
## Insight Library
The following patterns and techniques were extracted from prior task runs.
Apply the relevant ones to improve your accuracy and efficiency:

{insights_content}"""


def _build_system_prompt(workspace: Path, insights_content: Optional[str] = None) -> str:
    insights_section = ""
    if insights_content and insights_content.strip():
        insights_section = _INSIGHTS_SECTION_TEMPLATE.format(
            insights_content=insights_content.strip()
        )
    return _SYSTEM_PROMPT_TEMPLATE.format(
        workspace=workspace,
        insights_section=insights_section,
    )


def execute_vllm_task(
    *,
    task: Task,
    model_id: str,
    vllm_base_url: str,
    workspace: Path,
    timeout_seconds: float,
    insights_content: Optional[str] = None,
    max_iterations: int = 40,
    max_tokens_per_turn: int = 4096,
    temperature: float = 0.0,
    enable_thinking: bool = True,
) -> Dict[str, Any]:
    """
    Run a PinchBench task through a local vLLM Qwen3 agent.

    Returns a dict with the same keys as execute_openclaw_task so the rest
    of the pipeline (grading, insight extraction, metrics) works unchanged.
    """
    if not _OPENAI_AVAILABLE:
        return {
            "status": "error",
            "transcript": [],
            "usage": {},
            "workspace": str(workspace),
            "exit_code": -1,
            "timed_out": False,
            "execution_time": 0.0,
            "stdout": "",
            "stderr": "openai package not installed: pip install openai",
        }

    client = _openai.OpenAI(base_url=vllm_base_url, api_key="EMPTY")

    system_prompt = _build_system_prompt(workspace, insights_content)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]

    # Multi-session support (same as openclaw: sequential prompts)
    sessions = task.frontmatter.get("sessions", [])
    if sessions:
        prompts = []
        for s in sessions:
            if isinstance(s, str):
                prompts.append(s)
            elif isinstance(s, dict):
                prompts.append(s.get("prompt") or s.get("message", ""))
    else:
        prompts = [task.prompt]

    transcript: List[Dict[str, Any]] = []
    usage_totals: Dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "request_count": 0,
    }
    start_time = time.time()
    timed_out = False
    status = "success"

    for prompt_idx, prompt_text in enumerate(prompts):
        if time.time() - start_time >= timeout_seconds:
            timed_out = True
            break

        messages.append({"role": "user", "content": prompt_text})

        for iteration in range(max_iterations):
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                timed_out = True
                break

            # Extra kwargs for Qwen3 thinking budget (if supported by vLLM build)
            extra: Dict[str, Any] = {}
            if enable_thinking:
                try:
                    # vLLM ≥ 0.8 supports chat_template_kwargs for thinking
                    extra["chat_template_kwargs"] = {"enable_thinking": True}
                except Exception:
                    pass

            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    tools=_AGENT_TOOLS,
                    tool_choice="auto",
                    max_tokens=max_tokens_per_turn,
                    temperature=temperature,
                    **extra,
                )
            except _openai.APITimeoutError:
                timed_out = True
                break
            except _openai.APIConnectionError as exc:
                status = "error"
                transcript.append({
                    "type": "error",
                    "error": f"vLLM connection error: {exc}",
                })
                break
            except _openai.APIStatusError as exc:
                status = "error"
                transcript.append({
                    "type": "error",
                    "error": f"vLLM API error {exc.status_code}: {exc.message}",
                })
                break
            except Exception as exc:
                status = "error"
                transcript.append({"type": "error", "error": str(exc)})
                break

            choice = response.choices[0]
            assistant_msg = choice.message

            # Track token usage
            if response.usage:
                usage_totals["input_tokens"] += response.usage.prompt_tokens or 0
                usage_totals["output_tokens"] += response.usage.completion_tokens or 0
                usage_totals["total_tokens"] += response.usage.total_tokens or 0
                usage_totals["request_count"] += 1

            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            # Convert to dict for transcript and next messages
            assistant_dict: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant_msg.content or "",
            }
            if hasattr(assistant_msg, "reasoning_content") and assistant_msg.reasoning_content:
                assistant_dict["reasoning_content"] = assistant_msg.reasoning_content
            if assistant_msg.tool_calls:
                assistant_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ]

            transcript.append(_build_assistant_entry(assistant_dict, usage_info))
            messages.append(assistant_dict)

            # If no tool calls, the agent is done with this session
            if not assistant_msg.tool_calls:
                break

            # Execute each tool call
            for tc in assistant_msg.tool_calls:
                tool_name = tc.function.name
                raw_args = tc.function.arguments
                try:
                    tool_args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    tool_args = {"raw": raw_args}

                result_text = _execute_tool(tool_name, tool_args, workspace)
                transcript.append(_build_tool_result_entry(tc.id, tool_name, result_text))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })

        if timed_out:
            break

    execution_time = time.time() - start_time

    if timed_out:
        status = "timeout"
    elif not transcript:
        status = "error"

    return {
        "agent_id": model_id,
        "task_id": task.task_id,
        "command": f"vllm:{vllm_base_url}",
        "status": status,
        "transcript": transcript,
        "usage": usage_totals,
        "workspace": str(workspace),
        "exit_code": 0 if status == "success" else -1,
        "timed_out": timed_out,
        "execution_time": execution_time,
        "stdout": "",
        "stderr": "",
    }


# ---------------------------------------------------------------------------
# Workspace preparation (no OpenClaw dependency)
# ---------------------------------------------------------------------------

def prepare_vllm_task_workspace(
    skill_dir: Path,
    task: Task,
    workspace_root: Path,
    insights_content: Optional[str] = None,
) -> Path:
    workspace = workspace_root / task.task_id
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    for file_spec in task.workspace_files:
        if "content" in file_spec:
            dest = workspace / file_spec["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(file_spec["content"], encoding="utf-8")
        else:
            source = skill_dir / "assets" / file_spec["source"]
            dest = workspace / file_spec["dest"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                dest.write_bytes(source.read_bytes())
            except FileNotFoundError:
                print(f"  Warning: workspace asset not found: {source}")

    # Write INSIGHTS.md so the agent can also read it as a file
    if insights_content and insights_content.strip():
        (workspace / "INSIGHTS.md").write_text(insights_content, encoding="utf-8")

    return workspace


# ---------------------------------------------------------------------------
# Local vLLM judge
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_MSG = (
    "You are a strict grading function. "
    "Respond with ONLY a JSON object, no prose, no markdown fences, no extra text."
)


def _judge_via_local_vllm(
    *,
    prompt: str,
    model_id: str,
    vllm_base_url: str,
    timeout_seconds: float = 120.0,
) -> Dict[str, Any]:
    """Call the local vLLM endpoint as a judge (no tools)."""
    payload = json.dumps({
        "model": model_id,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }).encode("utf-8")

    endpoint = vllm_base_url.rstrip("/") + "/chat/completions"
    req = request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        return {"status": "error", "text": "", "error": f"HTTP {exc.code}: {body}"}
    except error.URLError as exc:
        return {"status": "error", "text": "", "error": str(exc)}
    except TimeoutError:
        return {"status": "timeout", "text": "", "error": "Request timed out"}

    choices = data.get("choices", [])
    if not choices:
        return {"status": "error", "text": "", "error": "No choices in response"}
    text = choices[0].get("message", {}).get("content", "")
    return {"status": "success", "text": text}


# ---------------------------------------------------------------------------
# Transcript analysis helpers (copied from task_openclaw_pinchbench.py)
# ---------------------------------------------------------------------------

def _extract_transcript_text(transcript: List[Dict[str, Any]]) -> str:
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
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "") or block.get("content", "")
                    if text:
                        parts.append(f"[{role}]: {text}")
        elif role == "assistant":
            parts.append(f"[assistant]: {content}")
    return "\n\n".join(parts)


def _iter_dict_nodes(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_dict_nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_dict_nodes(value)


def _analyze_tool_calls(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    calls: List[Dict[str, Any]] = []
    calls_by_id: Dict[str, Dict[str, Any]] = {}

    def _extract_name(node):
        for key in ("name", "tool_name", "tool", "function", "function_name"):
            v = node.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        fn = node.get("function")
        if isinstance(fn, dict):
            n = fn.get("name")
            if isinstance(n, str) and n.strip():
                return n.strip()
        return None

    def _extract_call_id(node):
        for key in ("call_id", "tool_call_id", "toolUseId", "id", "tool_use_id"):
            v = node.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def _looks_like_error(node):
        if node.get("is_error") is True or node.get("isError") is True:
            return True
        status = node.get("status")
        if isinstance(status, str) and status.lower() in {"error", "failed", "failure"}:
            return True
        return bool(node.get("error"))

    # Scan for content blocks with type containing "tool"
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
            if "tool" not in block_type:
                continue
            if block_type == "toolcall":
                name = _extract_name(block) or "unknown_tool"
                call = {
                    "name": name,
                    "call_id": block.get("id") or _extract_call_id(block),
                    "status": "unknown",
                    "error": None,
                }
                calls.append(call)
                if call["call_id"]:
                    calls_by_id[call["call_id"]] = call
            elif block_type == "toolresult":
                call_id = _extract_call_id(block)
                has_error = _looks_like_error(block)
                linked = calls_by_id.get(call_id) if call_id else None
                if linked:
                    linked["status"] = "error" if has_error else "ok"
                    linked["error"] = block.get("error") if has_error else None

    # Mark still-unknown calls as ok (tool results came back via separate messages)
    for call in calls:
        if call["status"] == "unknown":
            call["status"] = "ok"

    tool_names = sorted({c["name"] for c in calls if c.get("name")})
    tool_counter = collections.Counter(c["name"] for c in calls if c.get("name"))
    ok_calls = sum(1 for c in calls if c.get("status") == "ok")
    error_calls = sum(1 for c in calls if c.get("status") == "error")
    unknown_calls = sum(1 for c in calls if c.get("status") == "unknown")

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
# Insights writer
# ---------------------------------------------------------------------------

def _load_insights_content(encyclopedia_path: str) -> Optional[str]:
    try:
        with open(encyclopedia_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  Warning: failed to load encyclopedia: {exc}")
        return None

    if isinstance(data, dict) and set(data.keys()) == {"insight"}:
        body = data["insight"]
    elif isinstance(data, dict):
        lines = []
        for name, desc in data.items():
            lines.append(f"### {name}\n{desc}\n")
        body = "\n".join(lines)
    else:
        body = str(data)

    return (
        "# Insight Library\n\n"
        "Reasoning traces and techniques from prior runs. Apply relevant insights "
        "before starting each task.\n\n"
        f"{body}\n"
    )


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class VLLMQwenPinchBenchPipeline:
    """
    PinchBench pipeline backed by a local vLLM Qwen3 agent.

    Mirrors OpenClawPinchBenchPipeline with the same output format, metrics,
    insight extraction, and encyclopedia aggregation.
    """

    def __init__(
        self,
        model_id: str,
        vllm_base_url: str = "http://localhost:8000/v1",
        output_dir: str = "qwen_pinchbench_output",
        suite: str = "all",
        pinchbench_dir: Optional[str] = None,
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
        api_model: str = "gemini-3-pro-preview",
        timeout_multiplier: float = 1.0,
        encyclopedia_path: Optional[str] = None,
        judge_model: Optional[str] = None,
        thinking_level: Optional[str] = "high",
        enable_thinking: bool = True,
        max_tokens_per_turn: int = 4096,
        max_agent_iterations: int = 40,
    ):
        self.model_id = model_id
        self.vllm_base_url = vllm_base_url.rstrip("/")
        self.output_dir = output_dir
        self.suite = suite
        self.use_api = use_api
        self.api_provider = api_provider
        self.api_key = api_key or (
            os.getenv("GEMINI_API_KEY") if api_provider == "gemini" else os.getenv("OPENROUTER_API_KEY")
        )
        self.api_model = api_model
        self.timeout_multiplier = timeout_multiplier
        self.encyclopedia_path = encyclopedia_path
        self.judge_model = judge_model or model_id
        self.thinking_level = thinking_level
        self.enable_thinking = enable_thinking
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_agent_iterations = max_agent_iterations

        if self.api_provider == "gemini" and self.api_key:
            os.environ["GEMINI_API_KEY"] = self.api_key

        self.skill_dir = Path(pinchbench_dir) if pinchbench_dir else Path(__file__).parent / "pinchbench"
        self.tasks_dir = self.skill_dir / "tasks"

        # Workspaces live under output_dir/workspaces/
        self.workspace_root = Path(self.output_dir) / "workspaces"
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        os.makedirs(self.output_dir, exist_ok=True)

        self._client: Optional[ChainOfThoughtReader] = None
        self._metrics_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # vLLM connectivity check
    # ------------------------------------------------------------------

    def _check_vllm(self) -> None:
        """Verify the vLLM server is reachable and serving the target model."""
        models_url = self.vllm_base_url.rstrip("/") + "/models"
        try:
            req = request.Request(
                models_url,
                headers={"Authorization": "Bearer EMPTY"},
                method="GET",
            )
            with request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            available = [m.get("id", "") for m in data.get("data", [])]
            if available:
                print(f"  vLLM models available: {available}")
            else:
                print("  Warning: vLLM /models returned empty list")
        except Exception as exc:
            print(f"\n{'='*70}")
            print(f"ERROR: Cannot reach vLLM server at {self.vllm_base_url}")
            print(f"  {exc}")
            print("\nStart a vLLM server first:")
            print(f"  vllm serve {self.model_id} \\")
            print(f"      --tensor-parallel-size <NUM_GPUS> \\")
            print(f"      --enable-auto-tool-choice \\")
            print(f"      --tool-call-parser hermes \\")
            print(f"      --trust-remote-code \\")
            print(f"      --host 0.0.0.0 --port 8000")
            print("=" * 70 + "\n")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Task loading
    # ------------------------------------------------------------------

    def _load_tasks(self) -> List[Task]:
        loader = TaskLoader(self.tasks_dir)
        tasks = loader.load_all_tasks()
        if self.suite == "all":
            return tasks
        if self.suite == "automated-only":
            return [t for t in tasks if t.grading_type == "automated"]
        ids = {tid.strip() for tid in self.suite.split(",") if tid.strip()}
        return [t for t in tasks if t.task_id in ids]

    # ------------------------------------------------------------------
    # Insight extraction client
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

    def _call_for_extraction(self, prompt: str, max_new_tokens: int) -> Tuple[str, Dict]:
        if self.use_api and self.api_provider == "gemini" and self.thinking_level:
            return call_gemini_thinking(
                api_key=self.api_key,
                model_name=self.api_model,
                prompt=prompt,
                thinking_level=self.thinking_level,
                max_new_tokens=max_new_tokens,
            )
        return self._ensure_client()._call_model(prompt, max_new_tokens=max_new_tokens)

    def _apply_reflection_and_extraction(
        self, task_prompt: str, agent_response: str
    ) -> Dict[str, Any]:
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

        insights: Dict[str, str] = {}
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
            print(f"  Warning: insight parse error: {exc}")

        print(f"  Extracted {len(insights)} insights")
        return {
            "insight_book": insights,
            "output_tokens": token_info.get("output_tokens", 0),
        }

    # ------------------------------------------------------------------
    # Grading (routes vllm/* judge prefix to local endpoint)
    # ------------------------------------------------------------------

    def _grade_task(self, task: Task, exec_result: Dict[str, Any]):
        judge = self.judge_model

        # Route vllm/ prefix to local endpoint
        if judge.startswith("vllm/"):
            from lib_grading import (
                _grade_automated,
                _grade_llm_judge,
                _combine_grades,
                GradeResult,
            )
            from lib_grading import _build_judge_prompt, _summarize_transcript, _read_workspace_files, _parse_judge_text, _normalize_judge_response, _format_grading_criteria

            vllm_model_id = judge[len("vllm/"):]
            transcript = exec_result.get("transcript", [])
            workspace_content = _read_workspace_files(exec_result.get("workspace", ""))
            rubric = task.llm_judge_rubric or _format_grading_criteria(task)
            prompt = _build_judge_prompt(task, _summarize_transcript(transcript), rubric, workspace_content)

            judge_result = _judge_via_local_vllm(
                prompt=prompt,
                model_id=vllm_model_id,
                vllm_base_url=self.vllm_base_url,
                timeout_seconds=180.0,
            )
            raw_parsed = _parse_judge_text(judge_result.get("text", ""))
            parsed = _normalize_judge_response(raw_parsed)
            breakdown = parsed.get("scores", {})
            total = parsed.get("total")
            from lib_grading import GradeResult, _normalize_score_dict
            return GradeResult(
                task_id=task.task_id,
                score=float(total) if total is not None else 0.0,
                max_score=1.0,
                grading_type="llm_judge",
                breakdown=_normalize_score_dict(breakdown),
                notes=str(parsed.get("notes", "")),
            )

        # All other judge models: use existing call_judge_api dispatch
        return grade_task(
            task=task,
            execution_result=exec_result,
            skill_dir=self.skill_dir,
            judge_model=judge,
            judge_backend="api",
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _metrics_log_path(self) -> Path:
        return Path(self.output_dir) / "metrics_log.json"

    def _build_metrics_summary(self, task_metrics: List[Dict[str, Any]], library_output_tokens: Optional[int]) -> Dict[str, Any]:
        graded = [t for t in task_metrics if t.get("grade", {}).get("score") is not None]
        score_sum = sum(float(t["grade"].get("score", 0.0)) for t in graded)
        max_sum = sum(float(t["grade"].get("max_score", 0.0)) for t in graded)
        accuracy = (score_sum / max_sum * 100.0) if max_sum > 0 else None

        tool_name_counts: Dict[str, int] = {}
        total_calls = successful_calls = error_calls = unknown_calls = 0
        for t in task_metrics:
            tools = t.get("tools", {})
            total_calls += int(tools.get("total_calls", 0) or 0)
            successful_calls += int(tools.get("successful_calls", 0) or 0)
            error_calls += int(tools.get("error_calls", 0) or 0)
            unknown_calls += int(tools.get("unknown_status_calls", 0) or 0)
            for name, count in (tools.get("name_counts", {}) or {}).items():
                tool_name_counts[name] = tool_name_counts.get(name, 0) + int(count)

        total_exec_time = sum(float(t.get("execution_time_seconds", 0.0) or 0.0) for t in task_metrics)
        total_agent_tokens = sum(int((t.get("output_tokens") or {}).get("agent", 0) or 0) for t in task_metrics)
        total_extract_tokens = sum(int((t.get("output_tokens") or {}).get("extraction", 0) or 0) for t in task_metrics)

        return {
            "tasks_total": len(task_metrics),
            "tasks_graded": len(graded),
            "graded_score_sum": score_sum,
            "graded_max_score_sum": max_sum,
            "overall_accuracy_pct": accuracy,
            "output_tokens_agent_total": total_agent_tokens,
            "output_tokens_extraction_total": total_extract_tokens,
            "output_tokens_total": total_agent_tokens + total_extract_tokens,
            "tool_names": sorted(tool_name_counts.keys()),
            "tool_name_counts": dict(sorted(tool_name_counts.items())),
            "total_tool_calls": total_calls,
            "successful_tool_calls": successful_calls,
            "error_tool_calls": error_calls,
            "unknown_status_tool_calls": unknown_calls,
            "execution_time_total_seconds": total_exec_time,
            "library_output_tokens": int(library_output_tokens) if library_output_tokens is not None else None,
        }

    def _write_metrics_log(self, task_metrics: List[Dict[str, Any]], *, library_output_tokens: Optional[int] = None) -> None:
        if library_output_tokens is None:
            library_output_tokens = self._metrics_cache.get("summary", {}).get("library_output_tokens")

        payload = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "output_dir": str(Path(self.output_dir).resolve()),
            "model": self.model_id,
            "vllm_base_url": self.vllm_base_url,
            "judge_model": self.judge_model,
            "suite": self.suite,
            "summary": self._build_metrics_summary(task_metrics, library_output_tokens),
            "tasks": task_metrics,
        }
        self._metrics_log_path().write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self._metrics_cache = payload

    def _write_overall_metrics_log(self, root_output_dir: str, elapsed_seconds: float) -> None:
        root = Path(root_output_dir)
        iter_logs = sorted(root.glob("iter_*/metrics_log.json"))
        if not iter_logs:
            return

        summaries = []
        for log_path in iter_logs:
            try:
                p = json.loads(log_path.read_text(encoding="utf-8"))
                summaries.append({"iteration": log_path.parent.name, "metrics_log": str(log_path), "summary": p.get("summary", {})})
            except Exception:
                continue

        if not summaries:
            return

        def _sum(key):
            return sum(int(s["summary"].get(key, 0) or 0) for s in summaries)

        def _fsum(key):
            return sum(float(s["summary"].get(key, 0.0) or 0.0) for s in summaries)

        gscore = _fsum("graded_score_sum")
        gmax = _fsum("graded_max_score_sum")
        accuracy = (gscore / gmax * 100.0) if gmax > 0 else None

        tool_counts: Dict[str, int] = {}
        for s in summaries:
            for name, count in (s["summary"].get("tool_name_counts", {}) or {}).items():
                tool_counts[name] = tool_counts.get(name, 0) + int(count)

        overall = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "output_dir": str(root.resolve()),
            "iterations": summaries,
            "summary": {
                "iterations": len(summaries),
                "tasks_total": _sum("tasks_total"),
                "tasks_graded": _sum("tasks_graded"),
                "graded_score_sum": gscore,
                "graded_max_score_sum": gmax,
                "overall_accuracy_pct": accuracy,
                "output_tokens_agent_total": _sum("output_tokens_agent_total"),
                "output_tokens_extraction_total": _sum("output_tokens_extraction_total"),
                "output_tokens_total": _sum("output_tokens_total"),
                "tool_names": sorted(tool_counts.keys()),
                "tool_name_counts": dict(sorted(tool_counts.items())),
                "total_tool_calls": _sum("total_tool_calls"),
                "successful_tool_calls": _sum("successful_tool_calls"),
                "error_tool_calls": _sum("error_tool_calls"),
                "execution_time_total_seconds": _fsum("execution_time_total_seconds"),
                "library_output_tokens_total": _sum("library_output_tokens"),
                "pipeline_elapsed_seconds": elapsed_seconds,
            },
        }
        (root / "metrics_log_overall.json").write_text(
            json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Step 1 + 2/3: Task execution + insight extraction
    # ------------------------------------------------------------------

    def run_tasks_and_extract(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        self._check_vllm()

        insights_content: Optional[str] = None
        if self.encyclopedia_path and Path(self.encyclopedia_path).exists():
            print(f"\nLoading encyclopedia from {self.encyclopedia_path}")
            insights_content = _load_insights_content(self.encyclopedia_path)
        else:
            print("\nNo encyclopedia — running without prior insights")

        tasks = self._load_tasks()
        if not tasks:
            print(f"No tasks found in {self.tasks_dir}")
            return []

        print(f"\nLoaded {len(tasks)} tasks (suite='{self.suite}')")
        print("=" * 80)

        if run_id is None:
            run_id = f"vllm_{int(time.time())}"

        results = []
        task_metrics: List[Dict[str, Any]] = []
        task_counter = 0

        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] Task: {task.task_id} — {task.name}")
            print("-" * 60)

            # Prepare workspace
            workspace = prepare_vllm_task_workspace(
                skill_dir=self.skill_dir,
                task=task,
                workspace_root=self.workspace_root,
                insights_content=insights_content,
            )

            # ----------------------------------------------------------------
            # Step 1: Run task with vLLM agent
            # ----------------------------------------------------------------
            timeout_seconds = task.timeout_seconds * self.timeout_multiplier
            try:
                exec_result = execute_vllm_task(
                    task=task,
                    model_id=self.model_id,
                    vllm_base_url=self.vllm_base_url,
                    workspace=workspace,
                    timeout_seconds=timeout_seconds,
                    insights_content=insights_content,
                    max_iterations=self.max_agent_iterations,
                    max_tokens_per_turn=self.max_tokens_per_turn,
                    enable_thinking=self.enable_thinking,
                )
            except Exception as exc:
                print(f"  Error executing task: {exc}")
                task_metrics.append({
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "execution_status": "error",
                    "execution_time_seconds": None,
                    "grade": {"score": None, "max_score": None, "accuracy_pct": None, "grading_type": task.grading_type},
                    "output_tokens": {"agent": 0, "extraction": 0, "total": 0},
                    "tools": {"names": [], "name_counts": {}, "total_calls": 0, "successful_calls": 0, "error_calls": 0, "unknown_status_calls": 0, "calls": []},
                    "insights_extracted": 0,
                    "error": str(exc),
                })
                self._write_metrics_log(task_metrics)
                continue

            status = exec_result.get("status", "error")
            transcript = exec_result.get("transcript", [])
            print(f"  Status: {status} | transcript entries: {len(transcript)}")

            # ----------------------------------------------------------------
            # Grade the task
            # ----------------------------------------------------------------
            try:
                grade = self._grade_task(task, exec_result)
                score_pct = grade.score / grade.max_score * 100 if grade.max_score > 0 else 0
                print(f"  Grade: {grade.score:.2f}/{grade.max_score:.2f} ({score_pct:.0f}%) [{grade.grading_type}]")
            except Exception as exc:
                print(f"  Warning: grading failed: {exc}")
                grade = None

            usage = exec_result.get("usage", {}) or {}
            agent_output_tokens = int(usage.get("output_tokens", 0) or 0)
            extraction_output_tokens = 0
            execution_time = exec_result.get("execution_time")
            if execution_time is not None:
                execution_time = float(execution_time)

            tool_stats = _analyze_tool_calls(transcript)
            task_metric: Dict[str, Any] = {
                "task_id": task.task_id,
                "task_name": task.name,
                "execution_status": status,
                "execution_time_seconds": execution_time,
                "grade": {
                    "score": grade.score if grade else None,
                    "max_score": grade.max_score if grade else None,
                    "accuracy_pct": (grade.score / grade.max_score * 100.0) if grade and grade.max_score > 0 else None,
                    "grading_type": grade.grading_type if grade else task.grading_type,
                    "notes": grade.notes if grade else "",
                },
                "output_tokens": {"agent": agent_output_tokens, "extraction": extraction_output_tokens, "total": agent_output_tokens},
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

            # ----------------------------------------------------------------
            # Steps 2 & 3: Reflection + insight extraction
            # ----------------------------------------------------------------
            agent_response = _extract_transcript_text(transcript)
            if not agent_response.strip():
                print("  Skipping insight extraction — no extractable text")
                task_metrics.append(task_metric)
                self._write_metrics_log(task_metrics)
                continue

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

            insight_book = extraction.get("insight_book", {}) or {}
            extraction_output_tokens = int(extraction.get("output_tokens", 0) or 0)
            task_metric["output_tokens"] = {
                "agent": agent_output_tokens,
                "extraction": extraction_output_tokens,
                "total": agent_output_tokens + extraction_output_tokens,
            }
            task_metric["insights_extracted"] = len(insight_book)

            # Save problem_XXXX.json
            task_counter += 1
            output_file = os.path.join(self.output_dir, f"problem_{task_counter:04d}.json")
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
            s = sum(float(t["grade"].get("score", 0.0) or 0.0) for t in graded)
            m = sum(float(t["grade"].get("max_score", 0.0) or 0.0) for t in graded)
            print(f"Overall score:   {s/m*100:.1f}%  ({len(graded)} tasks graded, judge={self.judge_model})")
        print(f"Total insights extracted: {total_insights}")
        self._write_metrics_log(task_metrics)
        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_insights(self) -> Optional[str]:
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
        existing_tasks = self._metrics_cache.get("tasks", [])
        self._write_metrics_log(existing_tasks, library_output_tokens=int(total_output_tokens or 0))
        return encyclopedia_path

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_pipeline(self, iterations: int = 1, start_from_step2: bool = False) -> None:
        start_time = time.time()
        base_output_dir = self.output_dir
        current_encyclopedia: Optional[str] = self.encyclopedia_path

        for iteration in range(1, iterations + 1):
            print("\n" + "=" * 80)
            print(f"Iteration {iteration}/{iterations}")
            print("=" * 80)

            if iterations > 1:
                iter_dir = os.path.join(self.output_dir, f"iter_{iteration:02d}")
                os.makedirs(iter_dir, exist_ok=True)
                orig_output_dir = self.output_dir
                self.output_dir = iter_dir
                self.workspace_root = Path(iter_dir) / "workspaces"
                self.workspace_root.mkdir(parents=True, exist_ok=True)
            else:
                orig_output_dir = None

            if current_encyclopedia:
                self.encyclopedia_path = current_encyclopedia

            if not start_from_step2 or iteration > 1:
                print("\n--- Step 1/2/3: Task Execution + Insight Extraction ---")
                self.run_tasks_and_extract()
            else:
                print("Skipping task execution (start_from_step2=True)")

            print("\n--- Aggregation ---")
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
        description="vLLM Qwen PinchBench Pipeline — evaluate Qwen3 on PinchBench without OpenClaw"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct",
                        help="Model ID as registered in the vLLM server (default: Qwen/Qwen3-30B-A3B-Instruct)")
    parser.add_argument("--vllm-base-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM OpenAI-compatible API base URL (default: http://localhost:8000/v1)")
    parser.add_argument("--output-dir", type=str, default="qwen_pinchbench_output",
                        help="Directory to save insights and encyclopedia")
    parser.add_argument("--suite", type=str, default="all",
                        help='Tasks: "all", "automated-only", or comma-separated task IDs')
    parser.add_argument("--judge", type=str, default=None,
                        help=(
                            "Judge model for LLM-judge tasks. "
                            "Use 'vllm/<model-id>' for local vLLM judge, "
                            "'google/...' or 'openrouter/...' for remote. "
                            "Defaults to the same model as --model (via vllm/)."
                        ))
    parser.add_argument("--pinchbench-dir", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--encyclopedia", type=str, default=None,
                        help="Pre-existing encyclopedia.json to inject for iteration 1")
    parser.add_argument("--start-from-step2", action="store_true",
                        help="Skip task execution, run aggregation only")
    parser.add_argument("--use-api", action="store_true",
                        help="Use external API (Gemini/OpenRouter) for insight extraction")
    parser.add_argument("--api-provider", type=str, default="gemini", choices=["gemini", "openrouter"])
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--api-model", type=str, default="gemini-3-pro-preview")
    parser.add_argument("--thinking-level", type=str, default="high", choices=["low", "medium", "high"])
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable Qwen3 thinking mode (adds /no_think to prompts)")
    parser.add_argument("--timeout-multiplier", type=float, default=1.0)
    parser.add_argument("--max-tokens-per-turn", type=int, default=4096,
                        help="Max tokens per agent turn (default: 4096)")
    parser.add_argument("--max-agent-iterations", type=int, default=40,
                        help="Max tool-call iterations per task (default: 40)")

    args = parser.parse_args()

    if not _OPENAI_AVAILABLE and not args.start_from_step2:
        print("ERROR: 'openai' package is required for agent execution.")
        print("  pip install openai")
        sys.exit(1)

    model_id = args.model

    # Default judge: same local vLLM model
    judge_model = args.judge or f"vllm/{model_id}"

    pipeline = VLLMQwenPinchBenchPipeline(
        model_id=model_id,
        vllm_base_url=args.vllm_base_url,
        output_dir=args.output_dir,
        suite=args.suite,
        pinchbench_dir=args.pinchbench_dir,
        use_api=args.use_api,
        api_key=args.api_key,
        api_provider=args.api_provider,
        api_model=args.api_model,
        timeout_multiplier=args.timeout_multiplier,
        encyclopedia_path=args.encyclopedia,
        judge_model=judge_model,
        thinking_level=args.thinking_level if args.use_api else None,
        enable_thinking=not args.no_thinking,
        max_tokens_per_turn=args.max_tokens_per_turn,
        max_agent_iterations=args.max_agent_iterations,
    )

    pipeline.run_pipeline(
        iterations=args.iterations,
        start_from_step2=args.start_from_step2,
    )


if __name__ == "__main__":
    main()
