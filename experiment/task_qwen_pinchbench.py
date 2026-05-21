"""
Qwen PinchBench Pipeline (Direct HuggingFace, no vLLM, no OpenClaw)

Loads the Qwen3 model directly via HuggingFace Transformers across all
available GPUs (device_map="auto") and runs PinchBench tasks with a simple
agentic tool-call loop.

Tools available to the agent: read_file, write_file, list_dir, bash.
Grading, insight extraction, and encyclopedia aggregation are identical to
task_openclaw_pinchbench.py.

Usage
-----
    python task_qwen_pinchbench.py \\
        --model Qwen/Qwen3-30B-A3B-Instruct \\
        --output-dir qwen_pinchbench_output \\
        --suite automated-only \\
        --use-api --api-provider gemini --api-key YOUR_KEY \\
        --iterations 2

    # Use a different judge model for LLM-judge tasks:
        --judge google/gemini-2.5-pro-preview  (OpenRouter/Gemini)

Requirements
------------
    pip install transformers accelerate torch
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
from client import ChainOfThoughtReader
from server_text import TextBasedInsightAggregationServer
from utils import call_gemini_thinking


# ---------------------------------------------------------------------------
# Tool definitions (Qwen3 / OpenAI function-call format)
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace root."}
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
                    "path": {"type": "string", "description": "Destination path relative to workspace root."},
                    "content": {"type": "string", "description": "Text content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and subdirectories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory relative to workspace (default: workspace root)."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the workspace directory and return stdout+stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command to run."}
                },
                "required": ["command"],
            },
        },
    },
]


def _render_manual_chat(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> str:
    """Fallback renderer for tokenizers whose chat template rejects tool history.

    Qwen chat templates vary across releases. Some accept OpenAI-style
    assistant.tool_calls / role=tool messages; others crash after the first tool
    result. This fallback keeps the model in a plain system/user/assistant chat
    and gives an explicit <tool_call>{...}</tool_call> contract.
    """
    tool_specs = []
    for tool in tools:
        fn = tool.get("function", {})
        params = fn.get("parameters", {})
        tool_specs.append(
            f"- {fn.get('name')}: {fn.get('description', '')}\n"
            f"  parameters: {json.dumps(params, ensure_ascii=False)}"
        )

    rendered = [
        "<|im_start|>system",
        "You may call tools by responding with exactly one or more blocks like:",
        '<tool_call>{"name": "tool_name", "arguments": {"arg": "value"}}</tool_call>',
        "Available tools:",
        "\n".join(tool_specs),
        "<|im_end|>",
    ]
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        if role not in {"system", "user", "assistant"}:
            role = "user"
        rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    rendered.append("<|im_start|>assistant\n")
    return "\n".join(rendered)


def _tool_result_message(tool_name: str, call_id: str, result_text: str) -> Dict[str, str]:
    max_chars = 20000
    if len(result_text) > max_chars:
        result_text = (
            result_text[:max_chars]
            + f"\n\n[tool result truncated to {max_chars} chars; use targeted reads/commands for more]"
        )
    return {
        "role": "user",
        "content": (
            f"Tool result for {tool_name} ({call_id}):\n"
            f"{result_text}\n\n"
            "Continue from this tool result. If the task is complete, respond with "
            "a concise final summary and no tool calls."
        ),
    }


def _requires_manual_chat(messages: List[Dict[str, Any]]) -> bool:
    """Use manual rendering once the conversation contains executed tools.

    HF chat templates for tool-calling are not stable across Qwen releases. The
    first turn can use the model-provided template to advertise tools, but after
    we have tool calls/results in history, manual rendering is safer and avoids
    template exceptions that previously produced `Status: error | entries: 3`.
    """
    for message in messages:
        content = str(message.get("content", ""))
        if "<tool_call>" in content or content.startswith("Tool result for "):
            return True
    return False


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _safe_path(path_str: str, workspace: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = workspace / p
    p = p.resolve()
    try:
        p.relative_to(workspace.resolve())
    except ValueError:
        p = workspace / Path(path_str).name
    return p


def _run_tool(name: str, args: Dict[str, Any], workspace: Path) -> str:
    try:
        if name == "read_file":
            p = _safe_path(args.get("path", ""), workspace)
            if not p.exists():
                return f"Error: file not found: {args.get('path')}"
            return p.read_text(encoding="utf-8", errors="replace")

        if name == "write_file":
            p = _safe_path(args.get("path", ""), workspace)
            content = args.get("content", "")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Written {len(content)} chars to {p.name}"

        if name == "list_dir":
            d = workspace
            if args.get("path"):
                d = _safe_path(args["path"], workspace)
            if not d.exists():
                return f"Error: not found: {args.get('path', '.')}"
            entries = sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name))
            lines = [f"{e.name}{'/' if e.is_dir() else ''}" for e in entries]
            return "\n".join(lines) if lines else "(empty)"

        if name == "bash":
            cmd = args.get("command", "")
            if not cmd.strip():
                return "Error: empty command"
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                cwd=str(workspace), timeout=60,
            )
            out = result.stdout or ""
            err = result.stderr or ""
            combined = out + (f"\n[stderr]:\n{err}" if err else "")
            if result.returncode != 0:
                combined += f"\n[exit {result.returncode}]"
            return combined or "(no output)"

        return f"Error: unknown tool '{name}'"
    except subprocess.TimeoutExpired:
        return "Error: command timed out (60s)"
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Qwen3 model wrapper (loaded once, shared across tasks)
# ---------------------------------------------------------------------------

class QwenAgent:
    """Wraps a HuggingFace Qwen3 model for agentic multi-turn tool calling."""

    def __init__(self, model_name: str, max_new_tokens: int = 4096, enable_thinking: bool = True):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self._model = None
        self._tokenizer = None
        self.last_prompt_text: str = ""

    def _load(self) -> None:
        if self._model is not None:
            return
        print(f"Loading model: {self.model_name}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        print("Model loaded.")

    def generate(self, messages: List[Dict[str, Any]]) -> Tuple[str, int]:
        """
        Run one forward pass. Returns (raw_text, new_tokens_generated).
        Uses Qwen3 chat template with tools baked in.
        """
        import torch

        self._load()

        if _requires_manual_chat(messages):
            text = _render_manual_chat(messages, _TOOLS)
        else:
            # enable_thinking is a Qwen3-specific kwarg; fall back gracefully.
            # If the tokenizer template rejects tool-result history, fall back to a
            # plain chat rendering with an explicit <tool_call> contract.
            try:
                text = self._tokenizer.apply_chat_template(
                    messages,
                    tools=_TOOLS,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                try:
                    text = self._tokenizer.apply_chat_template(
                        messages,
                        tools=_TOOLS,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                except Exception:
                    text = _render_manual_chat(messages, _TOOLS)
            except Exception:
                text = _render_manual_chat(messages, _TOOLS)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        self.last_prompt_text = text
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = out[0][input_len:]
        new_token_count = len(new_tokens)
        decoded = self._tokenizer.decode(new_tokens, skip_special_tokens=False)
        return decoded, new_token_count

    def run_task(
        self,
        task: Task,
        workspace: Path,
        timeout_seconds: float,
        insights_content: Optional[str],
        max_iterations: int = 30,
    ) -> Dict[str, Any]:
        """Agentic loop: generate → parse tool calls → execute → repeat."""
        system_content = _build_system_prompt(workspace, insights_content)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content},
        ]

        sessions = task.frontmatter.get("sessions", [])
        prompts = []
        if sessions:
            for s in sessions:
                if isinstance(s, str):
                    prompts.append(s)
                elif isinstance(s, dict):
                    prompts.append(s.get("prompt") or s.get("message", ""))
        else:
            prompts = [task.prompt]

        transcript: List[Dict[str, Any]] = []
        total_output_tokens = 0
        total_input_tokens = 0
        start = time.time()
        timed_out = False
        status = "success"
        error_messages: List[str] = []

        for prompt_text in prompts:
            if time.time() - start >= timeout_seconds:
                timed_out = True
                break

            messages.append({"role": "user", "content": prompt_text})

            for _ in range(max_iterations):
                if time.time() - start >= timeout_seconds:
                    timed_out = True
                    break

                try:
                    raw, n_new = self.generate(messages)
                except Exception as exc:
                    status = "error"
                    error_text = f"generation failed: {type(exc).__name__}: {exc}"
                    error_messages.append(error_text)
                    transcript.append({"type": "error", "error": error_text})
                    break

                total_output_tokens += n_new

                thinking_text, reply_text = _split_thinking(raw)
                tool_calls = _parse_tool_calls(reply_text)
                if not tool_calls:
                    tool_calls = _parse_tool_calls(raw)

                # Strip <tool_call> blocks and special tokens from the visible text.
                # The tool calls are represented separately via the tool_calls field
                # in the assistant message — keeping them in content too would make
                # apply_chat_template see duplicates on the next turn.
                visible = _strip_special_tokens(
                    re.sub(r"<tool_call>.*?</tool_call>", "", reply_text, flags=re.DOTALL)
                ).strip()

                # Build content blocks for transcript
                content_blocks: List[Dict[str, Any]] = []
                if thinking_text:
                    content_blocks.append({"type": "thinking", "text": thinking_text})
                if visible:
                    content_blocks.append({"type": "text", "text": visible})
                for tc in tool_calls:
                    content_blocks.append({
                        "type": "toolCall",
                        "name": tc["name"],
                        "id": tc["id"],
                        "arguments": tc["arguments"],
                    })

                transcript.append({
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": content_blocks,
                        "usage": {"output": n_new},
                    },
                })

                # Keep conversation history in plain chat roles. Several HF
                # Qwen templates reject OpenAI-style assistant.tool_calls plus
                # role=tool messages on the next turn, which caused tool-using
                # tasks to stop after one action. Keeping the raw <tool_call>
                # block in assistant content and feeding tool results back as a
                # user message is accepted by all Qwen chat templates we need.
                history_reply = _strip_special_tokens(reply_text).strip()
                messages.append({
                    "role": "assistant",
                    "content": history_reply or visible or "",
                })

                if not tool_calls:
                    # No more tool calls → agent finished
                    break

                # Execute tools
                for tc in tool_calls:
                    result_text = _run_tool(tc["name"], tc["arguments"], workspace)
                    transcript.append({
                        "type": "message",
                        "message": {
                            "role": "toolResult",
                            "toolCallId": tc["id"],
                            "toolName": tc["name"],
                            "content": [{"type": "text", "text": result_text}],
                        },
                    })
                    messages.append(_tool_result_message(tc["name"], tc["id"], result_text))

            if timed_out:
                break

        if timed_out:
            status = "timeout"
        elif not transcript:
            status = "error"

        return {
            "agent_id": self.model_name,
            "task_id": task.task_id,
            "command": f"hf:{self.model_name}",
            "status": status,
            "transcript": transcript,
            "usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "request_count": sum(1 for e in transcript if e.get("type") == "message" and e.get("message", {}).get("role") == "assistant"),
            },
            "workspace": str(workspace),
            "exit_code": 0 if status == "success" else -1,
            "timed_out": timed_out,
            "execution_time": time.time() - start,
            "stdout": "",
            "stderr": "\n".join(error_messages),
        }


# ---------------------------------------------------------------------------
# Prompt and output parsing helpers
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are a capable AI assistant. You are working in workspace: {workspace}
Current local date/time: {now}

Use tools to complete tasks. Tools: read_file, write_file, list_dir, bash.
All file paths are relative to the workspace directory.
When you need a tool, respond with a JSON block inside <tool_call> tags, for example:
<tool_call>{{"name": "write_file", "arguments": {{"path": "answer.txt", "content": "..."}}}}</tool_call>
Do not wrap tool calls in markdown fences. Do not claim a file was created,
edited, read, or verified unless you actually used a tool to do it. For tasks
that ask you to save/create a file, you must call write_file or bash.
When the task is fully done, respond with a plain text summary and call no more tools.
{insights}"""

_INSIGHTS_HEADER = "\n## Insight Library (apply before starting)\n"


def _build_system_prompt(workspace: Path, insights_content: Optional[str]) -> str:
    insights = ""
    if insights_content and insights_content.strip():
        insights = _INSIGHTS_HEADER + insights_content.strip()
    now = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
    return _SYSTEM_TEMPLATE.format(workspace=workspace, now=now, insights=insights)


def _split_thinking(raw: str) -> Tuple[str, str]:
    """Split Qwen3 output into (thinking_text, reply_text)."""
    m = re.match(r"<think>(.*?)</think>(.*)", raw, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", raw


def _strip_special_tokens(text: str) -> str:
    """Remove leftover special tokens like <|im_end|>."""
    return re.sub(r"<\|[^|]+\|>", "", text).strip()


def _parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from Qwen3 <tool_call>...</tool_call> blocks."""
    calls = []
    for i, m in enumerate(re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)):
        raw_json = m.group(1).strip()
        try:
            obj = json.loads(raw_json)
            name = obj.get("name", "unknown")
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            calls.append({"id": f"call_{i}", "name": name, "arguments": args})
        except json.JSONDecodeError:
            calls.append({"id": f"call_{i}", "name": "unknown", "arguments": {"raw": raw_json}})
    if calls:
        return calls

    stripped = _strip_special_tokens(text).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    if not candidates and stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    for raw_json in candidates:
        try:
            obj = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        objects = obj if isinstance(obj, list) else [obj]
        for i, item in enumerate(objects):
            if not isinstance(item, dict) or "name" not in item:
                continue
            name = item.get("name", "unknown")
            args = item.get("arguments", item.get("parameters", {}))
            if not isinstance(args, dict):
                args = {"raw": args}
            calls.append({"id": f"call_{i}", "name": name, "arguments": args})
    return calls


def _write_debug_artifacts(output_dir: str, task: Task, exec_result: Dict[str, Any]) -> None:
    debug_dir = Path(output_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task.task_id)
    payload = {
        "task_id": task.task_id,
        "task_name": task.name,
        "status": exec_result.get("status"),
        "stderr": exec_result.get("stderr", ""),
        "stdout": exec_result.get("stdout", ""),
        "exit_code": exec_result.get("exit_code"),
        "timed_out": exec_result.get("timed_out"),
        "execution_time": exec_result.get("execution_time"),
        "workspace": exec_result.get("workspace"),
        "usage": exec_result.get("usage", {}),
        "transcript": exec_result.get("transcript", []),
    }
    (debug_dir / f"{safe_task_id}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _print_error_details(exec_result: Dict[str, Any], max_chars: int = 4000) -> None:
    parts = []
    stderr = str(exec_result.get("stderr") or "").strip()
    stdout = str(exec_result.get("stdout") or "").strip()
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    for entry in exec_result.get("transcript", []) or []:
        if entry.get("type") == "error":
            parts.append(f"transcript error:\n{entry.get('error')}")
    if not parts:
        return
    detail = "\n\n".join(parts)
    if len(detail) > max_chars:
        detail = detail[:max_chars] + f"\n...[truncated to {max_chars} chars]"
    print("  Detailed error:")
    for line in detail.splitlines():
        print(f"    {line}")


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

def prepare_workspace(skill_dir: Path, task: Task, workspace_root: Path, insights_content: Optional[str]) -> Path:
    workspace = workspace_root / task.task_id
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    for spec in task.workspace_files:
        if "content" in spec:
            dest = workspace / spec["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(spec["content"], encoding="utf-8")
        else:
            src = skill_dir / "assets" / spec["source"]
            dest = workspace / spec["dest"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                dest.write_bytes(src.read_bytes())
            except FileNotFoundError:
                print(f"  Warning: asset not found: {src}")

    if insights_content and insights_content.strip():
        (workspace / "INSIGHTS.md").write_text(insights_content, encoding="utf-8")
        (workspace / "BOOTSTRAP.md").write_text(
            "Your workspace contains `INSIGHTS.md`. Before starting the task, "
            "read it with `read_file`, identify relevant techniques, and apply "
            "them while completing the requested work.\n",
            encoding="utf-8",
        )

    main_skills_dir = Path.home() / ".openclaw" / "workspace" / "skills"
    if main_skills_dir.exists():
        dest_skills_dir = workspace / "skills"
        dest_skills_dir.mkdir(parents=True, exist_ok=True)
        for src_skill_dir in main_skills_dir.iterdir():
            if not src_skill_dir.is_dir():
                continue
            dest_skill_dir = dest_skills_dir / src_skill_dir.name
            if dest_skill_dir.exists():
                shutil.rmtree(dest_skill_dir)
            shutil.copytree(src_skill_dir, dest_skill_dir)

    return workspace


# ---------------------------------------------------------------------------
# Transcript utilities
# ---------------------------------------------------------------------------

def _extract_transcript_text(transcript: List[Dict[str, Any]]) -> str:
    parts = []
    for entry in transcript:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text and block.get("type") in ("text", "thinking"):
                        parts.append(f"[{role}]: {text}")
        elif isinstance(content, str) and role == "assistant":
            parts.append(f"[assistant]: {content}")
    return "\n\n".join(parts)


def _analyze_tool_calls(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    calls = []
    for entry in transcript:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "toolCall":
                calls.append({
                    "name": block.get("name", "unknown"),
                    "call_id": block.get("id"),
                    "status": "ok",
                    "error": None,
                })

    tool_names = sorted({c["name"] for c in calls})
    counter = collections.Counter(c["name"] for c in calls)
    return {
        "tool_names": tool_names,
        "tool_name_counts": dict(sorted(counter.items())),
        "total_tool_calls": len(calls),
        "successful_tool_calls": len(calls),
        "error_tool_calls": 0,
        "unknown_status_tool_calls": 0,
        "calls": calls,
    }


# ---------------------------------------------------------------------------
# Encyclopedia loader
# ---------------------------------------------------------------------------

def _load_insights(encyclopedia_path: str) -> Optional[str]:
    try:
        with open(encyclopedia_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  Warning: failed to load encyclopedia: {exc}")
        return None
    if isinstance(data, dict) and set(data.keys()) == {"insight"}:
        body = data["insight"]
    elif isinstance(data, dict):
        body = "\n".join(f"### {k}\n{v}\n" for k, v in data.items())
    else:
        body = str(data)
    return body


# ---------------------------------------------------------------------------
# Judge (external API — same as openclaw version)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are a strict grading function. "
    "Respond with ONLY a JSON object, no prose, no markdown fences, no extra text."
)


def _call_judge(prompt: str, model: str, timeout: float = 120.0) -> Dict[str, Any]:
    """Route judge call based on model prefix."""
    from lib_agent import call_judge_api
    return call_judge_api(prompt=prompt, model=model, timeout_seconds=timeout)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class QwenPinchBenchPipeline:
    """
    PinchBench pipeline using a local HuggingFace Qwen3 model.
    No OpenClaw, no vLLM server — model runs in-process across all GPUs.
    """

    def __init__(
        self,
        model_name: str,
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
        max_new_tokens: int = 8192,
        max_agent_iterations: int = 30,
    ):
        self.model_name = model_name
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
        self.judge_model = judge_model or "openrouter/anthropic/claude-opus-4.5"
        self.thinking_level = thinking_level
        self.enable_thinking = enable_thinking
        self.max_agent_iterations = max_agent_iterations

        if self.api_provider == "gemini" and self.api_key:
            os.environ["GEMINI_API_KEY"] = self.api_key

        self.skill_dir = Path(pinchbench_dir) if pinchbench_dir else Path(__file__).parent / "pinchbench"
        self.tasks_dir = self.skill_dir / "tasks"
        self.workspace_root = Path(output_dir) / "workspaces"

        os.makedirs(output_dir, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        self.agent = QwenAgent(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )

        self._client: Optional[ChainOfThoughtReader] = None
        self._metrics_cache: Dict[str, Any] = {}

    def _load_tasks(self) -> List[Task]:
        loader = TaskLoader(self.tasks_dir)
        tasks = loader.load_all_tasks()
        if self.suite == "all":
            return tasks
        if self.suite == "automated-only":
            return [t for t in tasks if t.grading_type == "automated"]
        ids = {tid.strip() for tid in self.suite.split(",") if tid.strip()}
        return [t for t in tasks if t.task_id in ids]

    def _ensure_client(self) -> ChainOfThoughtReader:
        if self._client is None:
            self._client = ChainOfThoughtReader(
                use_api=self.use_api, api_key=self.api_key, api_provider=self.api_provider,
            )
            if self.use_api and self.api_provider == "gemini" and self.api_model:
                from utils import setup_gemini
                self._client.gemini_model = setup_gemini(api_key=self.api_key, model_name=self.api_model)
        return self._client

    def _call_for_extraction(self, prompt: str, max_new_tokens: int) -> Tuple[str, Dict]:
        if self.use_api and self.api_provider == "gemini" and self.thinking_level:
            return call_gemini_thinking(
                api_key=self.api_key, model_name=self.api_model,
                prompt=prompt, thinking_level=self.thinking_level, max_new_tokens=max_new_tokens,
            )
        return self._ensure_client()._call_model(prompt, max_new_tokens=max_new_tokens)

    def _extract_insights(self, task_prompt: str, agent_response: str) -> Dict[str, Any]:
        client = self._ensure_client()

        print("  Step 2: Reflecting...")
        reflection, _ = self._call_for_extraction(
            client._get_reflection_prompt(task_prompt, agent_response), max_new_tokens=4096
        )

        print("  Step 3: Extracting insights...")
        extraction_response, token_info = self._call_for_extraction(
            client._get_behavior_prompt(task_prompt, agent_response, reflection), max_new_tokens=8192
        )

        insights: Dict[str, str] = {}
        try:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", extraction_response, re.DOTALL)
            json_str = m.group(1) if m else None
            if not json_str:
                start = extraction_response.find("{")
                if start != -1:
                    depth = 0
                    in_str = esc = False
                    for i, ch in enumerate(extraction_response[start:], start):
                        if esc:
                            esc = False; continue
                        if ch == "\\": esc = True; continue
                        if ch == '"': in_str = not in_str; continue
                        if not in_str:
                            if ch == "{": depth += 1
                            elif ch == "}":
                                depth -= 1
                                if depth == 0:
                                    json_str = extraction_response[start: i + 1]; break
            if json_str:
                json_str = re.sub(r",\s*}", "}", json_str)
                raw = json.loads(json_str)
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        name = k if k.startswith("insight_") else f"insight_{k}"
                        desc = re.sub(r"\s+", " ", str(v)).strip()
                        if len(desc) >= 20:
                            insights[name] = desc
        except Exception as exc:
            print(f"  Warning: insight parse error: {exc}")

        print(f"  Extracted {len(insights)} insights")
        return {"insight_book": insights, "output_tokens": token_info.get("output_tokens", 0)}

    def _grade(self, task: Task, exec_result: Dict[str, Any]):
        return grade_task(
            task=task,
            execution_result=exec_result,
            skill_dir=self.skill_dir,
            judge_model=self.judge_model,
            judge_backend="api",
        )

    def _metrics_log_path(self) -> Path:
        return Path(self.output_dir) / "metrics_log.json"

    def _write_metrics_log(self, task_metrics: List[Dict[str, Any]], *, lib_tokens: Optional[int] = None) -> None:
        if lib_tokens is None:
            lib_tokens = self._metrics_cache.get("summary", {}).get("library_output_tokens")

        graded = [t for t in task_metrics if t.get("grade", {}).get("score") is not None]
        s = sum(float(t["grade"].get("score", 0.0)) for t in graded)
        m = sum(float(t["grade"].get("max_score", 0.0)) for t in graded)

        tool_counts: Dict[str, int] = {}
        for t in task_metrics:
            for name, count in (t.get("tools", {}).get("name_counts", {}) or {}).items():
                tool_counts[name] = tool_counts.get(name, 0) + int(count)

        payload = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": self.model_name,
            "judge_model": self.judge_model,
            "suite": self.suite,
            "summary": {
                "tasks_total": len(task_metrics),
                "tasks_graded": len(graded),
                "overall_accuracy_pct": (s / m * 100.0) if m > 0 else None,
                "tool_names": sorted(tool_counts.keys()),
                "tool_name_counts": dict(sorted(tool_counts.items())),
                "total_tool_calls": sum(int(t.get("tools", {}).get("total_calls", 0) or 0) for t in task_metrics),
                "output_tokens_agent_total": sum(int((t.get("output_tokens") or {}).get("agent", 0) or 0) for t in task_metrics),
                "output_tokens_extraction_total": sum(int((t.get("output_tokens") or {}).get("extraction", 0) or 0) for t in task_metrics),
                "execution_time_total_seconds": sum(float(t.get("execution_time_seconds", 0.0) or 0.0) for t in task_metrics),
                "library_output_tokens": int(lib_tokens) if lib_tokens is not None else None,
            },
            "tasks": task_metrics,
        }
        self._metrics_log_path().write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self._metrics_cache = payload

    def run_tasks_and_extract(self) -> List[Dict[str, Any]]:
        insights_content: Optional[str] = None
        if self.encyclopedia_path and Path(self.encyclopedia_path).exists():
            print(f"\nLoading encyclopedia: {self.encyclopedia_path}")
            insights_content = _load_insights(self.encyclopedia_path)
        else:
            print("\nNo encyclopedia — running without prior insights")

        tasks = self._load_tasks()
        if not tasks:
            print(f"No tasks found in {self.tasks_dir}")
            return []

        # Load model before starting tasks (only once)
        self.agent._load()

        print(f"\nLoaded {len(tasks)} tasks (suite='{self.suite}')")
        print("=" * 80)

        results = []
        task_metrics: List[Dict[str, Any]] = []
        task_counter = 0

        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] {task.task_id} — {task.name}")
            print("-" * 60)

            workspace = prepare_workspace(self.skill_dir, task, self.workspace_root, insights_content)
            timeout = task.timeout_seconds * self.timeout_multiplier

            try:
                exec_result = self.agent.run_task(
                    task=task,
                    workspace=workspace,
                    timeout_seconds=timeout,
                    insights_content=insights_content,
                    max_iterations=self.max_agent_iterations,
                )
            except Exception as exc:
                print(f"  Error: {exc}")
                task_metrics.append({
                    "task_id": task.task_id, "task_name": task.name,
                    "execution_status": "error", "execution_time_seconds": None,
                    "grade": {"score": None, "max_score": None, "accuracy_pct": None, "grading_type": task.grading_type},
                    "output_tokens": {"agent": 0, "extraction": 0, "total": 0},
                    "tools": {"names": [], "name_counts": {}, "total_calls": 0, "successful_calls": 0, "error_calls": 0, "unknown_status_calls": 0, "calls": []},
                    "insights_extracted": 0, "error": str(exc),
                })
                self._write_metrics_log(task_metrics)
                continue

            status = exec_result["status"]
            transcript = exec_result["transcript"]
            print(f"  Status: {status} | entries: {len(transcript)}")
            if status != "success":
                _print_error_details(exec_result)
            _write_debug_artifacts(self.output_dir, task, exec_result)

            try:
                grade = self._grade(task, exec_result)
                pct = grade.score / grade.max_score * 100 if grade.max_score > 0 else 0
                print(f"  Grade: {grade.score:.2f}/{grade.max_score:.2f} ({pct:.0f}%) [{grade.grading_type}]")
            except Exception as exc:
                print(f"  Warning: grading failed: {exc}")
                grade = None

            usage = exec_result.get("usage", {}) or {}
            agent_tokens = int(usage.get("output_tokens", 0) or 0)
            tool_stats = _analyze_tool_calls(transcript)

            task_metric: Dict[str, Any] = {
                "task_id": task.task_id, "task_name": task.name,
                "execution_status": status,
                "execution_time_seconds": exec_result.get("execution_time"),
                "grade": {
                    "score": grade.score if grade else None,
                    "max_score": grade.max_score if grade else None,
                    "accuracy_pct": (grade.score / grade.max_score * 100.0) if grade and grade.max_score > 0 else None,
                    "grading_type": grade.grading_type if grade else task.grading_type,
                    "notes": grade.notes if grade else "",
                },
                "output_tokens": {"agent": agent_tokens, "extraction": 0, "total": agent_tokens},
                "tools": {
                    "names": tool_stats["tool_names"],
                    "name_counts": tool_stats["tool_name_counts"],
                    "total_calls": tool_stats["total_tool_calls"],
                    "successful_calls": tool_stats["successful_tool_calls"],
                    "error_calls": tool_stats["error_tool_calls"],
                    "unknown_status_calls": tool_stats["unknown_status_tool_calls"],
                    "calls": tool_stats["calls"],
                },
                "insights_extracted": 0,
            }
            if exec_result.get("stderr"):
                task_metric["error"] = str(exec_result["stderr"])

            agent_response = _extract_transcript_text(transcript)
            if not transcript or not agent_response.strip():
                task_metrics.append(task_metric)
                self._write_metrics_log(task_metrics)
                continue

            try:
                extraction = self._extract_insights(task.prompt, agent_response)
            except Exception as exc:
                print(f"  Insight extraction error: {exc}")
                task_metric["extraction_error"] = str(exc)
                task_metrics.append(task_metric)
                self._write_metrics_log(task_metrics)
                continue

            insight_book = extraction.get("insight_book", {}) or {}
            extract_tokens = int(extraction.get("output_tokens", 0) or 0)
            task_metric["output_tokens"] = {"agent": agent_tokens, "extraction": extract_tokens, "total": agent_tokens + extract_tokens}
            task_metric["insights_extracted"] = len(insight_book)

            task_counter += 1
            out_file = os.path.join(self.output_dir, f"problem_{task_counter:04d}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({
                    "task_id": task.task_id, "task_name": task.name,
                    "task_prompt": task.prompt, "execution_status": status,
                    "output_tokens": extract_tokens,
                    "grade": {"score": grade.score if grade else None, "max_score": grade.max_score if grade else None,
                              "grading_type": grade.grading_type if grade else task.grading_type, "notes": grade.notes if grade else ""},
                    "insight_book": insight_book,
                }, f, indent=2, ensure_ascii=False)

            print(f"  Saved {len(insight_book)} insights → {out_file}")
            results.append({"task_id": task.task_id, "status": status,
                            "score": grade.score if grade else None, "insights_extracted": len(insight_book)})
            task_metrics.append(task_metric)
            self._write_metrics_log(task_metrics)

        print("\n" + "=" * 80)
        graded = [t for t in task_metrics if t.get("grade", {}).get("score") is not None]
        if graded:
            s = sum(float(t["grade"].get("score", 0.0)) for t in graded)
            m = sum(float(t["grade"].get("max_score", 0.0)) for t in graded)
            print(f"Overall: {s/m*100:.1f}% ({len(graded)} tasks graded)")
        self._write_metrics_log(task_metrics)
        return results

    def aggregate_insights(self) -> Optional[str]:
        print("\n" + "=" * 80 + "\nAggregating Insights\n" + "=" * 80)
        json_files = sorted(Path(self.output_dir).glob("problem_*.json"))
        if not json_files:
            print("No problem_*.json files found")
            return None

        server = TextBasedInsightAggregationServer(
            use_api=self.use_api, api_key=self.api_key,
            api_provider=self.api_provider, input_dirs=[self.output_dir],
        )
        result = server.aggregate_and_build_encyclopedia(
            json_files=[str(f) for f in json_files], output_dir=self.output_dir,
        )

        enc_path = os.path.join(self.output_dir, "encyclopedia.json")
        enc_dict = server._try_parse_json(server.encyclopedia)
        if enc_dict is None:
            enc_dict = server._try_parse_json(server._extract_json_only(server.encyclopedia))

        if enc_dict is None:
            enc_path = os.path.join(self.output_dir, "encyclopedia.txt")
            Path(enc_path).write_text(server.encyclopedia, encoding="utf-8")
            print(f"Encyclopedia saved as plain text: {enc_path}")
        else:
            with open(enc_path, "w", encoding="utf-8") as f:
                json.dump(enc_dict, f, indent=2, ensure_ascii=False)
            print(f"Encyclopedia saved: {enc_path}")

        lib_tokens = result.get("total_output_tokens", 0)
        self._write_metrics_log(self._metrics_cache.get("tasks", []), lib_tokens=int(lib_tokens or 0))
        return enc_path

    def run_pipeline(self, iterations: int = 1, start_from_step2: bool = False) -> None:
        t0 = time.time()
        base_dir = self.output_dir
        current_enc: Optional[str] = self.encyclopedia_path

        for iteration in range(1, iterations + 1):
            print(f"\n{'='*80}\nIteration {iteration}/{iterations}\n{'='*80}")

            if iterations > 1:
                iter_dir = os.path.join(base_dir, f"iter_{iteration:02d}")
                os.makedirs(iter_dir, exist_ok=True)
                self.output_dir = iter_dir
                self.workspace_root = Path(iter_dir) / "workspaces"
                self.workspace_root.mkdir(parents=True, exist_ok=True)

            if current_enc:
                self.encyclopedia_path = current_enc

            if not start_from_step2 or iteration > 1:
                self.run_tasks_and_extract()
            else:
                print("Skipping task execution (start_from_step2=True)")

            enc = self.aggregate_insights()
            if enc:
                current_enc = enc

            if iterations > 1:
                self.output_dir = base_dir

        elapsed = time.time() - t0
        print(f"\n{'='*80}\nPipeline Complete — {elapsed:.1f}s")
        if current_enc:
            print(f"Final encyclopedia: {current_enc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Qwen PinchBench — local HuggingFace model, no OpenClaw, no vLLM")
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct",
                   help="HuggingFace model ID (default: Qwen/Qwen3-30B-A3B-Instruct)")
    p.add_argument("--output-dir", default="qwen_pinchbench_output")
    p.add_argument("--suite", default="all",
                   help='"all", "automated-only", or comma-separated task IDs')
    p.add_argument("--judge", default=None,
                   help="Judge model (e.g. google/gemini-2.5-pro-preview via OpenRouter). "
                        "Default: openrouter/anthropic/claude-opus-4.5")
    p.add_argument("--pinchbench-dir", default=None)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--encyclopedia", default=None, help="Pre-existing encyclopedia.json")
    p.add_argument("--start-from-step2", action="store_true")
    p.add_argument("--use-api", action="store_true",
                   help="Use external API (Gemini/OpenRouter) for insight extraction")
    p.add_argument("--api-provider", default="gemini", choices=["gemini", "openrouter"])
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-model", default="gemini-3-pro-preview")
    p.add_argument("--thinking-level", default="high", choices=["low", "medium", "high"])
    p.add_argument("--no-thinking", action="store_true", help="Disable Qwen3 thinking mode")
    p.add_argument("--timeout-multiplier", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=8192, help="Max new tokens per agent turn")
    p.add_argument("--max-agent-iterations", type=int, default=30, help="Max tool-call iterations per task")

    args = p.parse_args()

    pipeline = QwenPinchBenchPipeline(
        model_name=args.model,
        output_dir=args.output_dir,
        suite=args.suite,
        pinchbench_dir=args.pinchbench_dir,
        use_api=args.use_api,
        api_key=args.api_key,
        api_provider=args.api_provider,
        api_model=args.api_model,
        timeout_multiplier=args.timeout_multiplier,
        encyclopedia_path=args.encyclopedia,
        judge_model=args.judge or "openrouter/anthropic/claude-opus-4.5",
        thinking_level=args.thinking_level if args.use_api else None,
        enable_thinking=not args.no_thinking,
        max_new_tokens=args.max_new_tokens,
        max_agent_iterations=args.max_agent_iterations,
    )

    pipeline.run_pipeline(iterations=args.iterations, start_from_step2=args.start_from_step2)


if __name__ == "__main__":
    main()
