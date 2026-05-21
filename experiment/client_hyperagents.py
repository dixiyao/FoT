"""
DGM-HyperAgents Client
Implements the DGM-HyperAgents (Digitally Generated Meta-Agents) framework from:
"HyperAgents: Metacognitive Self-Modification with Language Model Agents" (2603.19461)

Instead of modifying source code files (as in the original paper), this adaptation
evolves prompt templates for problem-solving. The "hyperagent" is the combination of
a task prompt + meta prompt, both of which the meta agent can modify.

Key concepts:
- Task agent: the evolved problem-solving prompt template
- Meta agent: the prompt-optimizer that analyzes task agent performance and proposes
  improvements to the task prompt AND (metacognitively) to its own meta prompt
- Archive: open-ended pool of (task_prompt, meta_prompt) variants with fitness scores
- Parent selection: score-child-prop (paper Appendix A.2 + real repo gl_utils.py)
  sigmoid fitness × exp(-(n_children/8)^3) novelty penalty
- Insights: the best evolved task prompts, stored as reusable encyclopedia entries

Paper defaults:
  iterations = 100  (paper review + robotics)
  lambda_     = 10  (sigmoid sharpness, Appendix A.2)
  top_k_mid   = 3   (agents used to compute alpha_mid)
"""

import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils import (
    HAS_GEMINI,
    check_cuda,
    setup_gemini,
    call_gemini,
    call_openrouter,
    load_hf_model,
    call_hf_model,
)


# ---------------------------------------------------------------------------
# Exact prompts from the paper
# ---------------------------------------------------------------------------

# Appendix A.1 — initial task agent prompt (adapted for text-based problem solving)
_INITIAL_TASK_PROMPT = """\
You are an expert problem solver.

Task input:
'''
{problem}
'''

Solve this problem carefully, step by step. Be thorough, systematic, and precise.

Before finalizing, verify your solution against the problem requirements.

Respond with your complete solution.\
"""

# Appendix A.1 — initial meta agent prompt
# In the original paper this is "Modify any part of the codebase at '{repo_path}'."
# Adapted here to prompt optimization: the meta agent modifies the task prompt text
# (and optionally its own meta prompt) rather than source files.
_INITIAL_META_PROMPT = """\
You are a meta-agent. Your role is to improve problem-solving prompts.

You will be given:
1. The current task agent prompt template
2. A problem that was attempted using that prompt
3. The agent's response
4. A self-evaluation score (0.0 = poor, 1.0 = excellent)

Propose ONE specific, concrete improvement to the task agent prompt that would make
the agent perform better on similar problems.

You may also modify this meta prompt itself if you believe a change here would lead
to better future improvements (metacognitive self-modification).

Output ONLY a valid JSON object (no markdown, no explanation outside the JSON):
{
  "log_summarization": "<brief summary of what the agent did and where it struggled>",
  "potential_improvements": "<what aspects of the current task prompt could be improved>",
  "improvement_proposal": "<the ONE specific improvement you propose and why>",
  "new_task_prompt": "<complete new task agent prompt with the improvement applied>",
  "new_meta_prompt": <null or the complete new meta prompt if you want to update it>
}\
"""

# Self-evaluation prompt — ask the model to rate its own solution
_SELF_EVAL_PROMPT = """\
Problem:
{problem}

Proposed solution:
{solution}

Rate the quality of this solution on a scale from 0.0 to 1.0, where:
  0.0 = completely wrong or missing
  0.5 = partially correct but with significant errors or gaps
  1.0 = fully correct, complete, and well-reasoned

Consider: correctness, completeness, clarity of reasoning, and absence of errors.

Output ONLY a single floating-point number between 0.0 and 1.0, nothing else.\
"""

# Insight extraction prompt — convert best evolved prompts into encyclopedia entries
_INSIGHT_FROM_PROMPT = """\
You are analyzing an evolved problem-solving prompt template that has been optimized
through iterative self-improvement.

Evolved task prompt:
{task_prompt}

Performance score: {score:.3f}

Extract the key techniques and strategies embedded in this prompt that make it
effective. Format these as reusable insights for similar problems.

Output ONLY a valid JSON object:
{{"insight_hyperagent_<name>": "<description of what this prompt teaches and why it works>"}}

Use at most 3 insight entries. Each description must be at least 40 characters.\
"""


# ---------------------------------------------------------------------------
# Helper: parent selection (score-child-prop, Appendix A.2)
# ---------------------------------------------------------------------------

def _score_child_prop(
    archive: List[Dict],
    lambda_: float = 10.0,
    top_k_mid: int = 3,
) -> List[float]:
    """
    Compute normalized selection probabilities for each archive entry.

    Matches utils/gl_utils.py from facebookresearch/HyperAgents exactly:

      mid_point = np.mean(sorted(scores, reverse=True)[:top_k_mid])
      s_i  = sigmoid(lambda_ * (alpha_i - mid_point))
      h_i  = exp(-(n_children_i / 8) ** 3)   [novelty/child penalty]
      p_i  ∝  s_i * h_i

    The h_i penalty comes directly from gl_utils.py:
      penalties = [math.exp(-(child_counts[commit]/8)**3) for commit in commits]
    """
    scores = [a["score"] for a in archive]
    # mid_point: mean of top-k_mid scores (real repo uses np.mean)
    mid_point = float(np.mean(sorted(scores, reverse=True)[:top_k_mid]))

    # sigmoid fitness scores
    sig_scores = [1.0 / (1.0 + math.exp(-lambda_ * (s - mid_point))) for s in scores]

    # child-count penalties: exp(-(n/8)^3)  — from real repo's gl_utils.py
    penalties = [math.exp(-(a["n_children"] / 8) ** 3) for a in archive]

    combined = [s * p for s, p in zip(sig_scores, penalties)]
    total = sum(combined)
    if total == 0:
        return [1.0 / len(archive)] * len(archive)
    return [w / total for w in combined]


def _sample_parent(archive: List[Dict], lambda_: float = 10.0, top_k_mid: int = 3) -> int:
    """
    Sample a parent index from the archive using score-child-prop.

    Uses random.choices() matching the real repo's:
      random.choices(commits, weights=probabilities)[0]
    """
    probs = _score_child_prop(archive, lambda_, top_k_mid)
    indices = list(range(len(archive)))
    return random.choices(indices, weights=probs, k=1)[0]


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class HyperAgentsClient:
    """
    DGM-HyperAgents client for problem-solving with metacognitive prompt evolution.

    Each call to solve_problem():
      1. Selects a parent from the archive (score-child-prop)
      2. Solves the problem using the parent's task prompt
      3. Self-evaluates the solution
      4. Runs the parent's meta agent → proposes new (task_prompt, meta_prompt)
      5. Adds the new hyperagent to the archive
      6. Returns solution + insight_book containing evolved prompts

    After processing many problems, the archive's best prompts are surfaced as
    reusable insights (encyclopedia entries) compatible with server_text.py.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1",
        task: Optional[str] = None,
        device: Optional[str] = None,
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
        output_dir: str = "output",
        load_in_8bit: bool = False,
        # HyperAgents-specific
        iterations: int = 100,     # paper default for paper-review / robotics tasks
        lambda_: float = 10.0,     # sigmoid sharpness (Appendix A.2)
        top_k_mid: int = 3,        # agents used to compute alpha_mid
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.task = task or "Solve the given problem step by step."
        self.load_in_8bit = load_in_8bit
        self.insight_book: Dict[str, str] = {}

        # HyperAgents hyperparameters
        self.iterations = iterations
        self.lambda_ = lambda_
        self.top_k_mid = top_k_mid

        # Archive of hyperagents: list of dicts
        # {task_prompt, meta_prompt, score, n_children, parent_id, iteration}
        self.archive: List[Dict] = []

        # Seed the archive with the initial (paper Appendix A.1) hyperagent
        self.archive.append({
            "task_prompt": _INITIAL_TASK_PROMPT,
            "meta_prompt": _INITIAL_META_PROMPT,
            "score": 0.5,      # neutral prior
            "n_children": 0,
            "parent_id": -1,
            "iteration": 0,
        })

        # Encyclopedia support (for loading external insights)
        self.encyclopedia = ""
        self.encyclopedia_dict: Dict[str, str] = {}
        self.encyclopedia_loaded = False

        # API support
        self.use_api = use_api
        self.api_provider = api_provider
        self.api_key = api_key or (os.getenv("GEMINI_API_KEY") if api_provider == "gemini" else os.getenv("OPENROUTER_API_KEY"))
        self.api_model_name = None  # set externally if needed
        if self.use_api and self.api_provider == "gemini":
            self.gemini_model = setup_gemini(
                api_key=self.api_key, model_name="gemini-3-pro-preview"
            )

        # HuggingFace model (lazy)
        self.model = None
        self.tokenizer = None
        self.device = device or ("cuda" if check_cuda() else "cpu")

        # Track which iteration we're on
        self._iteration_counter = 0

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        if self.model is not None and self.tokenizer is not None:
            return
        self.model, self.tokenizer = load_hf_model(
            self.model_name, self.device, self.load_in_8bit
        )

    def _call_model(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict]:
        if self.use_api:
            if self.api_provider == "openrouter":
                model = getattr(self, "api_model_name", None) or "anthropic/claude-opus-4.6"
                return call_openrouter(self.api_key, model, prompt, system_prompt, max_new_tokens)
            return call_gemini(self.gemini_model, prompt, system_prompt, max_new_tokens)
        self._load_model()
        return call_hf_model(
            self.model, self.tokenizer, self.model_name,
            prompt, system_prompt, max_new_tokens, self.device,
        )

    # ------------------------------------------------------------------
    # Core steps
    # ------------------------------------------------------------------

    def _step_solve(self, problem: str, task_prompt: str) -> Tuple[str, Dict]:
        """Run the task agent: fill task_prompt template and call model."""
        filled = task_prompt.format(problem=problem)

        # Prepend encyclopedia insights if loaded
        if self.encyclopedia_loaded and self.encyclopedia_dict:
            enc_section = (
                "## Learned Insights (from prior experience)\n"
                + json.dumps(self.encyclopedia_dict, indent=2)
                + "\n\n"
            )
            filled = enc_section + filled

        response, token_info = self._call_model(filled, max_new_tokens=32768)
        print(f"[HyperAgents] Task agent response: {response[:200]}...")
        return response, token_info

    def _step_self_eval(self, problem: str, solution: str) -> float:
        """Ask the model to score its own solution (0.0 – 1.0)."""
        prompt = _SELF_EVAL_PROMPT.format(problem=problem, solution=solution)
        response, _ = self._call_model(prompt, max_new_tokens=16)
        response = response.strip()
        # Extract first float from response
        match = re.search(r"([0-9]+\.?[0-9]*)", response)
        if match:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))
        print(f"[HyperAgents] Could not parse self-eval score from: {response!r}, defaulting to 0.5")
        return 0.5

    def _step_meta(
        self,
        problem: str,
        solution: str,
        score: float,
        parent: Dict,
    ) -> Dict:
        """
        Run the meta agent (Appendix A.1 + B).

        The meta agent analyzes:
          - the current task_prompt
          - the problem + response
          - the self-eval score
        and proposes ONE improvement, outputting:
          new_task_prompt + optionally new_meta_prompt
        """
        meta_prompt_filled = parent["meta_prompt"]  # meta prompt is its own template

        # Build the meta-agent user message.
        # Real repo MetaAgent.forward() calls:
        #   chat_with_agent("Modify any part of the codebase at '{repo_path}'.")
        # Adapted here to prompt evolution: the meta agent modifies the task prompt
        # (and optionally its own meta prompt) rather than actual source files.
        user_msg = (
            "Modify any part of the task agent prompt at the following location.\n\n"
            "Current task agent prompt:\n"
            f"'''\n{parent['task_prompt']}\n'''\n\n"
            f"Problem attempted:\n'''\n{problem}\n'''\n\n"
            f"Agent's response:\n'''\n{solution}\n'''\n\n"
            f"Self-evaluation score: {score:.3f}\n\n"
            "Now produce your improvement proposal as JSON."
        )

        response, _ = self._call_model(
            user_msg,
            system_prompt=meta_prompt_filled,
            max_new_tokens=4096,
        )
        print(f"[HyperAgents] Meta agent raw output: {response[:300]}...")

        # Parse JSON from meta agent output
        parsed = self._parse_json(response)
        return parsed

    def _parse_json(self, text: str) -> Dict:
        """Extract JSON object from model output, with fallback."""
        # Try markdown code block first
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            start = text.find("{")
            if start == -1:
                return {}
            # Find matching closing brace
            brace_count = 0
            in_string = False
            escape_next = False
            end = start
            for i in range(start, len(text)):
                c = text[i]
                if escape_next:
                    escape_next = False
                    continue
                if c == "\\":
                    escape_next = True
                    continue
                if c == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if c == "{":
                        brace_count += 1
                    elif c == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end = i
                            break
            json_str = text[start:end + 1]

        try:
            json_str = re.sub(r",\s*}", "}", json_str)
            json_str = re.sub(r",\s*]", "]", json_str)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[HyperAgents] JSON parse error: {e}")
            return {}

    def _extract_insights_from_prompt(
        self, task_prompt: str, score: float
    ) -> Dict[str, str]:
        """Convert a high-scoring evolved prompt into encyclopedia insight entries."""
        prompt = _INSIGHT_FROM_PROMPT.format(task_prompt=task_prompt, score=score)
        response, _ = self._call_model(prompt, max_new_tokens=2048)
        parsed = self._parse_json(response)
        insights = {}
        for k, v in parsed.items():
            name = k if k.startswith("insight_") else f"insight_{k}"
            desc = str(v).strip()
            if len(desc) >= 40:
                insights[name] = desc
        return insights

    # ------------------------------------------------------------------
    # Public interface (drop-in for client.py)
    # ------------------------------------------------------------------

    def solve_problem(
        self,
        task: Optional[str] = None,
        custom_solution_instruction: Optional[str] = None,
        insights_section: Optional[str] = None,
    ) -> Dict:
        """
        Solve a problem using DGM-HyperAgents evolution.

        One iteration of the DGM-H loop (Algorithm 1, paper §3):
          1. Select parent p_i from archive via score-child-prop
          2. Solve problem with p_i's task_prompt  (task agent)
          3. Self-evaluate the solution → alpha
          4. Run p_i's meta_prompt on (task_prompt, problem, solution, alpha)
             → new_task_prompt [+ new_meta_prompt]  (meta agent)
          5. Add new hyperagent to archive with score alpha
          6. Increment p_i.n_children

        Returns a dict with the same shape as client.py's solve_problem().
        """
        if task is not None:
            self.task = task
        problem = self.task

        self._iteration_counter += 1
        print(f"\n[HyperAgents] Iteration {self._iteration_counter} | Archive size: {len(self.archive)}")
        print(f"Problem: {problem}\n")

        self.insight_book = {}

        # Step 1 — select parent
        parent_idx = _sample_parent(self.archive, self.lambda_, self.top_k_mid)
        parent = self.archive[parent_idx]
        print(f"[HyperAgents] Selected parent {parent_idx} (score={parent['score']:.3f}, children={parent['n_children']})")

        # Step 2 — solve
        print("[HyperAgents] Step 1: Task agent solving problem...")
        solution, token_info = self._step_solve(problem, parent["task_prompt"])
        time.sleep(0.5)

        # Step 3 — self-evaluate
        print("[HyperAgents] Step 2: Self-evaluating solution...")
        alpha = self._step_self_eval(problem, solution)
        print(f"[HyperAgents] Self-eval score: {alpha:.3f}")
        time.sleep(0.5)

        # Step 4 — meta agent: propose improved hyperagent
        print("[HyperAgents] Step 3: Meta agent proposing improvements...")
        meta_output = self._step_meta(problem, solution, alpha, parent)
        time.sleep(0.5)

        new_task_prompt = meta_output.get("new_task_prompt") or parent["task_prompt"]
        new_meta_prompt_raw = meta_output.get("new_meta_prompt")
        new_meta_prompt = (
            new_meta_prompt_raw
            if isinstance(new_meta_prompt_raw, str) and len(new_meta_prompt_raw.strip()) > 20
            else parent["meta_prompt"]
        )

        # Step 5 — add new hyperagent to archive
        new_agent = {
            "task_prompt": new_task_prompt,
            "meta_prompt": new_meta_prompt,
            "score": alpha,
            "n_children": 0,
            "parent_id": parent_idx,
            "iteration": self._iteration_counter,
        }
        self.archive.append(new_agent)

        # Step 6 — increment parent's child count
        self.archive[parent_idx]["n_children"] += 1

        # Extract insights from this new agent if score is good
        if alpha >= 0.6:
            evolved_insights = self._extract_insights_from_prompt(new_task_prompt, alpha)
            self.insight_book.update(evolved_insights)

        # Also surface the best archive agent's prompt as a top-level insight
        best = max(self.archive, key=lambda a: a["score"])
        if best["score"] >= 0.7 and best["task_prompt"] != _INITIAL_TASK_PROMPT:
            key = f"insight_hyperagent_best_prompt_{len(self.insight_book):04d}"
            self.insight_book[key] = (
                f"Evolved task prompt (score={best['score']:.3f}): {best['task_prompt'][:500]}"
            )

        result = {
            "problem": problem,
            "task": self.task,
            "solution": solution,
            "reflection": meta_output.get("log_summarization", ""),
            "skills_extracted": self.insight_book,
            "skills_used": list(self.insight_book.keys()),
            "validation_errors": [],
            "insight_book": self.insight_book,
            "total_steps": 3,
            "token_info": token_info,
            # HyperAgents-specific fields
            "self_eval_score": alpha,
            "parent_idx": parent_idx,
            "archive_size": len(self.archive),
            "meta_proposal": meta_output.get("improvement_proposal", ""),
        }

        print(f"[HyperAgents] Done. Score={alpha:.3f}, archive size={len(self.archive)}, insights={len(self.insight_book)}")
        return result

    # ------------------------------------------------------------------
    # Encyclopedia / persistence (matching client.py interface)
    # ------------------------------------------------------------------

    def load_encyclopedia(self, encyclopedia_path: str, mode: str = "text"):
        """Load a single encyclopedia file."""
        try:
            if mode == "text":
                with open(encyclopedia_path, "r", encoding="utf-8") as f:
                    self.encyclopedia_dict = json.load(f)
                self.encyclopedia = json.dumps(self.encyclopedia_dict, indent=2)
                print(f"Loaded encyclopedia.json from {encyclopedia_path} ({len(self.encyclopedia_dict)} insights)")
            else:
                with open(encyclopedia_path, "r", encoding="utf-8") as f:
                    self.encyclopedia = f.read()
                print(f"Loaded encyclopedia from {encyclopedia_path} ({len(self.encyclopedia)} characters)")
            self.encyclopedia_loaded = True
        except Exception as e:
            raise FileNotFoundError(f"Failed to load encyclopedia from {encyclopedia_path}: {e}")

    def load_encyclopedias(self, encyclopedia_paths: List[str], mode: str = "text"):
        """Load and merge multiple encyclopedias (matches client.py interface exactly)."""
        if not encyclopedia_paths:
            print("No encyclopedias provided to load.")
            return

        used: List[str] = []
        merged_dict: Dict[str, str] = {}
        merged_text_parts: List[str] = []
        canonical_entries: Dict[str, List[Dict[str, str]]] = {}
        skipped_exact_dupes: int = 0
        collision_variants_added: int = 0

        for ep in encyclopedia_paths:
            try:
                if not ep or not os.path.exists(ep):
                    continue
                if mode == "text":
                    with open(ep, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        for k, v in data.items():
                            name = k if isinstance(k, str) else str(k)
                            desc = v if isinstance(v, str) else str(v)
                            cname = re.sub(r"\s+", " ", name.strip().lower())
                            entries = canonical_entries.get(cname, [])
                            if any(e.get("desc", "") == desc for e in entries):
                                skipped_exact_dupes += 1
                                continue
                            idx = len(entries) + 1
                            new_name = f"{name}_{idx}"
                            while new_name in merged_dict:
                                idx += 1
                                new_name = f"{name}_{idx}"
                            merged_dict[new_name] = desc
                            entries.append({"name": new_name, "desc": desc})
                            canonical_entries[cname] = entries
                            if idx > 1:
                                collision_variants_added += 1
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                name = item.get("name") or item.get("insight")
                                desc = item.get("description") or item.get("desc") or ""
                                if name:
                                    cname = re.sub(r"\s+", " ", name.strip().lower())
                                    entries = canonical_entries.get(cname, [])
                                    if any(e.get("desc", "") == desc for e in entries):
                                        skipped_exact_dupes += 1
                                        continue
                                    idx = len(entries) + 1
                                    new_name = f"{name}_{idx}"
                                    while new_name in merged_dict:
                                        idx += 1
                                        new_name = f"{name}_{idx}"
                                    merged_dict[new_name] = desc
                                    entries.append({"name": new_name, "desc": desc})
                                    canonical_entries[cname] = entries
                                    if idx > 1:
                                        collision_variants_added += 1
                else:
                    with open(ep, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    if text:
                        merged_text_parts.append(text)
                used.append(ep)
            except Exception as e:
                print(f"Warning: failed to load encyclopedia {ep}: {e}")

        if mode == "text":
            self.encyclopedia_dict = merged_dict
            self.encyclopedia = json.dumps(self.encyclopedia_dict, indent=2)
            print(
                f"Loaded {len(used)} encyclopedias (JSON), total insights {len(self.encyclopedia_dict)} "
                f"(skipped {skipped_exact_dupes} exact duplicates, added {collision_variants_added} collision variants)"
            )
        else:
            self.encyclopedia = "\n\n".join(merged_text_parts)
            self.encyclopedia_dict = {}
            print(f"Loaded {len(used)} encyclopedias (text), total chars {len(self.encyclopedia)}")
        self.encyclopedia_loaded = True

    def save_reasoning(self, reasoning_result: Dict, output_path: Optional[str] = None):
        """Save insight book as simple JSON: {"insight_name": "description"} (matches client.py)."""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        insight_book = reasoning_result.get("insight_book", {})
        if not insight_book:
            print("No skills to save")
            return

        if output_path is None:
            safe_name = re.sub(r"[^\w\s-]", "", reasoning_result.get("problem", "hyperagents")[:50])
            safe_name = re.sub(r"[-\s]+", "_", safe_name)
            output_path = str(output_dir / f"{safe_name}.json")
        else:
            if not os.path.isabs(output_path):
                output_path = str(output_dir / output_path)
            if not output_path.endswith(".json"):
                output_path += ".json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(insight_book, f, indent=2, ensure_ascii=False)
        print(f"Saved insight book to: {output_path}")

    def save_archive(self, path: Optional[str] = None):
        """Save the full hyperagent archive for later resumption (JSON format)."""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        archive_path = path or str(output_dir / "hyperagent_archive.json")
        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(self.archive, f, indent=2, ensure_ascii=False)
        print(f"Saved archive ({len(self.archive)} agents) to: {archive_path}")

    def save_archive_jsonl(self, path: Optional[str] = None):
        """
        Append archive snapshot to a JSONL file matching the real repo's archive.jsonl format.

        Real repo format (generate_loop.py):
          Each line: {"current_genid": <genid>, "archive": {<genid>: <entry>, ...}}
        Genids in real repo are 'initial', 0, 1, 2, ...
        Here we map them to 'initial' for agent 0 and iteration index for rest.
        """
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = path or str(output_dir / "archive.jsonl")

        # Build archive dict: {'initial': entry0, 0: entry1, 1: entry2, ...}
        archive_dict = {}
        for i, agent in enumerate(self.archive):
            genid = "initial" if i == 0 else i - 1
            archive_dict[str(genid)] = agent

        current_genid = (
            "initial" if len(self.archive) <= 1
            else str(len(self.archive) - 2)
        )
        record = {"current_genid": current_genid, "archive": archive_dict}

        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Appended archive snapshot to: {jsonl_path}")

    def load_archive(self, path: str):
        """Load a previously saved hyperagent archive (JSON or JSONL)."""
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                # Read last line of JSONL (most recent snapshot)
                last_line = None
                for line in f:
                    line = line.strip()
                    if line:
                        last_line = line
                if last_line is None:
                    print(f"Empty JSONL file: {path}")
                    return
                record = json.loads(last_line)
                archive_dict = record.get("archive", {})
                # Reconstruct ordered list: 'initial' first, then 0, 1, 2, ...
                ordered = []
                if "initial" in archive_dict:
                    ordered.append(archive_dict["initial"])
                idx = 0
                while str(idx) in archive_dict:
                    ordered.append(archive_dict[str(idx)])
                    idx += 1
                self.archive = ordered
            else:
                self.archive = json.load(f)
        print(f"Loaded archive ({len(self.archive)} agents) from: {path}")
