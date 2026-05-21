"""
Test-time Recursive Thinking (TRT) Client

Implements Algorithm 1 from:
  "Test-time Recursive Thinking: Self-Improvement without External Feedback"

Algorithm:
  Require: Problem P, rounds T, rollouts per round K
  Initialize: knowledge list K ← ∅, solution pool S ← ∅
  for t = 1 to T do
    // Generate
    for k = 1 to K do
      Design strategy s_k based on K
      r_k ← LLM(P, K, s_k)
      S ← S ∪ {r_k}
    end for
    // Select
    r* ← SELECT(S)      ← model self-ranks and picks best
    // Reflect
    for each r in current round where r ≠ r* do
      Extract insights by comparing r to r*
      K ← K ∪ {insights}
    end for
  end for
  return r*

Output format is identical to client.py so this is a drop-in replacement
in task_benchmark_domain.py, task_paper_insight_reading.py, etc.
"""

import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

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
# Prompts (from paper Appendix A.7)
# ---------------------------------------------------------------------------

# -- Solver system prompt (paper: "Solver System Prompt") ------------------
_SOLVER_SYSTEM_PROMPT = """\
You are an expert problem solver.

## YOUR TASK:
1. Read the problem statement carefully
2. If you have previous attempts/solutions shown, analyze them critically for errors
3. Provide a complete, thorough solution
4. If a reference solution exists, analyze it critically for errors. Your job is to
   improve it.

## VERIFICATION:
Before finalizing, verify your solution against the problem requirements.
If your reasoning has errors, fix them before submitting."""

# -- Knowledge manager system prompt (paper: "Knowledge Manager System Prompt") --
_KNOWLEDGE_MANAGER_SYSTEM_PROMPT = """\
You are a technical reviewer analyzing solutions to a problem.

Your job:
1. Rank solutions by CORRECTNESS first, then quality of reasoning
2. Extract SPECIFIC, ACTIONABLE lessons from failures (not generic tips)
3. DEDUPLICATE insights — don't repeat what's already in the knowledge base

DO NOT solve the problem yourself. Focus only on evaluation and knowledge curation."""

# -- Strategy prompt (paper §3.1: model designs strategies conditioned on K) --
_STRATEGY_PROMPT = """\
Problem:
{problem}

{knowledge_section}

Design a specific strategy for your next solution attempt that avoids the known \
failure modes above and explores a new direction.
- Be concrete and actionable (not generic advice)
- Phrase around what NOT to do (based on the don'ts list) and what to try instead
- If no prior knowledge, choose a fresh principled approach

Output your strategy in 2-4 sentences only."""

# -- Rollout: initial prompt (paper: "AIME Initial Prompt", generalised) ---
_ROLLOUT_INITIAL_PROMPT = """\
{problem}

Guideline: Let's solve this problem. Be thorough.

Strategy for this attempt:
{strategy}

## Output format (Use exact headers including square brackets):
[Summary]: A paragraph of detailed step-by-step summary of your solution, write \
thoroughly and in detail, note down every step and what the final answer is.
[Answer]: Your final answer.

Let's think step by step. Follow the output format strictly."""

# -- Rollout: iterative refinement prompt (paper: "AIME Iterative Refinement Prompt") --
_ROLLOUT_ITERATIVE_PROMPT = """\
{problem}

Let's solve this problem. I have some additional information that might help.
Examine them carefully and see if they can help you solve the problem more accurately.

{knowledge_text}

### Reference Solution
Take this information with a grain of salt — it might be wrong or incomplete.
Try to spot the mistakes in the solution and see if there is a more accurate approach.
{reference_solution}

Strategy for this attempt:
{strategy}

### Output format (Use exact headers including square brackets):
[Why the reference solution is wrong?]: If you get a different solution than \
the reference solution, explain here in a stand-alone manner what the reference \
solution's final answer is and why it is incorrect. \
(or write "N/A" if you agree with the reference solution)
[Summary]: A paragraph of detailed step-by-step summary of your solution, write \
thoroughly and in detail, note down every step and what the final answer is.
[Answer]: Your final answer.

Let's think step by step. Follow the output format strictly."""

# -- Select prompt (paper: Knowledge Manager ranks by correctness first) ---
_SELECT_PROMPT = """\
Problem:
{problem}

Candidate solutions:
{candidates}

Rank solutions by CORRECTNESS first, then quality of reasoning.

Output exactly one line in this format:
BEST: <number>

Where <number> is the 1-based index of the best candidate. Output nothing else after that line."""

# -- Insight extraction prompt (paper: knowledge = "don'ts", negative constraints) --
_INSIGHT_PROMPT = """\
Problem:
{problem}

Best solution (r*):
{r_best}

Suboptimal solution:
{r_other}

Extract SPECIFIC, ACTIONABLE failure lessons from the suboptimal solution compared \
to the best. Phrase each insight as a negative constraint ("don't") that prevents \
similar mistakes in future attempts. Focus on concrete failure modes, not generic tips.

Output a JSON object where each key starts with "insight_" and the value is a \
description of at least 20 characters:

{{"insight_name": "don't ... because ..."}}

Output valid JSON only."""


# ---------------------------------------------------------------------------
# TRT Client
# ---------------------------------------------------------------------------

class TRTClient:
    """
    Test-time Recursive Thinking client.

    Drop-in replacement for ChainOfThoughtReader (client.py).
    solve_problem() returns the same dict shape so existing pipeline code
    (task_benchmark_domain, task_paper_insight_reading, etc.) works unchanged.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1",
        task: Optional[str] = None,
        device: Optional[str] = None,
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
        api_model: str = "gemini-3-pro-preview",
        output_dir: str = "output",
        load_in_8bit: bool = False,
        rounds: int = 64,      # T — paper default: 64 for math (AIME), 8 for code
        rollouts: int = 2,     # K — paper uses K=1 for AIME, K=2 for code; set to 2
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.load_in_8bit = load_in_8bit
        self.task = task or "Solve the given problem step by step."
        self.insight_book: Dict[str, str] = {}
        self.rounds = rounds
        self.rollouts = rollouts

        # Encyclopedia support (identical to client.py)
        self.encyclopedia = ""
        self.encyclopedia_dict: Dict[str, str] = {}
        self.encyclopedia_loaded = False

        self.use_api = use_api
        self.api_provider = api_provider
        self.api_key = api_key or (os.getenv("GEMINI_API_KEY") if api_provider == "gemini" else os.getenv("OPENROUTER_API_KEY"))
        self.api_model_name = api_model
        if self.use_api and self.api_provider == "gemini":
            self.gemini_model = setup_gemini(
                api_key=self.api_key, model_name=self.api_model_name
            )

        self.model = None
        self.tokenizer = None
        self.device = device or ("cuda" if check_cuda() else "cpu")

    # ------------------------------------------------------------------
    # Model call (identical to client.py)
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
    ) -> Tuple[str, dict]:
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
    # Encyclopedia support (identical to client.py)
    # ------------------------------------------------------------------

    def load_encyclopedia(self, encyclopedia_path: str, mode: str = "text"):
        try:
            if mode == "text":
                with open(encyclopedia_path, "r", encoding="utf-8") as f:
                    self.encyclopedia_dict = json.load(f)
                self.encyclopedia = json.dumps(self.encyclopedia_dict, indent=2)
            else:
                with open(encyclopedia_path, "r", encoding="utf-8") as f:
                    self.encyclopedia = f.read()
            self.encyclopedia_loaded = True
            print(f"Loaded encyclopedia from {encyclopedia_path}")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load encyclopedia: {e}")

    def load_encyclopedias(self, encyclopedia_paths: List[str], mode: str = "text"):
        """Load and merge multiple encyclopedias (drop-in for client.py)."""
        if not encyclopedia_paths:
            return
        merged_dict: Dict[str, str] = {}
        merged_text_parts: List[str] = []
        canonical_entries: Dict[str, List[Dict[str, str]]] = {}
        skipped = 0
        used = []
        for ep in encyclopedia_paths:
            try:
                if not ep or not os.path.exists(ep):
                    continue
                if mode == "text":
                    with open(ep, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    items = data.items() if isinstance(data, dict) else (
                        ((item.get("name") or item.get("insight"), item.get("description") or item.get("desc", ""))
                         for item in data if isinstance(item, dict))
                        if isinstance(data, list) else []
                    )
                    for name, desc in items:
                        if not name:
                            continue
                        cname = re.sub(r"\s+", " ", str(name).strip().lower())
                        entries = canonical_entries.get(cname, [])
                        if any(e.get("desc", "") == desc for e in entries):
                            skipped += 1
                            continue
                        idx = len(entries) + 1
                        new_name = f"{name}_{idx}"
                        while new_name in merged_dict:
                            idx += 1
                            new_name = f"{name}_{idx}"
                        merged_dict[new_name] = desc
                        entries.append({"name": new_name, "desc": desc})
                        canonical_entries[cname] = entries
                else:
                    with open(ep, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    if text:
                        merged_text_parts.append(text)
                used.append(ep)
            except Exception as exc:
                print(f"Warning: failed to load encyclopedia {ep}: {exc}")
        if mode == "text":
            self.encyclopedia_dict = merged_dict
            self.encyclopedia = json.dumps(self.encyclopedia_dict, indent=2)
            print(f"Loaded {len(used)} encyclopedias, total insights {len(self.encyclopedia_dict)}")
        else:
            self.encyclopedia = "\n\n".join(merged_text_parts)
            self.encyclopedia_dict = {}
            print(f"Loaded {len(used)} encyclopedias (text), total chars {len(self.encyclopedia)}")
        self.encyclopedia_loaded = True

    # ------------------------------------------------------------------
    # TRT steps
    # ------------------------------------------------------------------

    def _format_knowledge(self, knowledge: List[str]) -> str:
        """Format the knowledge list K as an Empirical Mistakes List (paper §3.1: 'don'ts')."""
        if not knowledge:
            return ""
        lines = [f"{i + 1}. {k}" for i, k in enumerate(knowledge)]
        return "## Empirical Mistakes List (Don'ts — avoid these failure modes):\n" + "\n".join(lines)

    def _step_strategy(self, problem: str, knowledge: List[str]) -> str:
        """Design strategy s_k based on current knowledge K (paper §3.1)."""
        knowledge_section = self._format_knowledge(knowledge)
        if not knowledge_section:
            knowledge_section = "(No prior knowledge yet — this is the first attempt.)"
        prompt = _STRATEGY_PROMPT.format(
            problem=problem,
            knowledge_section=knowledge_section,
        )
        response, _ = self._call_model(
            prompt, system_prompt=_KNOWLEDGE_MANAGER_SYSTEM_PROMPT, max_new_tokens=512
        )
        strategy = response.strip()
        print(f"    Strategy: {strategy[:120]}{'...' if len(strategy) > 120 else ''}")
        return strategy

    def _step_rollout(
        self,
        problem: str,
        knowledge: List[str],
        strategy: str,
        r_prev: str = "",          # previous round's best solution (reference)
    ) -> Tuple[str, dict]:
        """Generate rollout r_k = LLM(P, K, s_k).

        Round 1 (no prior knowledge/reference): uses initial prompt.
        Round 2+ (with knowledge + reference solution): uses iterative refinement
        prompt from paper Appendix A.7.1.
        """
        # Encyclopedia prepended to knowledge section
        enc_section = ""
        if self.encyclopedia_loaded and self.encyclopedia:
            enc_section = (
                "## Encyclopedia (cross-problem insights):\n"
                f"{self.encyclopedia}\n\n"
            )

        knowledge_text = self._format_knowledge(knowledge)
        if enc_section:
            knowledge_text = enc_section + knowledge_text

        if r_prev:
            # Iterative refinement prompt (paper: "AIME Iterative Refinement Prompt")
            prompt = _ROLLOUT_ITERATIVE_PROMPT.format(
                problem=problem,
                knowledge_text=knowledge_text if knowledge_text else "(No prior failure modes recorded yet.)",
                reference_solution=r_prev,
                strategy=strategy,
            )
        else:
            # Initial prompt (paper: "AIME Initial Prompt")
            prompt = _ROLLOUT_INITIAL_PROMPT.format(
                problem=problem,
                strategy=strategy,
            )

        response, token_info = self._call_model(
            prompt, system_prompt=_SOLVER_SYSTEM_PROMPT, max_new_tokens=32768
        )
        return response, token_info

    def _step_select(self, problem: str, round_rollouts: List[str]) -> int:
        """SELECT(S) — model self-judges and returns 0-based index of best rollout."""
        if len(round_rollouts) == 1:
            return 0

        candidates_text = ""
        for i, r in enumerate(round_rollouts, 1):
            candidates_text += f"\n--- Candidate {i} ---\n{r}\n"

        prompt = _SELECT_PROMPT.format(problem=problem, candidates=candidates_text)
        response, _ = self._call_model(
            prompt, system_prompt=_KNOWLEDGE_MANAGER_SYSTEM_PROMPT, max_new_tokens=64
        )

        # Parse "BEST: <n>"
        match = re.search(r"BEST:\s*(\d+)", response, re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1  # convert to 0-based
            idx = max(0, min(idx, len(round_rollouts) - 1))
            print(f"    Selected rollout {idx + 1}/{len(round_rollouts)}")
            return idx

        print(f"    Warning: could not parse selection from: {response[:100]} — defaulting to rollout 1")
        return 0

    def _step_extract_insights(
        self, problem: str, r_best: str, r_other: str
    ) -> Dict[str, str]:
        """Extract failure insights by comparing r_other to r_best (paper: knowledge = 'don'ts')."""
        prompt = _INSIGHT_PROMPT.format(
            problem=problem,
            r_best=r_best,
            r_other=r_other,
        )
        response, _ = self._call_model(
            prompt, system_prompt=_KNOWLEDGE_MANAGER_SYSTEM_PROMPT, max_new_tokens=4096
        )

        # Parse JSON (same robust logic as client.py)
        insights: Dict[str, str] = {}
        try:
            # Try markdown code block first
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                start = response.find("{")
                end = response.rfind("}")
                json_str = response[start: end + 1] if start != -1 and end != -1 else None

            if json_str:
                json_str = re.sub(r",\s*}", "}", json_str)
                raw = json.loads(json_str)
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        name = k if k.startswith("insight_") else f"insight_{k}"
                        desc = v if isinstance(v, str) else str(v)
                        desc = re.sub(r"\s+", " ", desc).strip()
                        if len(desc) >= 20:
                            insights[name] = desc
        except Exception as exc:
            print(f"    Warning: insight parse error: {exc}")

        print(f"    Extracted {len(insights)} insights from pairwise comparison")
        return insights

    def _extract_answer(self, rollout: str) -> str:
        """Extract the [Answer] field from a rollout (paper AIME output format)."""
        match = re.search(r"\[Answer\]\s*[:\-]?\s*(.+?)(?:\n|$)", rollout, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Fallback: look for \boxed{...}
        match = re.search(r"\\boxed\{([^}]+)\}", rollout)
        if match:
            return match.group(1).strip()
        return ""

    # ------------------------------------------------------------------
    # Main TRT loop
    # ------------------------------------------------------------------

    def solve_problem(
        self,
        task: Optional[str] = None,
        custom_solution_instruction: Optional[str] = None,
        insights_section: Optional[str] = None,
    ) -> Dict:
        """
        Run the TRT algorithm and return a dict compatible with client.py's
        solve_problem() output so existing pipeline code works unchanged.

        Returns:
            {
                "problem": str,
                "solution": str,           # best solution r*
                "insight_book": dict,      # {insight_name: description}
                "trt_knowledge": list,     # raw knowledge list K
                "trt_rounds": int,
                "trt_rollouts_per_round": int,
                "token_info": dict,
            }
        """
        if task is not None:
            self.task = task
        problem = self.task

        print(f"\nProblem: {problem[:200]}{'...' if len(problem) > 200 else ''}\n")
        print(f"TRT: {self.rounds} rounds × {self.rollouts} rollouts/round")
        print("=" * 60)

        # Algorithm state
        knowledge: List[str] = []       # K — grows across rounds (phrased as "don'ts")
        solution_pool: List[str] = []   # S — all rollouts accumulated across rounds
        best_solution: str = ""
        prev_answer: str = ""           # tracks last round's [Answer] for K=1 math
        last_token_info: dict = {}
        insight_book: Dict[str, str] = {}
        insight_counter = 0

        for t in range(1, self.rounds + 1):
            print(f"\n--- Round {t}/{self.rounds} ---")
            round_rollouts: List[str] = []

            # ----------------------------------------------------------------
            # Generate: K rollouts, each with a fresh strategy
            # Reference solution = best r* from previous round (empty in round 1)
            # ----------------------------------------------------------------
            for k in range(1, self.rollouts + 1):
                print(f"  Rollout {k}/{self.rollouts}")

                strategy = self._step_strategy(problem, knowledge)
                rollout, token_info = self._step_rollout(
                    problem, knowledge, strategy, r_prev=best_solution
                )
                last_token_info = token_info

                print(f"    Output tokens: {token_info.get('output_tokens', 0)}")
                round_rollouts.append(rollout)
                solution_pool.append(rollout)   # S ← S ∪ {r_k}
                time.sleep(0.5)

            # ----------------------------------------------------------------
            # Select: self-rank and pick best r* from current round's rollouts
            # ----------------------------------------------------------------
            print(f"  Selecting best rollout...")
            best_idx = self._step_select(problem, round_rollouts)
            r_best = round_rollouts[best_idx]
            best_solution = r_best

            # ----------------------------------------------------------------
            # Reflect: extract insights to grow K
            # K=1 (math): track self-rejected answers across rounds (paper §3.2)
            # K>1: pairwise compare each non-best to r* (paper Algorithm 1 line 13-15)
            # ----------------------------------------------------------------
            if self.rollouts == 1:
                # Paper §3.2: "We track all previously self-rejected answers in the
                # knowledge list to help model self-assess correctness each round."
                current_answer = self._extract_answer(r_best)
                if prev_answer and current_answer and prev_answer != current_answer:
                    insight_counter += 1
                    entry = (
                        f"insight_rejected_answer_{insight_counter:04d}: "
                        f"Don't answer '{prev_answer}' — this was self-rejected in "
                        f"round {t - 1} in favor of a different approach."
                    )
                    knowledge.append(entry)
                    insight_book[f"insight_rejected_answer_{insight_counter:04d}"] = entry.split(": ", 1)[1]
                prev_answer = current_answer
                print(f"  [Answer this round: {current_answer or '(not parsed)'}]")
            else:
                print(f"  Reflecting on {len(round_rollouts) - 1} suboptimal rollout(s)...")
                for k, r in enumerate(round_rollouts):
                    if k == best_idx:
                        continue
                    new_insights = self._step_extract_insights(problem, r_best, r)
                    for name, desc in new_insights.items():
                        insight_counter += 1
                        unique_name = f"{name}_{insight_counter:04d}"
                        insight_book[unique_name] = desc
                        knowledge.append(f"{unique_name}: {desc}")

        print(f"\n{'=' * 60}")
        print(f"TRT complete: {len(knowledge)} total insights across {self.rounds} rounds")

        self.insight_book = insight_book

        return {
            "problem": problem,
            "task": self.task,
            "solution": best_solution,
            "reflection": "",           # not used in TRT; kept for API compatibility
            "skills_extracted": insight_book,
            "skills_used": list(insight_book.keys()),
            "validation_errors": [],
            "insight_book": insight_book,
            "total_steps": self.rounds * self.rollouts,
            "token_info": last_token_info,
            # TRT-specific fields
            "trt_knowledge": knowledge,
            "trt_rounds": self.rounds,
            "trt_rollouts_per_round": self.rollouts,
        }

    def save_reasoning(self, reasoning_result: Dict, output_path: Optional[str] = None):
        """Save insight book to JSON (same interface as client.py save_reasoning).

        Saves insight_book as the primary output (identical to client.py),
        plus TRT-specific fields (knowledge list, rounds) for traceability.
        """
        from pathlib import Path

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        insight_book = reasoning_result.get("insight_book", {})
        if not insight_book:
            print("No insights to save")
            return

        if output_path is None:
            safe_name = re.sub(r"[^\w\s-]", "", reasoning_result.get("problem", "reasoning")[:50])
            safe_name = re.sub(r"[-\s]+", "_", safe_name)
            output_path = str(output_dir / f"{safe_name}.json")
        else:
            if not os.path.isabs(output_path):
                output_path = str(output_dir / output_path)
            if not output_path.endswith(".json"):
                output_path += ".json"

        # Save insight book (same format as client.py) + TRT trace fields
        save_data = dict(insight_book)  # flat {name: desc} — compatible with server_text.py
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"Saved insight book to: {output_path}")
