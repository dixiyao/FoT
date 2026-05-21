"""
ACE (Agentic Context Engineering) Client
Implements the ACE framework from:
"Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"
(arXiv:2510.04618v3, ICLR 2026)

Three-role architecture (§3, Figure 4):
  - Generator  : solves the problem guided by the evolving Playbook context
  - Reflector  : diagnoses reasoning quality and extracts lessons (up to 5 rounds)
  - Curator    : integrates lessons as incremental delta bullets into the Playbook

Key innovations over monolithic context rewriting:
  1. Dedicated Reflector — separates evaluation from curation, prevents brevity bias
  2. Incremental delta updates (§3.1) — no full rewrites, prevents context collapse
  3. Grow-and-refine (§3.2) — append bullets + periodic dedup/pruning

Paper defaults (§4.2, §A.6):
    max_reflector_rounds = 5          (Table 19: 5 rounds optimal on AppWorld)
    max_context_chars    = ~400_000   (100K tokens; pruning trigger, Table 21)
    dedup_threshold      = 0.90       (Table 20: 90% best on FiNER)

Playbook sections (Figure 3):
    STRATEGIES AND HARD RULES
    USEFUL APPROACHES
    DOMAIN CONCEPTS
    TROUBLESHOOTING AND PITFALLS

NOTE: Ground-truth labels are NOT available per-problem here.
      We use model self-evaluation (0-1 score) as environment feedback,
      matching the "ACE ✗ (no GT labels)" setting from Tables 1 & 2.
      This still delivers +14.8% avg on agents (Table 1, online, no GT).
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# Exact prompts from Appendix F of the paper
# ---------------------------------------------------------------------------

# Figure 12: ACE Generator prompt (domain-specific / FiNER variant)
# Used for problem solving with a loaded Playbook context.
_GENERATOR_PROMPT = """\
You are an analysis expert tasked with answering questions using your knowledge, a curated playbook of strategies and insights and a reflection that goes over the diagnosis of all previous mistakes made while answering the question.

Instructions: - Read the playbook carefully and apply relevant strategies, formulas, and insights - Pay attention to common mistakes listed in the playbook and avoid them 
- Show your reasoning step-by-step - Be concise but thorough in your analysis - If the playbook contains relevant code snippets or formulas, use them appropriately 
- Double-check your calculations and logic before providing the final answer

Your output should be a json object, which contains the following fields: - reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations 
- bullet_ids: each line in the playbook has a bullet_id. all bulletpoints in the playbook that's relevant, helpful for you to answer this question, you should include their bullet_id in this list 
- final_answer: your concise final answer

Playbook:
{playbook}

Reflection:
{reflection}

Question:
{problem}

Answer in this exact JSON format:
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
  "bullet_ids": ["shr-00001", "tip-00002"],
  "final_answer": "[Your concise final answer here]"
}}\
"""

# Figure 13: ACE Reflector prompt (domain-specific / FiNER variant)
# Adapted for the no-GT-label online setting: ground_truth is replaced by
# self-evaluation feedback (score + qualitative assessment).
_REFLECTOR_PROMPT = """\
You are an expert analyst and educator. Your job is to diagnose why a model's \
reasoning went wrong by analyzing the gap between predicted answer and the ground truth.

Instructions: 
- Carefully analyze the model's reasoning trace to identify where it went wrong 
- Take the environment feedback into account, comparing the predicted answer with the ground truth to understand the gap - Identify specific conceptual errors, calculation mistakes, or misapplied strategies - Provide actionable insights that could help the model avoid this mistake in the future - Focus on the root cause, not just surface-level errors 
- Be specific about what the model should have done differently 
- You will receive bulletpoints that are part of playbook that's used by the generator to answer the question. 
- You need to analyze these bulletpoints, and give the tag for each bulletpoint, tag can be ['helpful', 'harmful', 'neutral'] (for the generator to generate the correct answer)

Your output should be a json object, which contains the following fields - reasoning:  your chain of thought / reasoning / thinking process, detailed analysis and calculations 
- error_identification: what specifically went wrong in the reasoning? - \
root_cause_analysis: why did this error occur? What concept was misunderstood? - correct_approach: what should the model have done instead? - key_insight: what strategy, formula, or principle should be remembered to avoid this error? - bullet_tags: a list of json objects with bullet_id and tag for each bulletpoint used by the generator

Question:
{problem}

Model's Reasoning Trace:
{reasoning}

Model's Predicted Answer:
{solution}

Ground Truth Answer:
{ground_truth}

Environment Feedback:
{feedback}

Part of Playbook that's used by the generator to answer the question:
{relevant_bullets}

Answer in this exact JSON format:
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
  "error_identification": "[What specifically went wrong in the reasoning?]",
  "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the model have done instead?]",
  "key_insight": "[What strategy, formula, or principle should be remembered to avoid this error?]",
  "bullet_tags": [
    {{"id": "shr-00001", "tag": "helpful"}},
    {{"id": "tip-00002", "tag": "harmful"}}
  ]
}}\
"""

# Figure 14: ACE Curator prompt (domain-specific / FiNER variant)
_CURATOR_PROMPT = """\
You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

Context: - The playbook you created will be used to help answering similar questions. 
- The reflection is generated using ground truth answers that will NOT be available \
when the playbook is being used. So you need to come up with content that can aid the playbook user to create predictions that likely align with ground truth.

CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.

Instructions: - Review the existing playbook and the reflection from the previous attempt 
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook 
- Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement to the existing playbook - Do NOT regenerate the entire playbook - only provide the additions needed - Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one - Format your response as a PURE JSON object with specific sections - For any operation if no new content to add, return an empty list for the operations field - Be concise and specific - each addition should be actionable

Training Context:
Total token budget: {token_budget} tokens
Training progress: Sample {current_step} out of {total_samples}

Current Playbook Stats:
{playbook_stats}

Recent Reflection:
{recent_reflection}

Current Playbook:
{current_playbook}

Question Context:
{question_context}

Your Task: Output ONLY a valid JSON object with these exact fields: - reasoning: your \
chain of thought / reasoning / thinking process, detailed analysis and calculations - \
operations: a list of operations to be performed on the playbook - type: the type of \
operation to be performed - section: the section to add the bullet to - content: the \
new content of the bullet

Available Operations: 1. ADD: Create new bullet points with fresh IDs - section: the \
section to add the new bullet to - content: the new content of the bullet. Note: no need \
to include the bullet_id in the content like '[ctx-00263] helpful=1 harmful=0 ::', the \
bullet_id will be added by the system.

RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here]",
  "operations": [
    {{
      "type": "ADD",
      "section": "strategies_and_hard_rules",
      "content": "[New insight...]"
    }}
  ]
}}\
"""

# Self-evaluation prompt for no-GT-label feedback (online adaptation, §4.3)
_SELF_EVAL_PROMPT = """\
Problem: {problem}

Proposed solution: {solution}

Evaluate the quality of this solution on a scale from 0.0 to 1.0 where:
- 0.0 means completely wrong, missing the point, or harmful
- 0.5 means partially correct with significant gaps or errors
- 1.0 means completely correct, well-reasoned, and comprehensive

Consider: correctness, completeness, logical consistency, and reasoning quality.
Output ONLY a single floating-point number between 0.0 and 1.0.\
"""


# ---------------------------------------------------------------------------
# Playbook data structure (§3.1, Figure 3)
# ---------------------------------------------------------------------------

_PLAYBOOK_SECTIONS = [
    "strategies_and_hard_rules",
    "useful_approaches",
    "domain_concepts",
    "troubleshooting_and_pitfalls",
]

_SECTION_PREFIXES = {
    "strategies_and_hard_rules": "shr",
    "useful_approaches":         "app",
    "domain_concepts":           "dc",
    "troubleshooting_and_pitfalls": "tip",
}

_SECTION_HEADERS = {
    "strategies_and_hard_rules":    "STRATEGIES AND HARD RULES",
    "useful_approaches":            "USEFUL APPROACHES",
    "domain_concepts":              "DOMAIN CONCEPTS",
    "troubleshooting_and_pitfalls": "TROUBLESHOOTING AND PITFALLS",
}


class Playbook:
    """
    ACE evolving context — a structured collection of itemized bullets.
    Each bullet has: unique ID, content, helpful/harmful counters (metadata).
    Supports incremental ADD, tag updates, deduplication, and serialization.
    """

    def __init__(self):
        self.bullets: Dict[str, List[Dict[str, Any]]] = {
            s: [] for s in _PLAYBOOK_SECTIONS
        }
        self._counters: Dict[str, int] = {
            v: 0 for v in _SECTION_PREFIXES.values()
        }

    def _next_id(self, section: str) -> str:
        prefix = _SECTION_PREFIXES.get(section, "gen")
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}-{self._counters[prefix]:05d}"

    def add_bullet(self, section: str, content: str) -> str:
        """Append a new bullet to the given section. Returns the assigned ID."""
        if section not in self.bullets:
            section = "strategies_and_hard_rules"
        content = content.strip()
        if not content:
            return ""
        bullet_id = self._next_id(section)
        self.bullets[section].append({
            "id": bullet_id,
            "content": content,
            "helpful": 0,
            "harmful": 0,
        })
        return bullet_id

    def update_tags(self, bullet_tags: List[Dict[str, str]]):
        """Increment helpful/harmful counters from Reflector bullet_tags output."""
        id_map: Dict[str, Dict] = {}
        for bullets in self.bullets.values():
            for b in bullets:
                id_map[b["id"]] = b
        for item in bullet_tags:
            bid = item.get("id", "")
            tag = item.get("tag", "neutral")
            if bid in id_map:
                if tag == "helpful":
                    id_map[bid]["helpful"] += 1
                elif tag == "harmful":
                    id_map[bid]["harmful"] += 1

    def render(self) -> str:
        """Format playbook for insertion into LLM prompts (Figure 3 layout)."""
        parts = []
        for section in _PLAYBOOK_SECTIONS:
            bullets = self.bullets.get(section, [])
            if not bullets:
                continue
            parts.append(_SECTION_HEADERS[section])
            parts.append("")
            for b in bullets:
                parts.append(
                    f"[{b['id']}] helpful={b['helpful']} harmful={b['harmful']} ::"
                )
                parts.append(b["content"])
                parts.append("")
        return "\n".join(parts).strip()

    def total_chars(self) -> int:
        return len(self.render())

    def num_bullets(self) -> int:
        return sum(len(v) for v in self.bullets.values())

    def get_by_ids(self, bullet_ids: List[str]) -> str:
        """Return formatted text for a subset of bullet IDs."""
        id_map: Dict[str, Dict] = {}
        for bullets in self.bullets.values():
            for b in bullets:
                id_map[b["id"]] = b
        lines = []
        for bid in bullet_ids:
            if bid in id_map:
                b = id_map[bid]
                lines.append(f"[{b['id']}] :: {b['content']}")
        return "\n".join(lines)

    def deduplicate(self, threshold: float = 0.90):
        """
        Grow-and-refine (§3.2): remove near-duplicate bullets within each section
        using Jaccard word-overlap. When duplicates exist, keep the one with more
        helpful votes (or the existing one on tie).
        """
        for section in _PLAYBOOK_SECTIONS:
            bullets = self.bullets[section]
            if len(bullets) < 2:
                continue
            kept: List[Dict] = []
            for b in bullets:
                words_b = set(b["content"].lower().split())
                dup_of = None
                for k in kept:
                    words_k = set(k["content"].lower().split())
                    union = words_b | words_k
                    if not union:
                        continue
                    overlap = len(words_b & words_k) / len(union)
                    if overlap >= threshold:
                        dup_of = k
                        break
                if dup_of is None:
                    kept.append(b)
                elif b["helpful"] > dup_of["helpful"]:
                    # Replace kept entry with the higher-voted duplicate
                    kept.remove(dup_of)
                    kept.append(b)
            self.bullets[section] = kept

    def prune_harmful(self):
        """Remove bullets where harmful count strictly exceeds helpful count."""
        for section in _PLAYBOOK_SECTIONS:
            self.bullets[section] = [
                b for b in self.bullets[section]
                if b["helpful"] >= b["harmful"]
            ]

    def to_dict(self) -> Dict:
        return {"bullets": self.bullets, "counters": self._counters}

    @classmethod
    def from_dict(cls, data: Dict) -> "Playbook":
        pb = cls()
        pb.bullets = data.get("bullets", {s: [] for s in _PLAYBOOK_SECTIONS})
        for s in _PLAYBOOK_SECTIONS:
            pb.bullets.setdefault(s, [])
        pb._counters = data.get("counters", {
            v: 0 for v in _SECTION_PREFIXES.values()
        })
        # Re-sync counters to max observed ID to avoid collisions
        for section, bullets in pb.bullets.items():
            prefix = _SECTION_PREFIXES.get(section, "gen")
            for b in bullets:
                bid = b.get("id", "")
                m = re.match(rf"^{re.escape(prefix)}-(\d+)$", bid)
                if m:
                    pb._counters[prefix] = max(
                        pb._counters.get(prefix, 0), int(m.group(1))
                    )
        return pb


# ---------------------------------------------------------------------------
# ACE Client
# ---------------------------------------------------------------------------

class ACEClient:
    """
    Drop-in replacement for client.py implementing the ACE framework.

    solve_problem() runs one online adaptation step:
      Generator  → produces solution using current Playbook
      Reflector  → diagnoses quality, tags bullets (1-5 rounds)
      Curator    → proposes ADD operations (delta context items)
      Playbook   → updated incrementally; dedup when over budget

    The Playbook persists across calls (growing over time).
    Use load_encyclopedias() to warm-start from a prior run.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        task: Optional[str] = None,
        device: Optional[str] = None,
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
        api_model: str = "gemini-3-pro-preview",
        output_dir: str = "output",
        load_in_8bit: bool = False,
        # ACE-specific hyperparameters (paper §4.2, §A.6)
        max_reflector_rounds: int = 1,      # paper: 5; reduced for LLM-call cost
        max_context_chars: int = 400_000,   # ~100K tokens (Table 21)
        dedup_threshold: float = 0.90,      # Table 20: 90% best on FiNER
        playbook: Optional[Playbook] = None,
    ):
        self.model_name = model_name
        self.task = task or "Solve the given problem step by step."
        self.output_dir = output_dir
        self.load_in_8bit = load_in_8bit

        # Paper hyperparameters
        self._max_reflector_rounds = max_reflector_rounds
        self._max_context_chars = max_context_chars
        self._dedup_threshold = dedup_threshold
        self._step_counter = 1  # tracks "current_step" for Curator prompt

        # Evolving playbook (§3.1 — persistent across solve_problem calls)
        self._playbook = playbook if playbook is not None else Playbook()

        # insight_book accumulates bullets as flat k→v for server_text.py
        self.insight_book: Dict[str, str] = {}

        # Encyclopedia support (loaded external playbooks)
        self.encyclopedia = ""
        self.encyclopedia_dict: Dict[str, str] = {}
        self.encyclopedia_loaded = False

        # API / HuggingFace backend
        self.use_api = use_api
        self.api_provider = api_provider
        self.api_key = api_key or (os.getenv("GEMINI_API_KEY") if api_provider == "gemini" else os.getenv("OPENROUTER_API_KEY"))
        self.api_model_name = api_model
        if self.use_api and self.api_provider == "gemini":
            self.gemini_model = setup_gemini(
                api_key=self.api_key, model_name=self.api_model_name
            )
        else:
            self.gemini_model = None

        self.model = None
        self.tokenizer = None
        self.device = device or ("cuda" if check_cuda() else "cpu")

        self.reasoning_steps: List[Dict] = []

    # ------------------------------------------------------------------
    # Model dispatch
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
    # JSON parsing helper
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> Dict:
        """Best-effort JSON extraction from model output."""
        # Strip markdown code fences
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            candidate = m.group(1)
        else:
            # Find first { … } block
            start = text.find("{")
            if start == -1:
                return {}
            depth = 0
            in_str = False
            esc = False
            end = start
            for i in range(start, len(text)):
                c = text[i]
                if esc:
                    esc = False
                    continue
                if c == "\\":
                    esc = True
                    continue
                if c == '"' and not esc:
                    in_str = not in_str
                    continue
                if not in_str:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
            candidate = text[start:end + 1]

        # Light cleanup
        candidate = re.sub(r",\s*}", "}", candidate)
        candidate = re.sub(r",\s*]", "]", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return {}

    # ------------------------------------------------------------------
    # ACE roles (exact prompts from Appendix F)
    # ------------------------------------------------------------------

    def _run_generator(self, problem: str) -> Dict:
        """Generator role (Figure 9/12): solve problem using current Playbook."""
        playbook_str = self._playbook.render()
        prompt = _GENERATOR_PROMPT.format(
            playbook=playbook_str if playbook_str else "(empty — no strategies accumulated yet)",
            reflection="(none yet — this is the first attempt on this problem)",
            problem=problem,
        )
        raw, token_info = self._call_model(prompt, max_new_tokens=8192)
        print(f"[ACE Generator] raw output (first 300 chars): {raw[:300]}")
        parsed = self._parse_json(raw)
        if not parsed:
            parsed = {"reasoning": raw, "bullet_ids": [], "final_answer": raw}
        return parsed, token_info

    def _self_evaluate(self, problem: str, solution: str) -> float:
        """
        Self-evaluation feedback (no GT labels, online adaptation §4.3).
        Returns score in [0, 1].
        """
        prompt = _SELF_EVAL_PROMPT.format(problem=problem, solution=solution)
        raw, _ = self._call_model(prompt, max_new_tokens=32)
        raw = raw.strip()
        m = re.search(r"[-+]?\d*\.?\d+", raw)
        if m:
            score = float(m.group())
            return max(0.0, min(1.0, score))
        return 0.5

    def _run_reflector(
        self,
        problem: str,
        reasoning: str,
        solution: str,
        feedback: str,
        relevant_bullets: str,
    ) -> Dict:
        """Reflector role (Figure 10/13): diagnose quality, tag bullets."""
        prompt = _REFLECTOR_PROMPT.format(
            problem=problem,
            reasoning=reasoning,
            solution=solution,
            ground_truth="(not available — self-evaluation used as proxy)",
            feedback=feedback,
            relevant_bullets=relevant_bullets or "(none used by generator)",
        )
        raw, _ = self._call_model(prompt, max_new_tokens=4096)
        print(f"[ACE Reflector] raw output (first 300 chars): {raw[:300]}")
        parsed = self._parse_json(raw)
        if not parsed:
            parsed = {
                "reasoning": raw,
                "error_identification": "",
                "root_cause_analysis": "",
                "correct_approach": "",
                "key_insight": raw[:500],
                "bullet_tags": [],
            }
        return parsed

    def _run_curator(
        self,
        reflection: Dict,
        problem: str,
    ) -> Dict:
        """Curator role (Figure 11/14): produce ADD delta operations."""
        playbook_str = self._playbook.render()
        playbook_stats = (
            f"Total bullets: {self._playbook.num_bullets()}, "
            f"Approximate chars: {self._playbook.total_chars()}"
        )
        # Estimate token budget as chars (1 token ≈ 4 chars)
        token_budget_str = f"{self._max_context_chars // 4:,} tokens (≈{self._max_context_chars:,} chars)"

        prompt = _CURATOR_PROMPT.format(
            token_budget=token_budget_str,
            current_step=self._step_counter,
            total_samples=max(self._step_counter, 1),
            playbook_stats=playbook_stats,
            recent_reflection=json.dumps(reflection, indent=2)[:3000],
            current_playbook=playbook_str if playbook_str else "(empty)",
            question_context=problem[:500],
        )
        raw, _ = self._call_model(prompt, max_new_tokens=4096)
        print(f"[ACE Curator] raw output (first 300 chars): {raw[:300]}")
        parsed = self._parse_json(raw)
        if not parsed:
            parsed = {"reasoning": raw, "operations": []}
        return parsed

    def _apply_delta(self, curator_output: Dict):
        """Apply ADD operations from Curator to the Playbook (§3.1)."""
        operations = curator_output.get("operations", [])
        added = 0
        for op in operations:
            if op.get("type", "").upper() == "ADD":
                section = op.get("section", "strategies_and_hard_rules")
                content = op.get("content", "").strip()
                if content:
                    self._playbook.add_bullet(section, content)
                    added += 1
        print(f"[ACE Curator] applied {added} ADD operation(s), playbook now has {self._playbook.num_bullets()} bullets")

    def _playbook_to_insight_book(self) -> Dict[str, str]:
        """
        Convert Playbook bullets to the flat insight_book format used by
        server_text.py / save_reasoning / load_encyclopedias.
        Key format: "ace_<section>_<bullet_id>"
        """
        ib: Dict[str, str] = {}
        for section in _PLAYBOOK_SECTIONS:
            for b in self._playbook.bullets[section]:
                key = f"ace_{section}_{b['id']}"
                ib[key] = b["content"]
        return ib

    # ------------------------------------------------------------------
    # Encyclopedia / persistence interface (client.py-compatible)
    # ------------------------------------------------------------------

    def load_encyclopedias(
        self, encyclopedia_paths: List[str], mode: str = "text"
    ):
        """
        Load and merge multiple encyclopedias into the current Playbook.

        Entries with keys starting with "ace_" are parsed back as Playbook
        bullets. All other entries are added as strategies_and_hard_rules bullets.
        Exact-duplicate content is silently skipped.
        """
        if not encyclopedia_paths:
            print("No encyclopedias provided to load.")
            return

        existing_contents = set()
        for bullets in self._playbook.bullets.values():
            for b in bullets:
                existing_contents.add(b["content"])

        loaded = 0
        added = 0
        for ep in encyclopedia_paths:
            if not ep or not os.path.exists(ep):
                continue
            try:
                with open(ep, "r", encoding="utf-8") as f:
                    data = json.load(f)
                loaded += 1

                items: Dict[str, str] = {}
                if isinstance(data, dict):
                    # Check for embedded full playbook
                    if "ace_playbook" in data and isinstance(data["ace_playbook"], dict):
                        pb = Playbook.from_dict(data["ace_playbook"])
                        for section in _PLAYBOOK_SECTIONS:
                            for b in pb.bullets.get(section, []):
                                if b["content"] not in existing_contents:
                                    self._playbook.add_bullet(section, b["content"])
                                    existing_contents.add(b["content"])
                                    added += 1
                        continue
                    items = {str(k): str(v) for k, v in data.items()}
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            name = item.get("name") or item.get("insight", "")
                            desc = item.get("description") or item.get("desc", "")
                            if name and desc:
                                items[str(name)] = str(desc)

                for key, content in items.items():
                    if not content or content in existing_contents:
                        continue
                    # Infer section from key prefix
                    section = "strategies_and_hard_rules"
                    if key.startswith("ace_"):
                        for s in _PLAYBOOK_SECTIONS:
                            if key.startswith(f"ace_{s}_"):
                                section = s
                                break
                    self._playbook.add_bullet(section, content)
                    existing_contents.add(content)
                    added += 1

            except Exception as e:
                print(f"Warning: failed to load encyclopedia {ep}: {e}")

        # Rebuild flat encyclopedia_dict for compatibility
        self.encyclopedia_dict = self._playbook_to_insight_book()
        self.encyclopedia = json.dumps(self.encyclopedia_dict, indent=2)
        self.encyclopedia_loaded = self._playbook.num_bullets() > 0
        print(
            f"[ACE] Loaded {loaded} encyclopedia file(s), added {added} bullets "
            f"(playbook now has {self._playbook.num_bullets()} bullets)"
        )

    def load_encyclopedia(self, encyclopedia_path: str, mode: str = "text"):
        """Single-file convenience wrapper for load_encyclopedias."""
        self.load_encyclopedias([encyclopedia_path], mode)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def solve_problem(
        self,
        task: Optional[str] = None,
        custom_solution_instruction: Optional[str] = None,
        insights_section: Optional[str] = None,
    ) -> Dict:
        """
        One ACE online adaptation step: Generator → Reflector → Curator → Playbook.

        Args:
            task: Problem text to solve. If None, uses self.task.
            custom_solution_instruction: Ignored (kept for API compatibility).
            insights_section: Ignored (Playbook is used instead).

        Returns:
            Dict compatible with client.py solve_problem() output.
        """
        if task is not None:
            self.task = task
        problem = self.task

        print(f"\n[ACE] Problem: {problem[:120]}...\n")
        self.reasoning_steps = []
        total_steps = 0

        # ---- Step 1: Generator ----------------------------------------
        print("[ACE] Step 1: Generator")
        gen_out, token_info = self._run_generator(problem)
        reasoning  = gen_out.get("reasoning", "")
        solution   = gen_out.get("final_answer", reasoning)
        bullet_ids = gen_out.get("bullet_ids", [])
        total_steps += 1
        self.reasoning_steps.append({
            "step": 1, "name": "Generator",
            "reasoning": reasoning, "solution": solution,
            "bullet_ids": bullet_ids,
        })
        time.sleep(0.5)

        # ---- Step 2: Self-evaluation (environment feedback, no GT) ----
        print("[ACE] Step 2: Self-evaluation")
        score = self._self_evaluate(problem, solution)
        if score >= 0.8:
            qual = "The solution appears strong and well-reasoned."
        elif score >= 0.5:
            qual = "The solution has room for improvement in accuracy or completeness."
        else:
            qual = "The solution likely has significant errors or critical gaps."
        feedback = f"Self-evaluation score: {score:.2f}/1.00. {qual}"
        total_steps += 1
        time.sleep(0.5)

        # ---- Step 3: Reflector (iterative, up to max_reflector_rounds) -
        relevant_bullets = self._playbook.get_by_ids(bullet_ids)
        reflection: Dict = {}
        for rnd in range(self._max_reflector_rounds):
            print(f"[ACE] Step 3: Reflector (round {rnd + 1}/{self._max_reflector_rounds})")
            reflection = self._run_reflector(
                problem=problem,
                reasoning=reasoning,
                solution=solution,
                feedback=feedback,
                relevant_bullets=relevant_bullets,
            )
            # Enrich feedback with reflector's diagnosis for next round
            insight = reflection.get("key_insight", "")
            if insight:
                feedback += f"\nPrevious reflection key insight: {insight[:300]}"
            total_steps += 1
            time.sleep(0.5)

        # Update playbook bullet tags (helpful/harmful counters)
        self._playbook.update_tags(reflection.get("bullet_tags", []))

        # ---- Step 4: Curator ------------------------------------------
        print("[ACE] Step 4: Curator")
        curator_out = self._run_curator(reflection=reflection, problem=problem)
        total_steps += 1
        self._step_counter += 1
        time.sleep(0.5)

        # ---- Step 5: Apply delta & grow-and-refine --------------------
        self._apply_delta(curator_out)
        if self._playbook.total_chars() > self._max_context_chars:
            print("[ACE] Context over budget — running deduplication + prune")
            self._playbook.deduplicate(self._dedup_threshold)
            self._playbook.prune_harmful()

        # ---- Build output --------------------------------------------
        insight_book = self._playbook_to_insight_book()
        self.insight_book = insight_book  # persist on instance

        key_insight = reflection.get("key_insight", "")
        correct_approach = reflection.get("correct_approach", "")
        skills_extracted: Dict[str, str] = {}
        if key_insight:
            skills_extracted["ace_key_insight"] = key_insight
        if correct_approach:
            skills_extracted["ace_correct_approach"] = correct_approach
        # Include the newest bullets as extracted skills
        for section in _PLAYBOOK_SECTIONS:
            for b in self._playbook.bullets[section][-3:]:
                skills_extracted[f"ace_{section}_{b['id']}"] = b["content"]

        return {
            "problem": problem,
            "task": self.task,
            "solution": solution,
            "reflection": json.dumps(reflection, indent=2),
            "skills_extracted": skills_extracted,
            "skills_used": bullet_ids,
            "validation_errors": [],
            "insight_book": insight_book,
            "total_steps": total_steps,
            "token_info": {**token_info, "self_eval_score": score},
            # ACE-specific: full playbook for warm-starting future runs
            "ace_playbook": self._playbook.to_dict(),
        }

    def save_reasoning(
        self, reasoning_result: Dict, output_path: Optional[str] = None
    ):
        """
        Save insight_book as flat JSON {"insight_name": "description"}.
        Embeds "ace_playbook" for full playbook persistence.
        """
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        insight_book = reasoning_result.get("insight_book", {})
        if not insight_book:
            print("[ACE] No insights to save")
            return

        if output_path is None:
            safe_name = re.sub(
                r"[^\w\s-]", "", reasoning_result.get("problem", "reasoning")[:50]
            )
            safe_name = re.sub(r"[-\s]+", "_", safe_name)
            output_path = str(output_dir / f"{safe_name}.json")
        else:
            if not os.path.isabs(output_path):
                output_path = str(output_dir / output_path)
            if not output_path.endswith(".json"):
                output_path += ".json"

        # Save insight_book + embedded full playbook for warm-starting
        payload = dict(insight_book)
        pb_dict = reasoning_result.get("ace_playbook", self._playbook.to_dict())
        payload["ace_playbook"] = pb_dict  # type: ignore[assignment]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[ACE] Saved playbook ({self._playbook.num_bullets()} bullets) to: {output_path}")

    def save_playbook(self, path: str):
        """Directly save the current Playbook to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._playbook.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[ACE] Playbook saved to: {path}")

    def load_playbook(self, path: str):
        """Load a previously saved Playbook from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._playbook = Playbook.from_dict(data)
        print(
            f"[ACE] Loaded playbook from {path} "
            f"({self._playbook.num_bullets()} bullets)"
        )
