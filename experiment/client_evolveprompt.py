"""
PROMPTQUINE / EvolvePrompt Client
Implements the PROMPTQUINE evolutionary prompt pruning framework from:
"Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective"
(arXiv:2506.17930v1, ICML 2025)

Core insight from the paper: pruning ICL prompt tokens into seemingly incoherent
"gibberish" can counterintuitively match or surpass state-of-the-art automatic
prompt optimization across diverse tasks.

Search algorithms implemented (exactly as in paper):
  - TAPruning   : Threshold Accepting hill-climbing (Algorithm 1)
  - SAHCPruning : Steepest-Ascent Hill Climbing (Algorithm 2)
  - SSGA        : Steady-state Genetic Algorithm (Algorithm 5) — paper default for 1-shot
  - GGA         : Generational Genetic Algorithm (Algorithm 4) — faster, used for many-shot

Output: the best discovered pruned prompt template, stored as a reusable insight
compatible with server_text.py / load_encyclopedias pipeline.

Paper hyperparameters (Table 9, SSGA 1-shot ICL):
    population_size     = 30
    offspring_size      = 50
    mutation_rate       = [1, 2, 3, 4]   (uniformly sampled # tokens to prune)
    tournament_ratio    = 0.2
    num_iterations      = 10_000         (total prompts evaluated)
    min_prompt_length   = 15             (stop when mean token count < this)
    delta               = 0.96           (TAPruning acceptance threshold)
    elite_frac          = 0.05           (top-k% for re-ranking calibration)

NOTE: Paper uses accuracy on a 200-sample held-out set as fitness.
      This adaptation uses model self-evaluation (0–1 score) since no labeled
      held-out set is available per-problem. For LLM-call-based fitness,
      num_iterations defaults to 100 (paper: 10,000) for practical use.
"""

import json
import math
import os
import random
import re
import time
from pathlib import Path
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
# Prompt templates (adapted from paper Appendix B.2 CoT template + Section 3.1)
# ---------------------------------------------------------------------------

# Initial preamble — the pruneable instruction part.
# Paper CoT template (Appendix B.2, GSM8K/MAWPS):
#   "{Examples}\nQuestion: {Input}\nLet's think step by step."
# Here we adapt for text-based problem solving without fixed ICL examples.
_INITIAL_PREAMBLE = (
    "Let's think step by step to solve this problem carefully and thoroughly. "
    "Be systematic and verify your work before finalizing the answer."
)

# Full prompt assembled for model call: preamble + problem
_FULL_PROMPT_TEMPLATE = """\
{preamble}

Problem:
{problem}

Let's think step by step.\
"""

# Self-evaluation fitness proxy (paper §4.2: LLM-as-a-Judge fitness for generation tasks)
_SELF_EVAL_PROMPT = """\
Problem:
{problem}

Proposed solution:
{solution}

Rate the quality of this solution on a scale from 0.0 to 1.0, where:
  0.0 = completely wrong or missing
  0.5 = partially correct with significant errors or gaps
  1.0 = fully correct, complete, and well-reasoned

Consider correctness, completeness, and quality of reasoning.

Output ONLY a single floating-point number between 0.0 and 1.0, nothing else.\
"""

# Description prompt: extract insight from the optimized prompt
# (adapted from paper §7: "pruned prompts contain universal insights into LLM sensitivity")
_INSIGHT_FROM_PRUNED = """\
The following prompt preamble was discovered through evolutionary token pruning
(PROMPTQUINE) to improve problem-solving performance. Fitness score: {score:.3f}

Pruned preamble:
"{pruned_preamble}"

In one sentence, describe what structural property of this prompt makes it effective.
Then output ONLY a valid JSON object:
{{"insight_evolveprompt_{tag}": "<one-sentence description> | PROMPT: {pruned_preamble}"}}\
"""


# ---------------------------------------------------------------------------
# Word-level tokenizer (paper: token mask = binary genotype over prompt tokens)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Split prompt into word-level tokens (whitespace-delimited)."""
    return text.split()


def _apply_mask(tokens: List[str], mask: List[bool]) -> str:
    """Reconstruct prompt string from token list and boolean mask."""
    return " ".join(t for t, keep in zip(tokens, mask) if keep)


def _mask_length(mask: List[bool]) -> int:
    return sum(mask)


def _mean_mask_length(masks: List[List[bool]]) -> float:
    if not masks:
        return 0.0
    return sum(_mask_length(m) for m in masks) / len(masks)


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def _tournament_select(
    population: List[List[bool]],
    scores: List[float],
    k: int,
) -> int:
    """Tournament selection: sample k individuals, return index of best."""
    indices = random.sample(range(len(population)), min(k, len(population)))
    return max(indices, key=lambda i: scores[i])


def _copy_then_mutate(
    mask: List[bool],
    mutation_rate_choices: List[int],
) -> List[bool]:
    """
    Paper §3.4: Mutations are implemented via bit-flip 1-to-0 operations
    (pruning tokens). Mutation rate uniformly sampled from {1, 2, 3, 4}.
    """
    n_flip = random.choice(mutation_rate_choices)
    new_mask = mask[:]
    prunable = [i for i, keep in enumerate(new_mask) if keep]
    if not prunable:
        return new_mask
    n_flip = min(n_flip, len(prunable))
    for i in random.sample(prunable, n_flip):
        new_mask[i] = False
    return new_mask


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class EvolvePromptClient:
    """
    PROMPTQUINE evolutionary prompt pruning client.

    For each problem, runs an evolutionary search (SSGA by default) over
    pruned versions of the task prompt preamble to find the token subsequence
    that maximises the self-evaluated solution quality.

    The discovered pruned prompt is stored in insight_book as a reusable
    encyclopedia entry compatible with server_text.py.

    Search algorithm choices (paper):
      "ssga"   — Steady-state GA (Algorithm 5), default for 1-shot
      "gga"    — Generational GA (Algorithm 4), faster convergence
      "ta"     — TAPruning, Threshold Accepting hill-climbing (Algorithm 1)
      "sahc"   — SAHCPruning, Steepest-Ascent Hill Climbing (Algorithm 2)
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
        # PROMPTQUINE hyperparameters (paper Table 9)
        population_size: int = 30,
        offspring_size: int = 50,
        mutation_rate_choices: Optional[List[int]] = None,
        tournament_ratio: float = 0.2,
        num_iterations: int = 100,      # paper: 10_000; reduced for LLM-call fitness
        min_prompt_length: int = 15,    # paper Table 9
        delta: float = 0.96,            # TAPruning threshold (paper §3.1)
        elite_frac: float = 0.05,       # top-k% for re-ranking (paper §D.3)
        search_method: str = "ssga",    # "ssga" | "gga" | "ta" | "sahc"
        initial_preamble: Optional[str] = None,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.task = task or "Solve the given problem step by step."
        self.load_in_8bit = load_in_8bit
        self.insight_book: Dict[str, str] = {}

        # Search hyperparameters
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.mutation_rate_choices = mutation_rate_choices or [1, 2, 3, 4]
        self.tournament_ratio = tournament_ratio
        self.num_iterations = num_iterations
        self.min_prompt_length = min_prompt_length
        self.delta = delta
        self.elite_frac = elite_frac
        self.search_method = search_method.lower()

        # Initial preamble (pruneable part of the prompt)
        self.initial_preamble = initial_preamble or _INITIAL_PREAMBLE

        # Best discovered preamble across all calls (persistent across problems)
        self._best_preamble: Optional[str] = None
        self._best_score: float = 0.0

        # Encyclopedia support
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
    # Prompt assembly
    # ------------------------------------------------------------------

    def _build_prompt(self, preamble: str, problem: str) -> str:
        """Assemble full model prompt from preamble + problem."""
        # Prepend encyclopedia insights if loaded
        prefix = ""
        if self.encyclopedia_loaded and self.encyclopedia_dict:
            # Surface any previously discovered optimized preambles from encyclopedia
            enc_preambles = [
                v for k, v in self.encyclopedia_dict.items()
                if "insight_evolveprompt" in k and "PROMPT:" in v
            ]
            if enc_preambles:
                # Use the most recent encyclopedia preamble as additional context
                best_enc = enc_preambles[-1].split("PROMPT:")[-1].strip()
                prefix = best_enc + "\n\n"

        prompt = _FULL_PROMPT_TEMPLATE.format(preamble=prefix + preamble, problem=problem)
        return prompt

    # ------------------------------------------------------------------
    # Fitness function
    # ------------------------------------------------------------------

    def _evaluate_fitness(self, preamble: str, problem: str) -> Tuple[float, str]:
        """
        Paper fitness: accuracy on held-out set.
        Adaptation: model self-evaluation (0–1) — used when no labeled set is available.

        Returns (score, solution).
        """
        prompt = self._build_prompt(preamble, problem)
        solution, _ = self._call_model(prompt, max_new_tokens=4096)

        # Self-evaluate (paper §4.2: LLM-as-a-Judge for generation tasks)
        eval_prompt = _SELF_EVAL_PROMPT.format(problem=problem, solution=solution)
        score_text, _ = self._call_model(eval_prompt, max_new_tokens=16)
        score_text = score_text.strip()

        match = re.search(r"([0-9]+\.?[0-9]*)", score_text)
        score = float(match.group(1)) if match else 0.5
        score = max(0.0, min(1.0, score))

        return score, solution

    # ------------------------------------------------------------------
    # Algorithm 1 — TAPruning (Threshold Accepting)
    # ------------------------------------------------------------------

    def _ta_pruning(
        self,
        tokens: List[str],
        problem: str,
    ) -> Tuple[List[bool], float, str]:
        """
        Algorithm 1 from paper: Threshold Accepting (TAPruning).

        Left-to-right pruning order. Accept new prompt if it improves or
        degrades within threshold delta (≥ delta × f_optimal).
        Repeat until no token can be removed.
        """
        n = len(tokens)
        mask = [True] * n
        optimal_mask = mask[:]

        f_optimal, best_solution = self._evaluate_fitness(_apply_mask(tokens, mask), problem)
        print(f"[TAPruning] Initial fitness: {f_optimal:.3f}, tokens: {n}")

        iters_done = 0
        converged = False
        while not converged and iters_done < self.num_iterations:
            improved_this_pass = False
            active = [i for i, m in enumerate(mask) if m]

            for idx in active:
                if iters_done >= self.num_iterations:
                    break
                new_mask = mask[:]
                new_mask[idx] = False

                if _mask_length(new_mask) < self.min_prompt_length:
                    continue

                f_new, sol = self._evaluate_fitness(_apply_mask(tokens, new_mask), problem)
                iters_done += 1

                if f_new > f_optimal:
                    mask = new_mask
                    optimal_mask = new_mask[:]
                    f_optimal = f_new
                    best_solution = sol
                    improved_this_pass = True
                    print(f"[TAPruning] iter {iters_done}: improved to {f_optimal:.3f}, "
                          f"tokens remaining: {_mask_length(optimal_mask)}")
                elif f_new >= f_optimal * self.delta:
                    mask = new_mask  # accept within threshold, don't update optimal

            if not improved_this_pass:
                converged = True

        print(f"[TAPruning] Done. fitness={f_optimal:.3f}, "
              f"tokens={_mask_length(optimal_mask)}/{n}, iters={iters_done}")
        return optimal_mask, f_optimal, best_solution

    # ------------------------------------------------------------------
    # Algorithm 2 — SAHCPruning (Steepest-Ascent Hill Climbing)
    # ------------------------------------------------------------------

    def _sahc_pruning(
        self,
        tokens: List[str],
        problem: str,
    ) -> Tuple[List[bool], float, str]:
        """
        Algorithm 2 from paper: SAHCPruning.

        Evaluate ALL possible single-token removals, accept only the BEST.
        Computationally expensive; included for completeness.
        """
        n = len(tokens)
        mask = [True] * n
        optimal_mask = mask[:]

        f_optimal, best_solution = self._evaluate_fitness(_apply_mask(tokens, mask), problem)
        print(f"[SAHCPruning] Initial fitness: {f_optimal:.3f}, tokens: {n}")

        iters_done = 0
        converged = False
        while not converged and iters_done < self.num_iterations:
            active = [i for i, m in enumerate(mask) if m]
            best_candidate_idx = -1
            best_candidate_score = f_optimal
            best_candidate_sol = best_solution

            for idx in active:
                if iters_done >= self.num_iterations:
                    break
                new_mask = mask[:]
                new_mask[idx] = False

                if _mask_length(new_mask) < self.min_prompt_length:
                    continue

                f_new, sol = self._evaluate_fitness(_apply_mask(tokens, new_mask), problem)
                iters_done += 1

                if f_new > best_candidate_score:
                    best_candidate_score = f_new
                    best_candidate_idx = idx
                    best_candidate_sol = sol

            if best_candidate_idx != -1:
                mask[best_candidate_idx] = False
                optimal_mask = mask[:]
                f_optimal = best_candidate_score
                best_solution = best_candidate_sol
                print(f"[SAHCPruning] iter {iters_done}: fitness={f_optimal:.3f}, "
                      f"tokens={_mask_length(optimal_mask)}")
            else:
                converged = True

        print(f"[SAHCPruning] Done. fitness={f_optimal:.3f}, "
              f"tokens={_mask_length(optimal_mask)}/{n}, iters={iters_done}")
        return optimal_mask, f_optimal, best_solution

    # ------------------------------------------------------------------
    # Algorithm 5 — SSGA (Steady-state GA, default)
    # ------------------------------------------------------------------

    def _ssga(
        self,
        tokens: List[str],
        problem: str,
    ) -> Tuple[List[bool], float, str]:
        """
        Algorithm 5 from paper: PROMPTQUINE Steady-state GA (SSGA).

        More exploratory than GGA. Default for 1-shot ICL pruning.
        Uses regularized evolution: only new offspring compete for
        population inclusion, preventing premature convergence (§D.7).
        """
        k = max(1, int(self.tournament_ratio * self.population_size))
        n = len(tokens)
        initial_mask = [True] * n

        # "We initialize the entire population with duplicates of ICL prompts" (paper §3.4)
        population: List[List[bool]] = [initial_mask[:] for _ in range(self.population_size)]
        scores: List[float] = []
        solutions: List[str] = []

        print(f"[SSGA] Evaluating initial population (size={self.population_size})...")
        for mask in population:
            s, sol = self._evaluate_fitness(_apply_mask(tokens, mask), problem)
            scores.append(s)
            solutions.append(sol)

        # H = history of all evaluated (mask, score, solution)
        history: List[Tuple[List[bool], float, str]] = list(
            zip(population, scores, solutions)
        )

        iters_done = len(population)
        best_score = max(scores)
        best_solution = solutions[scores.index(best_score)]
        print(f"[SSGA] Initial best fitness: {best_score:.3f}")

        iteration = 0
        while iters_done < self.num_iterations:
            iteration += 1

            # Check minimum length (paper termination criterion)
            if _mean_mask_length(population) < self.min_prompt_length:
                print(f"[SSGA] Stopping: mean prompt length below {self.min_prompt_length}")
                break

            new_offspring_masks: List[List[bool]] = []
            new_offspring_scores: List[float] = []
            new_offspring_sols: List[str] = []

            for _ in range(self.offspring_size):
                if iters_done >= self.num_iterations:
                    break

                # copy-then-mutate
                parent_idx = _tournament_select(population, scores, k)
                child_mask = _copy_then_mutate(population[parent_idx], self.mutation_rate_choices)

                if _mask_length(child_mask) < self.min_prompt_length:
                    continue

                child_score, child_sol = self._evaluate_fitness(
                    _apply_mask(tokens, child_mask), problem
                )
                iters_done += 1

                new_offspring_masks.append(child_mask)
                new_offspring_scores.append(child_score)
                new_offspring_sols.append(child_sol)
                history.append((child_mask, child_score, child_sol))

                if child_score > best_score:
                    best_score = child_score
                    best_solution = child_sol

            if not new_offspring_masks:
                break

            # "g ← P[#p :]" then "P ← g[:p]" — regularized evolution
            # New population = top #p from the new offspring
            offspring_ranked = sorted(
                zip(new_offspring_masks, new_offspring_scores, new_offspring_sols),
                key=lambda x: x[1],
                reverse=True,
            )
            take = min(self.population_size, len(offspring_ranked))
            population = [x[0] for x in offspring_ranked[:take]]
            scores = [x[1] for x in offspring_ranked[:take]]
            solutions = [x[2] for x in offspring_ranked[:take]]

            if iteration % 10 == 0:
                print(f"[SSGA] iter {iteration}: best={best_score:.3f}, "
                      f"pop_best={max(scores):.3f}, "
                      f"mean_len={_mean_mask_length(population):.1f}, "
                      f"evals={iters_done}")

        # Re-ranking: elite calibration (paper §4.1, §D.3)
        # "Select top k% from history by fitness, then pick highest-performant"
        history.sort(key=lambda x: x[1], reverse=True)
        elite_k = max(1, int(len(history) * self.elite_frac))
        elite = history[:elite_k]

        # Best from elite = final optimal prompt
        best_mask, best_score, best_solution = max(elite, key=lambda x: x[1])
        print(f"[SSGA] Done. evals={iters_done}, elite_size={elite_k}, "
              f"best_fitness={best_score:.3f}, "
              f"tokens={_mask_length(best_mask)}/{n}")
        return best_mask, best_score, best_solution

    # ------------------------------------------------------------------
    # Algorithm 4 — GGA (Generational GA)
    # ------------------------------------------------------------------

    def _gga(
        self,
        tokens: List[str],
        problem: str,
    ) -> Tuple[List[bool], float, str]:
        """
        Algorithm 4 from paper: PROMPTQUINE Generational GA (GGA).

        Well-suited for parallelization. Used for many-shot ICL pruning.
        Faster convergence than SSGA (typically within 3,000 iterations).
        """
        k = max(1, int(self.tournament_ratio * self.population_size))
        n = len(tokens)
        initial_mask = [True] * n

        population = [initial_mask[:] for _ in range(self.population_size)]
        scores = []
        solutions = []

        print(f"[GGA] Evaluating initial population (size={self.population_size})...")
        for mask in population:
            s, sol = self._evaluate_fitness(_apply_mask(tokens, mask), problem)
            scores.append(s)
            solutions.append(sol)

        history = list(zip(population[:], scores[:], solutions[:]))
        iters_done = len(population)
        best_score = max(scores)
        best_solution = solutions[scores.index(best_score)]
        print(f"[GGA] Initial best fitness: {best_score:.3f}")

        iteration = 0
        while iters_done < self.num_iterations:
            iteration += 1

            if _mean_mask_length(population) < self.min_prompt_length:
                print(f"[GGA] Stopping: mean prompt length below {self.min_prompt_length}")
                break

            # "Initialize g ← Empty"
            gen_masks: List[List[bool]] = []
            gen_scores: List[float] = []
            gen_sols: List[str] = []

            for _ in range(self.offspring_size):
                if iters_done >= self.num_iterations:
                    break

                parent_idx = _tournament_select(population, scores, k)
                child_mask = _copy_then_mutate(population[parent_idx], self.mutation_rate_choices)

                if _mask_length(child_mask) < self.min_prompt_length:
                    continue

                child_score, child_sol = self._evaluate_fitness(
                    _apply_mask(tokens, child_mask), problem
                )
                iters_done += 1
                gen_masks.append(child_mask)
                gen_scores.append(child_score)
                gen_sols.append(child_sol)
                history.append((child_mask, child_score, child_sol))

                if child_score > best_score:
                    best_score = child_score
                    best_solution = child_sol

            if not gen_masks:
                break

            # "Sort g in descending order of Score. Update population P ← g[:p]"
            gen_ranked = sorted(
                zip(gen_masks, gen_scores, gen_sols),
                key=lambda x: x[1],
                reverse=True,
            )
            take = min(self.population_size, len(gen_ranked))
            population = [x[0] for x in gen_ranked[:take]]
            scores = [x[1] for x in gen_ranked[:take]]
            solutions = [x[2] for x in gen_ranked[:take]]

            if iteration % 10 == 0:
                print(f"[GGA] gen {iteration}: best={best_score:.3f}, "
                      f"gen_best={max(gen_scores) if gen_scores else 0:.3f}, "
                      f"evals={iters_done}")

        # Re-ranking
        history.sort(key=lambda x: x[1], reverse=True)
        elite_k = max(1, int(len(history) * self.elite_frac))
        elite = history[:elite_k]
        best_mask, best_score, best_solution = max(elite, key=lambda x: x[1])

        print(f"[GGA] Done. evals={iters_done}, best_fitness={best_score:.3f}, "
              f"tokens={_mask_length(best_mask)}/{n}")
        return best_mask, best_score, best_solution

    # ------------------------------------------------------------------
    # Search dispatcher
    # ------------------------------------------------------------------

    def _run_search(
        self,
        problem: str,
    ) -> Tuple[str, float, str]:
        """
        Run the selected search algorithm on the current problem.

        Returns (best_preamble, best_score, best_solution).
        """
        # Determine starting preamble:
        # if we have a previously discovered best preamble, start from it;
        # if encyclopedia has optimized prompts, prefer those.
        start_preamble = self.initial_preamble
        if self._best_preamble is not None:
            start_preamble = self._best_preamble
        if self.encyclopedia_loaded and self.encyclopedia_dict:
            enc_prompts = {
                k: v for k, v in self.encyclopedia_dict.items()
                if "insight_evolveprompt" in k and "PROMPT:" in v
            }
            if enc_prompts:
                # Prefer the most recently added (last key)
                last_key = list(enc_prompts.keys())[-1]
                candidate = enc_prompts[last_key].split("PROMPT:")[-1].strip()
                if candidate:
                    start_preamble = candidate

        tokens = _tokenize(start_preamble)
        print(f"[EvolvePrompt] Starting search: method={self.search_method}, "
              f"preamble_tokens={len(tokens)}, max_evals={self.num_iterations}")

        if self.search_method == "ta":
            best_mask, best_score, best_solution = self._ta_pruning(tokens, problem)
        elif self.search_method == "sahc":
            best_mask, best_score, best_solution = self._sahc_pruning(tokens, problem)
        elif self.search_method == "gga":
            best_mask, best_score, best_solution = self._gga(tokens, problem)
        else:  # default: ssga
            best_mask, best_score, best_solution = self._ssga(tokens, problem)

        best_preamble = _apply_mask(tokens, best_mask)
        return best_preamble, best_score, best_solution

    # ------------------------------------------------------------------
    # Insight extraction
    # ------------------------------------------------------------------

    def _extract_insight(self, pruned_preamble: str, score: float) -> Dict[str, str]:
        """Convert the optimized pruned preamble into an encyclopedia insight entry."""
        tag = re.sub(r"\W+", "_", pruned_preamble[:30]).strip("_").lower()
        prompt = _INSIGHT_FROM_PRUNED.format(
            pruned_preamble=pruned_preamble,
            score=score,
            tag=tag,
        )
        response, _ = self._call_model(prompt, max_new_tokens=512)

        # Parse JSON
        start = response.find("{")
        if start == -1:
            key = f"insight_evolveprompt_{tag}"
            return {key: f"Evolved prompt (score={score:.3f}) | PROMPT: {pruned_preamble}"}

        brace_count = 0
        in_string = False
        escape_next = False
        end = start
        for i in range(start, len(response)):
            c = response[i]
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
        json_str = response[start:end + 1]
        try:
            json_str = re.sub(r",\s*}", "}", json_str)
            parsed = json.loads(json_str)
            insights = {}
            for k, v in parsed.items():
                name = k if k.startswith("insight_") else f"insight_{k}"
                desc = str(v).strip()
                # Ensure preamble is embedded in description for future loading
                if "PROMPT:" not in desc:
                    desc += f" | PROMPT: {pruned_preamble}"
                if len(desc) >= 20:
                    insights[name] = desc
            return insights
        except json.JSONDecodeError:
            key = f"insight_evolveprompt_{tag}"
            return {key: f"Evolved prompt (score={score:.3f}) | PROMPT: {pruned_preamble}"}

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
        Solve a problem using PROMPTQUINE evolutionary prompt search.

        Per-problem pipeline (mirrors Algorithm 3 from paper):
          1. Initialize token population from current best preamble
          2. Run evolutionary search (SSGA/GGA/TAPruning) to prune preamble
          3. Solve problem with best discovered pruned preamble
          4. Extract insight: store optimized preamble as encyclopedia entry
          5. Update persistent best preamble for next call

        Returns a dict with the same shape as client.py's solve_problem(),
        with insight_book containing the optimized pruned prompt.
        """
        if task is not None:
            self.task = task
        problem = self.task

        print(f"\n[EvolvePrompt] Problem: {problem[:100]}...\n")
        self.insight_book = {}

        # Run evolutionary search
        print("[EvolvePrompt] Step 1: Running evolutionary prompt search...")
        best_preamble, best_score, solution = self._run_search(problem)

        print(f"\n[EvolvePrompt] Best preamble found (score={best_score:.3f}):")
        print(f"  '{best_preamble}'")

        # Update persistent best
        if best_score > self._best_score:
            self._best_score = best_score
            self._best_preamble = best_preamble

        # Solve with the best prompt one final time (already done during search)
        # The solution from the search is already the best-preamble solution

        # Extract insight from the optimized preamble
        print("[EvolvePrompt] Step 2: Extracting insight from optimized preamble...")
        insights = self._extract_insight(best_preamble, best_score)
        self.insight_book.update(insights)

        result = {
            "problem": problem,
            "task": self.task,
            "solution": solution,
            "reflection": f"Evolved preamble via {self.search_method.upper()} "
                          f"(score={best_score:.3f}): '{best_preamble}'",
            "skills_extracted": self.insight_book,
            "skills_used": list(self.insight_book.keys()),
            "validation_errors": [],
            "insight_book": self.insight_book,
            "total_steps": 2,
            "token_info": {},
            # EvolvePrompt-specific fields
            "optimized_prompt": best_preamble,
            "fitness_score": best_score,
            "search_method": self.search_method,
            "original_preamble": self.initial_preamble,
        }

        print(f"[EvolvePrompt] Done. insights={len(self.insight_book)}")
        return result

    # ------------------------------------------------------------------
    # Encyclopedia / persistence (matching client.py interface exactly)
    # ------------------------------------------------------------------

    def load_encyclopedia(self, encyclopedia_path: str, mode: str = "text"):
        """Load a single encyclopedia file."""
        try:
            if mode == "text":
                with open(encyclopedia_path, "r", encoding="utf-8") as f:
                    self.encyclopedia_dict = json.load(f)
                self.encyclopedia = json.dumps(self.encyclopedia_dict, indent=2)
                # Restore best preamble from encyclopedia if available
                for k, v in self.encyclopedia_dict.items():
                    if "insight_evolveprompt" in k and "PROMPT:" in v:
                        candidate = v.split("PROMPT:")[-1].strip()
                        if candidate:
                            self._best_preamble = candidate
                            break
                print(f"Loaded encyclopedia.json from {encyclopedia_path} "
                      f"({len(self.encyclopedia_dict)} insights)")
            else:
                with open(encyclopedia_path, "r", encoding="utf-8") as f:
                    self.encyclopedia = f.read()
                print(f"Loaded encyclopedia from {encyclopedia_path} "
                      f"({len(self.encyclopedia)} characters)")
            self.encyclopedia_loaded = True
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load encyclopedia from {encyclopedia_path}: {e}"
            )

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
            # Restore best preamble from merged encyclopedia
            for k, v in self.encyclopedia_dict.items():
                if "insight_evolveprompt" in k and "PROMPT:" in v:
                    candidate = v.split("PROMPT:")[-1].strip()
                    if candidate:
                        self._best_preamble = candidate
                        break
            print(
                f"Loaded {len(used)} encyclopedias (JSON), "
                f"total insights {len(self.encyclopedia_dict)} "
                f"(skipped {skipped_exact_dupes} exact duplicates, "
                f"added {collision_variants_added} collision variants)"
            )
        else:
            self.encyclopedia = "\n\n".join(merged_text_parts)
            self.encyclopedia_dict = {}
            print(f"Loaded {len(used)} encyclopedias (text), "
                  f"total chars {len(self.encyclopedia)}")
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
            safe_name = re.sub(
                r"[^\w\s-]", "", reasoning_result.get("problem", "evolveprompt")[:50]
            )
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

    def save_best_preamble(self, path: Optional[str] = None):
        """Save the best discovered preamble for resumption."""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = path or str(output_dir / "evolveprompt_best.json")
        data = {
            "best_preamble": self._best_preamble,
            "best_score": self._best_score,
            "search_method": self.search_method,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved best preamble (score={self._best_score:.3f}) to: {save_path}")

    def load_best_preamble(self, path: str):
        """Load a previously saved best preamble."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._best_preamble = data.get("best_preamble")
        self._best_score = data.get("best_score", 0.0)
        print(f"Loaded best preamble (score={self._best_score:.3f}) from: {path}")
