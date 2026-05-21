"""
Metacognitive Reuse Client
Implements the exact pipeline from:
"Metacognitive Reuse: Turning Recurring LLM Reasoning Into Concise Behaviors"
(https://arxiv.org/abs/2509.13237)

Three-phase behavior extraction pipeline:
  Phase 1: Solution Generation — LLM solves the problem producing reasoning trace + answer
  Phase 2: Self-Reflection    — LLM evaluates logical soundness and identifies reusable behaviors
  Phase 3: Behavior Distillation — LLM converts (question, solution, reflection) into (name, instruction) behavior pairs

Behavior-Conditioned Inference (BCI):
  Given question Q and relevant behaviors B from the handbook, produce solution S = LLM(B, Q)
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from utils import (
    HAS_GEMINI,
    check_cuda,
    setup_gemini,
    call_gemini,
    call_openrouter,
    load_hf_model,
    call_hf_model,
)


class MetacognitiveClient:
    """
    Metacognitive Reuse client implementing the exact pipeline from the paper.

    Phases:
      1. Solution Generation: Solve the problem with full reasoning trace
      2. Self-Reflection: Evaluate solution and identify reusable behaviors
      3. Behavior Distillation: Extract (name → instruction) behavior pairs

    Behavior-Conditioned Inference:
      Use stored behaviors to guide solving new problems.
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
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.reasoning_steps = []
        self.load_in_8bit = load_in_8bit
        self.task = task or "Solve the given problem step by step."
        self.behavior_book = {}  # Store extracted behaviors: {name: instruction}

        # API support
        self.use_api = use_api
        self.api_provider = api_provider
        self.api_key = api_key or (os.getenv("GEMINI_API_KEY") if api_provider == "gemini" else os.getenv("OPENROUTER_API_KEY"))
        self.api_model_name = None  # set externally if needed
        if self.use_api and self.api_provider == "gemini":
            self.gemini_model = setup_gemini(
                api_key=self.api_key,
                model_name="gemini-3-pro-preview",
            )

        # Model and tokenizer will be loaded lazily on first use (only for HuggingFace models)
        self.model = None
        self.tokenizer = None
        self.device = device or ("cuda" if check_cuda() else "cpu")

    def _load_model(self):
        """Lazy load the Hugging Face model and tokenizer"""
        if self.model is not None and self.tokenizer is not None:
            return
        self.model, self.tokenizer = load_hf_model(
            self.model_name, device=self.device, load_in_8bit=self.load_in_8bit,
        )

    def _call_model(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> tuple:
        """
        Call the language model (HuggingFace or Gemini API).

        Returns:
            Tuple of (generated_text, token_info_dict)
        """
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
    # Phase 1: Solution Generation
    # ------------------------------------------------------------------
    def _get_solution_prompt(
        self,
        problem: str,
        custom_instruction: Optional[str] = None,
        behavior_section: Optional[str] = None,
    ) -> str:
        """Generate solution prompt.

        Following the paper: the Solution Prompt maps questions to solutions
        containing reasoning traces and final answers.

        Args:
            problem: The problem text to solve
            custom_instruction: Optional custom instruction to append
            behavior_section: Pre-formatted behavior section for BCI
        """
        behavior_text = behavior_section or ""
        custom_section = ""
        if custom_instruction:
            custom_section = f"\n\n{custom_instruction}"

        prompt = f"""{behavior_text}Problem: {problem}{custom_section}"""
        return prompt

    def _step_solution(
        self,
        problem: str,
        custom_instruction: Optional[str] = None,
        behavior_section: Optional[str] = None,
    ) -> Dict:
        """Phase 1: Generate solution with full reasoning trace.

        The Metacognitive Strategist (LLM) produces a solution containing
        the reasoning trace and final answer.
        """
        prompt = self._get_solution_prompt(
            problem,
            custom_instruction=custom_instruction,
            behavior_section=behavior_section,
        )

        response, token_info = self._call_model(prompt, None, max_new_tokens=32768)
        print(f"Solution Response: {response}")
        print(f"Phase 1 Output Tokens: {token_info['output_tokens']}")

        step_result = {
            "step": 1,
            "name": "Solution Generation",
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
            "token_info": token_info,
        }

        self.reasoning_steps.append(step_result)
        return step_result

    # ------------------------------------------------------------------
    # Phase 2: Self-Reflection
    # ------------------------------------------------------------------
    def _get_reflection_prompt(self, problem: str, solution: str) -> str:
        """Phase 2: Self-Reflection Prompt.

        Following the paper: the model evaluates whether reasoning is
        logically sound, the answer correct, and whether any new, reusable
        behaviors can be distilled to streamline future problem solving.
        """
        prompt = f"""You are a metacognitive evaluator. Analyze the following solution to determine:
1. Whether the reasoning is logically sound
2. Whether the answer is correct
3. Whether any new, reusable behaviors can be distilled to streamline future problem solving

Problem:
{problem}

Solution:
{solution}

Provide your self-reflection covering:

### Logical Soundness
- Is the reasoning chain valid? Are there any logical gaps or errors?
- Are all steps properly justified?

### Answer Correctness
- Does the final answer follow from the reasoning?
- Are there any computational or conceptual errors?

### Reusable Behaviors
- What recurring reasoning patterns or strategies were used?
- What concrete, actionable procedures could be extracted as reusable behaviors?
- What step-by-step methods could help solve similar problems in the future?

Focus on identifying concrete, reusable behaviors that can be distilled into concise (name, instruction) pairs for future problem solving."""
        return prompt

    def _step_reflection(self, problem: str, solution: str) -> Dict:
        """Phase 2: Self-Reflection.

        The model evaluates the solution's logical soundness, correctness,
        and identifies reusable behaviors that can be distilled.
        """
        prompt = self._get_reflection_prompt(problem, solution)

        response, token_info = self._call_model(prompt, None, max_new_tokens=4096)
        print(f"Reflection Response: {response}")

        step_result = {
            "step": 2,
            "name": "Self-Reflection",
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
        }

        self.reasoning_steps.append(step_result)
        return step_result

    # ------------------------------------------------------------------
    # Phase 3: Behavior Distillation
    # ------------------------------------------------------------------
    def _get_behavior_prompt(self, problem: str, solution: str, reflection: str) -> str:
        """Phase 3: Behavior Distillation Prompt.

        Following the paper: convert (question, solution, reflection) into
        (name, instruction) behavior pairs to be appended to the behavior handbook.

        Behaviors are defined as "name + instruction" — concise procedural hints
        that provide actionable guidance for problem-solving.
        """
        prompt = f"""You are a behavior distillation system. Your task is to convert the following problem-solving trace into reusable behaviors.

A behavior is defined as a (name, instruction) pair where:
- **name**: A concise identifier describing the behavior (e.g., systematic_counting, modular_arithmetic_check)
- **instruction**: A concise, actionable procedural hint that provides guidance for solving similar problems. The instruction should be specific enough to be directly useful but general enough to apply across similar problems.

Problem:
{problem}

Solution:
{solution}

Self-Reflection:
{reflection}

**Your Task:**
Extract all reusable behaviors from this problem-solving trace. Each behavior should capture a recurring reasoning pattern or strategy that can streamline future problem solving.

**What Makes a Good Behavior:**
- Concise: A behavior is a short procedural hint, not a full solution
- Actionable: Provides clear guidance on what to do
- Reusable: Applies to a class of similar problems, not just this one
- Specific: Contains concrete steps or checks, not vague advice

**Output Format (JSON):**
Output a JSON object where keys are behavior names and values are instruction strings:

{{"behavior_name": "instruction"}}

Format Rules:
- Use valid JSON format
- Each behavior name should be descriptive using snake_case (e.g., behavior_systematic_counting)
- Each behavior name must start with "behavior_"
- Each instruction is a concise string describing when and how to apply this behavior
- Keep instructions focused and actionable — they are procedural hints, not full explanations
- Escape quotes in instructions with backslash: \\"

**Example:**
{{
  "behavior_systematic_counting": "When counting possibilities across multiple positions or categories, examine each position's contribution independently then combine. This prevents missed cases and double-counting.",
  "behavior_modular_arithmetic_check": "When working with divisibility or remainder problems, reduce expressions modulo the divisor at each step. Check intermediate results mod n to simplify calculations and verify the final answer.",
  "behavior_extreme_case_verification": "After finding a general solution, verify it against extreme or boundary cases (n=0, n=1, maximum values). This catches off-by-one errors and edge cases that algebraic manipulation might miss."
}}

**Output your response as a valid JSON object only:**"""
        return prompt

    def _step_behavior_distillation(
        self, problem: str, solution: str, reflection: str
    ) -> Dict:
        """Phase 3: Behavior Distillation.

        Convert (question, solution, reflection) into (name, instruction)
        behavior pairs appended to the behavior handbook.
        """
        prompt = self._get_behavior_prompt(problem, solution, reflection)

        response, token_info = self._call_model(prompt, None, max_new_tokens=32768)
        print(f"Behavior Distillation Response: {response}")

        # Parse behaviors from JSON response
        behaviors = {}
        validation_errors = []

        try:
            # Method 1: Extract JSON from markdown code blocks
            json_code_block = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL
            )
            if json_code_block:
                json_str = json_code_block.group(1)
            else:
                # Method 2: Find JSON object in response
                start_idx = response.find("{")
                if start_idx != -1:
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    json_str = None
                    for i in range(start_idx, len(response)):
                        char = response[i]
                        if escape_next:
                            escape_next = False
                            continue
                        if char == "\\":
                            escape_next = True
                            continue
                        if char == '"' and not escape_next:
                            in_string = not in_string
                            continue
                        if not in_string:
                            if char == "{":
                                brace_count += 1
                            elif char == "}":
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = response[start_idx : i + 1]
                                    break
                    if json_str is None:
                        last_brace = response.rfind("}", start_idx)
                        if last_brace != -1:
                            json_str = response[start_idx : last_brace + 1]
                else:
                    json_str = None

            if json_str:
                try:
                    json_str = re.sub(r",\s*}", "}", json_str)
                    json_str = re.sub(r",\s*]", "]", json_str)

                    json_data = json.loads(json_str)

                    if isinstance(json_data, dict):
                        for bname, binstr in json_data.items():
                            # Ensure behavior name starts with behavior_
                            if not bname.startswith("behavior_"):
                                bname = f"behavior_{bname}"

                            # Normalize value to string
                            if isinstance(binstr, dict):
                                binstr = str(binstr)
                            elif isinstance(binstr, list):
                                binstr = " ".join(str(item) for item in binstr)
                            elif not isinstance(binstr, str):
                                binstr = str(binstr)

                            binstr = re.sub(r"\s+", " ", binstr).strip()

                            if len(binstr) >= 20:
                                behaviors[bname] = binstr
                            else:
                                validation_errors.append(
                                    f"Behavior '{bname}' has too short instruction"
                                )

                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error: {e}")
                    validation_errors.append(f"JSON parsing error: {e}")

            # Method 3: Fallback regex extraction
            if not behaviors:
                print("Warning: JSON parsing failed. Attempting regex extraction.")
                behavior_pattern = r'"(behavior_\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"'
                matches = re.findall(behavior_pattern, response)
                for bname, binstr in matches:
                    binstr = binstr.replace('\\"', '"').replace("\\n", " ")
                    binstr = re.sub(r"\s+", " ", binstr).strip()
                    if len(binstr) >= 20:
                        behaviors[bname] = binstr

        except Exception as e:
            print(f"Warning: Error parsing behaviors: {e}")
            validation_errors.append(f"Exception during parsing: {e}")

        if not behaviors:
            validation_errors.append("Could not extract any behaviors from response")

        # Filter valid behaviors
        valid_behaviors = {}
        for k, v in behaviors.items():
            if not k.startswith("behavior_"):
                continue
            if isinstance(v, str) and len(v.strip()) >= 20:
                valid_behaviors[k] = v
        if not valid_behaviors:
            print("WARNING: No valid behaviors extracted from this problem!")

        if validation_errors:
            print(f"Validation warnings ({len(validation_errors)}):")
            for error in validation_errors[:5]:
                print(f"  - {error}")

        print(
            f"Extracted {len(valid_behaviors)} valid behaviors: {list(valid_behaviors.keys())}"
        )

        step_result = {
            "step": 3,
            "name": "Behavior Distillation",
            "prompt": prompt,
            "response": response,
            "behaviors": behaviors,
            "valid_behaviors": valid_behaviors,
            "validation_errors": validation_errors,
            "timestamp": time.time(),
        }

        self.reasoning_steps.append(step_result)
        return step_result

    # ------------------------------------------------------------------
    # Main Pipeline: 3-Phase Behavior Extraction
    # ------------------------------------------------------------------
    def solve_and_extract_behaviors(
        self,
        task: Optional[str] = None,
        custom_solution_instruction: Optional[str] = None,
        behavior_section: Optional[str] = None,
    ) -> Dict:
        """
        Solve a problem and extract behaviors using the 3-phase metacognitive pipeline.

        Phase 1: Solution Generation — produce reasoning trace + answer
        Phase 2: Self-Reflection — evaluate solution, identify reusable behaviors
        Phase 3: Behavior Distillation — extract (name, instruction) pairs

        Args:
            task: The problem to solve. If None, uses default task.
            custom_solution_instruction: Optional instruction for Phase 1.
            behavior_section: Pre-formatted behavior section for BCI.

        Returns:
            Dictionary containing solution, reflection, extracted behaviors, and behavior book.
        """
        if task is not None:
            self.task = task

        problem = self.task

        print(f"Problem: {problem}\n")

        # Reset per-problem state
        self.reasoning_steps = []
        self.behavior_book = {}

        # Phase 1: Solution Generation
        print("Phase 1: Generating solution...")
        step1 = self._step_solution(
            problem,
            custom_instruction=custom_solution_instruction,
            behavior_section=behavior_section,
        )
        solution = step1["response"]
        time.sleep(1)

        # Phase 2: Self-Reflection
        print("Phase 2: Self-reflection on solution...")
        step2 = self._step_reflection(problem, solution)
        reflection = step2["response"]
        time.sleep(1)

        # Phase 3: Behavior Distillation
        print("Phase 3: Distilling behaviors...")
        step3 = self._step_behavior_distillation(problem, solution, reflection)
        time.sleep(1)

        # Update behavior_book with extracted behaviors
        extracted_behaviors = step3.get("valid_behaviors", step3.get("behaviors", {}))
        if extracted_behaviors:
            self.behavior_book.update(extracted_behaviors)
            print(f"Added {len(extracted_behaviors)} behaviors to behavior book")
        else:
            print("WARNING: No behaviors extracted from this problem!")

        result = {
            "problem": problem,
            "task": self.task,
            "solution": solution,
            "reflection": reflection,
            "behaviors_extracted": step3.get(
                "valid_behaviors", step3.get("behaviors", {})
            ),
            "behaviors_used": list(
                step3.get("valid_behaviors", step3.get("behaviors", {})).keys()
            ),
            "validation_errors": step3.get("validation_errors", []),
            "behavior_book": self.behavior_book,
            "total_steps": len(self.reasoning_steps),
            "token_info": step1.get("token_info", {}),
        }

        return result

    # ------------------------------------------------------------------
    # Behavior-Conditioned Inference (BCI)
    # ------------------------------------------------------------------
    def behavior_conditioned_inference(
        self,
        problem: str,
        behaviors: Dict[str, str],
        custom_instruction: Optional[str] = None,
    ) -> Dict:
        """
        Behavior-Conditioned Inference (BCI) from the paper.

        Given question Q and relevant behaviors B from the handbook,
        produce solution S = LLM(B, Q).

        The behaviors are provided as in-context procedural hints that
        guide the LLM's reasoning process.

        Args:
            problem: The question to solve
            behaviors: Dict of {behavior_name: instruction} from the handbook
            custom_instruction: Optional additional instruction

        Returns:
            Dictionary with solution and token info
        """
        # Format behaviors into a behavior section
        behavior_section = self._format_behavior_section(behaviors)

        prompt = self._get_solution_prompt(
            problem,
            custom_instruction=custom_instruction,
            behavior_section=behavior_section,
        )

        response, token_info = self._call_model(prompt, None, max_new_tokens=32768)
        print(f"BCI Solution Response: {response}")
        print(f"BCI Output Tokens: {token_info['output_tokens']}")

        result = {
            "problem": problem,
            "solution": response,
            "behaviors_used": list(behaviors.keys()),
            "num_behaviors": len(behaviors),
            "token_info": token_info,
        }

        return result

    def _format_behavior_section(self, behaviors: Dict[str, str]) -> str:
        """Format behaviors into a section for behavior-conditioned inference.

        Following the paper: behaviors are provided as in-context procedural
        hints that guide the LLM's reasoning.
        """
        if not behaviors:
            return ""

        behavior_lines = []
        for bname, binstr in behaviors.items():
            # Format as "name: <instruction>" following the paper's convention
            behavior_lines.append(f"{bname}: {binstr}")

        behaviors_text = "\n".join(behavior_lines)

        section = f"""
A behavior is a note or skill to keep in mind while solving math problems. It can be a strategy, a trick, or a technique. It can also be a general rule or a common sense principle. The behavior is not a solution to the problem, but it can be used to solve the problem. 

Here is a list of behaviors:

{behaviors_text}

Now, solve the following math problem efciently and clearly. You can use any of the behaviors above to solve the problem. 

In your reasoning, when you use a behavior explicitly refer to the behaviors when you use them. 
"""
        return section

    # ------------------------------------------------------------------
    # Behavior Book I/O
    # ------------------------------------------------------------------
    def save_behavior_book(
        self, result: Dict, output_path: Optional[str] = None
    ):
        """Save behavior book as JSON: {"behavior_name": "instruction"}"""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        behavior_book = result.get("behavior_book", {})
        if not behavior_book:
            print("No behaviors to save")
            return

        if output_path is None:
            safe_name = re.sub(
                r"[^\w\s-]", "", result.get("problem", "behaviors")[:50]
            )
            safe_name = re.sub(r"[-\s]+", "_", safe_name)
            output_path = str(output_dir / f"{safe_name}.json")
        else:
            if not os.path.isabs(output_path):
                output_path = str(output_dir / output_path)
            if not output_path.endswith(".json"):
                output_path += ".json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(behavior_book, f, indent=2, ensure_ascii=False)

        print(f"Saved behavior book to: {output_path}")

    @staticmethod
    def load_behavior_book(path: str) -> Dict[str, str]:
        """Load a behavior book from JSON file.

        Args:
            path: Path to JSON file containing behaviors

        Returns:
            Dict of {behavior_name: instruction}
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle wrapped format (from benchmark output)
        if isinstance(data, dict):
            if "behavior_book" in data:
                return data["behavior_book"]
            # Flat behavior dict
            return {k: v for k, v in data.items() if isinstance(v, str)}

        return {}

    @staticmethod
    def merge_behavior_books(behavior_books: List[Dict[str, str]]) -> Dict[str, str]:
        """Merge multiple behavior books into one combined book.

        Keeps first occurrence of each behavior name. If names collide
        with different instructions, suffixes _1, _2 etc. are added.

        Args:
            behavior_books: List of behavior book dicts

        Returns:
            Merged behavior book
        """
        merged = {}
        for book in behavior_books:
            for bname, binstr in book.items():
                if bname not in merged:
                    merged[bname] = binstr
                elif merged[bname] != binstr:
                    # Name collision with different instruction — add suffix
                    idx = 2
                    new_name = f"{bname}_{idx}"
                    while new_name in merged:
                        idx += 1
                        new_name = f"{bname}_{idx}"
                    merged[new_name] = binstr
        return merged

    @staticmethod
    def collect_behaviors_from_dir(directory: str) -> Dict[str, str]:
        """Collect all behaviors from problem_*.json files in a directory.

        Each problem file is expected to have a "behavior_book" key.

        Args:
            directory: Path to directory containing problem_*.json files

        Returns:
            Combined behavior book from all files
        """
        all_behaviors = {}
        if not os.path.isdir(directory):
            return all_behaviors

        problem_files = sorted(
            f
            for f in os.listdir(directory)
            if f.startswith("problem_") and f.endswith(".json")
        )

        for fname in problem_files:
            fpath = os.path.join(directory, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                behavior_book = data.get("behavior_book", {})
                for bname, binstr in behavior_book.items():
                    if bname not in all_behaviors:
                        all_behaviors[bname] = binstr
                    elif all_behaviors[bname] != binstr:
                        idx = 2
                        new_name = f"{bname}_{idx}"
                        while new_name in all_behaviors:
                            idx += 1
                            new_name = f"{bname}_{idx}"
                        all_behaviors[new_name] = binstr
            except Exception as e:
                print(f"Warning: Failed to read {fpath}: {e}")

        return all_behaviors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Metacognitive Reuse Client — 3-Phase Behavior Extraction Pipeline"
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default=None,
        help="Problem/question to solve (required for behavior extraction)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1",
        help="Hugging Face model name to use (default: deepseek-ai/DeepSeek-R1)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda' or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use an API provider instead of HuggingFace model",
    )
    parser.add_argument(
        "--api-provider",
        type=str,
        default="gemini",
        choices=["gemini", "openrouter"],
        help="Which API provider to use (default: gemini)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the chosen provider (or set GEMINI_API_KEY / OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output",
        help="Output directory for saving results (default: output)",
    )
    parser.add_argument(
        "--load-in-8bit",
        type=bool,
        default=False,
        help="Load model with 8-bit quantization (default: False, uses FP16 instead)",
    )

    args = parser.parse_args()

    client = MetacognitiveClient(
        model_name=args.model,
        task=args.task,
        device=args.device,
        use_api=args.use_api,
        api_key=args.api_key,
        api_provider=args.api_provider,
        output_dir=args.output,
        load_in_8bit=args.load_in_8bit,
    )

    try:
        if not args.task:
            print("Error: Please provide a problem/question using -t or --task")
            print(
                "Example: python client_metacognitive.py -t 'Find the area of a circle with radius 4'"
            )
            exit(1)

        # Run the 3-phase pipeline
        result = client.solve_and_extract_behaviors(task=args.task)
        client.save_behavior_book(result)

        if result:
            print("\n" + "=" * 80)
            print("METACOGNITIVE REUSE PIPELINE COMPLETE")
            print("=" * 80)
            print(f"Solution: {result.get('solution', 'N/A')}")
            print(
                f"\nBehaviors Extracted: {len(result.get('behaviors_extracted', {}))}"
            )
            print(f"Behaviors Used: {result.get('behaviors_used', [])}")
            if result.get("validation_errors"):
                print(
                    f"Validation Warnings: {len(result.get('validation_errors', []))}"
                )
            print("\n" + "=" * 80)
            print("EXTRACTED BEHAVIORS")
            print("=" * 80)
            for bname, binstr in result.get("behavior_book", {}).items():
                display_name = bname.replace("behavior_", "").replace("_", " ")
                print(f"\n{display_name} -> {binstr}")
            print("\n" + "=" * 80)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print("\nMake sure you have:")
        print("1. Installed required packages: pip install -r requirements.txt")
        print("2. For GPU support, ensure CUDA is properly installed")
        print("\nExample usage:")
        print(
            "  python client_metacognitive.py -t 'Find the area of a circle with radius 4'"
        )
        print(
            "  python client_metacognitive.py -t 'Solve: 2x + 3 = 7' --use-api"
        )
