"""
Behavior Curation Pipeline Client
Implements metacognitive reuse for LLM reasoning based on:
"Metacognitive Reuse: Turning Recurring LLM Reasoning Into Concise Behaviors"
Uses a three-stage pipeline: Solution → Reflection → Insight Extraction
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


class ChainOfThoughtReader:
    """
    A task-agnostic client for behavior curation pipeline based on
    "Metacognitive Reuse: Turning Recurring LLM Reasoning Into Concise Behaviors"
    Implements three-stage pipeline: Solution → Reflection → Insight Extraction
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
        self.insight_book = {}  # Store extracted behaviors

        # Encyclopedia support (for solving with learned insights)
        self.encyclopedia = ""
        self.encyclopedia_dict = {}
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

        # Model and tokenizer will be loaded lazily on first use (only for HuggingFace models)
        self.model = None
        self.tokenizer = None
        self.device = device or ("cuda" if check_cuda() else "cpu")

    def _load_model(self):
        """Lazy load the Hugging Face model and tokenizer"""
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

    def load_encyclopedia(self, encyclopedia_path: str, mode: str = "text"):
        """Load encyclopedia for solving problems with learned insights.

        Args:
            encyclopedia_path: Path to encyclopedia file
            mode: "text" for encyclopedia.json or "normal" for encyclopedia.txt
        """
        try:
            if mode == "text":
                # Text mode: Load encyclopedia.json
                with open(encyclopedia_path, "r", encoding="utf-8") as f:
                    self.encyclopedia_dict = json.load(f)
                self.encyclopedia = json.dumps(self.encyclopedia_dict, indent=2)
                print(
                    f"Loaded encyclopedia.json from {encyclopedia_path} ({len(self.encyclopedia_dict)} insights)"
                )
            else:
                # Normal mode: Load encyclopedia.txt
                with open(encyclopedia_path, "r", encoding="utf-8") as f:
                    self.encyclopedia = f.read()
                print(
                    f"Loaded encyclopedia from {encyclopedia_path} ({len(self.encyclopedia)} characters)"
                )

            self.encyclopedia_loaded = True
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load encyclopedia from {encyclopedia_path}: {e}"
            )

    def load_encyclopedias(self, encyclopedia_paths: List[str], mode: str = "text"):
        """Load and merge multiple encyclopedias.

        - In "text" mode: merges JSON dictionaries (or lists of {name, description}).
        - In "normal" mode: concatenates text files.
        The first occurrence of an insight name is kept when merging.
        """
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
                            # Skip if exact duplicate description already recorded
                            if any(e.get("desc", "") == desc for e in entries):
                                skipped_exact_dupes += 1
                                continue
                            # Determine unique suffixed name (name_1, name_2, ...)
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
                f"Loaded {len(used)} encyclopedias (JSON), total insights {len(self.encyclopedia_dict)} (skipped {skipped_exact_dupes} exact duplicates, added {collision_variants_added} collision variants)"
            )
        else:
            self.encyclopedia = "\n\n".join(merged_text_parts)
            self.encyclopedia_dict = {}
            print(
                f"Loaded {len(used)} encyclopedias (text), total chars {len(self.encyclopedia)}"
            )
        self.encyclopedia_loaded = True

    def load_rag_store(self, file_search_store_name: str, api_key: str):
        """Load a Google File Search store for RAG-based problem solving."""
        try:
            import google.genai as genai_new
            from google.genai import types
        except ImportError:
            raise ImportError("google-genai package is required for RAG functionality")

        # Initialize client
        genai_client = genai_new.Client(api_key=api_key)

        # Store the RAG store reference
        self.rag_store_name = file_search_store_name
        self.rag_client = genai_client
        self.rag_types = types
        self.rag_loaded = True

        print(f"Loaded RAG store: {file_search_store_name}")

    # REMOVED: grade_proof_with_gemini moved to math_datasets/imo_benchmark.py
    # This function is specific to IMO-ProofBench and should not be in the general client

    def _get_solution_prompt(
        self,
        problem: str,
        custom_instruction: Optional[str] = None,
        insights_section: Optional[str] = None,
    ) -> str:
        """Generate solution prompt with or without encyclopedia/RAG.

        Args:
            problem: The problem text to solve
            custom_instruction: Optional custom instruction to append to the prompt
            insights_section: Pre-formatted insights section (from caller)
        """
        # Use provided insights_section or empty string
        insights_text = insights_section or ""

        # Add RAG retrieval if RAG is loaded
        rag_section = ""
        if hasattr(self, 'rag_loaded') and self.rag_loaded:
            try:
                # Perform RAG retrieval using file search store
                search_response = self.rag_client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=f"Search for relevant reasoning traces and techniques for solving this problem: {problem}",
                    config=self.rag_types.GenerateContentConfig(
                        tools=[self.rag_types.Tool(
                            file_search=self.rag_types.FileSearch(
                                file_search_store=self.rag_store_name
                            )
                        )]
                    )
                )
                
                # Extract relevant traces from search results
                if search_response.candidates and search_response.candidates[0].content:
                    rag_results = []
                    for part in search_response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            rag_results.append(part.text)
                    
                    if rag_results:
                        rag_section = "\n\nRelevant Reasoning Traces:\n" + "\n".join(rag_results[:3])  # Limit to top 3 results
                        
            except Exception as e:
                print(f"Warning: RAG retrieval failed: {e}")
                rag_section = ""

        # Add custom instruction if provided
        custom_section = ""
        if custom_instruction:
            custom_section = f"\n\n{custom_instruction}"

        prompt = f"""{insights_text}{rag_section}Problem: {problem}{custom_section}"""
        return prompt

    def _get_reflection_prompt(self, problem: str, solution: str) -> str:
        """Step 2: Extract procedural knowledge and reusable patterns for reasoning traces"""
        prompt = f"""
Analyze the solution below to extract procedural knowledge that reflect the reasoning traces.

Problem:
{problem}

Step-by-Step Solution:
{solution}

Your task: Extract the fundamental techniques used in reslution that can be packaged as reasoning traces. Focus on:

1. What step-by-step procedures were used? How can these be repeated?
2. What conditions determined which approach to use? When should each technique apply?
3. What methods, strategies, or workflows can be applied to similar problems?
4. What made this approach effective? What should someone know to use it correctly?
5. What types of problems would benefit from these techniques?

Output your analysis covering:

### I. Procedural Knowledge
- Break down the solution into clear, repeatable procedures
- The extracted traces should be concrete solutions rather than general principles.

### II. Reusable Techniques and Methods
- List specific techniques, strategies, or workflows used
- The techniques should be solid on practical questions rather than very general and high-level principles.
- For each technique, identify:
  * When it should be used (conditions/triggers)
  * How it was applied (concrete steps)
  * Why it was effective (insights)
  * What problems it could solve (applicability)

### III. Critical Insights and Guidelines
- What key insights made this solution work?
- What common pitfalls should be avoided?
- What variations or edge cases should be considered?

Focus on extracting actionable, procedural knowledge that can be packaged as reusable insights for similar problems.
"""
        return prompt

    def _get_behavior_prompt(self, problem: str, solution: str, reflection: str) -> str:
        """Step 3: Generate insights following Anthropic format - procedural knowledge with instructions"""
        prompt_template = """
Extract reasoning traces from the solution below. Analyze the solution and reflection to identify concrete, actionable traces that similar problems can be solved via the traces

Problem: {problem}

Solution: {solution}

Reflection: {reflection}

**Your Task:**
Identify and extract all reusable reasoning traces, techniques, and methods used in the solution. Each trace should be a concrete procedure that can guide someone to solve similar problems.

**What Makes a Good Reasoning Trace:**
- A specific technique or method that was used in the solution.
- Something that can be applied to similar problems, not just this one.
- Includes guidance on when and how to use it with clear steps that can be followed if necessary.
- Not repeatance of already well-known or commonly adopted techniques.
- Not too general and high-level but contains actional procedural knowledge.

**Description Must Include:**
1. **Core idea**: The fundamental concept of what this trace is about. What is the main technique or method? What does it do?

2. **When to use**: Explain when this skill should be applied. What types of problems? What conditions must be met? What situations trigger this skill?

**Output Format (Simple JSON):**
Output a simple JSON object with skill names as keys and descriptions as string values:

{{"trace_name": "description"}}

Format Rules:
- Use valid JSON format
- Each trace name must start with "trace_"
- Keep JSON simple - no nested objects, just key-value pairs
- Escape quotes in descriptions with backslash: \\"

**Example:**
{{
  "trace_polynomialFactoring": "The major idea is how we can turn a polynomial into a product of simpler expressions. This skill is particularly useful for quadratic and higher-degree polynomial equations where factoring can simplify the problem. Factoring reduces complex polynomials to simpler equations. When solving equations with polynomial expressions that can be factored, especially when the polynomial has recognizable patterns like difference of squares (a²-b²), perfect square trinomials (a²±2ab+b²), or common factors.  
  "trace_depthFirstSearchImplementation": "This algorithm is essential for problems involving path finding, cycle detection, topological sorting, connected components, or exploring all possible solutions in a search space. DFS explores depth before breadth, using stack-based recursion or explicit stack. It is memory-efficient for deep structures and naturally handles backtracking. The visited set prevents infinite loops and redundant work. DFS is the foundation for many graph algorithms including topological sort, strongly connected components, and maze solving. When you need to explore or traverse a graph, tree, or nested structure systematically, going as deep as possible before backtracking. Use DFS when you need to visit all nodes in a connected component, find paths between nodes, detect cycles, or explore recursive structures like file systems, nested data, or game states. "
}}

**Output your response as a valid JSON object only:**
"""
        prompt = prompt_template.format(
            problem=problem, solution=solution, reflection=reflection
        )
        return prompt

    def _step_solution(
        self,
        problem: str,
        custom_instruction: Optional[str] = None,
        insights_section: Optional[str] = None,
    ) -> Dict:
        """Step 1: Generate solution using Solution Prompt

        Args:
            problem: The problem to solve
            custom_instruction: Optional custom instruction to append to the prompt
            insights_section: Pre-formatted insights section (from caller)
        """
        prompt = self._get_solution_prompt(
            problem,
            custom_instruction=custom_instruction,
            insights_section=insights_section,
        )

        system_prompt = None
        response, token_info = self._call_model(
            prompt, system_prompt, max_new_tokens=32768
        )
        print(f"Solution Response: {response}")

        # Log token usage for Step 1
        print(f"Step 1 Output Tokens: {token_info['output_tokens']}")

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

    def _step_reflection(self, problem: str, solution: str) -> Dict:
        """Step 2: Generate reflection using Reflection Prompt"""
        prompt = self._get_reflection_prompt(problem, solution)

        system_prompt = None
        # Step 2: Use 4096 tokens for reflection (needs more tokens for detailed critique)
        response, token_info = self._call_model(
            prompt, system_prompt, max_new_tokens=4096
        )
        print(f"Reflection Response: {response}")

        step_result = {
            "step": 2,
            "name": "Reflection & Insight Extraction",
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
        }

        self.reasoning_steps.append(step_result)
        return step_result

    def _step_behavior_extraction(
        self, problem: str, solution: str, reflection: str
    ) -> Dict:
        """Step 3: Extract actionable skills using enhanced Behavior Prompt"""
        prompt = self._get_behavior_prompt(problem, solution, reflection)

        system_prompt = None
        response, token_info = self._call_model(
            prompt, system_prompt, max_new_tokens=32768
        )
        print(f"Skill Extraction Response: {response}")

        # Simple JSON extraction: parse skills from JSON format
        skills = {}
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
                    # Find matching closing brace
                    brace_count = 0
                    in_string = False
                    escape_next = False
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
                    else:
                        # If no complete match, try last brace
                        last_brace = response.rfind("}", start_idx)
                        if last_brace != -1:
                            json_str = response[start_idx : last_brace + 1]
                        else:
                            json_str = None
                else:
                    json_str = None

            if json_str:
                try:
                    # Simple cleanup
                    json_str = re.sub(r",\s*}", "}", json_str)
                    json_str = re.sub(r",\s*]", "]", json_str)

                    # Parse JSON
                    json_data = json.loads(json_str)

                    if isinstance(json_data, dict):
                        for insight_name, insight_desc in json_data.items():
                            # Ensure skill name starts with insight_
                            if not insight_name.startswith("insight_"):
                                insight_name = f"insight_{insight_name}"

                            # Convert to string and normalize
                            if isinstance(insight_desc, dict):
                                insight_desc = str(insight_desc)
                            elif isinstance(insight_desc, list):
                                insight_desc = " ".join(
                                    str(item) for item in insight_desc
                                )
                            elif not isinstance(insight_desc, str):
                                insight_desc = str(insight_desc)

                            # Normalize whitespace
                            insight_desc = re.sub(r"\s+", " ", insight_desc).strip()

                            # Validate
                            if len(insight_desc) >= 20:
                                skills[insight_name] = insight_desc
                            else:
                                validation_errors.append(
                                    f"Skill '{insight_name}' has too short description"
                                )

                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error: {e}")
                    validation_errors.append(f"JSON parsing error: {e}")

            # Method 3: Fallback - extract using regex if JSON parsing failed
            if not skills:
                print("Warning: JSON parsing failed. Attempting regex extraction.")
                # Extract insight_name: "description" patterns
                insight_pattern = r'"insight_\w+"\s*:\s*"((?:[^"\\]|\\.)*)"'
                name_pattern = r'"insight_\w+"'
                names = re.findall(name_pattern, response)
                descriptions = re.findall(insight_pattern, response)

                for i, name in enumerate(names):
                    if i < len(descriptions):
                        insight_name = name.strip('"')
                        insight_desc = (
                            descriptions[i].replace('\\"', '"').replace("\\n", " ")
                        )
                        insight_desc = re.sub(r"\s+", " ", insight_desc).strip()
                        if len(insight_desc) >= 20:
                            skills[insight_name] = insight_desc

        except Exception as e:
            print(f"Warning: Error parsing skills: {e}")
            validation_errors.append(f"Exception during parsing: {e}")

        if not skills:
            validation_errors.append("Could not extract any skills from response")

        # Filter valid skills
        valid_skills = {}
        for k, v in skills.items():
            if not k.startswith("insight_"):
                continue
            if isinstance(v, str) and len(v.strip()) >= 20:
                valid_skills[k] = v
        if not valid_skills:
            print("WARNING: No valid skills extracted from this problem!")

        # Report validation results
        if validation_errors:
            print(f"Validation warnings ({len(validation_errors)}):")
            for error in validation_errors[:5]:
                print(f"  - {error}")

        print(
            f"Extracted {len(valid_skills)} valid skills: {list(valid_skills.keys())}"
        )

        step_result = {
            "step": 3,
            "name": "Insight Extraction",
            "prompt": prompt,
            "response": response,
            "skills": skills,
            "valid_skills": valid_skills,
            "validation_errors": validation_errors,
            "timestamp": time.time(),
        }

        self.reasoning_steps.append(step_result)
        return step_result

    def solve_problem(
        self,
        task: Optional[str] = None,
        custom_solution_instruction: Optional[str] = None,
        insights_section: Optional[str] = None,
    ) -> Dict:
        """
        Solve a problem and extract skills using the behavior curation pipeline.
        Implements the three-stage pipeline: Solution → Reflection → Behavior Extraction

        Args:
            task: The problem/task to solve and extract skills from.
                 If None, uses the default task.
            custom_solution_instruction: Optional custom instruction to append to step 1 prompt
            insights_section: Pre-formatted insights section string (from caller)

        Returns:
            Dictionary containing solution, reflection, extracted skills, and insight book.
        """
        # Update task if provided
        if task is not None:
            self.task = task

        problem = self.task

        print(f"Problem: {problem}\n")

        # Reset reasoning steps and behavior book
        self.reasoning_steps = []
        self.insight_book = {}

        # Step 1: Solution Generation
        print("Step 1: Generating solution...")
        step1 = self._step_solution(
            problem,
            custom_instruction=custom_solution_instruction,
            insights_section=insights_section,
        )
        solution = step1["response"]
        time.sleep(1)

        # Step 2: Extract Insights and Learnings
        print("Step 2: Extracting insights and learnings from solution...")
        step2 = self._step_reflection(problem, solution)
        reflection = step2["response"]
        time.sleep(1)

        # Step 3: Insight Extraction
        print("Step 3: Extracting actionable insights...")
        step3 = self._step_behavior_extraction(problem, solution, reflection)
        time.sleep(1)

        # Update insight_book with extracted insights
        extracted_insights = step3.get("valid_skills", step3.get("skills", {}))
        if extracted_insights:
            self.insight_book.update(extracted_insights)
            print(f"Added {len(extracted_insights)} insights to insight book")
        else:
            print("WARNING: No skills extracted from this problem!")

        # Compile results
        result = {
            "problem": problem,
            "task": self.task,
            "solution": solution,
            "reflection": reflection,
            "skills_extracted": step3.get("valid_skills", step3.get("skills", {})),
            "skills_used": list(
                step3.get("valid_skills", step3.get("skills", {})).keys()
            ),
            "validation_errors": step3.get("validation_errors", []),
            "insight_book": self.insight_book,
            "total_steps": len(self.reasoning_steps),
            "token_info": step1.get("token_info", {}),
        }

        return result

    def _format_complete_reasoning(self) -> str:
        """Format all reasoning steps into a complete reasoning process"""
        formatted = []
        for step in self.reasoning_steps:
            formatted.append(f"\n{'='*80}")
            formatted.append(f"STEP {step['step']}: {step['name']}")
            formatted.append(f"{'='*80}\n")
            formatted.append(step["response"])
            formatted.append("\n")
        return "\n".join(formatted)

    def save_reasoning(self, reasoning_result: Dict, output_path: Optional[str] = None):
        """Save only skill book as simple JSON: {"insight_name": "description"}"""
        # Ensure output directory exists
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        insight_book = reasoning_result.get("insight_book", {})
        if not insight_book:
            print("No skills to save")
            return

        if output_path is None:
            # Create a safe filename from the problem/question
            safe_name = re.sub(
                r"[^\w\s-]", "", reasoning_result.get("problem", "reasoning")[:50]
            )
            safe_name = re.sub(r"[-\s]+", "_", safe_name)
            output_path = str(output_dir / f"{safe_name}.json")
        else:
            # If relative path, make it relative to output_dir
            if not os.path.isabs(output_path):
                output_path = str(output_dir / output_path)
            # Ensure .json extension
            if not output_path.endswith(".json"):
                output_path += ".json"

        # Save only skill book as simple JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(insight_book, f, indent=2, ensure_ascii=False)

        print(f"Saved skill book to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Behavior Curation Pipeline - Metacognitive Reuse for LLM Reasoning"
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

    # Example usage
    reader = ChainOfThoughtReader(
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
            print("Example: python client.py -t 'Find the area of a circle with radius 4'")
            exit(1)

        # Solve the problem using the 3-step pipeline
        result = reader.solve_problem(task=args.task)
        reader.save_reasoning(result)

        if result:
            print("\n" + "=" * 80)
            print("BEHAVIOR CURATION PIPELINE COMPLETE")
            print("=" * 80)
            print(f"Solution: {result.get('solution', 'N/A')}")
            print(f"\nInsights Extracted: {len(result.get('skills_extracted', {}))}")
            print(f"Insights Used: {result.get('skills_used', [])}")
            if result.get("validation_errors"):
                print(
                    f"Validation Warnings: {len(result.get('validation_errors', []))}"
                )
            print("\n" + "=" * 80)
            print("EXTRACTED INSIGHTS")
            print("=" * 80)
            for insight_name, insight_desc in result.get("insight_book", {}).items():
                print(f"\n{insight_name}: {insight_desc}")
            print("\n" + "=" * 80)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print("\nMake sure you have:")
        print("1. Installed required packages: pip install -r requirements.txt")
        print("2. For GPU support, ensure CUDA is properly installed")
        print("\nExample usage:")
        print("  python client.py -t 'Find the area of a circle with radius 4'")
        print("  python client.py -t 'Solve: 2x + 3 = 7' --use-api")
