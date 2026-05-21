"""
Text-Based Insight Aggregation Server
Implements a purely text-based approach for building an Encyclopedia from multiple insight books.
Uses LLM prompts to analyze relationships, merge insights, and extract general knowledge.

Pipeline:
1. Collect Insight Books → Aggregate Insight Store
2. Text-Based Profiling → Analyze relationships, merge same insights, cluster related insights
3. Knowledge Extraction → Extract general, fundamental knowledge that can derive collected insights
"""

import argparse
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from utils import (
    HAS_GEMINI,
    check_cuda,
    setup_gemini,
    call_gemini,
    call_openrouter,
    load_hf_model,
    call_hf_model,
)


class TextBasedInsightAggregationServer:
    """
    Text-based server that aggregates insight books using LLM prompts
    to analyze relationships and extract general knowledge.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device: Optional[str] = None,
        input_dirs: Optional[List[str]] = None,
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
        num_insights: Optional[int] = None,
        max_files: Optional[int] = None,
        custom_prompt_section: str = "",
        seed: Optional[int] = None,
    ):
        self.model_name = model_name
        # Support both single dir and multiple dirs for backward compatibility
        if input_dirs is None:
            input_dirs = ["math_output"]
        elif isinstance(input_dirs, str):
            input_dirs = [input_dirs]
        self.input_dirs = input_dirs
        self.num_insights = num_insights
        self.max_files = max_files
        self.seed = seed
        self.insight_store = {}  # Aggregated insight store
        self.encyclopedia = ""  # Final encyclopedia
        self.aggregation_steps = []
        self.insight_relationships = {}  # Text-based profiling of relationships
        self.custom_prompt_section = (custom_prompt_section or "").strip()
        self.last_generation_info = {}

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
            self.model_name, device=self.device,
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
                text, token_info = call_openrouter(self.api_key, model, prompt, system_prompt, max_new_tokens)
            else:
                text, token_info = call_gemini(
                    self.gemini_model, prompt, system_prompt, max_new_tokens,
                )
            self.last_generation_info = token_info
            return text, token_info
        self._load_model()
        text, token_info = call_hf_model(
            self.model, self.tokenizer, self.model_name,
            prompt, system_prompt, max_new_tokens, self.device,
        )
        self.last_generation_info = token_info
        return text, token_info

    def _prepend_custom_prompt(self, prompt: str) -> str:
        """Prepend optional custom prompt section before the main prompt."""
        if not self.custom_prompt_section:
            return prompt
        return f"{self.custom_prompt_section}\n\n{prompt}"

    def collect_insight_books(self, json_files: Optional[List[str]] = None) -> Dict:
        """
        Step 1: Collect ALL insights from problem*.json files.

        Simple approach:
        1. Find all problem*.json files recursively under input_dirs (one or more directories)
        2. Extract behavior_book from each file
        3. Store all insights with indexed keys (no deduplication)
        4. Return insight store for text profiling

        Args:
            json_files: List of JSON file paths. If None, finds all problem*.json recursively from all input_dirs.

        Returns:
            Dictionary containing collected insights.
        """
        # Find all problem*.json and paper*.json files recursively from all input directories
        if json_files is None:
            json_files = []
            print(f"Searching for problem*.json and paper*.json files under {self.input_dirs}...")
            for input_dir_str in self.input_dirs:
                input_path = Path(input_dir_str)
                if not input_path.exists():
                    print(f"  Warning: Input directory does not exist: {input_path}")
                    continue
                print(f"  Scanning: {input_path}")
                json_files.extend([str(f) for f in input_path.rglob("problem_*.json")])
                json_files.extend([str(f) for f in input_path.rglob("paper_*.json")])
            json_files = sorted(set(json_files))  # Remove duplicates if same file appears in multiple scans

            # Sort files by numeric suffix (e.g. paper_0001.json -> 1, problem_0042.json -> 42)
            def _extract_number(filepath):
                basename = Path(filepath).stem  # e.g. "paper_0001"
                match = re.search(r'_(\d+)$', basename)
                return int(match.group(1)) if match else float('inf')

            json_files.sort(key=_extract_number)

            print(f"Found {len(json_files)} problem/paper*.json files")

            # Limit to max_files if specified, with optional random shuffle
            if self.max_files is not None and self.max_files < len(json_files):
                rng = random.Random(self.seed)
                rng.shuffle(json_files)
                json_files = json_files[:self.max_files]
                seed_str = str(self.seed) if self.seed is not None else "random"
                print(f"Randomly sampled {self.max_files} files (seed={seed_str})")

        if not json_files:
            print("ERROR: No problem*.json or paper*.json files found!")
            return {
                "step": 1,
                "name": "Collect Insights",
                "files_processed": 0,
                "error": "No files found",
            }

        all_insights = {}  # Store all insights with indexed keys
        insight_counter = 0  # Global counter for all insights
        files_processed = 0

        print(f"Collecting insights from {len(json_files)} files...")
        
        # Debug: show first few files
        if len(json_files) > 0:
            print(f"  Sample files:")
            for f in json_files[:3]:
                print(f"    - {f}")

        # Process each JSON file
        for json_file in json_files:
            try:
                # Load JSON file - it contains insights directly
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Debug: show what we loaded
                if not isinstance(data, dict):
                    print(f"  Warning: {json_file} is not a dict!")
                    print(f"    Type: {type(data)}")
                    print(f"    Content preview: {str(data)[:200]}")
                    continue
                
                # Check if wrapped in insight_book/behavior_book
                if "insight_book" in data:
                    insights_dict = data["insight_book"]
                elif "behavior_book" in data:
                    insights_dict = data["behavior_book"]
                else:
                    # Data itself is the insights dictionary
                    insights_dict = data

                if not isinstance(insights_dict, dict):
                    print(f"  Warning: insights in {json_file} is not a dict, skipping")
                    continue

                # Count insights (excluding metadata keys)
                file_insight_count = 0
                for insight_name, insight_desc in insights_dict.items():
                    # Skip metadata keys
                    if insight_name in ["paper_name", "problem", "problem_id", "iteration", "is_correct", "number_output_tokens", "loop_count"]:
                        continue
                    
                    # Add insight with indexed key
                    insight_counter += 1
                    file_insight_count += 1
                    indexed_key = f"{insight_name}_{insight_counter:06d}"
                    all_insights[indexed_key] = insight_desc

                if file_insight_count > 0:
                    files_processed += 1
                    
                    # Progress logging every 100 files
                    if files_processed % 100 == 0:
                        print(f"  Processed {files_processed} files, collected {insight_counter} insights...")
                else:
                    print(f"  Warning: No insights found in {json_file}")
                    print(f"    Keys in file: {list(insights_dict.keys())[:10]}")

            except Exception as e:
                print(f"  Warning: Failed to read {json_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Store in insight_store
        self.insight_store = all_insights

        step_result = {
            "step": 1,
            "name": "Collect Insights",
            "files_processed": files_processed,
            "total_insights_collected": insight_counter,
            "insight_store": self.insight_store,
            "timestamp": time.time(),
        }

        self.aggregation_steps.append(step_result)
        print(f"\nCollected ALL {insight_counter} insights from {files_processed} files")
        print(f"Insight store contains {len(all_insights)} entries (no deduplication)")
        
        if files_processed == 0 and len(json_files) > 0:
            print(f"\nWARNING: Found {len(json_files)} files but processed 0!")
            print("This likely means all files have empty 'insight_book' dictionaries.")
            print("Check that client.py successfully extracted insights during step 1.")

        return step_result

    def _get_text_profiling_prompt(self, insight_store: Dict) -> str:
        """Step 2: Prompt for text-based profiling of insight relationships

        Based on research in hierarchical insight learning, insight composition, and knowledge graphs.
        References:
        - Generalizable Hierarchical Insight Learning (GIL): Object-centric insight primitives
        - Insight Chaining: Composing insights for complex tasks
        - Knowledge Graph approaches: Relationship mapping and ontology construction

        Note: insight_store now contains ALL collected skills without deduplication.
        Skills with similar names (e.g., insight_name_001, insight_name_002) are different instances.
        """
        insights_text = "\n".join(
            [f"- {name}: {desc}" for name, desc in insight_store.items()]
        )

        prompt = f"""
You are analyzing a collection of reasoning traces generated for problem-solving.  Understand their relationships and structure and build a profiling of their relationships.

**Collected {len(insight_store)} traces in total:**
Note: This collection includes ALL traces from all problems without deduplication. Similar names with different indices (e.g., _001, _002) represent different occurrences that may have variations in their descriptions.

{insights_text}

**Your Task:**
Analyze these traces and build a profiling of their relationships:

1. **Identify Clusters**:
   Group related traces that share:
   - Resolve the same or similar problem
   - Similar approaches or techniques
   - Nearly identical traces (e.g., same trace with minor variations in description or parameters)
   - Traces in the same cluster should be higly similar.

2. **Build Trace Relationships**:
Record all important relationships - traces don't exist in isolation and build a relationship graph that records:
   - **Prerequisite relationships**: Traces that must be learned/used before others
   - **Composition relationships**: Traces that can be chained/composed together
   - **Alternative relationships**: Different approaches to the same problem
   - **Complementary relationships**: Traces that work better together than individually used
   - **Derivation relationships**: Traces derived from or based on others
   - **Similar relationships**: Traces that are similar but not identical
Map relationships between traces within clusters and across clusters.

# Output Format:
{{
  "clusters": [
    {{
      "cluster_id": 0,
      "cluster_name": "Domain/Theme Name",
      "traces": ["name1", "name2", "name3"],
      "theme": "What is the high-level techniqual idea of the traces in this cluster?",
    }}
  ],
  "relationships": [
    {{
      "trace_a": "trace_name1",
      "trace_b": "trace_name2",
      "relationship_type": "prerequisite/complementary/alternative/similar/derived_from/composes_with",
      "description": "How these traces relate to each other and Why"
    }}
  ]
}}

**Output your analysis as JSON only:**

"""
        return prompt

    def _step_text_profiling(self, insight_store: Dict) -> Dict:
        """Step 2: Text-based profiling of insight relationships

        Note: insight_store contains ALL collected skills without deduplication.
        """
        print("Building text-based profiling of insight relationships...")
        print(f"Analyzing ALL {len(insight_store)} collected insights (no deduplication)...")

        prompt = self._get_text_profiling_prompt(insight_store)
        prompt = self._prepend_custom_prompt(prompt)
        system_prompt = None

        if "deepseek" in self.model_name.lower():
            max_tokens=32768
        else:
            max_tokens=65536

        response,output_tokens = self._call_model(prompt, system_prompt, max_new_tokens=max_tokens)
        print(f"Text profiling response received (output tokens: {output_tokens['output_tokens']}):")

        # Extract JSON from response
        json_content = self._extract_json_only(response)
        profiling_data = self._try_parse_json(json_content)

        if profiling_data is None:
            print(
                "Warning: Could not parse profiling JSON. Falling back to minimal structure."
            )
            profiling_data = {
                "clusters": [],
                "relationships": [],
                "insights": list(insight_store.keys()),
                "raw_response": response,
            }

        self.insight_relationships = profiling_data

        step_result = {
            "step": 2,
            "name": "Text-Based Profiling",
            "prompt": prompt,
            "response": response,
            "profiling": self.insight_relationships,
            "output_tokens": output_tokens.get("output_tokens", 0),
            "timestamp": time.time(),
        }

        self.aggregation_steps.append(step_result)
        return step_result

    def _get_knowledge_extraction_prompt(
        self, insight_store: Dict, profiling: Dict, existing_encyclopedia: str = ""
    ) -> str:
        """Step 3: Prompt for extracting general, fundamental knowledge

        Based on research in:
        - Knowledge Distillation: Extracting higher-level abstractions
        - Hierarchical Insight Learning: Multi-level insight organization
        - Cross-domain Transfer: Identifying universal patterns
        - Insight Composition: Creating composable, reusable insights
        - Anthropic Insights: Composable, portable insight structure
        """

        # Format clusters
        clusters_text = ""
        if isinstance(profiling, dict) and "clusters" in profiling:
            clusters_text = "\n".join(
                [
                    f"- Cluster {cluster.get('cluster_id', '?')} ({cluster.get('cluster_name', 'unnamed')}): "
                    f"{', '.join(cluster.get('insights', []))}"
                    for cluster in profiling["clusters"]
                ]
            )

        # Format all insights from insight_store (for reference)
        all_insights_text = ";".join(
            [f"{name}: {desc}" for name, desc in insight_store.items()]
        )

        # Include full previous encyclopedia if available
        if existing_encyclopedia:
            try:
                # Try to parse as JSON to extract insights
                enc_data = json.loads(existing_encyclopedia)
                if isinstance(enc_data, dict):
                    # Extract insights if they're in a dictionary format
                    enc_insights = []
                    if "general_insights" in enc_data:
                        for insight in enc_data["general_insights"]:
                            if isinstance(insight, dict):
                                insight_name = insight.get("insight_name", "")
                                insight_desc = insight.get("description", "")
                                if insight_name and insight_desc:
                                    enc_insights.append(
                                        f"{insight_name}: {insight_desc}"
                                    )
                    elif isinstance(enc_data, dict):
                        # If it's a flat dictionary of insights
                        for name, desc in enc_data.items():
                            if name.startswith("insight_"):
                                enc_insights.append(f"{name}: {desc}")
            except Exception as e:
                print(f"Warning: Failed to parse existing encyclopedia as JSON: {e}")
                enc_insights = [existing_encyclopedia]

        if not existing_encyclopedia:
            print("No existing encyclopedia provided.")

        proper_number = f"{self.num_insights}" if self.num_insights is not None else (int(math.log10(len(insight_store))*10 + 1) if len(insight_store) > 0 else "a reasonable number")
        print(f"Proper number of insights to extract: {proper_number} over {len(insight_store)} collected insights")

        prompt = f"""
**Your Task:**
You are extracting fundamental insights from a collection of problem-solving traces.

**Output Requirements (STRICT):**
1. Return EXACTLY one valid JSON object and nothing else.
2. Do NOT output markdown code fences.
3. Do NOT output explanations, notes, reasoning, prefixes, suffixes, or `<think>` content.
4. Do NOT output list/array at top-level.
5. Every key must start with "insight_".
6. Every value must be a single string.
7. No nested objects, no nested arrays.

**Required JSON shape:**
{{
    "insight_name1": "description string",
    "insight_name2": "description string"
}}

**Formatting Rules:**
- Use valid JSON syntax only.
- Keep top-level as key-value pairs only.
- Escape quotes in descriptions with backslash: \\" 
- If uncertain, still output a valid JSON object (possibly with fewer insights), never free text.

Your goal is to extract a comprehensive set of fundamental, cross-domain insights that can be derived and applied beyond their original domain meet following requirements: 
- Combine previous insights (if any): {existing_encyclopedia if existing_encyclopedia else "None"} with new insights.
- Extract your insights based on all client reasoning traces: {all_insights_text}. These traces are derived from solving specific problems (bottom-up approach)
- Use clusters of reasoning traces: {clusters_text if clusters_text else "None identified"} to help organize.
- Use relationships between traces: {json.dumps(profiling.get('relationships', []), indent=2) if isinstance(profiling, dict) else "None identified"} to help organization
- Your task is to extract multi-disciplinary, fundamental knowledge (top-down approach) which can be generalized to multi-domain problem-solving.
- The extracted insights should be able to DERIVE and GUIDE the use of the collected insights
- The extraced insights cannot be too general. They are not supposed to be knowledge which can be applied to any problem. They should be fundamental knowlege to particular several domains but specific.
- You should extra {proper_number} insights. Not too few. Not too many.Do not over simplified or too detailed.
- DO NOT over-merge insights.

Insights should have following properties:
1. **Extract Reusable Primitives**:
   - For EACH cluster, extract multiple fundamental insights capturing core essence and variations (DO NOT over-merge)
   - Identify cross-domain patterns that apply to multiple fields
   - Create reusable, composable primitives specific enough to be actionable

2. **Knolwedge to include**:
   - **Fundamental Level**: Core principles underlying multiple domains
   - **General Level**: Broad techniques for related problem types
   - **Cross-Domain**: Insights transferable beyond origin field

3. **Preserve While Generalizing**:
   - Create fundamental versions that can guide/derive original insights
   - Maintain important variations rather than collapsing into single insight

4. **Description Format:**
   Each description is a single string containing:
   - What the insight is and how it solves problems
   - When to use: problem types, conditions, triggers (be comprehensive and specific)

**Example 1 - Transformer Architecture:**

Input reasoning traces:
- reasoning_trace_VisionTransformerImageClassification: "I need to classify medical X-ray images into disease categories. CNNs aren't working well - they can't capture relationships between distant regions like how fluid in the lower right lung might relate to heart enlargement. Let me try Vision Transformer (ViT). I'll divide each X-ray into 16x16 patches - a 224x224 image gives 196 patches. Each patch becomes like a token in NLP. I flatten each to a 256-dim vector. Since transformers don't know spatial positions, I add position embeddings so the model knows patch [0,0] is top-left. I prepend a [CLS] token to gather global info. Feeding through 12 transformer encoder layers - the self-attention lets every patch attend to every other patch directly, so patch [2,5] can look at patch [10,12] even though they're far apart spatially. This is exactly what I need! After 12 layers, I extract the [CLS] token and feed to MLP classifier. Training on 50k chest X-rays: 94.2% accuracy, beating ResNet-50's 89.3%. The attention maps show it's correctly attending to both lungs simultaneously for pneumonia detection, linking heart size to lung fluid - this cross-region reasoning is what CNNs miss. The key: treating image patches as tokens with self-attention enables global spatial reasoning."

- reasoning_trace_TransformerNextWordPrediction: "I'm building autocomplete for a text editor. The challenge: predict next word given arbitrary-length context. RNNs struggle with long sequences - the hidden state forgets earlier context. Let me use a transformer decoder. I tokenize 'The cat sat on the' using WordPiece → tokens [254, 8901, 4523, 651, 278]. Convert each to 512-dim embedding and add positional encodings. Critical part: causal masking so the model can't cheat by seeing future words. When predicting token at position 4, it should only see 0-3. I implement lower-triangular attention mask. Processing through 6 decoder layers with masked self-attention. At position 4 ('the'), attention computes similarity between its query and keys of previous words. It attends strongly to 'sat' (0.8) and 'on' (0.7), weakly to 'cat' (0.2). Using 12 heads helps - different heads capture different patterns: head-2 learns syntax (preposition+article), head-5 learns semantics (actions+objects), head-8 learns long-range dependencies. After final layer, project last hidden state through 50k-dim softmax. For 'the': top predictions are 'mat' (0.73), 'floor' (0.12), 'rug' (0.08). Deployed in production - users accept top-3 suggestion 85% of the time, reducing typing by 40%. Transformer's self-attention captures context way better than RNN's sequential processing."

Output aggregated insight:
{{
  "insight_transformerArchitecture": "This fundamental neural network architecture applies across natural language processing, computer vision, time series analysis, graph neural networks, and multi-modal learning. This insight is essential for modern AI applications including language models, image processing, code generation, and scientific computing. When you need to capture relationships between all elements simultaneously (self-attention), you're working with sequences of variable length, you need parallel processing of sequences, or when the problem involves understanding context and relationships. Details: 1) Design input representation - convert your data into embeddings (token embeddings for text, patch embeddings for images, node embeddings for graphs), add positional encodings to preserve sequence information, and prepare input for transformer blocks 2) Create models with transformer blocks 3) Apply task-specific architecture - use encoder-only for understanding (BERT, ViT), decoder-only for generation (GPT), or encoder-decoder for translation"
}}

**Example 2 - Surface-Enhanced Raman Spectroscopy:**

Input reasoning traces:
- reasoning_trace_SERSMedicalDetectionR6G: "I need to detect cancer biomarkers in blood at incredibly low concentrations - 10^-12 M, like finding molecules in a swimming pool. ELISA only goes to 10^-9 M, not sensitive enough for early diagnosis. Let me try SERS - Surface-Enhanced Raman Spectroscopy. Metal nanoparticles create huge EM field enhancements. I synthesize 60nm gold nanoparticles via citrate reduction. At 785nm laser, these have plasmon resonance amplifying local field by ~10^6. But I need selectivity too - can't detect everything. So I functionalize the gold with anti-PSA antibodies for prostate cancer. When I add patient serum, only PSA proteins bind. Here's the clever part: I add R6G (Rhodamine 6G) reporter molecules. R6G has enormous Raman cross-section and when it sits in nanogaps between gold particles, field enhancement shoots to 10^8 or 10^10. Incubate 30 min for PSA binding, add R6G which sticks near bound PSA. Hit with 785nm laser at 5mW - I see characteristic R6G peaks at 1650, 1510, 1310 cm^-1. Peak intensity directly proportional to PSA amount. Integrate 60 sec for good SNR. Comparing to calibration: detecting PSA at 0.1 ng/mL - that's 10,000x more sensitive than ELISA! On clinical samples, detected prostate cancer 3-6 months earlier than conventional tests. The breakthrough: combining selective antibody recognition with SERS amplification gives single-molecule sensitivity while maintaining specificity."

- reasoning_trace_SERSPollutantDetection: "Monitoring river water for pesticides. EPA limit for malathion is 0.1 ppb but standard chromatography needs 1 ppb minimum. I need 10x better for early warning. SERS might work. Instead of spherical particles, I'll fabricate silver nanorod arrays - sharp tips create hotter hotspots than spheres. Using oblique angle deposition: 80nm nanorods with 4:1 aspect ratio on silicon. Gaps between rods only 5-10nm - perfect for trapping molecules. I calculate enhancement should hit 10^10 at 532nm. Collect river water, filter through 0.2μm to remove debris and bacteria. Drop 50μL onto nanorod substrate. During 5-min adsorption, pesticide molecules diffuse into nanogaps. Small gap means molecules guaranteed in enhancement zone (<10nm from metal). Rinse gently - removes interfering organics/salts but leaves adsorbed pesticides. Excite with 532nm at 2mW, matching silver plasmon peak. Even at 0.01 ppb malathion, clear peaks at 1440 cm^-1 (P=S stretch), 1080 cm^-1 (P-O-C), 640 cm^-1 (C-S). Measuring 1440 peak height vs calibration standards for quantification. Tested 50 river sites, cross-validated against LC-MS: R^2=0.97 correlation. Best part: do this in field with portable Raman - no lab needed. Real-time monitoring at 10x below regulatory limits. The nanorod geometry is critical - those sharp tips and tight gaps push enhancement to 10^10."

Output aggregated insight:
{{
  "insight_surfaceEnhancedRamanSpectroscopy": "This powerful technique applies across analytical chemistry, materials science, biosensing, pharmaceutical analysis, environmental monitoring, and forensics. The technique achieves single-molecule sensitivity (10^6-10^11 enhancement) while providing molecular structural information through vibrational fingerprints. When to use: When you need ultra-sensitive detection below conventional analytical limits, when you want label-free molecular identification, when analyzing trace contaminants or biomarkers, or when field-portable real-time analysis is required. Common steps: 1) Prepare SERS-active substrate - synthesize plasmonic nanostructures (gold/silver nanoparticles, nanorods, nanostars) optimizing particle size (20-100 nm), shape, and inter-particle spacing (1-10 nm gaps) to maximize electromagnetic field enhancement at laser wavelength 2) Functionalize substrate if needed - modify metallic surface with antibodies, aptamers, or molecular recognition elements for selective analyte binding and improved specificity 3) Prepare and apply sample - process sample (filter, dilute, concentrate as needed), deposit onto SERS substrate via drop-casting or flow-through, allow adsorption time for molecules to enter hot spots (<10 nm from metal surface) 4) Select laser parameters - choose wavelength matching plasmon resonance (532, 633, or 785 nm), optimize power (0.1-10 mW) to avoid sample damage while maximizing signal 5) Acquire SERS spectrum - collect Raman scattered light with appropriate integration time, record vibrational spectrum showing characteristic molecular peaks 6) Analyze spectral fingerprint - identify molecules by comparing peak positions to reference spectra, quantify concentration from peak intensities using calibration curves, assess molecular orientation from peak ratios 7) Validate and control quality - average multiple spots for reproducibility, use internal standards, verify with orthogonal methods, consider substrate heterogeneity and enhancement factor variations"
}}
"""
        
        return prompt

    def _step_knowledge_extraction(
        self, insight_store: Dict, profiling: Dict, existing_encyclopedia: str = ""
    ) -> Dict:
        """Step 3: Extract general, fundamental knowledge"""
        print("Extracting general, fundamental knowledge...")

        prompt = self._get_knowledge_extraction_prompt(
            insight_store, profiling, existing_encyclopedia
        )
        prompt = self._prepend_custom_prompt(prompt)
        system_prompt = None

        if "deepseek" in self.model_name.lower():   
            max_tokens = 32768
        else:
            max_tokens = 65536 
        response,output_tokens = self._call_model(
            prompt,
            system_prompt,
            max_new_tokens=max_tokens,
        )
        print(response)
        print(f"Knowledge extraction response generated {output_tokens['output_tokens']} tokens")
        if self.last_generation_info:
            print(
                "Step 3 generation stop reason: "
                f"{self.last_generation_info.get('finish_reason', 'unknown')} "
                f"(output_tokens={self.last_generation_info.get('output_tokens', 'n/a')}, "
                f"max_new_tokens={self.last_generation_info.get('max_new_tokens', 'n/a')})"
            )

        # Extract JSON from response
        json_content = self._extract_json_only(response)
        encyclopedia_dict = self._try_parse_json(json_content)

        # If JSON parsing fails, extract insights using pattern matching
        if encyclopedia_dict is None:
            print("JSON parsing failed - extracting insights using pattern matching...")
            encyclopedia_dict = self._extract_insights_from_text(response)
            if encyclopedia_dict:
                print(f"Successfully extracted {len(encyclopedia_dict)} insights from text")
                json_content = json.dumps(encyclopedia_dict, indent=2, ensure_ascii=False)
            else:
                error_msg = (
                        "ERROR: Could not extract insights from response after retry.\n"
                        f"Initial response:\n{response}"
                    )
                print(error_msg)
                raise ValueError(error_msg)

        # Update encyclopedia
        self.encyclopedia = json_content

        step_result = {
            "step": 3,
            "name": "Knowledge Extraction",
            "prompt": prompt,
            "response": response,
            "encyclopedia": json_content,
            "encyclopedia_dict": encyclopedia_dict,
            "output_tokens": output_tokens.get("output_tokens", 0),
            "timestamp": time.time(),
        }

        self.aggregation_steps.append(step_result)
        return step_result

    def _extract_json_only(self, text: str) -> str:
        """Extract JSON content from response, removing any explanatory text"""
        # Try to find JSON in code blocks first
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()

        # Try to find JSON object
        start_idx = text.find("{")
        if start_idx != -1:
            # Find matching closing brace
            brace_count = 0
            in_string = False
            escape_next = False
            for i in range(start_idx, len(text)):
                char = text[i]
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
                            return text[start_idx : i + 1]
            # If no complete match, try last brace
            last_brace = text.rfind("}", start_idx)
            if last_brace != -1:
                return text[start_idx : last_brace + 1]

        return text

    def _try_parse_json(self, text: str) -> Optional[Dict]:
        """Safely try to parse JSON; return None if parsing fails"""
        try:
            return json.loads(text)
        except Exception:
            return None

    def _extract_insights_from_text(self, text: str) -> Dict:
        """Extract insights from text when JSON parsing fails.

        Searches for patterns like:
        - "insight_name": "description"
        - insight_name: description
        - **insight_name**: description

        Returns dict of extracted insights.
        """
        insights = {}

        # Pattern 1: JSON-style with quotes: "insight_xxx": "description..."
        pattern1 = r'"(insight_\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"'
        matches1 = re.findall(pattern1, text, re.DOTALL)
        for name, desc in matches1:
            insights[name] = desc.replace('\\"', '"').replace('\\n', '\n').strip()

        # Pattern 2: Markdown style: **insight_xxx**: description or ## insight_xxx
        pattern2 = r'(?:\*\*|##)\s*(insight_\w+)\s*(?:\*\*|:)\s*(.+?)(?=(?:\*\*|##)\s*insight_|\Z)'
        matches2 = re.findall(pattern2, text, re.DOTALL)
        for name, desc in matches2:
            if name not in insights:  # Don't overwrite pattern1 matches
                insights[name] = desc.strip()

        # Pattern 3: Simple format: insight_xxx: description (looking for "When to use:" as content marker)
        pattern3 = r'(insight_\w+):\s*(When to use:.*?)(?=insight_\w+:|$)'
        matches3 = re.findall(pattern3, text, re.DOTALL | re.IGNORECASE)
        for name, desc in matches3:
            if name not in insights:  # Don't overwrite previous matches
                insights[name] = desc.strip()

        return insights

    # REMOVED: No longer using fallback encyclopedias
    # def _build_fallback_encyclopedia(self, insight_store: Dict) -> Dict:
    #     """Fallback encyclopedia: use collected insights when model output is not valid JSON"""
    #     if not insight_store:
    #         return {}
    #     return {name: desc for name, desc in insight_store.items() if desc}

    def _load_existing_encyclopedia(self, output_dir: str) -> str:
        """Load existing encyclopedia from output directory if it exists"""
        encyclopedia_path = os.path.join(output_dir, "encyclopedia.txt")
        if os.path.exists(encyclopedia_path):
            try:
                with open(encyclopedia_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    print(
                        f"Loaded existing encyclopedia from {encyclopedia_path} ({len(content)} characters)"
                    )
                    return content
            except Exception as e:
                print(f"Warning: Could not load existing encyclopedia: {e}")
        return ""

    def aggregate_and_build_encyclopedia(
        self,
        json_files: Optional[List[str]] = None,
        output_dir: str = "math_output",
    ) -> Dict:
        """
        Main method to aggregate insight books and build the Encyclopedia using text-based approach.

        Args:
            json_files: List of JSON file paths. If None, scans input_dir.
            output_dir: Output directory to check for existing encyclopedia.

        Returns:
            Dictionary containing all aggregation steps and final encyclopedia.
        """
        # Step 1: Collect insight Books
        print("\n" + "=" * 80)
        print("STEP 1: Collecting insight Books")
        print("=" * 80)
        collection_result = self.collect_insight_books(json_files)
        time.sleep(1)

        insight_datastore_tokens = 0
        try:
            insight_datastore_text = json.dumps(self.insight_store, ensure_ascii=False)
            if self.use_api:
                # API tokenizer is not exposed in this code path; use a rough estimate.
                insight_datastore_tokens = len(re.findall(r"\S+", insight_datastore_text))
            else:
                if self.tokenizer is None:
                    self._load_model()
                model_max_length = getattr(self.tokenizer, "model_max_length", 65536)
                if model_max_length is None or model_max_length > 1_000_000:
                    model_max_length = 65536
                model_max_length = int(model_max_length)

                # Count full datastore tokens without triggering over-length warnings.
                tokenized = self.tokenizer(
                    insight_datastore_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=model_max_length,
                    return_overflowing_tokens=True,
                )
                insight_datastore_tokens = sum(
                    len(chunk_ids) for chunk_ids in tokenized["input_ids"]
                )

                if insight_datastore_tokens > model_max_length:
                    print(
                        "Insight datastore exceeds single-pass context window: "
                        f"{insight_datastore_tokens} > {model_max_length}"
                    )
        except Exception as e:
            print(f"Warning: Failed to count insight datastore tokens: {e}")
            insight_datastore_tokens = 0

        print(f"Total insight datastore tokens: {insight_datastore_tokens}")

        if not self.insight_store:
            files_processed = collection_result.get("files_processed", 0)
            print(f"Warning: No insights found in {files_processed} collected files!")
            return {
                "error": "No insights found",
                "files_processed": files_processed,
                "collection_result": collection_result,
                "aggregation_steps": self.aggregation_steps,
            }

        # Step 2: Text-Based Profiling
        print("\n" + "=" * 80)
        print("STEP 2: Text-Based Profiling of insight Relationships")
        print("=" * 80)
        profiling_result = self._step_text_profiling(self.insight_store)
        time.sleep(1)

        # Step 3: Knowledge Extraction
        print("\n" + "=" * 80)
        print("STEP 3: Extracting General, Fundamental Knowledge")
        print("=" * 80)

        # Load existing encyclopedia if available
        existing_encyclopedia = self._load_existing_encyclopedia(output_dir)

        extraction_result = self._step_knowledge_extraction(
            self.insight_store,
            self.insight_relationships,
            existing_encyclopedia,
        )

        # Combine all results
        total_output_tokens = (
            profiling_result.get("output_tokens", 0)
            + extraction_result.get("output_tokens", 0)
        )
        result = {
            "collection": collection_result,
            "profiling": profiling_result,
            "extraction": extraction_result,
            "aggregation_steps": self.aggregation_steps,
            "encyclopedia": self.encyclopedia,
            "insight_store": self.insight_store,  # Preserve original insights
            "insight_relationships": self.insight_relationships,
            "insight_datastore_tokens": insight_datastore_tokens,
            "total_output_tokens": total_output_tokens,
        }

        return result

    def save_results(self, result: Dict, output_dir: str = "math_output"):
        """Save only encyclopedia.json with format {"insight_name": "description"}"""
        os.makedirs(output_dir, exist_ok=True)
        encyclopedia_path = os.path.join(output_dir, "encyclopedia.json")

        # Parse encyclopedia JSON string and save as formatted JSON
        encyclopedia_dict = self._try_parse_json(self.encyclopedia)

        if encyclopedia_dict is None:
            # If not valid JSON, try to extract JSON from the string
            json_content = self._extract_json_only(self.encyclopedia)
            encyclopedia_dict = self._try_parse_json(json_content)

        if encyclopedia_dict is None:
            error_msg = f"ERROR: Could not parse encyclopedia as JSON. Encyclopedia content:\n{self.encyclopedia}"
            print(error_msg)
            raise ValueError(error_msg)

        with open(encyclopedia_path, "w", encoding="utf-8") as f:
            json.dump(encyclopedia_dict, f, indent=2, ensure_ascii=False)
        print(f"Encyclopedia saved to: {encyclopedia_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text-Based insight Aggregation Server - Build Encyclopedia using LLM prompts"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        nargs="+",
        default=["math_output"],
        help="Input directory/directories containing insight JSON files. Can specify multiple dirs (default: math_output)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Hugging Face model name (default: deepseek-ai/DeepSeek-R1-Distill-Llama-8B)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda' or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="math_output",
        help="Output directory for encyclopedia (default: math_output)",
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
        "--num-insights",
        type=int,
        default=None,
        help="Number of insights to extract. If not provided, the model decides a proper number.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of JSON files to use, randomly sampled. If not provided, all files are used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for file sampling when --max-files is set. If not provided, sampling is non-deterministic.",
    )

    args = parser.parse_args()

    # Create server
    server = TextBasedInsightAggregationServer(
        model_name=args.model,
        device=args.device,
        input_dirs=args.input_dir,
        use_api=args.use_api,
        api_key=args.api_key,
        api_provider=args.api_provider,
        num_insights=args.num_insights,
        max_files=args.max_files,
        seed=args.seed,
    )

    try:
        # Aggregate and build encyclopedia
        result = server.aggregate_and_build_encyclopedia(output_dir=args.output_dir)

        # Save results
        server.save_results(result, output_dir=args.output_dir)

        print("\n" + "=" * 80)
        print("AGGREGATION COMPLETE")
        print("=" * 80)
        print(f"Total insights collected: {len(server.insight_store)}")
        print(f"Encyclopedia length: {len(server.encyclopedia)} characters")
        print(f"Insight library output tokens: {result.get('total_output_tokens', 0)}")
        print(f"Output directory: {args.output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print("\nMake sure you have:")
        print("1. Installed required packages: pip install -r requirements.txt")
        print("2. insight JSON files in the input directory")
        print("3. For GPU support, ensure CUDA is properly installed")
