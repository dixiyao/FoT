"""
Insight Aggregation Server
Implements the aggregation phase for building an Encyclopedia from multiple insight books.
Follows the pipeline: Collect Insight Books → Aggregate Insight Store → Reflection → Encyclopedia Chapter → Encyclopedia
"""

import argparse
import json
import os
import re
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    HAS_GEMINI,
    check_cuda,
    setup_gemini,
    call_gemini,
    call_openrouter,
    load_hf_model,
    call_hf_model,
)

# Suppress transformers loading warnings for embedding models
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
warnings.filterwarnings("ignore", message=".*Materializing param.*")

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. GraphRAG features will be limited.")


class InsightAggregationServer:
    """
    Server that aggregates insight books from multiple client results
    and constructs an Encyclopedia through reflection and synthesis.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1",
        device: Optional[str] = None,
        input_dir: str = "build/log",
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
    ):
        self.model_name = model_name
        self.input_dir = input_dir
        self.insight_store = (
            {}
        )  # Aggregated insight store (compatible with behavior_book from client)
        self.encyclopedia = ""  # Final encyclopedia
        self.aggregation_steps = []

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

        # Embedding model for knowledge graph (loaded lazily)
        self.embedding_model = None
        self.embedding_model_name = "BAAI/bge-base-en-v1.5"

    def _load_model(self):
        """Lazy load the Hugging Face model and tokenizer"""
        if self.model is not None and self.tokenizer is not None:
            return
        self.model, self.tokenizer = load_hf_model(
            self.model_name, device=self.device,
        )

    def _call_model(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call the language model (HuggingFace or Gemini API).

        Returns:
            Generated text response
        """
        if self.use_api:
            if self.api_provider == "openrouter":
                model = getattr(self, "api_model_name", None) or "anthropic/claude-opus-4.6"
                text, _ = call_openrouter(self.api_key, model, prompt, system_prompt)
            else:
                text, _ = call_gemini(self.gemini_model, prompt, system_prompt)
            return text
        self._load_model()
        text, _ = call_hf_model(
            self.model, self.tokenizer, self.model_name,
            prompt, system_prompt, 78632, self.device,
        )
        return text

    def collect_insight_books(self, json_files: Optional[List[str]] = None) -> Dict:
        """
        Step 1: Collect insight books from multiple client result JSON files sequentially.

        Args:
            json_files: List of JSON file paths. If None, scans input_dir for JSON files.

        Returns:
            Dictionary containing collected insight books and metadata.
        """
        if json_files is None:
            # Scan input directory for JSON files
            input_path = Path(self.input_dir)
            json_files = list(input_path.glob("*.json"))
            # Filter out non-skill files (metadata, summary, results, etc.)
            # Process problem_*.json (from math_pipeline), paper_*.json (from client.py), or other skill files
            json_files = [
                str(f)
                for f in json_files
                if "metadata" not in f.name.lower()
                and "summary" not in f.name.lower()
                and "results" not in f.name.lower()
                and "encyclopedia" not in f.name.lower()
            ]

        print(f"Collecting skill books from {len(json_files)} files...")

        collected_books = {}
        all_insights = {}
        insight_counts = {}  # Track how many times each skill appears
        total_skills_count = 0  # Total count including duplicates
        problems = []

        # Read JSON files sequentially
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract skill book (client.py uses "behavior_book" key but contains skills)
                # Handle both dict and list formats
                insight_book_raw = (
                    data.get("behavior_book")
                    or data.get("behaviors")
                    or data.get("skills")
                )

                # Convert list format to dict if needed
                insight_book = {}
                if isinstance(insight_book_raw, dict):
                    insight_book = insight_book_raw
                elif isinstance(insight_book_raw, list):
                    # Convert list of dicts to dict format
                    # Expected format: [{"behavior": "name", "description": "desc"}, ...]
                    for item in insight_book_raw:
                        if isinstance(item, dict):
                            # Try different key names
                            insight_name = (
                                item.get("behavior")
                                or item.get("skill")
                                or item.get("name")
                            )
                            insight_desc = item.get("description") or item.get("desc")
                            if insight_name and insight_desc:
                                # Ensure skill name starts with "skill_" prefix
                                if not insight_name.startswith("skill_"):
                                    insight_name = f"skill_{insight_name}"
                                insight_book[insight_name] = insight_desc
                elif isinstance(data, dict) and not insight_book_raw:
                    # Check if data itself is a skill book (direct format from client.py)
                    # If all keys start with "skill_" or most keys are strings with descriptions
                    insight_keys = [
                        k
                        for k in data.keys()
                        if isinstance(k, str) and k.startswith("skill_")
                    ]
                    if len(insight_keys) > 0 and len(insight_keys) >= len(data) * 0.8:
                        # This is likely a direct skill book dictionary
                        insight_book = data
                    elif all(
                        isinstance(v, str) and len(v) > 20
                        for v in data.values()
                        if isinstance(v, str)
                    ):
                        # All values are strings (likely descriptions), treat as skill book
                        insight_book = data

                if insight_book:
                    filename = Path(json_file).stem
                    collected_books[filename] = {
                        "problem": data.get("problem", "Unknown"),
                        "insight_book": insight_book,
                        "behavior_book": insight_book,  # Keep for compatibility
                        "solution": data.get("solution", ""),
                        "reflection": data.get("reflection", ""),
                    }

                    # Count and aggregate all skills
                    total_skills_count += len(insight_book)
                    for insight_name, insight_desc in insight_book.items():
                        # Update skill store (keep latest description if duplicate)
                        all_insights[insight_name] = insight_desc
                        # Count occurrences
                        insight_counts[insight_name] = (
                            insight_counts.get(insight_name, 0) + 1
                        )

                    problems.append(data.get("problem", "Unknown"))

                    print(f"  Collected insights from {filename}")
            except Exception as e:
                print(f"  Warning: Failed to read {json_file}: {e}")
                continue

        # Create aggregated skill store
        self.insight_store = all_insights

        step_result = {
            "step": 1,
            "name": "Collect Insight Books",
            "files_processed": len(collected_books),
            "total_skills_collected": total_skills_count,  # Total including duplicates
            "unique_skills": len(all_insights),  # Unique skill count
            "insight_counts": insight_counts,  # Count of each skill
            "collected_books": collected_books,
            "insight_store": self.insight_store,
            "behavior_bookstore": self.insight_store,  # Keep for compatibility
            "problems": problems,
            "timestamp": time.time(),
        }

        self.aggregation_steps.append(step_result)
        print(
            f"\nCollected insights ({len(all_insights)} unique) from {len(collected_books)} files"
        )

        return step_result

    def _load_embedding_model(self):
        """Load the embedding model for knowledge graph clustering"""
        if self.embedding_model is not None:
            return

        try:
            print(f"Loading embedding model: {self.embedding_model_name}")
            # Suppress transformers loading warnings and progress bars
            import logging

            # Suppress transformers warnings
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            logging.getLogger("transformers.configuration_utils").setLevel(
                logging.ERROR
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Ensure device is properly set for embedding model
                embedding_device = (
                    self.device
                    if self.device
                    else ("cuda" if torch.cuda.is_available() else "cpu")
                )
                self.embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    device=embedding_device,
                )
                print(f"Embedding model device: {embedding_device}")
            print("Embedding model loaded successfully!")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model {self.embedding_model_name}: {e}"
            )

    def _step_knowledge_graph_clustering(
        self, insight_store: Dict, r1: float = 0.9, r2: float = 0.4
    ) -> Dict:
        """Step 2: Build knowledge graph and cluster skills using embeddings"""
        print("Building knowledge graph with embeddings...")

        # Load embedding model
        self._load_embedding_model()

        # Prepare skill texts for embedding
        insight_names = list(insight_store.keys())
        insight_texts = [f"{name}: {desc}" for name, desc in insight_store.items()]

        # Compute embeddings
        print(f"Computing embeddings for {len(insight_texts)} skills...")
        # Use show_progress_bar based on environment - disable in non-interactive environments
        import sys

        show_progress = (
            sys.stdout.isatty()
        )  # Only show progress bar if running in terminal
        embeddings = self.embedding_model.encode(
            insight_texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=32,  # Process in batches for better performance
        )

        # Normalize embeddings for cosine similarity
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute pairwise cosine similarity
        print("Computing pairwise cosine similarities...")
        similarity_matrix = cosine_similarity(embeddings_norm)

        # Normalize similarity to [0, 1] range (cosine similarity is already in [-1, 1])
        # Map from [-1, 1] to [0, 1]
        similarity_matrix = (similarity_matrix + 1) / 2

        # Build graph: nodes are skills, edges are linked skills (r2 <= similarity < r1)
        same_skills = defaultdict(set)  # Groups of identical skills (similarity >= r1)
        graph_edges = []  # Edges for linked skills (r2 <= similarity < r1)
        skill_to_group = {}  # Map skill to its same_skills group

        print(f"Building graph with thresholds: r1={r1} (same skill), r2={r2} (linked)")
        for i in range(len(insight_names)):
            for j in range(i + 1, len(insight_names)):
                sim = similarity_matrix[i][j]

                if sim >= r1:
                    # Same skill - merge into group (not added as edge)
                    if i not in skill_to_group and j not in skill_to_group:
                        # Create new group
                        group_id = len(same_skills)
                        same_skills[group_id].add(i)
                        same_skills[group_id].add(j)
                        skill_to_group[i] = group_id
                        skill_to_group[j] = group_id
                    elif i in skill_to_group:
                        # Add j to i's group
                        group_id = skill_to_group[i]
                        same_skills[group_id].add(j)
                        skill_to_group[j] = group_id
                    elif j in skill_to_group:
                        # Add i to j's group
                        group_id = skill_to_group[j]
                        same_skills[group_id].add(i)
                        skill_to_group[i] = group_id
                elif sim >= r2:
                    # Linked skills - add edge (r2 <= similarity < r1)
                    graph_edges.append((i, j, sim))

        # Build clusters from connected components in the graph
        # Each connected component is a cluster
        print("Building clusters from graph connected components...")
        adjacency = defaultdict(set)
        for i, j, sim in graph_edges:
            adjacency[i].add(j)
            adjacency[j].add(i)

        # Find connected components (clusters)
        clusters = []
        visited = set()

        def get_connected_component(node: int) -> Set[int]:
            """Get all nodes in the connected component"""
            component = set()
            stack = [node]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                if current in adjacency:
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            return component

        for i in range(len(insight_names)):
            if i in visited:
                continue

            # Get connected component
            component = get_connected_component(i)

            if len(component) > 1:  # Only create cluster if more than 1 node
                clusters.append(component)

        # Convert to skill names
        same_skills_groups = {
            group_id: [insight_names[i] for i in group]
            for group_id, group in same_skills.items()
        }

        clusters_named = [[insight_names[i] for i in cluster] for cluster in clusters]

        print(f"Found {len(same_skills_groups)} groups of identical skills")
        print(f"Found {len(clusters_named)} clusters (connected components)")

        # Build GraphRAG database
        graphrag_data = self._build_graphrag_database(
            insight_names,
            insight_store,
            embeddings_norm,
            similarity_matrix,
            same_skills_groups,
            clusters_named,
            graph_edges,
            r1,
            r2,
        )

        step_result = {
            "step": 2,
            "name": "Knowledge Graph Clustering",
            "same_skills_groups": same_skills_groups,
            "clusters": clusters_named,
            "similarity_matrix": similarity_matrix.tolist(),
            "graph_edges": [
                (insight_names[i], insight_names[j], float(sim))
                for i, j, sim in graph_edges
            ],
            "graphrag_data": graphrag_data,
            "timestamp": time.time(),
        }

        self.aggregation_steps.append(step_result)
        return step_result

    def _build_graphrag_database(
        self,
        insight_names: List[str],
        insight_store: Dict,
        embeddings: np.ndarray,
        similarity_matrix: np.ndarray,
        same_skills_groups: Dict,
        clusters: List[List[str]],
        graph_edges: List[Tuple[int, int, float]],
        r1: float,
        r2: float,
    ) -> Dict:
        """Build GraphRAG database with graph structure and embeddings"""
        print("Building GraphRAG database...")

        # Create graph structure
        if HAS_NETWORKX:
            G = nx.Graph()
        else:
            G = None

        # Add nodes (skills)
        nodes = {}
        for idx, insight_name in enumerate(insight_names):
            node_data = {
                "insight_name": insight_name,
                "description": insight_store[insight_name],
                "embedding": embeddings[idx].tolist(),  # Store normalized embedding
                "index": idx,
            }
            nodes[insight_name] = node_data
            if G is not None:
                G.add_node(insight_name, **node_data)

        # Add edges based on similarity
        edges = []
        for i, j, sim in graph_edges:
            skill_i = insight_names[i]
            skill_j = insight_names[j]
            edge_type = "same" if sim >= r1 else "linked"
            edge_data = {
                "similarity": float(sim),
                "type": edge_type,
            }
            edges.append({"source": skill_i, "target": skill_j, **edge_data})
            if G is not None:
                G.add_edge(skill_i, skill_j, **edge_data)

        # Store same skills groups as node attributes
        for group_id, group_skills in same_skills_groups.items():
            for insight_name in group_skills:
                if insight_name in nodes:
                    nodes[insight_name]["same_skills_group"] = group_id
                    if G is not None:
                        G.nodes[insight_name]["same_skills_group"] = group_id

        # Store cluster information
        for cluster_id, cluster_skills in enumerate(clusters):
            for insight_name in cluster_skills:
                if insight_name in nodes:
                    if "clusters" not in nodes[insight_name]:
                        nodes[insight_name]["clusters"] = []
                    nodes[insight_name]["clusters"].append(cluster_id)
                    if G is not None:
                        if "clusters" not in G.nodes[insight_name]:
                            G.nodes[insight_name]["clusters"] = []
                        G.nodes[insight_name]["clusters"].append(cluster_id)

        graphrag_data = {
            "nodes": nodes,
            "edges": edges,
            "same_skills_groups": same_skills_groups,
            "clusters": {i: cluster for i, cluster in enumerate(clusters)},
            "r1": r1,
            "r2": r2,
            "similarity_matrix": similarity_matrix.tolist(),
            "graph_format": "networkx" if HAS_NETWORKX else "dict",
        }

        print(f"GraphRAG database built: {len(nodes)} nodes, {len(edges)} edges")
        return graphrag_data

    def _get_cluster_summary_prompt(
        self,
        same_skills_groups: Dict,
        clusters: List[List[str]],
        insight_store: Dict,
        existing_encyclopedia: str = "",
        r1: float = 0.9,
        r2: float = 0.4,
    ) -> str:
        """Get the prompt for merging same skills and summarizing clusters"""
        # Format same skills groups
        same_skills_text = ""
        for group_id, insight_names in same_skills_groups.items():
            same_skills_text += f"\nGroup {group_id} (merge these into one skill):\n"
            for insight_name in insight_names:
                same_skills_text += (
                    f"  - {insight_name}: {insight_store.get(insight_name, 'N/A')}\n"
                )

        # Format clusters
        clusters_text = ""
        for cluster_id, cluster_skills in enumerate(clusters):
            clusters_text += f"\nCluster {cluster_id}:\n"
            clusters_text += "  Skills in this cluster:\n"
            for insight_name in cluster_skills:
                clusters_text += (
                    f"    - {insight_name}: {insight_store.get(insight_name, 'N/A')}\n"
                )

        # Get all unique skills (not in same_skills_groups)
        all_insight_names = set(insight_store.keys())
        merged_insight_names = set()
        for group in same_skills_groups.values():
            merged_insight_names.update(group)
        standalone_skills = all_insight_names - merged_insight_names

        standalone_text = ""
        if standalone_skills:
            standalone_text = "\nStandalone Skills (keep as-is):\n"
            for insight_name in standalone_skills:
                standalone_text += (
                    f"  - {insight_name}: {insight_store.get(insight_name, 'N/A')}\n"
                )

        # Format existing encyclopedia section
        existing_encyclopedia_section = ""
        if existing_encyclopedia and existing_encyclopedia.strip():
            existing_encyclopedia_section = f"""
### Existing Encyclopedia (merge with this):
{existing_encyclopedia}

Note: You should merge new skills with the existing encyclopedia, combining similar skills and adding new ones.
"""

        prompt = f"""
You are building a comprehensive Skills Encyclopedia from clustered skills.
{existing_encyclopedia_section}
------------------------------------------------------------
TASK: Merge Same Skills and Summarize Clusters
------------------------------------------------------------

### Instructions:

1. MERGE WITH EXISTING ENCYCLOPEDIA (if provided)
   - If an existing encyclopedia is provided above, merge the new skills with it
   - If a skill in the new set already exists in the encyclopedia, merge them intelligently
   - Combine descriptions, keeping all important technical details from both
   - Add new skills that don't exist in the encyclopedia
   - Maintain the structure and organization of the existing encyclopedia when possible

2. MERGE SAME SKILLS (similarity >= {r1})
   - For each group of same skills below, merge them into ONE unified skill
   - These skills have cosine similarity >= {r1}, indicating they are essentially the same
   - Combine their descriptions, keeping all important technical details
   - Create a single, comprehensive skill that captures all variations
   - Skills should be very detailed and specific, going into deep technical details
   - DO NOT abstract to general principles - keep technical specificity

3. SUMMARIZE CLUSTERS (similarity {r2} to {r1})
   - For each cluster below, create a cluster summary with:
     * Cluster topic/theme (what connects these skills)
     * Potential use cases (when to use skills from this cluster)
     * **REQUIRED: List ALL skills in the cluster in the "skills" array** (DO NOT merge them - keep them separate)
   - Skills in a cluster have cosine similarity between {r2} and {r1}, meaning they are related but distinct
   - **CRITICAL: Each cluster MUST include a "skills" array containing all skills from that cluster with their full descriptions**
   - Keep them as separate entries within the cluster

4. KEEP STANDALONE SKILLS
   - Standalone skills (not in any group or cluster) should be kept as-is

### Same Skills Groups (MERGE these):
{same_skills_text}

### Clusters (SUMMARIZE these, but keep skills separate):
{clusters_text}

**IMPORTANT**: For each cluster above, you MUST include ALL skills from that cluster in the "skills" array. Each skill should have "insight_name" and "description" fields. Do NOT omit the skills array - it is REQUIRED.

### Standalone Skills:
{standalone_text}

### Output Format:
Output ONLY a JSON object with this structure:

```json
{{
  "title": "Problem-Solving Skills Encyclopedia",
  "merged_skills": [
    {{
      "insight_name": "...",
      "description": "...",
      "merged_from": ["skill1", "skill2", ...]
    }}
  ],
  "clusters": [
    {{
      "cluster_id": 0,
      "topic": "...",
      "use_cases": ["...", "..."],
      "skills": [
        {{
          "insight_name": "...",
          "description": "..."
        }},
        {{
          "insight_name": "...",
          "description": "..."
        }}
      ],
      "standalone": false
    }}
  ],
  "standalone_skills": [
    {{
      "insight_name": "...",
      "description": "..."
    }}
  ]
}}
```

**CRITICAL REQUIREMENTS**:
1. Every cluster MUST have a "skills" array containing ALL skills from that cluster
2. The "skills" array cannot be empty - it must contain at least one skill object
3. Each skill in the "skills" array must have both "insight_name" and "description" fields
4. Do NOT omit the "skills" array from any cluster - it is REQUIRED

Output ONLY the JSON object, nothing else.
"""

        return prompt

    def _extract_json_only(self, text: str) -> str:
        """Extract only JSON content from response, removing any explanatory text"""
        try:
            # Strategy 1: Look for JSON in code blocks
            json_code_block = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL
            )
            if json_code_block:
                return json_code_block.group(1).strip()

            # Strategy 2: Look for JSON object (find the first { and matching })
            # Count braces to find the complete JSON object
            start_idx = text.find("{")
            if start_idx != -1:
                brace_count = 0
                for i in range(start_idx, len(text)):
                    if text[i] == "{":
                        brace_count += 1
                    elif text[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = text[start_idx : i + 1]
                            # Validate it's valid JSON
                            json.loads(json_str)
                            return json_str.strip()

            # Strategy 3: Try to find any JSON object
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Clean up and validate
                json_str = json_str.strip()
                json_str = re.sub(r",\s*}", "}", json_str)
                json_str = re.sub(r",\s*]", "]", json_str)
                json.loads(json_str)  # Validate
                return json_str

            # If no JSON found, return original (shouldn't happen)
            return text
        except (json.JSONDecodeError, AttributeError):
            # If extraction fails, return original text
            return text

    def _step_cluster_summary(
        self,
        same_skills_groups: Dict,
        clusters: List[List[str]],
        insight_store: Dict,
        existing_encyclopedia: str = "",
        r1: float = 0.9,
        r2: float = 0.4,
    ) -> Dict:
        """Step 3: Merge same skills and summarize clusters"""
        prompt = self._get_cluster_summary_prompt(
            same_skills_groups, clusters, insight_store, existing_encyclopedia, r1, r2
        )

        system_prompt = None
        response = self._call_model(prompt, system_prompt)
        print(f"Cluster summary generated ({len(response)} characters)")

        # Extract only JSON content, removing any explanatory text
        json_content = self._extract_json_only(response)

        # Update the encyclopedia with only JSON content
        self.encyclopedia = json_content

        step_result = {
            "step": 3,
            "name": "Cluster Summary and Skill Merging",
            "prompt": prompt,
            "response": response,
            "encyclopedia": json_content,  # Store only JSON content
            "timestamp": time.time(),
        }

        self.aggregation_steps.append(step_result)
        return step_result

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

    def _save_graphrag_database(self, graphrag_data: Dict, output_dir: str):
        """Save GraphRAG database to disk"""
        os.makedirs(output_dir, exist_ok=True)

        # Save as JSON (embeddings will be large but manageable)
        graphrag_path = os.path.join(output_dir, "graphrag_db.json")
        with open(graphrag_path, "w", encoding="utf-8") as f:
            json.dump(graphrag_data, f, indent=2, ensure_ascii=False)

        # Also save embeddings separately as numpy array for faster loading
        embeddings_path = os.path.join(output_dir, "graphrag_embeddings.npy")
        embeddings_list = [
            node["embedding"] for node in graphrag_data["nodes"].values()
        ]
        if embeddings_list:
            embeddings_array = np.array(embeddings_list)
            np.save(embeddings_path, embeddings_array)

        print("GraphRAG database saved to:")
        print(f"  - {graphrag_path}")
        print(f"  - {embeddings_path}")

    def aggregate_and_build_encyclopedia(
        self,
        json_files: Optional[List[str]] = None,
        r1: float = 0.9,
        r2: float = 0.4,
        output_dir: str = "build/log",
    ) -> Dict:
        """
        Main method to aggregate skill books and build the Encyclopedia.

        Args:
            json_files: List of JSON file paths. If None, scans input_dir.
            r1: Threshold for same skills (cosine similarity >= r1 means same skill).
            r2: Threshold for linked skills (r2 <= similarity < r1 means linked).
            output_dir: Output directory to check for existing encyclopedia.

        Returns:
            Dictionary containing all aggregation steps and final encyclopedia.
        """
        # Step 1: Collect Insight Books (append all skills together)
        print("\n" + "=" * 80)
        print("STEP 1: Collecting Skill Books")
        print("=" * 80)
        collection_result = self.collect_insight_books(json_files)
        time.sleep(1)

        if not self.insight_store:
            files_processed = collection_result.get("files_processed", 0)
            print(f"Warning: No skills found in {files_processed} collected files!")
            return {
                "error": "No skills found",
                "files_processed": files_processed,
                "collection_result": collection_result,
                "aggregation_steps": self.aggregation_steps,
            }

        # Step 2: Knowledge Graph Clustering
        print("\n" + "=" * 80)
        print("STEP 2: Knowledge Graph Clustering")
        print(f"Using thresholds: r1={r1} (same skill), r2={r2} (linked)")
        print("=" * 80)
        clustering_result = self._step_knowledge_graph_clustering(
            self.insight_store, r1=r1, r2=r2
        )
        same_skills_groups = clustering_result["same_skills_groups"]
        clusters = clustering_result["clusters"]
        time.sleep(1)

        # Step 3: Save GraphRAG database and merge/summarize
        print("\n" + "=" * 80)
        print("STEP 3: Saving GraphRAG Database and Merging Skills")
        print("=" * 80)

        # Save GraphRAG database
        graphrag_data = clustering_result.get("graphrag_data", {})
        if graphrag_data:
            self._save_graphrag_database(graphrag_data, output_dir)

        # Load existing encyclopedia if it exists
        existing_encyclopedia = self._load_existing_encyclopedia(output_dir)
        if existing_encyclopedia:
            print("Found existing encyclopedia - will merge with new skills")
        self._step_cluster_summary(
            same_skills_groups,
            clusters,
            self.insight_store,
            existing_encyclopedia,
            r1=r1,
            r2=r2,
        )

        # Compile results
        result = {
            "insight_store": self.insight_store,
            "behavior_bookstore": self.insight_store,  # Keep for compatibility
            "collection_metadata": {
                "files_processed": collection_result.get("files_processed", 0),
                "total_skills_collected": collection_result.get(
                    "total_skills_collected", 0
                ),
                "unique_skills": collection_result.get("unique_skills", 0),
                "problems": collection_result.get("problems", []),
                "collected_books": collection_result.get("collected_books", {}),
            },
            "clustering": {
                "same_skills_groups": same_skills_groups,
                "clusters": clusters,
            },
            "encyclopedia": self.encyclopedia,  # Final encyclopedia containing aggregated skills
            "aggregation_steps": self.aggregation_steps,
            "total_skills": len(self.insight_store),
            "total_behaviors": len(self.insight_store),  # Keep for compatibility
            "total_steps": len(self.aggregation_steps),
        }

        return result

    def save_results(self, result: Dict, output_dir: str = "build/log"):
        """Save aggregation results - only the encyclopedia"""
        os.makedirs(output_dir, exist_ok=True)

        # Save only the encyclopedia (main output)
        encyclopedia_path = os.path.join(output_dir, "encyclopedia.txt")
        with open(encyclopedia_path, "w", encoding="utf-8") as f:
            f.write(result.get("encyclopedia", "No encyclopedia generated."))

        print("\nEncyclopedia saved to:")
        print(f"  - {encyclopedia_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Skill Aggregation Server - Build Encyclopedia from Skill Books"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="build/log",
        help="Directory containing client result JSON files (default: build/log)",
    )
    parser.add_argument(
        "-f",
        "--files",
        type=str,
        nargs="+",
        default=None,
        help="Specific JSON files to process (default: all JSON files in input-dir)",
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
        "-o",
        "--output-dir",
        type=str,
        default="build/log",
        help="Output directory for results (default: build/log)",
    )
    parser.add_argument(
        "--r1",
        type=float,
        default=0.9,
        help="Threshold r1 for same skills (cosine similarity >= r1 means same skill, default: 0.9)",
    )
    parser.add_argument(
        "--r2",
        type=float,
        default=0.4,
        help="Threshold r2 for linked skills (r2 <= similarity < r1 means linked, default: 0.4)",
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

    args = parser.parse_args()

    # Create server instance
    server = SkillAggregationServer(
        model_name=args.model,
        device=args.device,
        input_dir=args.input_dir,
        use_api=args.use_api,
        api_key=args.api_key,
        api_provider=args.api_provider,
    )

    try:
        # Run aggregation pipeline
        result = server.aggregate_and_build_encyclopedia(
            json_files=args.files, r1=args.r1, r2=args.r2, output_dir=args.output_dir
        )

        # Save results
        server.save_results(result, output_dir=args.output_dir)

        print("\n" + "=" * 80)
        print("AGGREGATION COMPLETE")
        print("=" * 80)
        print(
            f"Total skills: {result.get('total_skills', result.get('total_behaviors', 0))}"
        )
        print(f"Encyclopedia length: {len(result.get('encyclopedia', ''))} characters")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print("\nMake sure you have:")
        print("1. Run client.py to generate skill book JSON files")
        print("2. Installed required packages: pip install -r requirements.txt")
        print("3. For GPU support, ensure CUDA is properly installed")
