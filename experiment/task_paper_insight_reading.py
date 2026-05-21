"""
Paper Insight Reading Pipeline
Extracts insights from scientific papers using the client + server_text pipeline.

This file handles the paper-reading specific task:
- Reading papers from papers_dir
- Extracting and formatting paper content
- Generating insights using client.solve_problem()
- Aggregating insights using server_text pipeline

The generic 3-step pipeline (Solution → Reflection → Behavior) stays in client.py
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

from client import ChainOfThoughtReader
from client_metacognitive import MetacognitiveClient
from client_trt import TRTClient
from client_hyperagents import HyperAgentsClient
from client_evolveprompt import EvolvePromptClient
from client_ace import ACEClient
from server_text import TextBasedInsightAggregationServer

_CLIENT_CHOICES = ["default", "metacognitive", "trt", "hyperagents", "evolveprompt", "ace"]


def _paper_file_sort_key(path: Path):
    """Sort by leading numeric prefix first (e.g., 001_), then filename/path."""
    filename = path.name
    match = re.match(r"^(\d+)_", filename)
    has_prefix = 0 if match else 1
    prefix_value = int(match.group(1)) if match else 10**9
    return (has_prefix, prefix_value, filename.lower(), str(path).lower())


def _build_client(
    client_type: str,
    model_name: str,
    device,
    use_api: bool,
    api_key,
    api_provider: str,
    output_dir: str,
    load_in_8bit: bool,
):
    """Factory: return the requested client instance."""
    common = dict(
        model_name=model_name,
        device=device,
        use_api=use_api,
        api_key=api_key,
        api_provider=api_provider,
        output_dir=output_dir,
        load_in_8bit=load_in_8bit,
    )
    if client_type == "metacognitive":
        return MetacognitiveClient(**common)
    elif client_type == "trt":
        return TRTClient(**common)
    elif client_type == "hyperagents":
        return HyperAgentsClient(**common)
    elif client_type == "evolveprompt":
        return EvolvePromptClient(**common)
    elif client_type == "ace":
        return ACEClient(**common)
    else:  # "default"
        return ChainOfThoughtReader(
            model_name=model_name,
            device=device,
            use_api=use_api,
            api_key=api_key,
            api_provider=api_provider,
            load_in_8bit=load_in_8bit,
        )


class PaperInsightReader:
    """
    Pipeline for extracting insights from scientific papers.
    Similar structure to BenchmarkDomainPipeline but for paper reading.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device: Optional[str] = None,
        papers_dir: str = "papers",
        output_dir: str = "paper_insights",
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
        load_in_8bit: bool = False,
        client_type: str = "default",
    ):
        """
        Initialize Paper Insight Reader.

        Args:
            model_name: Model for insight extraction
            device: Device for model (cuda/cpu)
            papers_dir: Directory containing papers to read
            output_dir: Directory to save extracted insights
            use_api: Whether to use an API provider
            api_key: API key for the chosen provider
            api_provider: Which API provider to use (gemini/openrouter)
            load_in_8bit: Whether to load model in 8-bit
            client_type: Which client algorithm to use (default/metacognitive/trt/hyperagents/evolveprompt/ace)
        """
        self.model_name = model_name
        self.device = device
        self.papers_dir = papers_dir
        self.output_dir = output_dir
        self.use_api = use_api
        self.api_key = api_key
        self.api_provider = api_provider
        self.load_in_8bit = load_in_8bit
        self.client_type = client_type

        # Initialize client (generic pipeline)
        self.client = None

        # Initialize server_text (generic aggregation)
        self.server_text = None

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def _ensure_client(self):
        """Lazy load client"""
        if self.client is None:
            self.client = _build_client(
                client_type=self.client_type,
                model_name=self.model_name,
                device=self.device,
                use_api=self.use_api,
                api_key=self.api_key,
                api_provider=self.api_provider,
                output_dir=self.output_dir,
                load_in_8bit=self.load_in_8bit,
            )
            print(f"[PaperInsightReader] Using client: {self.client_type} ({type(self.client).__name__})")

    def _read_paper_content(self, paper_path: str) -> Optional[str]:
        """
        Read paper content from file.

        Supports:
        - .txt: Plain text
        - .md: Markdown
        - .pdf: PDF (using PyPDF2)
        - .json: JSON with 'content' field

        Args:
            paper_path: Path to paper file

        Returns:
            Paper content as string, or None if failed
        """
        try:
            paper_path = Path(paper_path)

            if paper_path.suffix == ".txt":
                with open(paper_path, "r", encoding="utf-8") as f:
                    return f.read()

            elif paper_path.suffix == ".md":
                with open(paper_path, "r", encoding="utf-8") as f:
                    return f.read()

            elif paper_path.suffix == ".json":
                with open(paper_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("content") or data.get("text") or data.get("abstract")

            elif paper_path.suffix == ".pdf":
                if not HAS_PYPDF2:
                    print(f"  Warning: PyPDF2 not installed. Install with: pip install PyPDF2")
                    return None

                try:
                    text = ""
                    with open(paper_path, "rb") as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                    return text.strip()
                except Exception as e:
                    print(f"  Warning: Failed to extract PDF text from {paper_path}: {e}")
                    return None

            else:
                print(f"  Warning: Unsupported file format {paper_path.suffix}")
                return None

        except Exception as e:
            print(f"  Error reading {paper_path}: {e}")
            return None

    def _format_paper_batch_for_insight_extraction(
        self, paper_items: List[tuple]
    ) -> str:
        """
        Format a batch of papers into one extraction prompt.

        Args:
            paper_items: List of (paper_name, paper_content) pairs

        Returns:
            Combined prompt for the batch
        """
        sections = ["Complete following tasks one by one:"]
        for paper_name, paper_content in paper_items:
            sections.append(f"""
            ### Answer research question of the paper: {paper_name} based on paper content:{paper_content}
            """)
        sections.append(""" All papers need to be considered in the analysis. """)
        return "\n\n".join(sections)

    def extract_insights_from_papers(
        self,
        max_papers: Optional[int] = None,
        file_pattern: str = "*.txt",
        agent_read_num: int = 1,
    ) -> Dict:
        """
        Extract insights from papers in papers_dir.

        Similar to task_benchmark_domain._extract_insights_for_dataset()

        Args:
            max_papers: Maximum number of papers to process
            file_pattern: Glob pattern for paper files (e.g., "*.txt", "*.pdf")
            agent_read_num: Number of papers to concatenate into one step-1 read

        Returns:
            Dictionary with extraction results
        """
        self._ensure_client()

        # Find paper files
        papers_path = Path(self.papers_dir)
        paper_files = sorted(papers_path.rglob(file_pattern), key=_paper_file_sort_key)

        if not paper_files:
            print(f"No papers found in {self.papers_dir} matching pattern {file_pattern}")
            return {"papers_processed": 0, "insights_extracted": 0}

        if max_papers:
            paper_files = paper_files[:max_papers]

        if agent_read_num < 1:
            raise ValueError("agent_read_num must be at least 1")

        print(f"Found {len(paper_files)} papers to process")
        if agent_read_num > 1:
            print(f"Batching papers in groups of {agent_read_num}")
        print("=" * 80)

        results = []
        insights_count = 0

        for batch_start in range(0, len(paper_files), agent_read_num):
            batch_files = paper_files[batch_start : batch_start + agent_read_num]
            batch_end = batch_start + len(batch_files)
            batch_label = f"{batch_start + 1:04d}_{batch_end:04d}"
            print(f"\n[{batch_start + 1}-{batch_end}/{len(paper_files)}] Processing batch")

            paper_items = []
            skipped_names = []
            for paper_path in batch_files:
                paper_name = paper_path.stem
                paper_content = self._read_paper_content(paper_path)
                if not paper_content:
                    skipped_names.append(paper_name)
                    print(f"  Skipping {paper_name} - could not read content")
                    continue
                paper_items.append((paper_name, paper_content))

            if not paper_items:
                continue

            task_text = self._format_paper_batch_for_insight_extraction(paper_items)

            try:
                result = self.client.solve_problem(task=task_text)

                output_file = os.path.join(self.output_dir, f"paper_{batch_label}.json")
                insight_book = result.get("insight_book", {})

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(insight_book, f, indent=2, ensure_ascii=False)

                insights_count += len(insight_book)
                print(f"  Extracted {len(insight_book)} insights")
                print(f"  Saved to: {output_file}")

                results.append(
                    {
                        "paper_names": [name for name, _ in paper_items],
                        "insight_count": len(insight_book),
                        "batch_range": [batch_start + 1, batch_end],
                        "skipped_papers": skipped_names,
                    }
                )

            except Exception as e:
                print(f"  Error processing batch {batch_label}: {e}")
                continue

        print("\n" + "=" * 80)
        print(f"Processed {len(paper_files)} papers in {len(results)} batches")
        print(f"Total insights extracted: {insights_count}")

        return {
            "papers_processed": len(paper_files),
            "batches_processed": len(results),
            "insights_extracted": insights_count,
            "results": results,
        }

    def aggregate_insights(self, r1: float = 0.95, r2: float = 0.4) -> str:
        """
        Aggregate insights from all paper results using server_text pipeline.

        Similar to task_benchmark_domain.generate_combined_encyclopedia()

        Args:
            r1: Threshold for insight aggregation
            r2: Threshold for insight relationships

        Returns:
            Path to combined encyclopedia
        """
        print("\n" + "=" * 80)
        print("Aggregating Insights from All Papers")
        print("=" * 80)

        # Find all paper result files
        json_files = list(Path(self.output_dir).glob("paper_*.json"))

        if not json_files:
            print("No paper result files found!")
            return None

        print(f"Found {len(json_files)} paper result files")

        # Initialize server_text (generic aggregation pipeline)
        self.server_text = TextBasedInsightAggregationServer(
            model_name=self.model_name,
            device=self.device,
            input_dirs=[self.output_dir],
            use_api=self.use_api,
            api_key=self.api_key,
            api_provider=self.api_provider,
        )

        # Run aggregation pipeline
        result = self.server_text.aggregate_and_build_encyclopedia(
            json_files=[str(f) for f in json_files], output_dir=self.output_dir
        )

        # Save encyclopedia
        encyclopedia_path = os.path.join(self.output_dir, "paper_encyclopedia.json")
        encyclopedia_dict = self.server_text._try_parse_json(
            self.server_text.encyclopedia
        )

        if encyclopedia_dict is None:
            json_content = self.server_text._extract_json_only(
                self.server_text.encyclopedia
            )
            encyclopedia_dict = self.server_text._try_parse_json(json_content)

        if encyclopedia_dict is None:
            print("Warning: Could not parse encyclopedia as JSON")
            # Save as text instead
            encyclopedia_path = encyclopedia_path.replace(".json", ".txt")
            with open(encyclopedia_path, "w", encoding="utf-8") as f:
                f.write(self.server_text.encyclopedia)
        else:
            with open(encyclopedia_path, "w", encoding="utf-8") as f:
                json.dump(encyclopedia_dict, f, indent=2, ensure_ascii=False)

        print(f"\nPaper encyclopedia saved to: {encyclopedia_path}")
        return encyclopedia_path

    def run_pipeline(
        self,
        max_papers: Optional[int] = None,
        file_pattern: str = "*.txt",
        agent_read_num: int = 1,
        r1: float = 0.95,
        r2: float = 0.4,
        start_from_step2: bool = False,
    ):
        """
        Run the complete paper insight extraction pipeline.

        Args:
            max_papers: Maximum number of papers to process
            file_pattern: Glob pattern for paper files
            agent_read_num: Number of papers to concatenate into one step-1 read
            r1: Aggregation threshold
            r2: Relationship threshold
            start_from_step2: If True, skip step 1 and start from aggregation
        """
        start_time = time.time()

        extraction_result = None
        
        # Step 1: Extract insights from papers
        if not start_from_step2:
            print("=" * 80)
            print("STEP 1: Extracting Insights from Papers")
            print("=" * 80)
            extraction_result = self.extract_insights_from_papers(
                max_papers=max_papers,
                file_pattern=file_pattern,
                agent_read_num=agent_read_num,
            )
        else:
            print("Skipping STEP 1 (start_from_step2=True)")

        # Step 2: Aggregate insights
        print("\n" + "=" * 80)
        print("STEP 2: Aggregating Insights")
        print("=" * 80)
        encyclopedia_path = self.aggregate_insights(r1=r1, r2=r2)

        # Summary
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("Pipeline Complete")
        print("=" * 80)
        if extraction_result:
            print(f"Papers processed: {extraction_result['papers_processed']}")
            print(f"Insights extracted: {extraction_result['insights_extracted']}")
        print(f"Encyclopedia: {encyclopedia_path}")
        print(f"Total time: {total_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Extract insights from scientific papers using client + server_text pipeline"
    )
    parser.add_argument(
        "-p",
        "--papers-dir",
        type=str,
        default="papers",
        help="Directory containing papers to read",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="paper_insights",
        help="Directory to save extracted insights",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name for insight extraction",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda' or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--max-papers", type=int, default=None, help="Maximum number of papers to process"
    )
    parser.add_argument(
        "--agent-read-num",
        type=int,
        default=1,
        help="Number of papers to concatenate into one agent read for step 1",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*.txt",
        help="Glob pattern for paper files (e.g., *.txt, *.pdf)",
    )
    parser.add_argument("--use-api", action="store_true", help="Use an API provider instead of HuggingFace model")
    parser.add_argument("--api-provider", type=str, default="gemini", choices=["gemini", "openrouter"], help="Which API provider to use (default: gemini)")
    parser.add_argument("--api-key", type=str, help="API key for the chosen provider")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument(
        "--r1", type=float, default=0.95, help="Aggregation threshold"
    )
    parser.add_argument(
        "--r2", type=float, default=0.4, help="Relationship threshold"
    )
    parser.add_argument(
        "--start-from-step2",
        action="store_true",
        help="Skip insight extraction and start from aggregation (step 2)",
    )
    parser.add_argument(
        "--client",
        type=str,
        default="default",
        choices=_CLIENT_CHOICES,
        help=(
            "Client algorithm to use for insight extraction. "
            "default=ChainOfThoughtReader (client.py), "
            "metacognitive=MetacognitiveClient, "
            "trt=TRTClient, "
            "hyperagents=HyperAgentsClient, "
            "evolveprompt=EvolvePromptClient, "
            "ace=ACEClient."
        ),
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = PaperInsightReader(
        model_name=args.model,
        device=args.device,
        papers_dir=args.papers_dir,
        output_dir=args.output_dir,
        use_api=args.use_api,
        api_key=args.api_key,
        api_provider=args.api_provider,
        load_in_8bit=args.load_in_8bit,
        client_type=args.client,
    )

    # Run pipeline
    pipeline.run_pipeline(
        max_papers=args.max_papers,
        file_pattern=args.file_pattern,
        agent_read_num=args.agent_read_num,
        r1=args.r1,
        r2=args.r2,
        start_from_step2=args.start_from_step2,
    )


if __name__ == "__main__":
    main()
