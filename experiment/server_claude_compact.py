"""
Claude Compact Server - Collects all insights from JSON files, then uses
the Claude Cookbooks guided summarization with chunking approach to compress
them into a single encyclopedia entry.

Step 1: Same as server_text.py - collect all insights from problem/paper JSON files.
Step 2: Domain-specific guided summarization following Claude Cookbooks:
        - Chunk the insights into manageable pieces
        - Summarize each chunk individually
        - Consolidate chunk summaries into a final compacted summary
Output: encyclopedia.json with {"insight": <compacted_summary>}

Reference: https://github.com/anthropics/claude-cookbooks/tree/main/capabilities/summarization
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from utils import setup_gemini, call_gemini


# ---------------------------------------------------------------------------
# Summarization prompts (following Claude Cookbooks guided summarization)
# ---------------------------------------------------------------------------

CHUNK_SUMMARY_PROMPT = """\
Summarize the following collection of research insights/knowledge. Focus on these key aspects:

1. Core findings (main discoveries, novel contributions, key results)
2. Methodological patterns (common techniques, approaches, frameworks used)
3. Domain relationships (how insights connect, overlap, or build on each other)
4. Actionable knowledge (practical guidelines, best practices, design principles)

Provide the summary in bullet points nested within the XML header for each section.
If any information is not explicitly stated, note it as "Not specified".
Do not preamble.

Insights:
{text}
"""

FINAL_CONSOLIDATION_PROMPT = """\
Combine the following chunked summaries of research insights into a coherent overall summary.
Deduplicate overlapping insights, merge related findings, and organize the knowledge
into a unified, structured encyclopedia entry.

Focus on:
1. Core findings - main discoveries and novel contributions
2. Methodological patterns - common techniques and frameworks
3. Domain relationships - how insights connect and build on each other
4. Actionable knowledge - practical guidelines and design principles

Remove redundancy across chunks. Produce a concise, comprehensive summary that captures
all unique knowledge from the original insights.
Do not preamble.

Chunked summaries:
{text}
"""


# ---------------------------------------------------------------------------
# Insight collection (same as server_text.py Step 1)
# ---------------------------------------------------------------------------

def collect_insight_books(input_dir: str, max_files: Optional[int] = None) -> Dict[str, str]:
    """
    Collect ALL insights from problem*.json and paper*.json files.

    Args:
        input_dir: Directory containing insight JSON files.
        max_files: Maximum number of files to process (sorted by numeric suffix).

    Returns:
        Dictionary of {insight_name: description}.
    """
    input_path = Path(input_dir)
    print(f"Searching for problem*.json and paper*.json files under {input_path}...")
    json_files = list(input_path.rglob("problem_*.json")) + list(input_path.rglob("paper_*.json"))
    json_files = [str(f) for f in json_files]

    # Sort files by numeric suffix (e.g. paper_0001.json -> 1, problem_0042.json -> 42)
    def _extract_number(filepath):
        basename = Path(filepath).stem
        match = re.search(r'_(\d+)$', basename)
        return int(match.group(1)) if match else float('inf')

    json_files.sort(key=_extract_number)

    print(f"Found {len(json_files)} problem/paper*.json files")

    # Limit to max_files if specified
    if max_files is not None and max_files < len(json_files):
        json_files = json_files[:max_files]
        print(f"Using first {max_files} files (sorted by number)")

    if not json_files:
        print("ERROR: No problem*.json or paper*.json files found!")
        return {}

    all_insights = {}
    insight_counter = 0
    files_processed = 0

    print(f"Collecting insights from {len(json_files)} files...")

    # Debug: show first few files
    if len(json_files) > 0:
        print(f"  Sample files:")
        for f in json_files[:3]:
            print(f"    - {f}")

    # Metadata keys to skip
    metadata_keys = {
        "paper_name", "problem", "problem_id", "iteration",
        "is_correct", "number_output_tokens", "loop_count",
    }

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

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
                insights_dict = data

            if not isinstance(insights_dict, dict):
                print(f"  Warning: insights in {json_file} is not a dict, skipping")
                continue

            file_insight_count = 0
            for insight_name, insight_desc in insights_dict.items():
                if insight_name in metadata_keys:
                    continue

                insight_counter += 1
                file_insight_count += 1
                indexed_key = f"{insight_name}_{insight_counter:06d}"
                all_insights[indexed_key] = insight_desc

            if file_insight_count > 0:
                files_processed += 1
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

    print(f"\nCollected ALL {insight_counter} insights from {files_processed} files")
    print(f"Insight store contains {len(all_insights)} entries (no deduplication)")

    if files_processed == 0 and len(json_files) > 0:
        print(f"\nWARNING: Found {len(json_files)} files but processed 0!")
        print("This likely means all files have empty 'insight_book' dictionaries.")

    return all_insights


# ---------------------------------------------------------------------------
# Gemini model setup/call (shared utils)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Chunking & guided summarization (following Claude Cookbooks pattern)
# ---------------------------------------------------------------------------

def format_insights_as_text(insights: Dict[str, str]) -> str:
    """Format all collected insights into a single text block."""
    lines = []
    for name, desc in insights.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def chunk_text(text: str, chunk_size: int = 50000) -> List[str]:
    """
    Split text into chunks by character count.
    Following Claude Cookbooks chunking pattern.

    Args:
        text: Full text to chunk.
        chunk_size: Max characters per chunk.

    Returns:
        List of text chunks.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def summarize_with_chunking(
    insights_text: str,
    model,
    chunk_size: int = 50000,
    max_output_tokens: int = 32768,
) -> tuple:
    """
    Guided summarization with chunking, following Claude Cookbooks pattern:
    1. Chunk insights into manageable pieces
    2. Summarize each chunk individually
    3. Consolidate chunk summaries into final summary

    Returns:
        (final_summary, total_output_tokens)
    """
    chunks = chunk_text(insights_text, chunk_size=chunk_size)
    print(f"  Split insights into {len(chunks)} chunks (chunk_size={chunk_size})")
    total_output_tokens = 0

    if len(chunks) == 1:
        # Single chunk - directly summarize
        print("  Single chunk, summarizing directly...")
        prompt = CHUNK_SUMMARY_PROMPT.format(text=chunks[0])
        summary, token_info = call_gemini(model, prompt, max_new_tokens=max_output_tokens)
        total_output_tokens += token_info.get("output_tokens", 0)
        # Extract content within <summary> tags if present
        match = re.search(r"<summary>(.*?)</summary>", summary, re.DOTALL)
        if match:
            summary = match.group(1).strip()
        return summary, total_output_tokens

    # Multiple chunks - hierarchical summarization
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"  Summarizing chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)...")
        prompt = CHUNK_SUMMARY_PROMPT.format(text=chunk)
        chunk_summary, token_info = call_gemini(model, prompt, max_new_tokens=max_output_tokens)
        total_output_tokens += token_info.get("output_tokens", 0)
        # Extract content within <summary> tags if present
        match = re.search(r"<summary>(.*?)</summary>", chunk_summary, re.DOTALL)
        if match:
            chunk_summary = match.group(1).strip()
        chunk_summaries.append(chunk_summary)
        print(f"    Chunk {i + 1} summary: {len(chunk_summary)} chars")

    # Consolidate all chunk summaries
    combined = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{s}" for i, s in enumerate(chunk_summaries)
    )
    print(f"  Consolidating {len(chunk_summaries)} chunk summaries ({len(combined)} chars)...")

    # If combined summaries are still too large, recursively chunk
    if len(combined) > chunk_size:
        print(f"  Combined summaries exceed chunk_size, recursing...")
        final_summary, rec_tok = summarize_with_chunking(combined, model, chunk_size, max_output_tokens)
        return final_summary, total_output_tokens + rec_tok

    prompt = FINAL_CONSOLIDATION_PROMPT.format(text=combined)
    final_summary, token_info = call_gemini(model, prompt, max_new_tokens=max_output_tokens)
    total_output_tokens += token_info.get("output_tokens", 0)

    # Extract content within <summary> tags if present
    match = re.search(r"<summary>(.*?)</summary>", final_summary, re.DOTALL)
    if match:
        final_summary = match.group(1).strip()

    return final_summary, total_output_tokens


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_encyclopedia(summary: str, output_dir: str):
    """Save compacted summary as encyclopedia.json with {"insight": summary} format."""
    os.makedirs(output_dir, exist_ok=True)
    encyclopedia_path = os.path.join(output_dir, "encyclopedia.json")

    encyclopedia = {"insight": summary}

    with open(encyclopedia_path, "w", encoding="utf-8") as f:
        json.dump(encyclopedia, f, indent=2, ensure_ascii=False)
    print(f"Encyclopedia saved to: {encyclopedia_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Claude Compact Server - Collect insights and compact via guided summarization with chunking"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="math_output",
        help="Input directory containing insight JSON files (default: math_output)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="math_output",
        help="Output directory for encyclopedia (default: math_output)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of JSON files to use, sorted by number. If not provided, all files are used.",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Google Gemini API key (or set GEMINI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-3-pro-preview",
        help="Gemini model name (default: gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Max characters per chunk for guided summarization (default: 50000)",
    )

    args = parser.parse_args()

    start_time = time.time()

    # Step 1: Collect all insights (same as server_text.py Step 1)
    print("=" * 80)
    print("STEP 1: Collecting Insights")
    print("=" * 80)
    insights = collect_insight_books(args.input_dir, max_files=args.max_files)

    if not insights:
        print("No insights collected. Exiting.")
        exit(1)

    # Step 2: Guided summarization with chunking (replaces server_text.py Steps 2-3)
    print("\n" + "=" * 80)
    print("STEP 2: Guided Summarization with Chunking (Claude Cookbooks pattern)")
    print("=" * 80)
    gemini_model = setup_gemini(api_key=args.gemini_api_key, model_name=args.gemini_model)
    insights_text = format_insights_as_text(insights)
    print(f"Total insights text: {len(insights_text)} characters")

    summary, total_output_tokens = summarize_with_chunking(
        insights_text,
        model=gemini_model,
        chunk_size=args.chunk_size,
    )

    print(f"\nCompacted summary:\n{summary}")

    # Save as encyclopedia
    save_encyclopedia(summary, output_dir=args.output_dir)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("CLAUDE COMPACT COMPLETE")
    print("=" * 80)
    print(f"Total insights collected: {len(insights)}")
    print(f"Compacted summary length: {len(summary)} characters")
    print(f"Insight library output tokens: {total_output_tokens}")
    print(f"Output directory: {args.output_dir}")
    print(f"Time elapsed: {elapsed:.1f}s")
