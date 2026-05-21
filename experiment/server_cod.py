"""
Chain of Density (CoD) Server - Collects all insights from JSON files, then uses
the Chain of Density prompting method to generate increasingly dense summaries.

Step 1: Same as server_text.py - collect all insights from problem/paper JSON files.
Step 2: Apply Chain of Density (CoD) summarization from:
        "From Sparse to Dense: GPT-4 Summarization with Chain of Density Prompting"
        (Adams et al., 2023, https://aclanthology.org/2023.newsum-1.7.pdf)
        - Iteratively generate 5 increasingly dense summaries
        - Each iteration adds 1-3 missing entities while maintaining word count
        - Final (5th) summary is the most dense and is saved as encyclopedia
Output: encyclopedia.json with {"insight": <densest_summary>}
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
# Chain of Density prompt (exact prompt from Adams et al., 2023)
# ---------------------------------------------------------------------------

COD_PROMPT = """\
Article: {article}
You will generate increasingly concise, entity-dense summaries of the above Article.
Repeat the following 2 steps 5 times.
Step 1. Identify 1-3 informative Entities (";" delimited) from the Article which are missing from the previously generated summary.
Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities.
A Missing Entity is: - Relevant: to the main story. - Specific: descriptive yet concise (5 words or fewer). - Novel: not in the previous summary. - Faithful: present in the Article. - Anywhere: located anywhere in the Article.
Guidelines:
- The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
- Make every word count: re-write the previous summary to improve flow and make space for additional entities.
- Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
- The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the Article.
- Missing entities can appear anywhere in the new summary.
- Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
Remember, use the exact same number of words for each summary.
Answer in JSON. The JSON should be a list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary"."""


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
# Chain of Density summarization
# ---------------------------------------------------------------------------

def format_insights_as_text(insights: Dict[str, str]) -> str:
    """Format all collected insights into a single text block."""
    lines = []
    for name, desc in insights.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def chunk_text(text: str, chunk_size: int = 50000) -> List[str]:
    """Split text into chunks by character count."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def extract_json_from_response(response_text: str) -> list:
    """Extract JSON list from LLM response, handling markdown code blocks."""
    # Try to find JSON in code blocks first
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response_text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Try to find raw JSON list
    match = re.search(r"\[.*\]", response_text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    # Try parsing the whole response
    return json.loads(response_text)


def cod_summarize(article_text: str, model, max_output_tokens: int = 32768) -> tuple:
    """
    Apply Chain of Density prompting to a single article/text block.

    Returns (densest_summary, output_tokens).
    """
    prompt = COD_PROMPT.format(article=article_text)
    response, token_info = call_gemini(model, prompt, max_new_tokens=max_output_tokens)
    out_tok = token_info.get("output_tokens", 0)

    try:
        cod_results = extract_json_from_response(response)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Warning: Failed to parse CoD JSON response: {e}")
        print(f"  Raw response:\n{response}")
        return response, out_tok

    # Print each iteration
    for i, entry in enumerate(cod_results):
        missing = entry.get("Missing_Entities", "N/A")
        summary = entry.get("Denser_Summary", "")
        word_count = len(summary.split())
        print(f"  Iteration {i + 1}: +[{missing}] ({word_count} words)")

    # Return the densest (last) summary
    densest = cod_results[-1].get("Denser_Summary", "")
    return densest, out_tok


def cod_summarize_with_chunking(
    insights_text: str,
    model,
    chunk_size: int = 50000,
    max_output_tokens: int = 32768,
) -> tuple:
    """
    Apply Chain of Density summarization with chunking for large inputs.

    1. Chunk insights into manageable pieces
    2. Run CoD on each chunk to get dense summaries
    3. If multiple chunks, consolidate with a final CoD pass

    Returns:
        (final_summary, final_output_tokens) — tokens only for the last
        CoD call that produces the final insight library.
    """
    chunks = chunk_text(insights_text, chunk_size=chunk_size)
    print(f"  Split insights into {len(chunks)} chunks (chunk_size={chunk_size})")

    if len(chunks) == 1:
        # Single chunk - directly apply CoD
        print("  Single chunk, applying Chain of Density directly...")
        return cod_summarize(chunks[0], model, max_output_tokens)

    # Multiple chunks - CoD each chunk, then consolidate
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"\n  --- Chunk {i + 1}/{len(chunks)} ({len(chunk)} chars) ---")
        summary, _ = cod_summarize(chunk, model, max_output_tokens)
        chunk_summaries.append(summary)
        print(f"  Chunk {i + 1} densest summary: {len(summary.split())} words")

    # Consolidate: combine chunk summaries and run final CoD pass
    combined = "\n\n".join(
        f"[Section {i + 1}] {s}" for i, s in enumerate(chunk_summaries)
    )
    print(f"\n  --- Final consolidation ({len(combined)} chars) ---")

    # If combined is still too large, recursively chunk
    if len(combined) > chunk_size:
        print(f"  Combined summaries exceed chunk_size, recursing...")
        return cod_summarize_with_chunking(combined, model, chunk_size, max_output_tokens)

    # Final call — this produces the library; return its token count
    return cod_summarize(combined, model, max_output_tokens)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_encyclopedia(summary: str, output_dir: str):
    """Save densest summary as encyclopedia.json with {"insight": summary} format."""
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
        description="Chain of Density Server - Collect insights and summarize via CoD prompting (Adams et al., 2023)"
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
        default="gemini-2.0-flash",
        help="Gemini model name (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Max characters per chunk for CoD summarization (default: 50000)",
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

    # Step 2: Chain of Density summarization (replaces server_text.py Steps 2-3)
    print("\n" + "=" * 80)
    print("STEP 2: Chain of Density Summarization (Adams et al., 2023)")
    print("=" * 80)
    gemini_model = setup_gemini(api_key=args.gemini_api_key, model_name=args.gemini_model)
    insights_text = format_insights_as_text(insights)
    print(f"Total insights text: {len(insights_text)} characters")

    summary, total_output_tokens = cod_summarize_with_chunking(
        insights_text,
        model=gemini_model,
        chunk_size=args.chunk_size,
    )

    print(f"\nDensest summary:\n{summary}")

    # Save as encyclopedia
    save_encyclopedia(summary, output_dir=args.output_dir)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("CHAIN OF DENSITY COMPLETE")
    print("=" * 80)
    print(f"Total insights collected: {len(insights)}")
    print(f"Densest summary length: {len(summary.split())} words, {len(summary)} chars")
    print(f"Insight library output tokens: {total_output_tokens}")
    print(f"Output directory: {args.output_dir}")
    print(f"Time elapsed: {elapsed:.1f}s")
