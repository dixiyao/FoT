"""
Naive Append Server - Collects all insights from JSON files and saves them directly
as an encyclopedia without any LLM-based profiling or knowledge extraction.

This script follows the same input/output protocol as server_text.py but only
performs Step 1 (collect insights) and directly saves the insight_store as
encyclopedia.json in {insight_name: description} format.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional


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


def save_encyclopedia(insights: Dict[str, str], output_dir: str):
    """Save insights as encyclopedia.json in {insight_name: description} format."""
    os.makedirs(output_dir, exist_ok=True)
    encyclopedia_path = os.path.join(output_dir, "encyclopedia.json")

    with open(encyclopedia_path, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)
    print(f"Encyclopedia saved to: {encyclopedia_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Naive Append Server - Collect insights and save directly as encyclopedia (no LLM aggregation)"
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
        help="Maximum number of JSON files to use, sorted by number (e.g. paper_0001 before paper_0002). If not provided, all files are used.",
    )

    args = parser.parse_args()

    start_time = time.time()

    # Step 1: Collect all insights
    insights = collect_insight_books(args.input_dir, max_files=args.max_files)

    if not insights:
        print("No insights collected. Exiting.")
        exit(1)

    # Save directly as encyclopedia (skip profiling and knowledge extraction)
    save_encyclopedia(insights, output_dir=args.output_dir)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("NAIVE APPEND COMPLETE")
    print("=" * 80)
    print(f"Total insights collected: {len(insights)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Time elapsed: {elapsed:.1f}s")
