"""
Randomly sample N accepted ICLR papers for a given year and output their titles.

Usage:
  python checker_iclr_sample.py --year 2024 --sample-size 50 --output sampled_papers_2024.json
  python checker_iclr_sample.py --year 2025 --accept-oral --accept-spotlight \
      --or-username user@example.com --or-password secret --output sampled_2025.json
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional


def fetch_accept_tracks(
    year: int,
    accept_oral: bool = True,
    accept_spotlight: bool = False,
    accept_poster: bool = False,
    or_username: str = None,
    or_password: str = None,
) -> List[Dict]:
    """Fetch accepted papers via OpenReview client."""
    try:
        import openreview
    except ImportError:
        raise ImportError("Install openreview-py: pip install openreview-py")

    accept_any = accept_oral or accept_spotlight or accept_poster
    if not accept_any:
        accept_oral = True

    client = openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=or_username,
        password=or_password,
    )
    print(f"Fetching ICLR {year} submissions via OpenReview client...")
    submissions = list(
        client.get_all_notes(
            invitation=f"ICLR.cc/{year}/Conference/-/Submission",
            details="directReplies",
        )
    )
    print(f"Retrieved {len(submissions)} submissions, filtering by venue...")

    papers = []
    for sub in submissions:
        content = sub.content
        venue = str(content.get("venue", {}).get("value", ""))

        track = None
        if "oral" in venue.lower():
            track = "oral"
            if not accept_oral:
                continue
        elif "spotlight" in venue.lower():
            track = "spotlight"
            if not accept_spotlight:
                continue
        elif "poster" in venue.lower():
            track = "poster"
            if not accept_poster:
                continue
        else:
            continue

        papers.append(
            {
                "id": sub.id,
                "forum": sub.forum,
                "title": content.get("title", {}).get("value", ""),
                "abstract": content.get("abstract", {}).get("value", ""),
                "venue": venue,
                "track": track,
            }
        )

    track_order = {"oral": 0, "spotlight": 1, "poster": 2}
    papers.sort(key=lambda p: track_order.get(p.get("track", ""), 999))
    return papers


def main():
    parser = argparse.ArgumentParser(
        description="Sample N accepted ICLR papers for a given year and save titles + metadata"
    )
    parser.add_argument("--year", type=int, default=2024, help="ICLR year (default: 2024)")
    parser.add_argument(
        "--sample-size", type=int, default=50, help="Number of papers to sample (default: 50)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--accept-oral", action="store_true", help="Include Oral papers"
    )
    parser.add_argument(
        "--accept-spotlight", action="store_true", help="Include Spotlight papers"
    )
    parser.add_argument(
        "--accept-poster", action="store_true", help="Include Poster papers"
    )
    parser.add_argument(
        "--or-username",
        type=str,
        default=None,
        help="OpenReview account email (required for ICLR 2025+)",
    )
    parser.add_argument(
        "--or-password",
        type=str,
        default=None,
        help="OpenReview account password (required for ICLR 2025+)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: sampled_papers_<year>.json)",
    )
    args = parser.parse_args()

    output_path = args.output or f"sampled_papers_{args.year}.json"

    papers = fetch_accept_tracks(
        args.year,
        accept_oral=args.accept_oral,
        accept_spotlight=args.accept_spotlight,
        accept_poster=args.accept_poster,
        or_username=args.or_username,
        or_password=args.or_password,
    )
    if not papers:
        print("No accepted papers found.")
        return

    print(f"Total accepted papers fetched: {len(papers)}")

    random.seed(args.seed)
    sample_size = min(args.sample_size, len(papers))
    sampled = random.sample(papers, sample_size)

    # Sort sampled by track for readability
    track_order = {"oral": 0, "spotlight": 1, "poster": 2}
    sampled.sort(key=lambda p: track_order.get(p.get("track", ""), 999))

    print(f"\nSampled {sample_size} papers (seed={args.seed}):")
    for i, p in enumerate(sampled, 1):
        print(f"  [{i:2d}] [{p['track']:10s}] {p['title']}")

    output_data = {
        "year": args.year,
        "sample_size": sample_size,
        "seed": args.seed,
        "tracks": {
            "oral": args.accept_oral or not (args.accept_oral or args.accept_spotlight or args.accept_poster),
            "spotlight": args.accept_spotlight,
            "poster": args.accept_poster,
        },
        "papers": [
            {
                "id": p["id"],
                "forum": p["forum"],
                "title": p["title"],
                "abstract": p["abstract"],
                "track": p["track"],
                "venue": p["venue"],
            }
            for p in sampled
        ],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {sample_size} paper records to {output_path}")


if __name__ == "__main__":
    main()
