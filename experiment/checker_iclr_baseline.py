"""
Baseline: Check ICLR Accept papers using learned skills from encyclopedia.
- Fetches ICLR papers for a given year directly from OpenReview API.
- Uses learned skills encyclopedia to extract skill names and check if they guide papers.
- Each skill includes: year proposed, is_iclr2023 (boolean).
- Outputs: overall percentage, percentage for pre-2023 skills, percentage for ICLR2023 skills.

Usage example:
  python checker_iclr_baseline.py \
      --gemini-key $GEMINI_API_KEY \
      --year 2024 \
      --output baseline_results.json
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests

# Import shared functions from checker_iclr for maximum consistency
from checker_iclr import (GeminiClient, _fetch_paper_content, call_gemini,
                          score_paper)

# Import genai for file search (RAG mode only)
# Note: File search requires the new google.genai API, not the old google.generativeai
try:
    import google.genai as genai_new  # type: ignore
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    genai_new = None
    types = None

# Note: We keep fetch_accept_tracks and _hydrate_papers_from_client local
# because _hydrate_papers_from_client needs to extract keywords


def fetch_accept_tracks(
    year: int,
    max_papers: int = None,
    accept_oral: bool = True,
    accept_spotlight: bool = False,
    accept_poster: bool = False,
) -> List[Dict]:
    """Fetch accepted papers using OpenReview client.

    Uses openreview-py to query ICLR submissions and filter by venue field.
    """
    try:
        import openreview

        use_or_client = True
    except ImportError:
        use_or_client = False
        print("Warning: openreview-py not installed, falling back to requests")

    # Default to oral if nothing specified (backward compatible)
    accept_any = accept_oral or accept_spotlight or accept_poster
    accept_oral = accept_oral or not accept_any

    decisions: List[Dict] = []

    # Try OpenReview client first
    if use_or_client:
        try:
            client = openreview.api.OpenReviewClient(
                baseurl="https://api2.openreview.net"
            )
            print(f"Fetching ICLR {year} submissions via OpenReview client...")
            submissions = list(
                client.get_all_notes(
                    invitation=f"ICLR.cc/{year}/Conference/-/Submission",
                    details="directReplies",
                )
            )
            print(f"Retrieved {len(submissions)} submissions, filtering by venue...")

            for sub in submissions:
                content = sub.content
                # API v2 nests values
                venue = str(content.get("venue", {}).get("value", ""))

                # Check if accepted and what track
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

                decisions.append({"forum": sub.forum, "track": track})
                if max_papers and len(decisions) >= max_papers:
                    break

            if decisions:
                print(f"Found {len(decisions)} accepted papers via client")
                # Sort by track priority: oral > spotlight > poster
                track_order = {"oral": 0, "spotlight": 1, "poster": 2}
                decisions.sort(key=lambda p: track_order.get(p.get("track", ""), 999))
                # Store client for later use
                for decision in decisions:
                    decision["_client"] = client
                return decisions
            else:
                print("No accepted papers found via client")
                return []
        except Exception as e:
            print(f"OpenReview client error: {e}")
            return []

    # Fallback: old requests-based approach
    print("Using requests-based fallback (may not work for ICLR 2024+)...")
    return []


def generate_skills_from_keywords(
    model: GeminiClient,
    keywords: List[str],
) -> Tuple[List[Dict], Dict]:
    """Generate skills/insights from paper keywords using Gemini.

    For each keyword, asks Gemini to generate a corresponding skill/technique description.

    Returns:
        Tuple of (skills_list, token_info) where skills_list contains skill dicts with name/description
    """
    if not keywords:
        return [], {"output_tokens": 0}

    skills = []
    total_tokens = 0

    for keyword in keywords:
        prompt = f"""Generate a skill corresponding to the given keyword: {keyword}.

Provide a technical description and guidelines of using this skill/insight to resolve questions.

Respond in the following JSON format:
{{
  "skill_name": "concise name of the skill",
  "description": "detailed technical description and usage guidelines"
}}"""
        try:
            response, token_info = call_gemini(model, prompt)
            total_tokens += token_info.get("output_tokens", 0)

            # Try to parse JSON from response
            try:
                import re

                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    skill_data = json.loads(json_match.group())
                    skills.append(
                        {
                            "name": skill_data.get("skill_name", keyword),
                            "description": skill_data.get(
                                "description", response.strip()
                            ),
                        }
                    )
                else:
                    skills.append({"name": keyword, "description": response.strip()})
            except json.JSONDecodeError:
                skills.append({"name": keyword, "description": response.strip()})

            time.sleep(0.3)  # Rate limiting between keyword processing
        except Exception as e:
            print(f"    Warning: Failed to generate skill for keyword '{keyword}': {e}")
            skills.append(
                {"name": keyword, "description": f"Skill related to {keyword}"}
            )

    return skills, {"output_tokens": total_tokens}


def generate_skills_phase_from_iclr2023_keywords(
    model: GeminiClient,
    output_file: str,
) -> List[Dict]:
    """Phase 1, Mode 1: Generate skills from ICLR 2023 top25 paper keywords.

    Fetches ICLR 2023 top25 papers from OpenReview API using Blind_Submission invitation.
    Implements exponential backoff to handle rate limiting.
    """
    print("\n" + "=" * 80)
    print("PHASE 1: Generate Skills from ICLR 2023 Top25 Paper Keywords")
    print("=" * 80)

    # Fetch ICLR 2023 notable top25 papers from OpenReview API
    print("Fetching ICLR 2023 Notable Top 25% papers from OpenReview...")
    import requests

    papers = []
    api_url = "https://api.openreview.net/notes"
    offset = 0
    limit = 1000

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
    )

    while True:
        retry_count = 0
        max_retries = 5
        success = False

        while retry_count < max_retries and not success:
            try:
                params = {
                    "invitation": "ICLR.cc/2023/Conference/-/Blind_Submission",
                    "details": "replyCount,invitation,original",
                    "offset": offset,
                    "limit": limit,
                    "sort": "number:asc",
                }

                response = session.get(api_url, params=params, timeout=60)
                response.raise_for_status()

                data = response.json()
                notes = data.get("notes", [])

                if not notes:
                    success = True  # Reached end of results
                    break

                # Filter for notable top 25% papers
                for note in notes:
                    content = note.get("content", {})
                    venue = content.get("venue", "")

                    # Check if this is a notable top 25% paper
                    if (
                        "Notable Top 25%" in venue
                        or ("Notable" in venue and "Top 25%" in venue)
                        or "notable top 25%" in venue.lower()
                    ):
                        papers.append(note)

                offset += limit
                print(
                    f"  Searched offset {offset}, found {len(papers)} notable top 25% papers so far..."
                )
                success = True  # Successfully fetched this batch

            except requests.exceptions.HTTPError as e:
                error_code = e.response.status_code if hasattr(e, "response") else 0
                error_str = str(e)

                # Check if rate limited (429)
                if (
                    error_code == 429
                    or "429" in error_str
                    or "Too Many Requests" in error_str
                ):
                    retry_count += 1
                    if retry_count < max_retries:
                        # Exponential backoff: start at 10s, then 20s, 40s, 80s, 160s
                        wait_time = 10 * (2 ** (retry_count - 1))
                        print(
                            f"  Rate limited (429). Waiting {wait_time}s before retry ({retry_count}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                    else:
                        print(f"  Error: Rate limited - max retries exceeded")
                        break
                else:
                    print(f"  HTTP Error: {error_str}")
                    break
            except Exception as e:
                error_str = str(e)

                # Check if rate limited in error message
                if (
                    "429" in error_str
                    or "RateLimitError" in error_str
                    or "Too many requests" in error_str.lower()
                ):
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 10 * (2 ** (retry_count - 1))
                        print(
                            f"  Rate limited. Waiting {wait_time}s before retry ({retry_count}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                    else:
                        print(f"  Error: Rate limited - max retries exceeded")
                        break
                else:
                    print(f"  Error fetching papers: {e}")
                    break

        if not success or not notes:
            break

    if not papers:
        print("Error: No ICLR 2023 notable top 25% papers found")
        print(
            "Tip: OpenReview API rate limit is 60 requests/minute. Try again later or contact OpenReview support."
        )
        return []

    print(f"Retrieved {len(papers)} ICLR 2023 notable top 25% papers")

    # Extract all unique keywords from papers
    all_keywords = set()
    for note in papers:
        content = note.get("content", {})
        keywords_raw = content.get("keywords", {})

        if isinstance(keywords_raw, dict):
            keywords_raw = keywords_raw.get("value", [])

        if isinstance(keywords_raw, str):
            keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
        elif isinstance(keywords_raw, list):
            keywords = keywords_raw
        else:
            keywords = []

        all_keywords.update(keywords)

    all_keywords = sorted(list(all_keywords))
    print(f"Extracted {len(all_keywords)} unique keywords from ICLR 2023 papers")
    print(f"Sample keywords: {all_keywords[:5]}")

    # Generate skills from keywords
    print(f"\nGenerating skills from {len(all_keywords)} keywords...")
    skills, token_info = generate_skills_from_keywords(model, all_keywords)
    print(
        f"Generated {len(skills)} skills with {token_info.get('output_tokens', 0)} tokens"
    )

    # Save skills
    skills_data = {
        "source": "ICLR 2023 Notable Top 25% Papers",
        "num_papers": len(papers),
        "num_skills": len(skills),
        "skills": skills,
        "generation_tokens": token_info.get("output_tokens", 0),
    }
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(skills_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(skills)} skills to {output_file}")

    return skills


def generate_skills_phase_from_iclr2024_keywords(
    model: GeminiClient,
    output_file: str,
) -> List[Dict]:
    """Phase 1, Mode 1b: Generate skills from ICLR 2024 accepted paper keywords.

    Fetches ICLR 2024 accepted papers (Oral/Spotlight/Poster) from OpenReview API.
    Implements exponential backoff to handle rate limiting.
    """
    print("\n" + "=" * 80)
    print("PHASE 1: Generate Skills from ICLR 2024 Accepted Paper Keywords")
    print("=" * 80)

    # Fetch ICLR 2024 accepted papers
    print("Fetching ICLR 2024 accepted papers from OpenReview...")
    import requests

    papers = []
    api_url = "https://api2.openreview.net/notes"
    offset = 0
    limit = 200

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
    )

    while True:
        retry_count = 0
        max_retries = 5
        success = False

        while retry_count < max_retries and not success:
            try:
                params = {
                    "invitation": "ICLR.cc/2024/Conference/-/Submission",
                    "details": "replyCount",
                    "offset": offset,
                    "limit": limit,
                    "sort": "number:asc",
                }

                response = session.get(api_url, params=params, timeout=60)

                if response.status_code == 429:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 10 * (2 ** (retry_count - 1))
                        print(
                            f"  Rate limited (429). Waiting {wait_time}s before retry ({retry_count}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  Error: Rate limited - max retries exceeded")
                        break

                response.raise_for_status()

                data = response.json()
                notes = data.get("notes", [])

                if not notes:
                    success = True
                    break

                # Filter for accepted papers (Oral/Spotlight/Poster)
                for note in notes:
                    content = note.get("content", {})
                    venue_obj = content.get("venue", {})
                    venue_val = ""
                    if isinstance(venue_obj, dict):
                        venue_val = str(venue_obj.get("value", ""))
                    elif isinstance(venue_obj, str):
                        venue_val = venue_obj
                    venue_lower = venue_val.lower()

                    # Check if accepted (oral, spotlight, or poster)
                    if (
                        "oral" in venue_lower
                        or "spotlight" in venue_lower
                        or "poster" in venue_lower
                    ):
                        papers.append(note)

                offset += limit
                if offset % 1000 == 0:
                    print(
                        f"  Searched offset {offset}, found {len(papers)} accepted papers so far..."
                    )
                success = True
                time.sleep(1.0)  # Rate limit between requests

            except requests.exceptions.HTTPError as e:
                error_code = e.response.status_code if hasattr(e, "response") else 0
                error_str = str(e)

                if error_code == 429 or "429" in error_str:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 10 * (2 ** (retry_count - 1))
                        print(
                            f"  Rate limited (429). Waiting {wait_time}s before retry ({retry_count}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                    else:
                        print(f"  Error: Rate limited - max retries exceeded")
                        break
                else:
                    print(f"  HTTP Error: {error_str}")
                    break
            except Exception as e:
                error_str = str(e)

                if "429" in error_str or "RateLimitError" in error_str:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 10 * (2 ** (retry_count - 1))
                        print(
                            f"  Rate limited. Waiting {wait_time}s before retry ({retry_count}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                    else:
                        print(f"  Error: Rate limited - max retries exceeded")
                        break
                else:
                    print(f"  Error fetching papers: {e}")
                    break

        if not success or not notes:
            break

    if not papers:
        print("Error: No ICLR 2024 accepted papers found")
        return []

    print(f"Retrieved {len(papers)} ICLR 2024 accepted papers")

    # Extract all unique keywords from papers
    all_keywords = set()
    for note in papers:
        content = note.get("content", {})
        keywords_raw = content.get("keywords", {})

        if isinstance(keywords_raw, dict):
            keywords_raw = keywords_raw.get("value", [])

        if isinstance(keywords_raw, str):
            keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
        elif isinstance(keywords_raw, list):
            keywords = keywords_raw
        else:
            keywords = []

        all_keywords.update(keywords)

    all_keywords = sorted(list(all_keywords))
    print(f"Extracted {len(all_keywords)} unique keywords from ICLR 2024 papers")
    print(f"Sample keywords: {all_keywords[:5]}")

    # Generate skills from keywords
    print(f"\nGenerating skills from {len(all_keywords)} keywords...")
    skills, token_info = generate_skills_from_keywords(model, all_keywords)
    print(
        f"Generated {len(skills)} skills with {token_info.get('output_tokens', 0)} tokens"
    )

    # Save skills
    skills_data = {
        "source": "ICLR 2024 Accepted Papers (Oral/Spotlight/Poster)",
        "num_papers": len(papers),
        "num_skills": len(skills),
        "skills": skills,
        "generation_tokens": token_info.get("output_tokens", 0),
    }
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(skills_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(skills)} skills to {output_file}")

    return skills


def generate_skills_phase_from_general_knowledge(
    model: GeminiClient,
    num_skills: int,
    output_file: str,
    year: int = 2024,
) -> List[Dict]:
    """Phase 1, Mode 2: Generate x skills from general machine learning knowledge."""
    print("\n" + "=" * 80)
    print(f"PHASE 1: Generate {num_skills} Skills from General ML Knowledge")
    print("=" * 80)

    prompt = f"""Generate a list of {num_skills} important and fundamental skills/techniques in machine learning.

For each skill, provide:
1. A concise skill name
2. A detailed technical description and usage guidelines

Respond in JSON format with an array:
[
  {{
    "skill_name": "skill name",
    "description": "technical description and guidelines"
  }},
  ...
]"""

    print(f"Requesting {num_skills} skills from Gemini...")
    response, token_info = call_gemini(model, prompt)

    # Parse JSON array from response
    skills = []
    try:
        import re

        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            skills_data = json.loads(json_match.group())
            for skill_data in skills_data:
                skills.append(
                    {
                        "name": skill_data.get("skill_name", f"Skill {len(skills)+1}"),
                        "description": skill_data.get("description", ""),
                    }
                )
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Warning: Failed to parse JSON response: {e}")
        print("Response:", response[:200])

    print(
        f"Generated {len(skills)} skills with {token_info.get('output_tokens', 0)} tokens"
    )

    # Save skills
    output_filename = output_file or f"gemini_baseline_skills_{num_skills}_year{year}.json"
    skills_result = {
        "source": f"General ML Knowledge ({num_skills} skills requested)",
        "num_skills": len(skills),
        "skills": skills,
        "generation_tokens": token_info.get("output_tokens", 0),
    }
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(skills_result, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(skills)} skills to {output_filename}")

    return skills


def generate_skills_phase_from_rag(
    api_key: str,
    papers_dir: str,
    num_skills: int,
    output_file: str,
    year: int = 2024,
) -> List[Dict]:
    """Phase 1, Mode 3: Generate skills using RAG with file search store.

    Uploads papers from a directory to Google's file search store and generates
    skills using both general knowledge and the uploaded papers.

    Args:
        api_key: Google Gemini API key
        papers_dir: Directory containing PDF papers to upload
        num_skills: Number of skills to generate
        output_file: Path to save generated skills

    Returns:
        List of generated skills with name and description
    """
    print("\n" + "=" * 80)
    print(f"PHASE 1: Generate {num_skills} Skills using RAG with File Search")
    print("=" * 80)

    if not HAS_GENAI:
        raise ImportError(
            "google-genai package is required for RAG mode. "
            "Install with: pip install google-genai"
        )

    # Initialize client using the new API (required for file search)
    client = genai_new.Client(api_key=api_key)

    # Find all PDF files in the directory first
    papers_path = Path(papers_dir)
    if not papers_path.exists():
        raise ValueError(f"Papers directory does not exist: {papers_dir}")

    pdf_files = list(papers_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {papers_dir}")

    print(f"\nFound {len(pdf_files)} PDF files in {papers_dir}")

    # Create stable display name based on directory name
    dir_name = papers_path.name
    store_display_name = f"iclr-papers-{dir_name}"

    # Check if file search store with this name already exists
    print(f"\nChecking for existing file search store: {store_display_name}")
    file_search_store = None
    store_is_reused = False

    try:
        # List existing stores and find matching one
        stores = client.file_search_stores.list()
        for store in stores:
            if hasattr(store, 'display_name') and store.display_name == store_display_name:
                file_search_store = store
                store_is_reused = True
                print(f"✓ Found existing file search store: {store.name}")
                print(f"  Reusing existing store to avoid duplicates")
                break
    except Exception as e:
        print(f"Warning: Could not list existing stores: {e}")

    # Create new store if not found
    if file_search_store is None:
        print(f"Creating new file search store: {store_display_name}")
        file_search_store = client.file_search_stores.create(
            config={'display_name': store_display_name}
        )
        print(f"✓ Created file search store: {file_search_store.name}")

    # Upload papers only if store is newly created (not reused)
    uploaded_count = 0
    failed_count = 0

    if store_is_reused:
        print(f"\nSkipping upload - reusing existing store with previously uploaded papers")
        # Note: We assume the store has the papers. If you need to verify or update,
        # you can modify this logic to always upload or check file count.
        uploaded_count = len(pdf_files)  # Assume all papers were previously uploaded
    else:
        print(f"\nUploading {len(pdf_files)} papers to file search store...")
        uploaded_count = 0
        failed_count = 0

        for idx, pdf_file in enumerate(pdf_files, 1):
            print(f"[{idx}/{len(pdf_files)}] Uploading {pdf_file.name}...")
            try:
                operation = client.file_search_stores.upload_to_file_search_store(
                    file=str(pdf_file),
                    file_search_store_name=file_search_store.name,
                    config={
                        'display_name': pdf_file.stem,
                    }
                )

                # Wait for upload to complete
                retry_count = 0
                max_retries = 30  # 30 * 5s = 2.5 minutes max wait
                while not operation.done and retry_count < max_retries:
                    time.sleep(5)
                    operation = client.operations.get(operation)
                    retry_count += 1

                if operation.done:
                    uploaded_count += 1
                    print(f"  ✓ Uploaded successfully")
                else:
                    failed_count += 1
                    print(f"  ✗ Upload timeout")

            except Exception as e:
                failed_count += 1
                print(f"  ✗ Upload failed: {e}")

            # Rate limiting
            time.sleep(1)

        print(f"\nUpload complete: {uploaded_count} successful, {failed_count} failed")

        if uploaded_count == 0:
            raise RuntimeError("No papers were successfully uploaded to file search store")

    # Generate skills using file search
    print(f"\nGenerating {num_skills} skills using file search store...")

    prompt = f"""Based on your general machine learning knowledge AND the uploaded research papers, generate a list of {num_skills} important and fundamental skills/techniques in machine learning.

For each skill, provide:
1. A concise skill name
2. A detailed technical description and usage guidelines

Respond in JSON format with an array:
[
  {{
    "skill_name": "skill name",
    "description": "technical description and guidelines"
  }},
  ...
]"""

    try:
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=65536,
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[file_search_store.name]
                        )
                    )
                ]
            )
        )

        response_text = response.text
        # Estimate tokens (rough approximation: 1 token ≈ 4 chars)
        output_tokens = len(response_text) // 4

        print(f"Generated response with ~{output_tokens} tokens")

    except Exception as e:
        raise RuntimeError(f"Error generating skills with file search: {e}")

    # Parse JSON array from response
    skills = []
    try:
        import re

        json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if json_match:
            skills_data = json.loads(json_match.group())
            for skill_data in skills_data:
                skills.append(
                    {
                        "name": skill_data.get("skill_name", f"Skill {len(skills)+1}"),
                        "description": skill_data.get("description", ""),
                    }
                )
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Warning: Failed to parse JSON response: {e}")
        print("Response:", response_text[:200])

    print(f"Generated {len(skills)} skills")

    # Save skills
    output_filename = output_file or f"rag_baseline_skills_{num_skills}_year{year}.json"
    skills_result = {
        "source": f"RAG with File Search ({num_skills} skills requested from {len(pdf_files)} papers)",
        "papers_dir": papers_dir,
        "num_papers": len(pdf_files),
        "num_papers_uploaded": uploaded_count if not store_is_reused else 0,
        "store_reused": store_is_reused,
        "num_skills": len(skills),
        "skills": skills,
        "generation_tokens": output_tokens,
        "file_search_store": file_search_store.name,
        "file_search_store_display_name": store_display_name,
    }
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(skills_result, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(skills)} skills to {output_filename}")

    return skills


def main():
    parser = argparse.ArgumentParser(
        description="Two-phase baseline: Generate skill sets, then check if ICLR papers are guided by them"
    )

    # Phase control
    parser.add_argument(
        "--phase",
        type=str,
        choices=["generate", "check", "both"],
        default="both",
        help="Which phase to run: 'generate' (Phase 1), 'check' (Phase 2), or 'both' (default)",
    )

    # Phase 1: Generate skills
    parser.add_argument(
        "--generate-mode",
        type=str,
        choices=["iclr2023_keywords", "iclr2024_keywords", "general_knowledge", "rag"],
        default="iclr2023_keywords",
        help="Mode for Phase 1 skill generation (default: iclr2023_keywords)",
    )
    parser.add_argument(
        "--num-skills",
        type=int,
        default=50,
        help="Number of skills to generate in 'general_knowledge' or 'rag' mode (default: 50)",
    )
    parser.add_argument(
        "--rag-papers-dir",
        type=str,
        help="Directory containing PDF papers for RAG mode (required for 'rag' mode)",
    )
    parser.add_argument(
        "--skills-output",
        type=str,
        help="Output file for generated skills (Phase 1). Default: iclr2023_top25_baseline_skills.json, gemini_baseline_skills_{x}.json, or rag_baseline_skills_{x}.json",
    )

    # Phase 2: Check papers
    parser.add_argument(
        "--skills-file",
        type=str,
        help="Path to skills file from Phase 1 (required for Phase 2 if not running Phase 1)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="ICLR conference year for Phase 2 (default: 2024)",
    )
    parser.add_argument(
        "--accept-oral",
        action="store_true",
        help="Include Accept (Oral) papers in Phase 2",
    )
    parser.add_argument(
        "--accept-spotlight",
        action="store_true",
        help="Include Accept (Spotlight) papers in Phase 2",
    )
    parser.add_argument(
        "--accept-poster",
        action="store_true",
        help="Include Accept (Poster) papers in Phase 2",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Max papers to process in Phase 2 (default: all)",
    )
    parser.add_argument(
        "--check-output",
        type=str,
        required=False,
        help="Output JSON file for Phase 2 results (default: baseline_check_results_{year}.json)",
    )

    # Common arguments
    parser.add_argument(
        "--gemini-key",
        type=str,
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-3-pro-preview",
        help="Gemini model name (default: gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between Gemini calls (default: 0.5)",
    )

    args = parser.parse_args()

    api_key = args.gemini_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key is required. Provide --gemini-key or set GEMINI_API_KEY."
        )

    model = GeminiClient(api_key=api_key, model_name=args.gemini_model)

    # ========== PHASE 1: Generate Skills ==========
    skills = []
    skills_file = args.skills_file

    if args.phase in ["generate", "both"]:
        if args.generate_mode == "iclr2023_keywords":
            skills_output = args.skills_output or "iclr2023_top25_baseline_skills.json"
            skills = generate_skills_phase_from_iclr2023_keywords(model, skills_output)
            skills_file = skills_output
        elif args.generate_mode == "iclr2024_keywords":
            skills_output = (
                args.skills_output or "iclr2024_accepted_baseline_skills.json"
            )
            skills = generate_skills_phase_from_iclr2024_keywords(model, skills_output)
            skills_file = skills_output
        elif args.generate_mode == "rag":
            # RAG mode requires papers directory
            if not args.rag_papers_dir:
                raise ValueError(
                    "RAG mode requires --rag-papers-dir to specify the papers directory"
                )
            skills_output = (
                args.skills_output or f"rag_baseline_skills_{args.num_skills}_year{args.year}.json"
            )
            skills = generate_skills_phase_from_rag(
                api_key, args.rag_papers_dir, args.num_skills, skills_output, year=args.year
            )
            skills_file = skills_output
        else:  # general_knowledge
            skills_output = (
                args.skills_output or f"gemini_baseline_skills_{args.num_skills}_year{args.year}.json"
            )
            skills = generate_skills_phase_from_general_knowledge(
                model, args.num_skills, skills_output, year=args.year
            )
            skills_file = skills_output

    # ========== PHASE 2: Check Papers ==========
    if args.phase in ["check", "both"]:
        if not skills and not skills_file:
            raise ValueError(
                "Phase 2 requires either Phase 1 to run or --skills-file to be specified"
            )

        # Load skills if not already generated
        if not skills and skills_file:
            print(f"\nLoading skills from {skills_file}...")
            try:
                with open(skills_file, "r") as f:
                    skills_data = json.load(f)
                    skills = skills_data.get("skills", [])
                    print(f"Loaded {len(skills)} skills")
            except FileNotFoundError:
                print(f"Error: Skills file not found: {skills_file}")
                return

        check_papers_phase(
            model,
            skills,
            args.year,
            args.max_papers,
            args.accept_oral,
            args.accept_spotlight,
            args.accept_poster,
            args.check_output,
            args.sleep,
        )


def check_papers_phase(
    model: GeminiClient,
    skills: List[Dict],
    year: int,
    max_papers: int = None,
    accept_oral: bool = True,
    accept_spotlight: bool = False,
    accept_poster: bool = False,
    output_file: str = None,
    sleep_duration: float = 0.5,
):
    """Phase 2: Check if ICLR papers are guided by the skill set."""
    print("\n" + "=" * 80)
    print(f"PHASE 2: Check ICLR {year} Papers Against Skill Set")
    print("=" * 80)

    # Set defaults
    if not accept_oral and not accept_spotlight and not accept_poster:
        accept_oral = True

    if not output_file:
        output_file = f"baseline_check_results_{year}.json"

    papers = fetch_accept_tracks(
        year,
        max_papers=max_papers,
        accept_oral=accept_oral,
        accept_spotlight=accept_spotlight,
        accept_poster=accept_poster,
    )
    if not papers:
        print(f"No Accept papers found for ICLR {year}")
        return

    print(f"\nProcessing {len(papers)} papers (sorted: oral → spotlight → poster)...\n")

    results = []
    all_matched_skills = set()
    track_stats = {
        "oral": {"total": 0, "guided": 0},
        "spotlight": {"total": 0, "guided": 0},
        "poster": {"total": 0, "guided": 0},
    }

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
    )

    # Format skills for evaluation
    skills_list = skills if isinstance(skills, list) else skills.get("skills", [])
    skills_text = "\n".join(
        [
            f"{i+1}. {s.get('name', f'Skill {i+1}')}: {s.get('description', '')}"
            for i, s in enumerate(skills_list)
        ]
    )

    for idx, paper in enumerate(papers, 1):
        track_label = paper.get("track", "")
        forum_id = paper.get("forum") or paper.get("id")

        print(f"[{idx}/{len(papers)}] Processing ({track_label}): Paper {forum_id}")

        # Fetch full paper content
        print("  Fetching paper content...")
        paper_content = _fetch_paper_content(forum_id, session)
        paper["content"] = paper_content
        if paper_content:
            print(f"  Retrieved {len(paper_content)} characters")
        else:
            print("  No full content available, using title/abstract only")

        time.sleep(1)  # Rate limit between fetches

        # Evaluate if paper is guided by skill set
        print("  Evaluating guidance with Gemini...")
        guided = False
        matched_insights = []
        total_tokens = 0

        try:
            # Modify paper object to include the skills as insights
            paper_with_skills = paper.copy()
            verdict, token_info = score_paper(model, skills_text, paper_with_skills)
            total_tokens += token_info.get("output_tokens", 0)
            guided = bool(verdict.get("guided"))
            matched_insights = verdict.get("matched_insights") or []

            # Track matched skills
            for insight in matched_insights:
                all_matched_skills.add(insight)

            print(
                f"  Result: {'✓ GUIDED' if guided else '✗ Not guided'} | Matched: {len(matched_insights)} | Tokens: {total_tokens}"
            )
        except Exception as exc:
            print(f"  Gemini error during evaluation: {exc}")

        # Update statistics
        if track_label in track_stats:
            track_stats[track_label]["total"] += 1
            if guided:
                track_stats[track_label]["guided"] += 1

        results.append(
            {
                "id": paper.get("id"),
                "forum": paper.get("forum"),
                "title": paper.get("title", ""),
                "track": paper.get("track", ""),
                "venue": paper.get("venue", ""),
                "venueid": paper.get("venueid", ""),
                "guided": guided,
                "matched_skills": matched_insights,
                "output_tokens": total_tokens,
            }
        )

        time.sleep(max(sleep_duration, 0))

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print overall and per-track statistics
    total = len(papers)
    total_guided = sum(track_stats[t]["guided"] for t in track_stats)

    print(f"\n{'='*80}")
    print(f"PHASE 2 RESULTS: ICLR {year} Paper Guidance Analysis")
    print(f"{'='*80}")

    print(f"\nSkill set size: {len(skills_list)}")
    print(f"Unique skills matched: {len(all_matched_skills)}")
    print(
        f"\nOverall: {total_guided}/{total} papers guided ({total_guided/total*100:.1f}%)\n"
    )

    for track in ["oral", "spotlight", "poster"]:
        stats = track_stats[track]
        if stats["total"] > 0:
            pct = stats["guided"] / stats["total"] * 100
            print(
                f"  {track.capitalize():10s}: {stats['guided']}/{stats['total']} guided ({pct:.1f}%)"
            )

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
