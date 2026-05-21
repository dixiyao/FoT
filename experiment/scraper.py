"""
Scraper for ICLR papers from OpenReview and arXiv papers.
- Supports ICLR notable lists (Top 5%, Top 25%, Poster) and accepted tracks (Oral, Spotlight, Poster).
- Year can be specified; defaults to 2023 for notable lists or 2024 when using accept flags.
- Supports arXiv paper downloads by subject area (physics, chemistry, math, cs).
Downloads PDFs and metadata for papers from the OpenReview website or arXiv.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


class OpenReviewScraper:
    def __init__(self, output_dir="data/papers/iclr23_top5", year=2023):
        self.output_dir = output_dir
        self.year = year
        self.base_url = "https://openreview.net"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def _search_papers_by_title(self, title_filter):
        """
        Search for papers by title keyword using OpenReview API.
        This method searches across ALL papers by fetching all submissions and filtering client-side.
        This ensures we get all matching papers, not just the first 1000.

        Args:
            title_filter (str): Keyword to search for in paper titles (case-insensitive).

        Returns:
            list: List of papers matching the title filter.
        """
        papers = []
        api_url = "https://api.openreview.net/notes"

        # Search across all ICLR submissions for the configured year
        # Use pagination to get all results
        offset = 0
        limit = 1000
        total_searched = 0
        total_found = 0

        print(
            f"Searching for papers with '{title_filter}' in title (case-insensitive)..."
        )
        print(
            f"This may take a while as we search through all ICLR {self.year} submissions..."
        )

        while True:
            try:
                params = {
                    "invitation": f"ICLR.cc/{self.year}/Conference/-/Blind_Submission",
                    "details": "replyCount,invitation,original",
                    "offset": offset,
                    "limit": limit,
                    "sort": "number:asc",
                }

                response = self.session.get(api_url, params=params, timeout=60)
                response.raise_for_status()

                if response.status_code == 200:
                    data = response.json()
                    notes = data.get("notes", [])

                    if not notes:
                        break  # No more papers

                    # Filter by title (case-insensitive)
                    batch_found = 0
                    for note in notes:
                        content = note.get("content", {})
                        title = content.get("title", "")

                        # Case-insensitive search - check if keyword appears anywhere in title
                        if title and title_filter.lower() in title.lower():
                            papers.append(note)
                            total_found += 1
                            batch_found += 1

                    total_searched += len(notes)

                    # Progress update
                    if batch_found > 0:
                        print(
                            f"  Searched {total_searched} papers, found {total_found} matching (latest batch: {batch_found})..."
                        )
                    elif total_searched % 5000 == 0:
                        print(
                            f"  Searched {total_searched} papers, found {total_found} matching so far..."
                        )

                    # Check if we got fewer results than limit (last page)
                    if len(notes) < limit:
                        break

                    offset += limit
                    time.sleep(0.5)  # Rate limiting between API calls
                else:
                    print(
                        f"API returned status {response.status_code}, stopping pagination"
                    )
                    break

            except requests.exceptions.RequestException as e:
                print(f"Error during search pagination: {e}")
                print(f"  Found {total_found} papers so far before error")
                break
            except Exception as e:
                print(f"Unexpected error during search: {e}")
                print(f"  Found {total_found} papers so far before error")
                break

        print(
            f"\nSearch complete: Found {len(papers)} papers matching '{title_filter}' in title (searched {total_searched} total papers)"
        )
        return papers

    def _fetch_accept_tracks_via_api(
        self,
        accept_oral=False,
        accept_spotlight=False,
        accept_poster=False,
    ):
        """Fetch accepted papers (oral/spotlight/poster) via OpenReview API by venue label."""
        api_url = "https://api.openreview.net/notes"
        offset = 0
        limit = 1000
        max_papers_to_fetch = 50000
        papers = []

        try:
            print(
                "Fetching accepted papers via API (filtering by venue: Oral/Spotlight/Poster)..."
            )
            while offset < max_papers_to_fetch:
                params = {
                    "invitation": f"ICLR.cc/{self.year}/Conference/-/Blind_Submission",
                    "details": "replyCount,invitation,original",
                    "offset": offset,
                    "limit": limit,
                    "sort": "number:asc",
                }

                response = self.session.get(api_url, params=params, timeout=60)
                if response.status_code != 200:
                    print(
                        f"  API returned status {response.status_code}; stopping accept-track API fetch"
                    )
                    break

                data = response.json()
                notes = data.get("notes", [])
                if not notes:
                    break

                batch_found = 0
                for note in notes:
                    content = note.get("content", {})
                    venue = content.get("venue") or content.get("venueid") or ""
                    venue_lower = venue.lower()

                    include = False
                    if accept_oral and "oral" in venue_lower:
                        include = True
                    elif accept_spotlight and "spotlight" in venue_lower:
                        include = True
                    elif accept_poster and "poster" in venue_lower:
                        include = True

                    if include:
                        papers.append(note)
                        batch_found += 1

                if batch_found > 0:
                    print(
                        f"  Offset {offset}: found {batch_found} accepted papers (running total {len(papers)})"
                    )

                if len(notes) < limit:
                    break

                offset += limit
                time.sleep(1.0)

        except Exception as e:
            print(f"Accept-track API fetch failed: {e}")

        return papers

    def _fetch_accept_tracks_via_api2_submissions(
        self,
        accept_oral=False,
        accept_spotlight=False,
        accept_poster=False,
    ):
        """Fetch accepted papers from api2 Submission notes by venue value."""
        api_url = "https://api2.openreview.net/notes"
        offset = 0
        limit = 200
        max_papers_to_fetch = 50000
        papers = []
        max_retries = 5

        try:
            print(
                "Fetching accepted papers via api2 Submission notes (venue contains Oral/Spotlight/Poster)..."
            )
            while offset < max_papers_to_fetch:
                params = {
                    "invitation": f"ICLR.cc/{self.year}/Conference/-/Submission",
                    "offset": offset,
                    "limit": limit,
                    "details": "replyCount",
                    "sort": "number:asc",
                }

                retry = 0
                resp = None
                while retry <= max_retries:
                    resp = self.session.get(api_url, params=params, timeout=60)
                    if resp.status_code == 429:
                        retry += 1
                        wait = 5 * (2 ** (retry - 1))
                        print(
                            f"  api2 Submission 429. Waiting {wait}s (retry {retry}/{max_retries})..."
                        )
                        time.sleep(wait)
                        continue
                    break

                if resp is None or resp.status_code != 200:
                    status = resp.status_code if resp is not None else "n/a"
                    print(f"  api2 Submission returned status {status}; stopping")
                    break

                data = resp.json()
                notes = data.get("notes", [])
                if not notes:
                    break

                batch_found = 0
                for note in notes:
                    content = note.get("content", {})
                    venue_obj = content.get("venue", {})
                    venue_val = ""
                    if isinstance(venue_obj, dict):
                        venue_val = str(venue_obj.get("value", ""))
                    elif isinstance(venue_obj, str):
                        venue_val = venue_obj
                    venue_lower = venue_val.lower()

                    include = False
                    track = None
                    if accept_oral and "oral" in venue_lower:
                        include = True
                        track = "oral"
                    elif accept_spotlight and "spotlight" in venue_lower:
                        include = True
                        track = "spotlight"
                    elif accept_poster and "poster" in venue_lower:
                        include = True
                        track = "poster"

                    if include:
                        note = dict(note)
                        note["track"] = track
                        papers.append(note)
                        batch_found += 1

                if batch_found > 0:
                    print(
                        f"  Offset {offset}: found {batch_found} accepted papers (running total {len(papers)})"
                    )

                if len(notes) < limit:
                    break
                offset += limit
                time.sleep(1.0)

        except Exception as e:
            print(f"api2 Submission fetch failed: {e}")

        return papers

    def _fetch_accept_tracks_via_decisions(
        self,
        accept_oral=False,
        accept_spotlight=False,
        accept_poster=False,
    ):
        """Fetch accepted papers using Decision notes (Accept (Oral/Spotlight/Poster))."""
        decision_invitation = f"ICLR.cc/{self.year}/Conference/-/Decision"
        api_url = "https://api.openreview.net/notes"
        offset = 0
        limit = 200
        max_papers_to_fetch = 50000
        papers = []
        max_retries = 5

        decision_strings = []
        if accept_oral:
            decision_strings.append("accept (oral)")
        if accept_spotlight:
            decision_strings.append("accept (spotlight)")
        if accept_poster:
            decision_strings.append("accept (poster)")

        try:
            print(
                "Fetching accepted papers via Decision notes (Accept Oral/Spotlight/Poster)..."
            )
            while offset < max_papers_to_fetch:
                params = {
                    "invitation": decision_invitation,
                    "offset": offset,
                    "limit": limit,
                    "sort": "number:asc",
                }

                retry = 0
                resp = None
                while retry <= max_retries:
                    resp = self.session.get(api_url, params=params, timeout=60)
                    if resp.status_code == 429:
                        retry += 1
                        wait = 5 * (2 ** (retry - 1))
                        print(
                            f"  Decision API rate limited (429). Waiting {wait}s (retry {retry}/{max_retries})..."
                        )
                        time.sleep(wait)
                        continue
                    break

                if resp is None or resp.status_code != 200:
                    status = resp.status_code if resp is not None else "n/a"
                    print(f"  Decision API returned status {status}; stopping")
                    break

                data = resp.json()
                notes = data.get("notes", [])
                if not notes:
                    break

                batch_forums = []
                for note in notes:
                    decision_text = (
                        note.get("content", {}).get("decision") or ""
                    ).lower()
                    if any(tag in decision_text for tag in decision_strings):
                        forum_id = note.get("forum") or note.get("id")
                        if forum_id:
                            batch_forums.append(forum_id)

                if batch_forums:
                    # Fetch submission notes for these forums
                    for forum_id in batch_forums:
                        try:
                            sub_retry = 0
                            sub_resp = None
                            while sub_retry <= max_retries:
                                sub_resp = self.session.get(
                                    api_url,
                                    params={
                                        "id": forum_id,
                                        "details": "replyCount,invitation,original",
                                    },
                                    timeout=60,
                                )
                                if sub_resp.status_code == 429:
                                    sub_retry += 1
                                    wait = 5 * (2 ** (sub_retry - 1))
                                    print(
                                        f"    Submission fetch 429. Waiting {wait}s (retry {sub_retry}/{max_retries})..."
                                    )
                                    time.sleep(wait)
                                    continue
                                break

                            if sub_resp is None or sub_resp.status_code != 200:
                                continue
                            sub_data = sub_resp.json()
                            sub_notes = sub_data.get("notes", [])
                            if sub_notes:
                                papers.append(sub_notes[0])
                        except Exception:
                            continue

                if len(notes) < limit:
                    break
                offset += limit
                time.sleep(1.0)

        except Exception as e:
            print(f"Decision-based accept fetch failed: {e}")

        return papers

    def get_paper_list(
        self,
        title_filter=None,
        top5=False,
        top25=False,
        poster=False,
        accept_oral=False,
        accept_spotlight=False,
        accept_poster=False,
    ):
        """
        Fetch the list of papers from ICLR notable lists or accepted tracks for the configured year

        Args:
            title_filter (str, optional): Filter papers by keyword in title (case-insensitive).
                                         If provided, searches across ALL papers, not just notable ones.
            top5 (bool): If True, only get notable top 5% papers
            top25 (bool): If True, only get notable top 25% papers
            poster (bool): If True, only get poster papers
            accept_oral (bool): If True, scrape Accept (Oral) tab
            accept_spotlight (bool): If True, scrape Accept (Spotlight) tab
            accept_poster (bool): If True, scrape Accept (Poster) tab
        """
        # If using accepted tabs, go straight to web scraping (API invitation differs)
        if accept_oral or accept_spotlight or accept_poster:
            # Try API via Decision notes first (most reliable)
            papers_api = self._fetch_accept_tracks_via_decisions(
                accept_oral=accept_oral,
                accept_spotlight=accept_spotlight,
                accept_poster=accept_poster,
            )
            if papers_api:
                return papers_api

            # Try api2 Submission notes (venue value)
            papers_api = self._fetch_accept_tracks_via_api2_submissions(
                accept_oral=accept_oral,
                accept_spotlight=accept_spotlight,
                accept_poster=accept_poster,
            )
            if papers_api:
                return papers_api

            # Fallback: filter Blind_Submission by venue labels
            papers_api = self._fetch_accept_tracks_via_api(
                accept_oral=accept_oral,
                accept_spotlight=accept_spotlight,
                accept_poster=accept_poster,
            )
            if papers_api:
                return papers_api
            return self._scrape_web_page(
                title_filter=title_filter,
                top5=top5,
                top25=top25,
                poster=poster,
                accept_oral=accept_oral,
                accept_spotlight=accept_spotlight,
                accept_poster=accept_poster,
            )
        # If title filter is provided, use search API to get ALL matching papers
        if title_filter:
            papers = self._search_papers_by_title(title_filter)
            if papers:
                return papers
            # If search API doesn't work, fall back to web scraping
            print("Search API didn't return results, trying web scraping...")
            return self._scrape_web_page(
                title_filter=title_filter,
                top5=top5,
                top25=top25,
                poster=poster,
                accept_oral=accept_oral,
                accept_spotlight=accept_spotlight,
                accept_poster=accept_poster,
            )

        # If no title filter, use the original method
        # Try to get papers via API endpoint with pagination
        api_url = "https://api.openreview.net/notes"
        papers = []

        # Try with smaller limit first to avoid 400 errors
        offset = 0
        limit = 1000  # Reduced from 50000 to avoid API errors
        max_papers_to_fetch = 50000  # Maximum total papers to fetch

        try:
            print("Fetching papers from OpenReview API (this may take a while)...")
            while offset < max_papers_to_fetch:
                params = {
                    "invitation": f"ICLR.cc/{self.year}/Conference/-/Blind_Submission",
                    "details": "replyCount,invitation,original",
                    "offset": offset,
                    "limit": limit,
                    "sort": "number:asc",
                }

                response = self.session.get(api_url, params=params, timeout=60)

                if response.status_code != 200:
                    print(
                        f"API returned status {response.status_code}, trying web scraping..."
                    )
                    break

                data = response.json()
                notes = data.get("notes", [])

                if not notes:
                    break  # No more papers

                # Filter for notable papers based on flags
                batch_count = 0
                for note in notes:
                    content = note.get("content", {})
                    venue = content.get("venue", "")
                    title = content.get("title", "")

                    # Check venue based on flags
                    should_include = False
                    if top5 and (
                        "Notable Top 5%" in venue
                        or ("Notable" in venue and "Top 5%" in venue)
                    ):
                        should_include = True
                    elif top25 and (
                        "Notable Top 25%" in venue
                        or ("Notable" in venue and "Top 25%" in venue)
                        or "notable top 25%" in venue.lower()
                    ):
                        should_include = True
                    elif poster and ("Poster" in venue or f"ICLR {self.year}" in venue):
                        should_include = True
                    elif not top5 and not top25 and not poster:
                        # Default: get notable top 5%
                        if "Notable Top 5%" in venue or (
                            "Notable" in venue and "Top 5%" in venue
                        ):
                            should_include = True

                    if should_include:
                        papers.append(note)
                        batch_count += 1

                if batch_count > 0:
                    print(
                        f"  Fetched {len(notes)} papers, found {batch_count} notable papers (total: {len(papers)})"
                    )

                if len(notes) < limit:
                    break  # Last batch

                offset += limit
                time.sleep(0.5)  # Rate limiting

            if papers:
                print(f"Found {len(papers)} papers via API")
        except requests.exceptions.RequestException as e:
            print(f"API method failed: {e}")
            print("Trying web scraping method...")
            papers = self._scrape_web_page(
                title_filter=title_filter,
                top5=top5,
                top25=top25,
                poster=poster,
                accept_oral=accept_oral,
                accept_spotlight=accept_spotlight,
                accept_poster=accept_poster,
            )
        except Exception as e:
            print(f"Unexpected error in API call: {e}")
            print("Trying web scraping method...")
            papers = self._scrape_web_page(
                title_filter=title_filter,
                top5=top5,
                top25=top25,
                poster=poster,
                accept_oral=accept_oral,
                accept_spotlight=accept_spotlight,
                accept_poster=accept_poster,
            )

        return papers

    def _scrape_web_page(
        self,
        title_filter=None,
        top5=False,
        top25=False,
        poster=False,
        accept_oral=False,
        accept_spotlight=False,
        accept_poster=False,
    ):
        """
        Fallback method: scrape papers from multiple web pages

        Args:
            title_filter (str, optional): Filter papers by keyword in title (case-insensitive).
            top5 (bool): If True, scrape from notable top 5% papers
            top25 (bool): If True, scrape from notable top 25% papers
            poster (bool): If True, scrape from poster papers
            accept_oral (bool): If True, scrape Accept (Oral)
            accept_spotlight (bool): If True, scrape Accept (Spotlight)
            accept_poster (bool): If True, scrape Accept (Poster)
        """
        # Build list of URLs based on flags and configured year
        # If no flags are set, use all notable URLs (backward compatibility)
        urls = []
        if accept_oral or accept_spotlight or accept_poster:
            if accept_oral:
                urls.append(
                    (
                        f"ICLR {self.year} Accept (Oral)",
                        f"https://openreview.net/group?id=ICLR.cc/{self.year}/Conference#tab-accept-oral",
                    )
                )
            if accept_spotlight:
                urls.append(
                    (
                        f"ICLR {self.year} Accept (Spotlight)",
                        f"https://openreview.net/group?id=ICLR.cc/{self.year}/Conference#tab-accept-spotlight",
                    )
                )
            if accept_poster:
                urls.append(
                    (
                        f"ICLR {self.year} Accept (Poster)",
                        f"https://openreview.net/group?id=ICLR.cc/{self.year}/Conference#tab-accept-poster",
                    )
                )
        else:
            if top5 or (not top5 and not top25 and not poster):
                urls.append(
                    (
                        "Notable Top 5%",
                        f"https://openreview.net/group?id=ICLR.cc/{self.year}/Conference#notable-top-5-",
                    )
                )
            if top25 or (not top5 and not top25 and not poster):
                urls.append(
                    (
                        "Notable Top 25%",
                        f"https://openreview.net/group?id=ICLR.cc%2F{self.year}%2FConference#notable-top-25-",
                    )
                )
            if poster or (not top5 and not top25 and not poster):
                urls.append(
                    (
                        "Poster",
                        f"https://openreview.net/group?id=ICLR.cc%2F{self.year}%2FConference#poster",
                    )
                )

        all_papers = []
        seen_paper_ids = set()  # Track seen papers to avoid duplicates

        for url_name, url in urls:
            try:
                print(f"Scraping from {url_name}: {url}")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")

                # Try multiple selectors to find paper links
                # OpenReview uses various structures, so we'll try several
                paper_links = []

                # Method 1: Look for links with /forum?id=
                links1 = soup.find_all("a", href=re.compile(r"/forum\?id="))
                paper_links.extend(links1)

                # Method 2: Look for links with href containing paper IDs (forum pattern)
                links2 = soup.find_all("a", href=re.compile(r"forum.*id="))
                paper_links.extend(links2)

                # Method 3: Look for data attributes or IDs that might contain paper IDs
                # Some pages use data-note-id or similar attributes
                links3 = soup.find_all(attrs={"data-note-id": True})
                for elem in links3:
                    paper_id = elem.get("data-note-id")
                    if paper_id:
                        # Create a pseudo-link object
                        class PseudoLink:
                            def __init__(self, paper_id, title):
                                self.paper_id = paper_id
                                self.title = title

                            def get(self, key, default=None):
                                if key == "href":
                                    return f"/forum?id={self.paper_id}"
                                return default

                            def get_text(self, strip=False):
                                return self.title

                        title = elem.get_text(strip=True) or f"Paper_{paper_id}"
                        paper_links.append(PseudoLink(paper_id, title))

                # Method 4: Look for script tags that might contain JSON data with paper info
                scripts = soup.find_all("script")
                for script in scripts:
                    script_text = script.string
                    if script_text and "forum" in script_text and "id" in script_text:
                        # Try to extract paper IDs from JavaScript/JSON in script tags
                        # Look for patterns like "id": "..." or id="..."
                        ids = re.findall(
                            r'["\']id["\']\s*:\s*["\']([^"\']+)["\']', script_text
                        )
                        for paper_id in ids:
                            if (
                                len(paper_id) > 5
                            ):  # Filter out short IDs that are likely not paper IDs

                                class PseudoLink:
                                    def __init__(self, paper_id):
                                        self.paper_id = paper_id

                                    def get(self, key, default=None):
                                        if key == "href":
                                            return f"/forum?id={self.paper_id}"
                                        return default

                                    def get_text(self, strip=False):
                                        return f"Paper_{self.paper_id}"

                                paper_links.append(PseudoLink(paper_id))

                # Deduplicate paper_links by href
                seen_hrefs = set()
                unique_links = []
                for link in paper_links:
                    href = link.get("href", "")
                    if href and href not in seen_hrefs:
                        seen_hrefs.add(href)
                        unique_links.append(link)
                paper_links = unique_links

                for link in paper_links:
                    href = link.get("href", "")
                    paper_id = None

                    # Extract paper ID from href
                    if "id=" in href:
                        paper_id = href.split("id=")[-1].split("&")[0].split("#")[0]
                    elif hasattr(link, "paper_id"):
                        paper_id = link.paper_id

                    if paper_id and paper_id not in seen_paper_ids:
                        title = (
                            link.get_text(strip=True)
                            if hasattr(link, "get_text")
                            else (getattr(link, "title", None) or f"Paper_{paper_id}")
                        )
                        # Apply title filter if specified
                        if title_filter:
                            if title_filter.lower() in title.lower():
                                all_papers.append(
                                    {
                                        "id": paper_id,
                                        "title": title,
                                        "url": urljoin(
                                            self.base_url,
                                            (
                                                href
                                                if href.startswith("/")
                                                else f"/{href}"
                                            ),
                                        ),
                                    }
                                )
                                seen_paper_ids.add(paper_id)
                        else:
                            all_papers.append(
                                {
                                    "id": paper_id,
                                    "title": title,
                                    "url": urljoin(
                                        self.base_url,
                                        href if href.startswith("/") else f"/{href}",
                                    ),
                                }
                            )
                            seen_paper_ids.add(paper_id)

                print(f"  Found {len(paper_links)} papers from this page")
                time.sleep(1)  # Rate limiting between pages

            except Exception as e:
                print(f"  Warning: Failed to scrape {url_name} ({url}): {e}")
                continue

        print(f"Total unique papers found: {len(all_papers)}")
        return all_papers

    def download_paper(self, paper_info):
        """Download a paper PDF given paper information"""
        # Try to get paper ID from various possible fields
        # The forum ID is what's used in OpenReview URLs
        paper_id = (
            paper_info.get("forum") or paper_info.get("id") or paper_info.get("number")
        )
        if not paper_id:
            print(
                f"Error: No paper ID found for paper: {paper_info.get('title', 'Unknown')}"
            )
            return None

        def _extract_title(paper):
            content_title = paper.get("content", {}).get("title")
            if isinstance(content_title, dict):
                content_title = content_title.get("value")
            if content_title:
                return str(content_title)
            fallback = paper.get("title") or f"paper_{paper_id}"
            return str(fallback)

        title = _extract_title(paper_info)

        # Clean title for filename
        safe_title = re.sub(r"[^\w\s-]", "", title)[:100]
        safe_title = re.sub(r"[-\s]+", "-", safe_title)

        # Use the standard OpenReview PDF URL format: https://openreview.net/pdf?id={paper_id}
        # This is the most reliable method as shown in the example: https://openreview.net/pdf?id=4-k7kUavAj
        pdf_url = f"https://openreview.net/pdf?id={paper_id}"

        # Download PDF
        pdf_path = os.path.join(self.output_dir, f"{safe_title}_{paper_id}.pdf")

        try:
            response = self.session.get(pdf_url, timeout=60, stream=True)
            response.raise_for_status()

            # Check if response is actually a PDF
            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower() and not content_type.startswith(
                "application/octet-stream"
            ):
                # Try alternative: check if we got HTML (error page) instead of PDF
                if response.headers.get("content-type", "").startswith("text/html"):
                    print(
                        f"Warning: Received HTML instead of PDF for {title}. The paper might not be publicly available."
                    )
                    return None

            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verify file was written and has content
            if os.path.getsize(pdf_path) > 0:
                print(f"Downloaded: {title}")
                return pdf_path
            else:
                print(f"Error: Downloaded file is empty for {title}")
                os.remove(pdf_path)
                return None

        except requests.exceptions.HTTPError as e:
            print(f"Failed to download {title}: HTTP {e.response.status_code} - {e}")
            return None
        except Exception as e:
            print(f"Error downloading {title}: {e}")
            return None

    def save_metadata(self, papers):
        """Save paper metadata to JSON file"""
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata to {metadata_path}")

    def _query_arxiv_earliest_date(self, title):
        """Lookup earliest arXiv posting date by paper title.

        Uses arXiv's Atom API, sorting by submitted date ascending and taking the
        first result. Returns an ISO timestamp string or None if not found.
        """
        base_url = "https://export.arxiv.org/api/query"
        query = f'ti:"{title}"'
        params = {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
            "start": 0,
            "max_results": 1,
        }

        try:
            response = self.session.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            xml_text = response.text

            # Minimal XML parsing to avoid extra dependencies
            # arXiv Atom uses the Atom namespace; we only need the first published tag
            published_match = re.search(r"<published>([^<]+)</published>", xml_text)
            if not published_match:
                return None

            published_str = published_match.group(1)
            try:
                # Normalize to ISO date string (YYYY-MM-DD)
                dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                return dt.date().isoformat()
            except ValueError:
                return published_str
        except Exception as e:
            print(f"  arXiv lookup failed for '{title}': {e}")
            return None

    def check_arxiv_dates(self, papers, delay_seconds=1.0):
        """For each paper, attempt to find earliest arXiv posted date by title."""
        results = []
        for idx, paper in enumerate(papers, 1):
            title = paper.get("content", {}).get("title") or paper.get("title") or ""
            if not title:
                results.append({"title": None, "arxiv_first_posted": None})
                continue

            print(f"[{idx}/{len(papers)}] Checking arXiv for: {title}")
            date_str = self._query_arxiv_earliest_date(title)
            results.append(
                {
                    "title": title,
                    "arxiv_first_posted": date_str,
                }
            )
            time.sleep(delay_seconds)  # be polite with arXiv API

        output_path = os.path.join(self.output_dir, "arxiv_posted_dates.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved arXiv earliest dates to {output_path}")
        return results

    def scrape_all(
        self,
        max_papers=None,
        title_filter=None,
        top5=False,
        top25=False,
        poster=False,
        accept_oral=False,
        accept_spotlight=False,
        accept_poster=False,
        check_arxiv=False,
    ):
        """
        Main method to scrape papers from OpenReview

        Args:
            max_papers (int, optional): Maximum number of papers to scrape.
                                       If None, scrapes all available papers.
            title_filter (str, optional): Filter papers by keyword in title (case-insensitive).
                                         If provided, only papers with this keyword in title will be included.
            top5 (bool): If True, scrape from notable top 5% papers
            top25 (bool): If True, scrape from notable top 25% papers
            poster (bool): If True, scrape from poster papers
            accept_oral (bool): If True, scrape Accept (Oral)
            accept_spotlight (bool): If True, scrape Accept (Spotlight)
            accept_poster (bool): If True, scrape Accept (Poster)
        """
        print("Fetching paper list...")
        if title_filter:
            print(
                f"Searching for papers with '{title_filter}' in title (case-insensitive)..."
            )

        # Try API first, then fall back to web scraping with specified flags
        papers = self.get_paper_list(
            title_filter=title_filter,
            top5=top5,
            top25=top25,
            poster=poster,
            accept_oral=accept_oral,
            accept_spotlight=accept_spotlight,
            accept_poster=accept_poster,
        )

        # If API doesn't work or returns no papers, try web scraping
        if not papers:
            print("No papers from API, trying web scraping...")
            papers = self._scrape_web_page(
                title_filter=title_filter,
                top5=top5,
                top25=top25,
                poster=poster,
                accept_oral=accept_oral,
                accept_spotlight=accept_spotlight,
                accept_poster=accept_poster,
            )

        if not papers:
            print("No papers found. Please check the URL or API access.")
            return

        total_papers = len(papers)
        print(f"Found {total_papers} papers")

        # Save metadata (save all found papers before limiting)
        self.save_metadata(papers)

        # Limit papers if max_papers is specified
        if max_papers is not None and max_papers > 0:
            papers = papers[:max_papers]
            print(f"Limiting to {len(papers)} papers (requested: {max_papers})")

        # Download PDFs
        print(f"\nDownloading {len(papers)} papers...")
        downloaded = 0
        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] Processing paper...")
            result = self.download_paper(paper)
            if result:
                downloaded += 1
            time.sleep(1)  # Be respectful with rate limiting

        print(
            f"\nCompleted! Downloaded {downloaded}/{len(papers)} papers to {self.output_dir}"
        )

        if check_arxiv:
            print("\nChecking arXiv earliest posted dates by title...")
            self.check_arxiv_dates(papers)


class ArxivScraper:
    """Scraper for arXiv papers by subject category."""

    # arXiv subject category mappings
    CATEGORIES = {
        "physics": "physics.*",
        "math": "math.*",
        "cs": "cs.*",
        "chemistry": "physics.chem-ph",  # Chemistry papers are in physics category
    }

    def __init__(self, output_dir="data/papers/arxiv", subject="physics"):
        """
        Initialize arXiv scraper.

        Args:
            output_dir: Directory to save downloaded papers
            subject: Subject area (physics, math, cs, chemistry)
        """
        self.output_dir = output_dir
        self.subject = subject.lower()
        self.category = self.CATEGORIES.get(self.subject, "physics.*")
        self.base_url = "http://export.arxiv.org/api/query"
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; ArxivScraper/1.0)"}
        )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def fetch_latest_papers(self, max_results=100):
        """
        Fetch latest papers from arXiv in the specified subject category.

        Args:
            max_results: Maximum number of papers to fetch

        Returns:
            List of paper metadata dictionaries
        """
        print(f"Fetching {max_results} latest {self.subject} papers from arXiv...")
        print(f"Using category: {self.category}")

        # Build query parameters
        params = {
            "search_query": f"cat:{self.category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": 0,
            "max_results": max_results,
        }

        try:
            response = self.session.get(self.base_url, params=params, timeout=60)
            response.raise_for_status()

            # Parse Atom XML response
            papers = self._parse_arxiv_response(response.text)
            print(f"Successfully fetched {len(papers)} papers")
            return papers

        except requests.exceptions.RequestException as e:
            print(f"Error fetching papers from arXiv: {e}")
            return []

    def _parse_arxiv_response(self, xml_text):
        """
        Parse arXiv API XML response.

        Args:
            xml_text: XML response from arXiv API

        Returns:
            List of paper dictionaries
        """
        papers = []

        # Use regex to parse XML (simple approach to avoid extra dependencies)
        # arXiv API returns Atom feed with <entry> elements for each paper
        entries = re.findall(r"<entry>(.*?)</entry>", xml_text, re.DOTALL)

        for entry in entries:
            try:
                # Extract fields
                title_match = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
                summary_match = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
                published_match = re.search(r"<published>(.*?)</published>", entry)
                updated_match = re.search(r"<updated>(.*?)</updated>", entry)
                id_match = re.search(r"<id>(.*?)</id>", entry)

                # Extract authors
                authors = []
                author_entries = re.findall(r"<author>(.*?)</author>", entry, re.DOTALL)
                for author in author_entries:
                    name_match = re.search(r"<name>(.*?)</name>", author)
                    if name_match:
                        authors.append(name_match.group(1).strip())

                # Extract PDF link
                pdf_match = re.search(
                    r'<link.*?title="pdf".*?href="(.*?)"', entry, re.DOTALL
                )
                if not pdf_match:
                    # Alternative pattern
                    pdf_match = re.search(
                        r'<link.*?href="(.*?\.pdf)"', entry, re.DOTALL
                    )

                # Extract arXiv ID from the entry ID
                arxiv_id = None
                if id_match:
                    id_url = id_match.group(1)
                    arxiv_id = id_url.split("/")[-1]

                paper = {
                    "id": arxiv_id,
                    "title": (
                        title_match.group(1).strip().replace("\n", " ")
                        if title_match
                        else "Unknown"
                    ),
                    "summary": (
                        summary_match.group(1).strip().replace("\n", " ")
                        if summary_match
                        else ""
                    ),
                    "authors": authors,
                    "published": published_match.group(1) if published_match else "",
                    "updated": updated_match.group(1) if updated_match else "",
                    "pdf_url": pdf_match.group(1) if pdf_match else None,
                    "entry_url": id_match.group(1) if id_match else "",
                }

                # If no PDF link found in link elements, construct it from arXiv ID
                if not paper["pdf_url"] and arxiv_id:
                    paper["pdf_url"] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

                papers.append(paper)

            except Exception as e:
                print(f"Warning: Failed to parse entry: {e}")
                continue

        return papers

    def download_paper(self, paper):
        """
        Download a single paper PDF.

        Args:
            paper: Paper metadata dictionary

        Returns:
            Path to downloaded PDF or None if failed
        """
        if not paper.get("pdf_url"):
            print(f"No PDF URL for paper: {paper.get('title', 'Unknown')}")
            return None

        title = paper.get("title", "Unknown")
        arxiv_id = paper.get("id", "unknown")

        # Clean title for filename
        safe_title = re.sub(r"[^\w\s-]", "", title)[:100]
        safe_title = re.sub(r"[-\s]+", "-", safe_title)

        pdf_path = os.path.join(self.output_dir, f"{safe_title}_{arxiv_id}.pdf")

        try:
            response = self.session.get(paper["pdf_url"], timeout=60, stream=True)
            response.raise_for_status()

            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            if os.path.getsize(pdf_path) > 0:
                print(f"Downloaded: {title[:80]}...")
                return pdf_path
            else:
                print(f"Error: Empty file for {title}")
                os.remove(pdf_path)
                return None

        except Exception as e:
            print(f"Error downloading {title}: {e}")
            return None

    def save_metadata(self, papers):
        """Save paper metadata to JSON file."""
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata to {metadata_path}")

    def scrape_all(self, max_papers=100):
        """
        Main method to fetch and download arXiv papers.

        Args:
            max_papers: Maximum number of papers to download
        """
        # Fetch paper metadata
        papers = self.fetch_latest_papers(max_results=max_papers)

        if not papers:
            print("No papers found.")
            return

        # Save metadata
        self.save_metadata(papers)

        # Download PDFs
        print(f"\nDownloading {len(papers)} papers...")
        downloaded = 0
        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] Processing paper...")
            result = self.download_paper(paper)
            if result:
                downloaded += 1
            time.sleep(3)  # Be respectful with arXiv API rate limiting (3 seconds recommended)

        print(
            f"\nCompleted! Downloaded {downloaded}/{len(papers)} papers to {self.output_dir}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape papers from OpenReview (ICLR) or arXiv"
    )
    parser.add_argument(
        "-n",
        "--num-papers",
        type=int,
        default=None,
        help="Number of papers to scrape (default: all for ICLR, 100 for arXiv)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for downloaded papers (default: data/papers/iclr23_top5 for ICLR, data/papers/arxiv_<subject> for arXiv)",
    )

    # arXiv-specific arguments
    arxiv_group = parser.add_argument_group("arXiv options")
    arxiv_group.add_argument(
        "--arxiv-subject",
        type=str,
        choices=["physics", "math", "cs", "chemistry"],
        default=None,
        help="Download latest papers from arXiv by subject (physics, math, cs, chemistry). Example: --arxiv-subject physics -n 100",
    )

    # ICLR-specific arguments
    iclr_group = parser.add_argument_group("ICLR/OpenReview options")
    iclr_group.add_argument(
        "-f",
        "--filter",
        type=str,
        default=None,
        help="Filter papers by keyword in title (case-insensitive). Example: -f 'diffusion'",
    )
    iclr_group.add_argument(
        "--year",
        type=int,
        default=None,
        help="ICLR conference year (default: 2023 for notable lists or 2024 when using accept flags)",
    )
    iclr_group.add_argument(
        "--top5",
        action="store_true",
        help="Scrape from notable top 5% papers",
    )
    iclr_group.add_argument(
        "--top25",
        action="store_true",
        help="Scrape from notable top 25% papers",
    )
    iclr_group.add_argument(
        "--poster",
        action="store_true",
        help="Scrape from poster papers",
    )
    iclr_group.add_argument(
        "--accept-oral",
        action="store_true",
        help="Scrape ICLR Accept (Oral) for the selected year",
    )
    iclr_group.add_argument(
        "--accept-spotlight",
        action="store_true",
        help="Scrape ICLR Accept (Spotlight) for the selected year",
    )
    iclr_group.add_argument(
        "--accept-poster",
        action="store_true",
        help="Scrape ICLR Accept (Poster) for the selected year",
    )
    iclr_group.add_argument(
        "--check-arxiv",
        action="store_true",
        help="After scraping ICLR papers, query arXiv by title and record earliest posted dates",
    )

    args = parser.parse_args()

    # Determine mode: arXiv or OpenReview
    if args.arxiv_subject:
        # arXiv mode
        default_output = f"data/papers/arxiv_{args.arxiv_subject}"
        output_dir = args.output_dir if args.output_dir else default_output
        max_papers = args.num_papers if args.num_papers else 100

        print(f"=== arXiv Scraper Mode ===")
        print(f"Subject: {args.arxiv_subject}")
        print(f"Max papers: {max_papers}")
        print(f"Output directory: {output_dir}\n")

        scraper = ArxivScraper(output_dir=output_dir, subject=args.arxiv_subject)
        scraper.scrape_all(max_papers=max_papers)

    else:
        # OpenReview/ICLR mode (default)
        accept_mode = args.accept_oral or args.accept_spotlight or args.accept_poster
        default_year = 2024 if accept_mode else 2023
        effective_year = args.year if args.year is not None else default_year
        default_output = "data/papers/iclr23_top5"
        output_dir = args.output_dir if args.output_dir else default_output

        print(f"=== OpenReview/ICLR Scraper Mode ===")
        print(f"Year: {effective_year}")
        print(f"Output directory: {output_dir}\n")

        scraper = OpenReviewScraper(output_dir=output_dir, year=effective_year)
        scraper.scrape_all(
            max_papers=args.num_papers,
            title_filter=args.filter,
            top5=args.top5,
            top25=args.top25,
            poster=args.poster,
            accept_oral=args.accept_oral,
            accept_spotlight=args.accept_spotlight,
            accept_poster=args.accept_poster,
            check_arxiv=args.check_arxiv,
        )
