"""
Check ICLR Accept (Oral) papers for insight guidance using Gemini or OpenRouter API.
- Fetches ICLR papers for a given year directly from OpenReview API.
- Uses a provided insights encyclopedia (JSON mapping of name->description or plain text) as guidance.
- Sends title + abstract + insights to the API and records which insights apply.
- Outputs a summary count (guided/total) and a JSON report with per-paper results.

Usage example:
  python guided_accept_oral_checker.py \
      --api-type gemini \
      --key $API_KEY \
      --api-model gemini-3-pro-preview \
      --encyclopedia important_checkpoints/client_aime25_server_math500/encyclopedia.json \
      --year 2024 \
      --output guided_oral_results.json

  python guided_accept_oral_checker.py \
      --api-type openrouter \
      --key $OPENROUTER_API_KEY \
      --api-model openai/gpt-4o \
      --encyclopedia important_checkpoints/client_aime25_server_math500/encyclopedia.json \
      --year 2024 \
      --output guided_oral_results.json
"""

import argparse
import glob
import json
import os
import re
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# Prefer new google.genai; fall back to deprecated google.generativeai
HAS_GENAI = False
HAS_GEMINI = False
try:
    import google.genai as genai_new  # type: ignore

    HAS_GENAI = True
except Exception:
    HAS_GENAI = False
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        import google.generativeai as genai_old  # type: ignore
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False


class GeminiClient:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        if not (HAS_GENAI or HAS_GEMINI):
            raise ImportError(
                "Install google-genai (preferred) or google-generativeai. Example: pip install google-genai"
            )
        self.model_name = model_name
        self.backend = "new" if HAS_GENAI else "old"
        if self.backend == "new":
            self.client = genai_new.Client(api_key=api_key)
        else:
            genai_old.configure(api_key=api_key)
            self.model = genai_old.GenerativeModel(model_name)

    def generate_text(self, prompt: str, max_output_tokens: int = 16384) -> Tuple[str, Dict]:
        """Generate text and return (text, token_info) tuple."""
        if self.backend == "new":
            from google.genai import types
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_output_tokens
                )
            )
            # Try primary accessor
            text = None
            output_tokens = 0
            if hasattr(resp, "text") and resp.text:
                text = resp.text.strip()
            # Fallback: attempt to stitch candidate parts
            try:
                candidates = getattr(resp, "candidates", []) or []
                parts = []
                for c in candidates:
                    content = getattr(c, "content", None)
                    if content and getattr(content, "parts", None):
                        for p in content.parts:
                            if hasattr(p, "text") and p.text:
                                parts.append(p.text)
                if parts:
                    text = "\n".join(parts).strip()
                # Try to extract token usage from usage_metadata
                if hasattr(resp, "usage_metadata"):
                    usage = resp.usage_metadata
                    output_tokens = (
                        getattr(usage, "output_token_count", 0)
                        or getattr(usage, "candidates_token_count", 0)
                        or 0
                    )
            except Exception:
                pass
            if text:
                return text, {"output_tokens": output_tokens}
            raise RuntimeError("Failed to extract text from google.genai response")
        else:
            generation_config = {"max_output_tokens": max_output_tokens}
            resp = self.model.generate_content(prompt, generation_config=generation_config)
            text = None
            output_tokens = 0
            if hasattr(resp, "text") and resp.text:
                text = resp.text.strip()
            # Try to extract token usage
            try:
                if hasattr(resp, "usage_metadata"):
                    usage = resp.usage_metadata
                    output_tokens = (
                        getattr(usage, "output_token_count", 0)
                        or getattr(usage, "candidates_token_count", 0)
                        or 0
                    )
            except Exception:
                pass
            if text:
                return text, {"output_tokens": output_tokens}
            # Fallback similar to client.py logic
            try:
                candidate = resp.candidates[0]
                if getattr(
                    candidate, "finish_reason", None
                ) == "RECITATION" and getattr(candidate, "safety_ratings", None):
                    raise RuntimeError(
                        "Gemini API blocked the response due to recitation."
                    )
                if candidate.content and candidate.content.parts:
                    parts = [
                        part.text
                        for part in candidate.content.parts
                        if hasattr(part, "text") and part.text
                    ]
                    if parts:
                        text = "\n".join(parts).strip()
                        return text, {"output_tokens": output_tokens}
            except Exception:
                pass
            raise RuntimeError(
                "Failed to extract text from google.generativeai response"
            )


class OpenRouterClient:
    def __init__(self, api_key: str, model_name: str = "openai/gpt-4o"):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://openrouter.ai/api/v1"

    def generate_text(self, prompt: str, max_output_tokens: int = 16384) -> Tuple[str, Dict]:
        """Generate text using OpenRouter API and return (text, token_info) tuple."""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/dixiyao/Federation-of-Text",
            "X-Title": "ICLR Insight Checker",
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens,
        }
        
        max_retries = 5
        base_delay = 1.0  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                response.raise_for_status()
                
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    text = result["choices"][0]["message"]["content"].strip()
                    output_tokens = result.get("usage", {}).get("completion_tokens", 0)
                    return text, {"output_tokens": output_tokens}
                else:
                    raise RuntimeError("Failed to extract text from OpenRouter response")
                    
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Too Many Requests
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"    Rate limited (429). Retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"    Rate limited (429). Max retries exceeded.")
                        raise e
                else:
                    # For other HTTP errors, don't retry
                    raise e
            except Exception as e:
                # For non-HTTP errors, don't retry
                raise e
        
        raise RuntimeError(f"Failed after {max_retries} attempts")


class LocalHFClient:
    """HuggingFace local model wrapper with the same interface as API clients."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
    ):
        from utils import check_cuda, load_hf_model

        self.model_name = model_name
        self.device = device or ("cuda" if check_cuda() else "cpu")
        self.load_in_8bit = load_in_8bit
        self.model, self.tokenizer = load_hf_model(
            self.model_name,
            self.device,
            self.load_in_8bit,
        )

    def generate_text(self, prompt: str, max_output_tokens: int = 16384) -> Tuple[str, Dict]:
        import torch
        from utils import _resolve_hf_context_limit

        system_prompt = (
            "You are a strict JSON classifier. Return exactly one JSON object "
            "and no other text. Do not explain your reasoning."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            full_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            full_prompt = f"{system_prompt}\n\nUser:\n{prompt}\n\nAssistant:\n"

        model_context_limit = _resolve_hf_context_limit(self.model, self.tokenizer)
        input_max_length = min(int(model_context_limit), 65536)
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=input_max_length,
        ).to(self.device)
        input_token_count = int(inputs["input_ids"].shape[1])
        print(
            f"Input tokens: {input_token_count}, Max new tokens: {max_output_tokens}",
            flush=True,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_output_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        output_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return text, {
            "backend": "huggingface",
            "input_tokens": input_token_count,
            "output_tokens": int(output_ids.shape[0]),
            "input_truncated": bool(input_token_count >= input_max_length),
            "input_limit": int(input_max_length),
        }


def load_insights(encyclopedia_path: str) -> Tuple[List[Tuple[str, str]], str]:
    """Load insights from encyclopedia file.

    Returns:
        A list of (name, description) tuples and a formatted string for prompting.
    """
    if not os.path.exists(encyclopedia_path):
        raise FileNotFoundError(f"Encyclopedia not found at {encyclopedia_path}")

    def _parse_insight_item(item):
        if isinstance(item, dict):
            name = (
                item.get("name")
                or item.get("insight_name")
                or item.get("skill_name")
                or item.get("title")
                or item.get("key")
                or item.get("id")
            )
            desc = (
                item.get("description")
                or item.get("desc")
                or item.get("detail")
                or item.get("text")
                or item.get("insight")
                or item.get("skill")
                or ""
            )
            if not name and isinstance(desc, str) and len(desc.strip()) > 0:
                return ("insight", desc.strip())
            if name:
                return (str(name), str(desc) if desc is not None else "")
            return None
        if isinstance(item, str):
            return ("insight", item)
        return None

    def _extract_insights_from_data(data):
        extracted = []
        if isinstance(data, dict):
            if "skills" in data and isinstance(data["skills"], list):
                for item in data["skills"]:
                    parsed = _parse_insight_item(item)
                    if parsed:
                        extracted.append(parsed)
                if extracted:
                    return extracted
            if "insights" in data:
                insights_value = data["insights"]
                if isinstance(insights_value, dict):
                    for k, v in insights_value.items():
                        extracted.append((str(k), str(v) if v is not None else ""))
                    if extracted:
                        return extracted
                if isinstance(insights_value, list):
                    for item in insights_value:
                        parsed = _parse_insight_item(item)
                        if parsed:
                            extracted.append(parsed)
                    if extracted:
                        return extracted
            if "insight" in data:
                insight_value = data["insight"]
                if isinstance(insight_value, dict):
                    for k, v in insight_value.items():
                        extracted.append((str(k), str(v) if v is not None else ""))
                    if extracted:
                        return extracted
                if isinstance(insight_value, list):
                    for item in insight_value:
                        parsed = _parse_insight_item(item)
                        if parsed:
                            extracted.append(parsed)
                    if extracted:
                        return extracted
            # Legacy or flat mapping: use string values only, ignore metadata keys.
            candidate_keys = [
                k for k, v in data.items() if isinstance(v, (str, int, float, bool))
            ]
            if candidate_keys:
                for k in candidate_keys:
                    extracted.append((str(k), str(data[k])))
                return extracted
        elif isinstance(data, list):
            for item in data:
                parsed = _parse_insight_item(item)
                if parsed:
                    extracted.append(parsed)
            return extracted
        return extracted

    insights: List[Tuple[str, str]] = []
    if encyclopedia_path.endswith(".json"):
        with open(encyclopedia_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            cleaned = raw_text.strip()
            data = None
            if cleaned.startswith("{") or cleaned.startswith("["):
                import re

                json_match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
                if json_match:
                    candidate = json_match.group(1)
                    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        data = json.loads(candidate)
                    except json.JSONDecodeError:
                        data = None
            if data is None:
                insights = [("encyclopedia_text", raw_text.strip())]
        if data is not None:
            extracted = _extract_insights_from_data(data)
            if extracted:
                insights = extracted
            elif isinstance(data, dict):
                insights = [(k, str(v) if v is not None else "") for k, v in data.items()]
            elif isinstance(data, list):
                insights = [item for item in (_parse_insight_item(item) for item in data) if item]
            else:
                insights = [("encyclopedia_text", raw_text.strip())]
    else:
        with open(encyclopedia_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        insights = [("encyclopedia_text", text)]

    if not insights:
        raise ValueError("No insights found in encyclopedia")

    prompt_block = []
    for idx, (name, desc) in enumerate(insights, 1):
        prompt_block.append(f"{idx}. {name}: {desc}")
    return insights, "\n".join(prompt_block)


def find_encyclopedia_paths(encyclopedia_path: str) -> List[str]:
    """Return a list of encyclopedia JSON files for evaluation."""
    if os.path.isdir(encyclopedia_path):
        paths = sorted(
            [
                os.path.join(encyclopedia_path, fn)
                for fn in os.listdir(encyclopedia_path)
                if fn.lower().endswith(".json") and os.path.isfile(os.path.join(encyclopedia_path, fn))
            ]
        )
        if not paths:
            raise FileNotFoundError(
                f"No JSON encyclopedia files found in directory {encyclopedia_path}"
            )
        return paths
    if os.path.isfile(encyclopedia_path):
        return [encyclopedia_path]
    raise FileNotFoundError(f"Encyclopedia path not found: {encyclopedia_path}")


def fetch_accept_tracks(
    year: int,
    max_papers: int = None,
    accept_oral: bool = True,
    accept_spotlight: bool = False,
    accept_poster: bool = False,
    or_username: str = None,
    or_password: str = None,
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

                decisions.append(
                    {
                        "id": sub.id,
                        "forum": sub.forum,
                        "title": content.get("title", {}).get("value", ""),
                        "abstract": content.get("abstract", {}).get("value", ""),
                        "venue": venue,
                        "track": track,
                    }
                )
                if max_papers and len(decisions) >= max_papers:
                    break

            if decisions:
                print(
                    f"Found {len(decisions)} accepted papers via client (no bulk hydration; will fetch content on-demand)"
                )
                # Sort by track priority: oral > spotlight > poster
                track_order = {"oral": 0, "spotlight": 1, "poster": 2}
                decisions.sort(key=lambda p: track_order.get(p.get("track", ""), 999))
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


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract plain text from raw PDF bytes.

    Tries pypdf first (lightweight), then pdfminer.six as fallback.
    Returns empty string if neither is available or extraction fails.
    """
    # Try pypdf
    try:
        import io
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text)
        return "\n".join(parts)
    except ImportError:
        pass
    except Exception as e:
        print(f"    Warning: pypdf extraction failed: {e}")

    # Try pdfminer.six
    try:
        import io
        from pdfminer.high_level import extract_text as pdfminer_extract
        return pdfminer_extract(io.BytesIO(pdf_bytes))
    except ImportError:
        pass
    except Exception as e:
        print(f"    Warning: pdfminer extraction failed: {e}")

    return ""


def _fetch_paper_content(
    forum_id: str,
    session: requests.Session = None,
    or_client=None,
    cache_dir: str = "data/iclr25",
) -> str:
    """Fetch full paper content for scoring, with disk caching.

    On first fetch the extracted text is saved to
    <cache_dir>/<forum_id>.txt so subsequent runs skip the download.

    Strategy (in order):
      1. Return cached text if <cache_dir>/<forum_id>.txt exists.
      2. Download PDF via authenticated OpenReview client and extract text.
         This is the only reliable path for ICLR 2025+ where forum pages
         require authentication.
      3. Supplement with API metadata fields (abstract, keywords, etc.)
         in case PDF extraction yields little text.
      4. Fall back to unauthenticated HTML scrape for older years.

    Returns paper text (up to 50k chars) or empty string.
    """
    def _sanitize(s: str) -> str:
        """Replace surrogate / non-encodable characters with '?'."""
        return s.encode("utf-8", errors="replace").decode("utf-8")

    # --- Cache check ---
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{forum_id}.txt")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = f.read()
            print(f"    Loaded from cache: {len(cached)} chars")
            return cached

    # --- Check for local PDF files ---
    local_pdf_paths = [
        os.path.join("data", "papers", f"{forum_id}.pdf"),
        os.path.join("data", "papers", "iclr23_top5", f"{forum_id}.pdf"),
        os.path.join("data", "papers", "iclr23_diffusion", f"{forum_id}.pdf"),
    ]
    for pdf_path in local_pdf_paths:
        if os.path.exists(pdf_path):
            try:
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                pdf_text = _extract_text_from_pdf_bytes(pdf_bytes)
                if pdf_text.strip():
                    print(f"    Loaded from local PDF: {len(pdf_text)} chars")
                    # Save to cache for future use
                    if cache_path:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            f.write(pdf_text)
                    return pdf_text
            except Exception as e:
                print(f"    Warning: Failed to extract text from local PDF {pdf_path}: {e}")
                continue

    if or_client is not None:
        full_text_parts = []

        # --- Step 1: download and extract PDF text ---
        try:
            pdf_bytes = or_client.get_pdf(forum_id, is_reference=False)
            if pdf_bytes:
                pdf_text = _extract_text_from_pdf_bytes(pdf_bytes)
                if pdf_text.strip():
                    print(f"    PDF extracted: {len(pdf_text)} chars")
                    full_text_parts.append(pdf_text)
                else:
                    print(f"    Warning: PDF downloaded but text extraction yielded nothing")
        except Exception as e:
            print(f"    Warning: PDF download failed for {forum_id}: {e}")

        # --- Step 2: supplement with API metadata fields ---
        try:
            note = or_client.get_note(forum_id)
            content = note.content
            meta_fields = (
                "title", "abstract", "keywords", "tldr", "summary",
                "primary_area", "research_area",
            )
            meta_parts = []
            for field in meta_fields:
                val = content.get(field)
                if val is None:
                    continue
                if isinstance(val, dict):
                    val = val.get("value", "")
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val)
                val = str(val).strip()
                if val:
                    meta_parts.append(f"[{field}] {val}")
            if meta_parts:
                full_text_parts.insert(0, "\n".join(meta_parts))
        except Exception as e:
            print(f"    Warning: API metadata fetch failed for {forum_id}: {e}")

        if full_text_parts:
            text = "\n\n".join(full_text_parts)[:50000]
            text = _sanitize(text)
            if cache_path:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
            return text

    # --- Step 4: unauthenticated HTML scrape (ICLR 2024 and older) ---
    if session is None:
        session = requests.Session()
        session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        )
    try:
        url = f"https://openreview.net/forum?id={forum_id}"
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        content_parts = []
        for content_div in soup.find_all(
            ["div", "section"], class_=re.compile("note-content|paper-content", re.I)
        ):
            text = content_div.get_text(separator=" ", strip=True)
            if text and len(text) > 100:
                content_parts.append(text)

        if content_parts:
            text = " ".join(content_parts)[:50000]
            text = _sanitize(text)
            if cache_path:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
            return text

        page_text = soup.get_text(separator=" ", strip=True)
        if len(page_text) > 1000:
            text = page_text[:50000]
            text = _sanitize(text)
            if cache_path:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
            return text

        return ""
    except Exception as e:
        print(f"    Warning: Could not fetch paper content for {forum_id}: {e}")
        return ""


def call_api(client, prompt: str, max_output_tokens: int = 16384) -> Tuple[str, Dict]:
    """Call API via wrapper and return (raw_text, token_info) tuple."""
    return client.generate_text(prompt, max_output_tokens=max_output_tokens)


def parse_verdict_json(raw: str) -> Dict[str, Any]:
    """Parse checker verdict JSON from chatty/reasoning model output."""
    text = raw.replace("Ġ", " ").replace("Ċ", "\n").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")

    candidates = [text]
    for match in re.finditer(r"\{", text):
        start = match.start()
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start: idx + 1])
                    break

    last_error = None
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        normalized = candidate
        normalized = re.sub(r"\bTrue\b", "true", normalized)
        normalized = re.sub(r"\bFalse\b", "false", normalized)
        normalized = re.sub(r"\bNone\b", "null", normalized)
        normalized = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', normalized)
        normalized = re.sub(r"([{,]\s*)(guided|guid|matched_insights|insights)\s*:", r'\1"\2":', normalized)
        normalized = normalized.replace('"guid"', '"guided"')
        normalized = normalized.replace('"insights"', '"matched_insights"')
        try:
            parsed = json.loads(normalized)
            if isinstance(parsed, dict):
                if "guided" not in parsed and "guid" in parsed:
                    parsed["guided"] = parsed.pop("guid")
                if "matched_insights" not in parsed and "insights" in parsed:
                    parsed["matched_insights"] = parsed.pop("insights")
                parsed.setdefault("guided", bool(parsed.get("matched_insights")))
                parsed.setdefault("matched_insights", [])
                return parsed
        except json.JSONDecodeError as exc:
            last_error = exc

    lowered = text.lower()
    insight_names = []
    for pat in (
        r"matched_insights\s*[:=]\s*\[([^\]]*)\]",
        r"insights\s*[:=]\s*\[([^\]]*)\]",
    ):
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            insight_names = [
                item.strip().strip('"').strip("'")
                for item in m.group(1).split(",")
                if item.strip().strip('"').strip("'")
            ]
            break

    if re.search(r"\b(guided|guid)\s*[:=]\s*true\b", lowered) or insight_names:
        return {"guided": True, "matched_insights": insight_names}
    if re.search(r"\b(guided|guid)\s*[:=]\s*false\b", lowered) or re.search(r"\bno\b", lowered):
        return {"guided": False, "matched_insights": []}

    raise ValueError(f"Failed to parse verdict JSON: {last_error}; raw preview={raw[:500]}")


def score_paper(
    model: Any,
    insights_prompt: str,
    paper: Dict,
    max_output_tokens: int = 16384,
    max_paper_chars: int = 20000,
) -> Tuple[Dict, Dict]:
    """Ask Gemini which insights guide the paper's solutions.

    Returns:
        Tuple of (result_dict, token_info) where result_dict has guided/matched_insights
    """
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    content = paper.get("content", "")
    if max_paper_chars and len(content) > max_paper_chars:
        content = content[:max_paper_chars]

    paper_text = (
        f"Title: {title}\n\nAbstract: {abstract}\n\nFull Paper Content: {content}"
    )

    prompt = f"""
Classify whether the research paper's proposed solution is guided by or directly derived from any insight in the encyclopedia.

You must output exactly this JSON schema and nothing else:
{{"guided": false, "matched_insights": []}}

Insights:
{insights_prompt}

Research Paper:
{paper_text}

Evaluation Criteria - An insight guides the paper ONLY IF ALL of the following are true:

1. CONCRETE METHODOLOGY USAGE: The insight's methodology or approach is concretely used in the paper's methods/approach section, not just theoretically relevant or mentioned in motivation.

2. METHODS SECTION PRESENCE: The insight must be related to how the paper actually implements its solution (methods, algorithms, techniques), not just in problem statement or related work.

3. COUNTERFACTUAL TEST: The paper's core contribution would fundamentally differ or fail without this insight. Ask: "If the authors didn't know this insight, could they still arrive at the same core solution?"

4. SPECIFICITY: The insight must specifically address a key challenge or component of the paper's solution, not just be generally applicable background knowledge.

Response Format:
- Respond ONLY in valid JSON with keys: guided (boolean), matched_insights (array of insight names)
- Set guided=true ONLY when at least one insight passes ALL criteria above
- Use only exact insight names from the Insights list above
- Do not include markdown, comments, explanations, or Python-style booleans. Use JSON true/false.
- If unsure, return {{"guided": false, "matched_insights": []}}.
"""
    print(
        f"      Model input chars: insights={len(insights_prompt)} paper={len(paper_text)} max_output_tokens={max_output_tokens}",
        flush=True,
    )
    raw, token_info = call_api(model, prompt, max_output_tokens=max_output_tokens)
    print(
        f"      Model returned {len(raw)} chars; output_tokens={token_info.get('output_tokens', 0)}",
        flush=True,
    )
    return parse_verdict_json(raw), token_info


def main():
    parser = argparse.ArgumentParser(
        description="Check ICLR Accept papers (oral/spotlight/poster) for insight guidance using Gemini, OpenRouter, or a local HuggingFace model"
    )
    parser.add_argument(
        "--key",
        "--gemini-key",
        dest="api_key",
        type=str,
        default=None,
        help="API key (or set API_KEY or GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--api-model",
        "--gemini-model",
        dest="api_model",
        type=str,
        default="gemini-3-pro-preview",
        help="API model name (default: gemini-3-pro-preview for gemini, openai/gpt-4o for openrouter)",
    )
    parser.add_argument(
        "--api-type",
        type=str,
        choices=["gemini", "openrouter", "local"],
        default="gemini",
        help="API type to use (default: gemini)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Local HuggingFace model name/path. If provided, uses local model inference instead of API.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Device for local HuggingFace model (cuda or cpu).",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load local HuggingFace model with 8-bit quantization.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="Maximum generated tokens per checker call (default: 2048).",
    )
    parser.add_argument(
        "--max-paper-chars",
        type=int,
        default=20000,
        help="Maximum full-paper characters included in each checker prompt (default: 20000; use 0 for no truncation).",
    )
    parser.add_argument(
        "--encyclopedia",
        type=str,
        required=True,
        help="Path to an insights encyclopedia JSON file or a directory containing encyclopedia JSON files",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="ICLR conference year (default: 2024)",
    )
    parser.add_argument(
        "--accept-oral",
        action="store_true",
        help="Include Accept (Oral) papers",
    )
    parser.add_argument(
        "--accept-spotlight",
        action="store_true",
        help="Include Accept (Spotlight) papers",
    )
    parser.add_argument(
        "--accept-poster",
        action="store_true",
        help="Include Accept (Poster) papers",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Limit number of papers (for quick tests)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="guided_oral_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between API calls (default: 1.0)",
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

    args = parser.parse_args()

    use_local_model = args.api_type == "local" or bool(args.model)

    if args.api_type == "openrouter":
        api_key = args.api_key or os.getenv("API_KEY") or os.getenv("OPENROUTER_API_KEY")
    else:
        api_key = args.api_key or os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
    if not use_local_model and not api_key:
        raise ValueError(
            "API key is required. Provide --key or set API_KEY (or GEMINI_API_KEY)."
        )

    if use_local_model:
        if not args.model:
            raise ValueError("Provide --model when using --api-type local.")
        model = LocalHFClient(
            model_name=args.model,
            device=args.device,
            load_in_8bit=args.load_in_8bit,
        )
        active_model_name = args.model
    elif args.api_type == "gemini":
        model = GeminiClient(api_key=api_key, model_name=args.api_model)
        active_model_name = args.api_model
    elif args.api_type == "openrouter":
        # Use provided model or default to gpt-4o
        model_name = args.api_model if args.api_model != "gemini-3-pro-preview" else "openai/gpt-4o"
        model = OpenRouterClient(api_key=api_key, model_name=model_name)
        active_model_name = model_name
    else:
        raise ValueError(f"Unsupported API type: {args.api_type}")

    papers = fetch_accept_tracks(
        args.year,
        max_papers=args.max_papers,
        accept_oral=args.accept_oral,
        accept_spotlight=args.accept_spotlight,
        accept_poster=args.accept_poster,
        or_username=args.or_username,
        or_password=args.or_password,
    )
    if not papers:
        print("No Accept papers found.")
        return

    encyclopedia_paths = find_encyclopedia_paths(args.encyclopedia)
    print(f"\nFound {len(encyclopedia_paths)} encyclopedia file(s) to evaluate.", flush=True)
    print(f"Processing {len(papers)} papers (sorted: oral → spotlight → poster)...\n", flush=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
    )

    # Build authenticated OpenReview client for content fetching (ICLR 2025+)
    or_client = None
    if args.or_username and args.or_password:
        try:
            import openreview
            or_client = openreview.api.OpenReviewClient(
                baseurl="https://api2.openreview.net",
                username=args.or_username,
                password=args.or_password,
            )
            print("Authenticated OpenReview client ready for paper content fetching.")
        except Exception as e:
            print(f"Warning: could not build authenticated client for content fetching: {e}")

    # Pre-load all encyclopedias
    encyclopedias_data = []
    for enc_path in encyclopedia_paths:
        insights, insights_prompt = load_insights(enc_path)
        encyclopedias_data.append({
            'path': enc_path,
            'insights': insights,
            'prompt': insights_prompt,
            'results': [],
            'track_stats': {
                "oral": {"total": 0, "guided": 0},
                "spotlight": {"total": 0, "guided": 0},
                "poster": {"total": 0, "guided": 0}
            }
        })

    print("\nFetching and evaluating paper contents...", flush=True)
    for idx, paper in enumerate(papers, 1):
        track_label = paper.get("track", "")
        forum_id = paper.get("forum") or paper.get("id")

        print(
            f"[{idx}/{len(papers)}] Fetching content for ({track_label}): {paper.get('title', '')[:80]}",
            flush=True,
        )
        paper_content = _fetch_paper_content(forum_id, session, or_client=or_client)
        paper["content"] = paper_content
        if paper_content:
            print(f"  Retrieved {len(paper_content)} characters", flush=True)
        else:
            print(f"  No full content available, using title/abstract only", flush=True)

        # Evaluate against all encyclopedias
        for enc_data in encyclopedias_data:
            print(f"    Evaluating with encyclopedia {os.path.basename(enc_data['path'])} and model {active_model_name}...", flush=True)
            output_tokens = 0
            try:
                verdict, token_info = score_paper(
                    model,
                    enc_data['prompt'],
                    paper,
                    max_output_tokens=args.max_output_tokens,
                    max_paper_chars=args.max_paper_chars,
                )
                guided = bool(verdict.get("guided"))
                matched = verdict.get("matched_insights") or []
                output_tokens = token_info.get("output_tokens", 0)
                print(
                    f"    Result: {'✓ GUIDED' if guided else '✗ Not guided'} | Insights: {len(matched)} | Tokens: {output_tokens}",
                    flush=True,
                )
            except Exception as exc:
                print(f"    API/model error: {exc}", flush=True)
                guided = False
                matched = []

            if track_label in enc_data['track_stats']:
                enc_data['track_stats'][track_label]["total"] += 1
                if guided:
                    enc_data['track_stats'][track_label]["guided"] += 1

            enc_data['results'].append(
                {
                    "id": paper.get("id"),
                    "forum": paper.get("forum"),
                    "title": paper.get("title", ""),
                    "guided": guided,
                    "matched_insights": matched,
                    "track": paper.get("track", ""),
                    "venue": paper.get("venue", ""),
                    "venueid": paper.get("venueid", ""),
                    "output_tokens": output_tokens,
                }
            )
            time.sleep(max(args.sleep, 0))

    # Build evaluations from the collected data
    evaluations = []
    for enc_data in encyclopedias_data:
        total = len(papers)
        total_guided = sum(enc_data['track_stats'][t]["guided"] for t in enc_data['track_stats'])
        guidance_rate = total_guided / total if total else 0.0

        evaluation = {
            "encyclopedia": os.path.basename(enc_data['path']),
            "path": enc_data['path'],
            "metrics": {
                "total_papers": total,
                "guided_papers": total_guided,
                "guidance_rate": guidance_rate,
                "track_stats": enc_data['track_stats'],
            },
            "results": enc_data['results'],
        }
        evaluations.append(evaluation)

        print(f"\n[{len(evaluations)}/{len(encyclopedias_data)}] {os.path.basename(enc_data['path'])}: {total_guided}/{total} papers guided ({guidance_rate*100:.1f}%)")
        for track in ["oral", "spotlight", "poster"]:
            stats = enc_data['track_stats'][track]
            if stats["total"] > 0:
                pct = stats["guided"] / stats["total"] * 100
                print(
                    f"    {track.capitalize():10s}: {stats['guided']}/{stats['total']} guided ({pct:.1f}%)"
                )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if len(evaluations) == 1:
        output_data = {
            "encyclopedia": evaluations[0]["encyclopedia"],
            "path": evaluations[0]["path"],
            "metrics": evaluations[0]["metrics"],
            "results": evaluations[0]["results"],
        }
    else:
        output_data = {"evaluations": evaluations}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary in requested style
    if len(evaluations) > 1:
        print(f"\n{'='*80}")
        print("GUIDANCE RATE SUMMARY")
        print(f"{'='*80}")
        summary_parts = []
        for idx, eval_data in enumerate(evaluations, 1):
            metrics = eval_data["metrics"]
            overall_rate = metrics["guidance_rate"] * 100
            track_stats = metrics["track_stats"]
            oral_pct = track_stats["oral"]["guided"] / track_stats["oral"]["total"] * 100 if track_stats["oral"]["total"] > 0 else 0
            spotlight_pct = track_stats["spotlight"]["guided"] / track_stats["spotlight"]["total"] * 100 if track_stats["spotlight"]["total"] > 0 else 0
            poster_pct = track_stats["poster"]["guided"] / track_stats["poster"]["total"] * 100 if track_stats["poster"]["total"] > 0 else 0
            summary_parts.append(f"file {idx}: all:{overall_rate:.1f}% oral:{oral_pct:.1f}% spotlight:{spotlight_pct:.1f}% poster:{poster_pct:.1f}%")
        print(" ".join(summary_parts))
        print(f"{'='*80}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
