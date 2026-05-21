"""
Check a pre-sampled list of ICLR papers for insight guidance using an LLM.

Takes the JSON output of checker_iclr_sample.py (list of paper titles + metadata)
and runs the same insight-scoring pipeline as checker_iclr.py.

Usage:
  python checker_iclr_givensample.py \
      --sample sampled_papers_2024.json \
      --encyclopedia important_checkpoints/client_aime25_server_math500/encyclopedia.json \
      --api-type gemini \
      --key $GEMINI_API_KEY \
      --api-model gemini-3-pro-preview \
      --output results_sample_2024.json

  python checker_iclr_givensample.py \
      --sample sampled_papers_2024.json \
      --encyclopedia important_checkpoints/ \
      --api-type openrouter \
      --key $OPENROUTER_API_KEY \
      --api-model openai/gpt-4o \
      --output results_sample_2024.json
"""

import argparse
import json
import os
import re
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# LLM clients (identical to checker_iclr.py)
# ---------------------------------------------------------------------------

class GeminiClient:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        if not (HAS_GENAI or HAS_GEMINI):
            raise ImportError("Install google-genai: pip install google-genai")
        self.model_name = model_name
        self.backend = "new" if HAS_GENAI else "old"
        if self.backend == "new":
            self.client = genai_new.Client(api_key=api_key)
        else:
            genai_old.configure(api_key=api_key)
            self.model = genai_old.GenerativeModel(model_name)

    def generate_text(self, prompt: str, max_output_tokens: int = 16384) -> Tuple[str, Dict]:
        if self.backend == "new":
            from google.genai import types
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=max_output_tokens),
            )
            text = None
            output_tokens = 0
            if hasattr(resp, "text") and resp.text:
                text = resp.text.strip()
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
            resp = self.model.generate_content(
                prompt, generation_config={"max_output_tokens": max_output_tokens}
            )
            text = None
            output_tokens = 0
            if hasattr(resp, "text") and resp.text:
                text = resp.text.strip()
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
            try:
                candidate = resp.candidates[0]
                if candidate.content and candidate.content.parts:
                    parts = [p.text for p in candidate.content.parts if hasattr(p, "text") and p.text]
                    if parts:
                        return "\n".join(parts).strip(), {"output_tokens": output_tokens}
            except Exception:
                pass
            raise RuntimeError("Failed to extract text from google.generativeai response")


class OpenRouterClient:
    def __init__(self, api_key: str, model_name: str = "openai/gpt-4o"):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://openrouter.ai/api/v1"

    def generate_text(self, prompt: str, max_output_tokens: int = 16384) -> Tuple[str, Dict]:
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
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
                if "choices" in result and result["choices"]:
                    text = result["choices"][0]["message"]["content"].strip()
                    output_tokens = result.get("usage", {}).get("completion_tokens", 0)
                    return text, {"output_tokens": output_tokens}
                raise RuntimeError("Failed to extract text from OpenRouter response")
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429 and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"    Rate limited. Retrying in {delay:.1f}s... ({attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise e
            except Exception as e:
                raise e
        raise RuntimeError(f"Failed after {max_retries} attempts")


class LocalHFClient:
    """HuggingFace local model with the same interface as API clients."""

    def __init__(self, model_name: str, device: Optional[str] = None, load_in_8bit: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device} (8bit={load_in_8bit})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        else:
            load_kwargs["torch_dtype"] = torch.float16 if self.device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if not load_in_8bit:
            self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded.")

    def generate_text(self, prompt: str, max_output_tokens: int = 4096) -> Tuple[str, Dict]:
        import torch

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
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            full_prompt = f"{system_prompt}\n\nUser:\n{prompt}\n\nAssistant:\n"

        inputs = self.tokenizer(
            full_prompt, return_tensors="pt", truncation=True, max_length=65536
        ).to(self.device)
        input_len = int(inputs["input_ids"].shape[1])
        print(f"      Input tokens: {input_len}, max_new_tokens: {max_output_tokens}", flush=True)

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
        output_ids = outputs[0][input_len:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return text, {"output_tokens": int(output_ids.shape[0])}


# ---------------------------------------------------------------------------
# Paper content fetching (same as checker_iclr.py)
# ---------------------------------------------------------------------------

def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        import io
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        parts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(p for p in parts if p.strip())
    except ImportError:
        pass
    except Exception as e:
        print(f"    Warning: pypdf failed: {e}")
    try:
        import io
        from pdfminer.high_level import extract_text as pdfminer_extract
        return pdfminer_extract(io.BytesIO(pdf_bytes))
    except ImportError:
        pass
    except Exception as e:
        print(f"    Warning: pdfminer failed: {e}")
    return ""


def _fetch_paper_content(
    forum_id: str,
    session: requests.Session = None,
    or_client=None,
    cache_dir: str = "data/iclr_cache",
) -> str:
    def _sanitize(s: str) -> str:
        return s.encode("utf-8", errors="replace").decode("utf-8")

    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{forum_id}.txt")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = f.read()
            print(f"    Loaded from cache: {len(cached)} chars")
            return cached

    if or_client is not None:
        full_text_parts = []
        try:
            pdf_bytes = or_client.get_pdf(forum_id, is_reference=False)
            if pdf_bytes:
                pdf_text = _extract_text_from_pdf_bytes(pdf_bytes)
                if pdf_text.strip():
                    print(f"    PDF extracted: {len(pdf_text)} chars")
                    full_text_parts.append(pdf_text)
        except Exception as e:
            print(f"    Warning: PDF download failed for {forum_id}: {e}")
        try:
            note = or_client.get_note(forum_id)
            content = note.content
            meta_fields = ("title", "abstract", "keywords", "tldr", "summary", "primary_area")
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
            print(f"    Warning: metadata fetch failed for {forum_id}: {e}")
        if full_text_parts:
            text = "\n\n".join(full_text_parts)[:50000]
            text = _sanitize(text)
            if cache_path:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
            return text

    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
    try:
        url = f"https://openreview.net/forum?id={forum_id}"
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        content_parts = []
        for div in soup.find_all(["div", "section"], class_=re.compile("note-content|paper-content", re.I)):
            text = div.get_text(separator=" ", strip=True)
            if text and len(text) > 100:
                content_parts.append(text)
        if content_parts:
            text = _sanitize(" ".join(content_parts)[:50000])
            if cache_path:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
            return text
        page_text = soup.get_text(separator=" ", strip=True)
        if len(page_text) > 1000:
            text = _sanitize(page_text[:50000])
            if cache_path:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
            return text
    except Exception as e:
        print(f"    Warning: HTML scrape failed for {forum_id}: {e}")
    return ""


# ---------------------------------------------------------------------------
# Encyclopedia loading + scoring (identical to checker_iclr.py)
# ---------------------------------------------------------------------------

def load_insights(encyclopedia_path: str) -> Tuple[List[Tuple[str, str]], str]:
    if not os.path.exists(encyclopedia_path):
        raise FileNotFoundError(f"Encyclopedia not found: {encyclopedia_path}")

    def _parse_item(item):
        if isinstance(item, dict):
            name = (
                item.get("name") or item.get("insight_name") or item.get("skill_name")
                or item.get("title") or item.get("key") or item.get("id")
            )
            desc = (
                item.get("description") or item.get("desc") or item.get("detail")
                or item.get("text") or item.get("insight") or item.get("skill") or ""
            )
            if not name and isinstance(desc, str) and desc.strip():
                return ("insight", desc.strip())
            if name:
                return (str(name), str(desc) if desc is not None else "")
            return None
        if isinstance(item, str):
            return ("insight", item)
        return None

    def _extract(data):
        extracted = []
        if isinstance(data, dict):
            for key in ("skills", "insights", "insight"):
                val = data.get(key)
                if val is None:
                    continue
                if isinstance(val, list):
                    for item in val:
                        p = _parse_item(item)
                        if p:
                            extracted.append(p)
                    if extracted:
                        return extracted
                if isinstance(val, dict):
                    for k, v in val.items():
                        extracted.append((str(k), str(v) if v is not None else ""))
                    if extracted:
                        return extracted
            for k, v in data.items():
                if isinstance(v, (str, int, float, bool)):
                    extracted.append((str(k), str(v)))
            return extracted
        if isinstance(data, list):
            for item in data:
                p = _parse_item(item)
                if p:
                    extracted.append(p)
        return extracted

    if encyclopedia_path.endswith(".json"):
        with open(encyclopedia_path, "r", encoding="utf-8") as f:
            raw = f.read()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if data is not None:
            insights = _extract(data) or [(k, str(v)) for k, v in data.items()] if isinstance(data, dict) else []
        else:
            insights = [("encyclopedia_text", raw.strip())]
    else:
        with open(encyclopedia_path, "r", encoding="utf-8") as f:
            insights = [("encyclopedia_text", f.read().strip())]

    if not insights:
        raise ValueError("No insights found in encyclopedia")

    prompt_block = [f"{i}. {name}: {desc}" for i, (name, desc) in enumerate(insights, 1)]
    return insights, "\n".join(prompt_block)


def find_encyclopedia_paths(path: str) -> List[str]:
    if os.path.isdir(path):
        paths = sorted(
            os.path.join(path, fn)
            for fn in os.listdir(path)
            if fn.lower().endswith(".json") and os.path.isfile(os.path.join(path, fn))
        )
        if not paths:
            raise FileNotFoundError(f"No JSON files in {path}")
        return paths
    if os.path.isfile(path):
        return [path]
    raise FileNotFoundError(f"Encyclopedia path not found: {path}")


def parse_verdict_json(raw: str) -> Dict[str, Any]:
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
        normalized = re.sub(r"\bTrue\b", "true", candidate)
        normalized = re.sub(r"\bFalse\b", "false", normalized)
        normalized = re.sub(r"\bNone\b", "null", normalized)
        normalized = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', normalized)
        normalized = re.sub(r"([{,]\s*)(guided|guid|matched_insights|insights)\s*:", r'\1"\2":', normalized)
        normalized = normalized.replace('"guid"', '"guided"').replace('"insights"', '"matched_insights"')
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
    for pat in (r"matched_insights\s*[:=]\s*\[([^\]]*)\]", r"insights\s*[:=]\s*\[([^\]]*)\]"):
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
    if re.search(r"\b(guided|guid)\s*[:=]\s*false\b", lowered):
        return {"guided": False, "matched_insights": []}
    raise ValueError(f"Failed to parse verdict JSON: {last_error}; raw={raw[:500]}")


def score_paper(
    model: Any,
    insights_prompt: str,
    paper: Dict,
    max_output_tokens: int = 16384,
    max_paper_chars: int = 20000,
) -> Tuple[Dict, Dict]:
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    content = paper.get("content", "")
    if max_paper_chars and len(content) > max_paper_chars:
        content = content[:max_paper_chars]

    paper_text = f"Title: {title}\n\nAbstract: {abstract}\n\nFull Paper Content: {content}"

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
    raw, token_info = model.generate_text(prompt, max_output_tokens=max_output_tokens)
    print(
        f"      Model returned {len(raw)} chars; output_tokens={token_info.get('output_tokens', 0)}",
        flush=True,
    )
    return parse_verdict_json(raw), token_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Check pre-sampled ICLR papers (from checker_iclr_sample.py) for insight guidance"
    )
    parser.add_argument(
        "--sample",
        type=str,
        required=True,
        help="Path to sampled papers JSON produced by checker_iclr_sample.py",
    )
    parser.add_argument(
        "--encyclopedia",
        type=str,
        required=True,
        help="Path to an insights encyclopedia JSON file or directory of JSON files",
    )
    parser.add_argument(
        "--api-type",
        type=str,
        choices=["gemini", "openrouter", "local"],
        default="gemini",
        help="API type: gemini | openrouter | local (HuggingFace)",
    )
    parser.add_argument(
        "--key",
        "--api-key",
        dest="api_key",
        type=str,
        default=None,
        help="API key (or set API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY env vars). Not needed for local.",
    )
    parser.add_argument(
        "--api-model",
        type=str,
        default=None,
        help="API model name (gemini/openrouter). Not used for --api-type local.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name or path for --api-type local (e.g. Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for local model: cuda | cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load local model with 8-bit quantization (requires bitsandbytes)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="Max generated tokens per LLM call (default: 2048)",
    )
    parser.add_argument(
        "--max-paper-chars",
        type=int,
        default=20000,
        help="Max full-paper chars per prompt (default: 20000; 0 = no limit)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: results_sample_<year>.json)",
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
        help="OpenReview email for authenticated content fetching (ICLR 2025+)",
    )
    parser.add_argument(
        "--or-password",
        type=str,
        default=None,
        help="OpenReview password for authenticated content fetching (ICLR 2025+)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/iclr_cache",
        help="Directory for caching downloaded paper text (default: data/iclr_cache)",
    )
    args = parser.parse_args()

    # Load sampled papers
    with open(args.sample, "r", encoding="utf-8") as f:
        sample_data = json.load(f)
    papers = sample_data.get("papers", [])
    year = sample_data.get("year", "unknown")
    print(f"Loaded {len(papers)} sampled papers from ICLR {year}")

    output_path = args.output or f"results_sample_{year}.json"

    # Build LLM client
    if args.api_type == "local":
        if not args.model:
            raise ValueError("--model is required for --api-type local (e.g. Qwen/Qwen2.5-7B-Instruct)")
        model = LocalHFClient(
            model_name=args.model,
            device=args.device,
            load_in_8bit=args.load_in_8bit,
        )
        active_model_name = args.model
    elif args.api_type == "openrouter":
        api_key = args.api_key or os.getenv("API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("API key required for openrouter. Provide --key or set OPENROUTER_API_KEY.")
        model_name = args.api_model or "openai/gpt-4o"
        model = OpenRouterClient(api_key=api_key, model_name=model_name)
        active_model_name = model_name
    else:  # gemini
        api_key = args.api_key or os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key required for gemini. Provide --key or set GEMINI_API_KEY.")
        model_name = args.api_model or "gemini-3-pro-preview"
        model = GeminiClient(api_key=api_key, model_name=model_name)
        active_model_name = model_name

    # Build authenticated OpenReview client if credentials provided
    or_client = None
    if args.or_username and args.or_password:
        try:
            import openreview
            or_client = openreview.api.OpenReviewClient(
                baseurl="https://api2.openreview.net",
                username=args.or_username,
                password=args.or_password,
            )
            print("Authenticated OpenReview client ready for content fetching.")
        except Exception as e:
            print(f"Warning: could not build authenticated OR client: {e}")

    # Load encyclopedia (single file required for human-comparison mode)
    encyclopedia_paths = find_encyclopedia_paths(args.encyclopedia)
    if len(encyclopedia_paths) > 1:
        print(f"Note: {len(encyclopedia_paths)} encyclopedia files found; using first: {encyclopedia_paths[0]}")
    enc_path = encyclopedia_paths[0]
    insights, insights_prompt = load_insights(enc_path)
    encyclopedias_data = [{"path": enc_path, "insights": insights, "prompt": insights_prompt, "results": []}]

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})

    print(f"\nFetching content and evaluating {len(papers)} papers...\n")
    for idx, paper in enumerate(papers, 1):
        track_label = paper.get("track", "")
        forum_id = paper.get("forum") or paper.get("id")
        print(
            f"[{idx}/{len(papers)}] ({track_label}) {paper.get('title', '')[:80]}",
            flush=True,
        )

        paper_content = _fetch_paper_content(
            forum_id, session, or_client=or_client, cache_dir=args.cache_dir
        )
        paper["content"] = paper_content
        if paper_content:
            print(f"  Retrieved {len(paper_content)} chars", flush=True)
        else:
            print(f"  No full content — using title/abstract only", flush=True)

        enc_data = encyclopedias_data[0]  # single encyclopedia per run
        print(
            f"    Evaluating with {os.path.basename(enc_data['path'])} / {active_model_name}...",
            flush=True,
        )
        try:
            verdict, token_info = score_paper(
                model,
                enc_data["prompt"],
                paper,
                max_output_tokens=args.max_output_tokens,
                max_paper_chars=args.max_paper_chars,
            )
            guided = bool(verdict.get("guided"))
            matched = verdict.get("matched_insights") or []
            print(f"    => {'TRUE  (guided)' if guided else 'FALSE (not guided)'}", flush=True)
        except Exception as exc:
            print(f"    API error: {exc}", flush=True)
            guided = False
            matched = []

        enc_data["results"].append(
            {
                "index": idx,
                "title": paper.get("title", ""),
                "track": track_label,
                "guided": guided,
                "matched_insights": matched,
                "forum": paper.get("forum"),
            }
        )
        time.sleep(max(args.sleep, 0))

    # Build output: flat list ordered by paper index for direct comparison with human labels
    enc_data = encyclopedias_data[0]
    results = enc_data["results"]

    # Compact boolean array (same order as sampled_papers JSON)
    guided_array = [r["guided"] for r in results]

    output_data = {
        "year": year,
        "sample_source": args.sample,
        "encyclopedia": os.path.basename(enc_data["path"]),
        "model": active_model_name,
        # Primary comparison list: index-aligned True/False for each paper
        "guided_labels": guided_array,
        # Full per-paper detail for traceability
        "papers": [
            {
                "index": r["index"],
                "title": r["title"],
                "track": r["track"],
                "guided": r["guided"],
                "matched_insights": r["matched_insights"],
                "forum": r["forum"],
            }
            for r in results
        ],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print final comparison table
    print(f"\n{'='*72}")
    print(f"LLM GUIDED/NOT-GUIDED LABELS  (compare with human evaluation)")
    print(f"{'='*72}")
    print(f"{'#':>4}  {'Guided':6}  Title")
    print(f"{'-'*72}")
    for r in results:
        label = "TRUE " if r["guided"] else "FALSE"
        print(f"{r['index']:>4}  {label}   {r['title'][:60]}")
    print(f"{'-'*72}")
    print(f"guided_labels array: {guided_array}")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
