"""
Check overlap between an insight library or trace dataset and benchmark/paper/task documents.

Use exactly one of --encyclopedia or --trace_dataset. The --encyclopedia
argument is an insight library in the same format as checker_iclr.py
--encyclopedia. The --trace_dataset argument is a folder containing JSON trace
files. The --check_array argument lists document sources. A document source can
be:
  - a benchmark name from task_benchmark_domain.py, e.g. aime24, gpqa_diamond,
    livecodebench_lite
  - a paper file or folder, e.g. data/papers/iclr23_top5
  - pinchbench, which loads all pinchbench task_*.md files

Example:
  python checker_privacy.py \
      --encyclopedia important_checkpoints/run/encyclopedia.json \
      --check_array aime24 gpqa_diamond livecodebench_lite data/papers/iclr23_top5 pinchbench

  python checker_privacy.py \
      --trace_dataset important_checkpoints/client_aime25_server_math500 \
      --check_array aime24 aime25
"""

import argparse
import ast
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


TEXT_EXTENSIONS = {".txt", ".md", ".json", ".jsonl", ".csv"}
PAPER_EXTENSIONS = {".pdf", ".txt", ".md", ".json"}
DEFAULT_GEMINI_TOKEN_MODEL = "gemini-3-pro-preview"
KNOWN_GEMINI_CONTEXT_WINDOWS = {
    "gemini-3-pro-preview": {"input_token_limit": 1_048_576, "output_token_limit": 65_536},
    "gemini-3.1-pro-preview": {"input_token_limit": 1_048_576, "output_token_limit": 65_536},
}
EXTRA_DATASET_REGISTRY: Dict[str, Tuple[str, str, Optional[str], Optional[str]]] = {
    "hle": ("hf", "cais/hle", None, "test"),
}


@dataclass
class Insight:
    name: str
    text: str
    shingles: set[Tuple[str, ...]]
    shingle_counts: Dict[Tuple[str, ...], int]


@dataclass
class Document:
    source: str
    doc_id: str
    text: str


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def read_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def estimate_gemini_tokens(text: str) -> int:
    # Gemini docs describe one token as roughly four characters for text.
    return max(1, (len(text) + 3) // 4) if text else 0


def count_text_tokens(
    text: str,
    model_name: str,
    api_key: Optional[str],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "model": model_name,
        "characters": len(text),
        "estimated_tokens": estimate_gemini_tokens(text),
        "exact_tokens": None,
        "input_token_limit": None,
        "output_token_limit": None,
        "fits_input_window": None,
        "method": "estimate",
        "error": None,
    }

    known_window = KNOWN_GEMINI_CONTEXT_WINDOWS.get(model_name)
    if known_window:
        result.update(known_window)
        result["fits_input_window"] = result["estimated_tokens"] <= known_window["input_token_limit"]

    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key) if api_key else genai.Client()
        count_response = client.models.count_tokens(model=model_name, contents=text)
        exact_tokens = getattr(count_response, "total_tokens", None)
        if exact_tokens is None and isinstance(count_response, dict):
            exact_tokens = count_response.get("total_tokens") or count_response.get("totalTokens")
        if exact_tokens is not None:
            result["exact_tokens"] = int(exact_tokens)
            result["method"] = "gemini_count_tokens"

        try:
            model_info = client.models.get(model=model_name)
            input_limit = getattr(model_info, "input_token_limit", None)
            output_limit = getattr(model_info, "output_token_limit", None)
            if input_limit is not None:
                result["input_token_limit"] = int(input_limit)
            if output_limit is not None:
                result["output_token_limit"] = int(output_limit)
        except Exception as exc:
            if result["error"] is None:
                result["error"] = f"Could not fetch model context window: {exc}"

        limit = result.get("input_token_limit")
        token_count = result.get("exact_tokens") or result["estimated_tokens"]
        if limit is not None:
            result["fits_input_window"] = token_count <= int(limit)
    except Exception as exc:
        result["error"] = (
            f"Exact Gemini token count unavailable; using estimate. "
            f"Install google-genai and set GEMINI_API_KEY, or pass --gemini-api-key. Error: {exc}"
        )

    return result


def count_file_tokens(
    path: str,
    model_name: str,
    api_key: Optional[str],
) -> Dict[str, Any]:
    return count_text_tokens(read_text_file(path), model_name, api_key)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def token_shingles(text: str, shingle_size: int) -> List[Tuple[str, ...]]:
    tokens = tokenize(text)
    if not tokens:
        return []
    if len(tokens) < shingle_size:
        return [tuple(tokens)]
    return [tuple(tokens[i : i + shingle_size]) for i in range(len(tokens) - shingle_size + 1)]


def shingle_counts(text: str, shingle_size: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    for shingle in token_shingles(text, shingle_size):
        counts[shingle] = counts.get(shingle, 0) + 1
    return counts


def shingle_text(shingle: Tuple[str, ...]) -> str:
    return " ".join(shingle)


def shingle_hamming_distance(left: Tuple[str, ...], right: Tuple[str, ...]) -> int:
    max_len = max(len(left), len(right))
    distance = abs(len(left) - len(right))
    for idx in range(min(len(left), len(right))):
        if left[idx] != right[idx]:
            distance += 1
    return min(distance, max_len)


def problem_number_from_name(name: str) -> Optional[int]:
    match = re.search(r"problem[_-](\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def compare_doc_to_reference(
    doc: Document,
    insight: Insight,
    max_mismatches: int,
    shingle_size: int,
) -> Dict[str, Any]:
    doc_counts = shingle_counts(doc.text, shingle_size)
    doc_shingles = set(doc_counts)
    exact_overlap = doc_shingles & insight.shingles
    union_size = len(doc_shingles | insight.shingles)
    jaccard = len(exact_overlap) / union_size if union_size else 0.0

    sliding_match_count = 0
    sliding_examples = []
    for doc_shingle in doc_shingles:
        for ref_shingle in insight.shingles:
            distance = shingle_hamming_distance(doc_shingle, ref_shingle)
            if distance <= max_mismatches:
                doc_occurrences = doc_counts.get(doc_shingle, 1)
                ref_occurrences = insight.shingle_counts.get(ref_shingle, 1)
                sliding_match_count += doc_occurrences * ref_occurrences
                if len(sliding_examples) < 10:
                    sliding_examples.append(
                        {
                            "document_window": shingle_text(doc_shingle),
                            "reference_window": shingle_text(ref_shingle),
                            "token_mismatches": distance,
                        }
                    )

    return {
        "insight": insight.name,
        "direct_4gram_intersection_count": len(exact_overlap),
        "direct_4gram_jaccard": round(jaccard, 6),
        "sliding_window_match_count": sliding_match_count,
        "direct_4gram_examples": [
            shingle_text(shingle)
            for shingle in sorted(exact_overlap)[:10]
        ],
        "sliding_window_examples": sliding_examples,
    }


def load_insights(encyclopedia_path: str, shingle_size: int) -> List[Insight]:
    if not os.path.exists(encyclopedia_path):
        raise FileNotFoundError(f"Insight library not found: {encyclopedia_path}")

    path = Path(encyclopedia_path)
    entries: List[Tuple[str, str]] = []
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            entries = [(str(k), normalize_text(v)) for k, v in data.items()]
        elif isinstance(data, list):
            for idx, item in enumerate(data, 1):
                if isinstance(item, dict):
                    name = item.get("name") or item.get("insight") or item.get("title") or f"insight_{idx}"
                    desc = (
                        item.get("description")
                        or item.get("desc")
                        or item.get("text")
                        or item.get("content")
                        or item
                    )
                    entries.append((str(name), normalize_text(desc)))
                else:
                    entries.append((f"insight_{idx}", normalize_text(item)))
        else:
            raise ValueError(f"Unsupported JSON insight library format: {encyclopedia_path}")
    else:
        entries = [("encyclopedia_text", path.read_text(encoding="utf-8"))]

    insights = []
    for name, text in entries:
        combined_text = f"{name}\n{text}"
        counts = shingle_counts(combined_text, shingle_size)
        if counts:
            insights.append(
                Insight(
                    name=name,
                    text=text,
                    shingles=set(counts),
                    shingle_counts=counts,
                )
            )
    if not insights:
        raise ValueError(f"No insights found in {encyclopedia_path}")
    return insights


TRACE_KEYS = {
    "reasoning_trace",
    "reasoning_traces",
    "trace",
    "traces",
    "trajectory",
    "transcript",
    "solution",
    "reflection",
    "insight_book",
    "behavior_book",
    "skills",
}


def flatten_trace_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            text = flatten_trace_value(item)
            if text.strip():
                parts.append(f"{key}: {text}")
        return "\n".join(parts)
    if isinstance(value, list):
        return "\n".join(part for part in (flatten_trace_value(item) for item in value) if part.strip())
    return normalize_text(value)


def extract_trace_text(data: Any) -> str:
    if isinstance(data, list):
        parts = [extract_trace_text(item) for item in data]
        parts = [part for part in parts if part.strip()]
        if parts:
            return "\n\n".join(parts)
        if all(isinstance(item, str) for item in data):
            return "\n".join(data)
        return ""
    if not isinstance(data, dict):
        return ""

    if {"nodes", "edges", "similarity_matrix", "graph_format"}.issubset(data.keys()):
        return ""

    if "role" in data and "content" in data:
        content = flatten_trace_value(data.get("content"))
        return f"{data.get('role')}: {content}" if content.strip() else ""

    parts = []
    for key, value in data.items():
        if str(key).lower() in TRACE_KEYS:
            text = flatten_trace_value(value)
            if text.strip():
                parts.append(f"{key}:\n{text}")
        elif isinstance(value, (dict, list)):
            text = extract_trace_text(value)
            if text.strip():
                parts.append(text)

    if parts:
        return "\n\n".join(parts)

    string_values = {
        str(key): value
        for key, value in data.items()
        if isinstance(value, str) and value.strip()
    }
    if string_values and len(string_values) >= max(1, len(data) // 2):
        return flatten_trace_value(string_values)

    return ""


def load_trace_dataset(trace_dataset: str, shingle_size: int) -> Tuple[List[Insight], str, Dict[str, Any]]:
    trace_dir = Path(trace_dataset)
    if not trace_dir.is_dir():
        raise NotADirectoryError(f"--trace_dataset must be a folder: {trace_dataset}")

    trace_files = sorted(
        path
        for path in trace_dir.rglob("*.json")
        if path.is_file() and "encyclopedia" not in path.name.lower()
    )
    if not trace_files:
        raise FileNotFoundError(f"No *.json trace files found under {trace_dataset}")

    traces: List[Tuple[str, str]] = []
    skipped = []
    for path in trace_files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            text = extract_trace_text(data)
        except Exception as exc:
            skipped.append({"path": str(path), "error": str(exc)})
            continue
        if text.strip():
            traces.append((str(path.relative_to(trace_dir)), text))

    if not traces:
        raise ValueError(f"No reasoning trace text extracted from {trace_dataset}")

    references = []
    for name, text in traces:
        counts = shingle_counts(text, shingle_size)
        if counts:
            references.append(
                Insight(
                    name=name,
                    text=text,
                    shingles=set(counts),
                    shingle_counts=counts,
                )
            )
    if not references:
        raise ValueError(f"No token shingles extracted from {trace_dataset}")
    appended_text = "\n\n".join(f"### {name}\n{text}" for name, text in traces)
    metadata = {
        "trace_dataset": trace_dataset,
        "json_files_found": len(trace_files),
        "traces_loaded": len(traces),
        "skipped_files": skipped,
    }
    return references, appended_text, metadata


def load_dataset_registry() -> Dict[str, Tuple[str, str, Optional[str], Optional[str]]]:
    registry_path = Path(__file__).with_name("task_benchmark_domain.py")
    tree = ast.parse(registry_path.read_text(encoding="utf-8"))
    registry: Dict[str, Tuple[str, str, Optional[str], Optional[str]]] = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "DATASET_REGISTRY":
            registry = ast.literal_eval(node.value)
            break
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == "DATASET_REGISTRY":
                    registry = ast.literal_eval(node.value)
                    break
    registry.update(EXTRA_DATASET_REGISTRY)
    return registry


def first_matching_field(row: Dict[str, Any], candidates: Sequence[str]) -> str:
    lower_to_key = {str(key).lower(): key for key in row.keys()}
    for candidate in candidates:
        key = lower_to_key.get(candidate.lower())
        if key is not None:
            value = row.get(key)
            if value is not None and str(value).strip():
                return normalize_text(value)
    return ""


def document_text_from_record(record: Dict[str, Any]) -> str:
    question = first_matching_field(
        record,
        [
            "problem",
            "question",
            "Question",
            "problem_text",
            "task",
            "statement",
            "prompt",
            "description",
            "text",
            "question_content",
            "input",
        ],
    )
    answer = first_matching_field(
        record,
        [
            "answer",
            "Correct Answer",
            "correct_answer",
            "solution",
            "final_answer",
            "answer_text",
            "short answer",
            "short_answer",
            "response",
            "test_cases",
        ],
    )

    extras = []
    for key in ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "starter_code"):
        if key in record and record[key]:
            extras.append(f"{key}: {normalize_text(record[key])}")

    parts = []
    if question:
        parts.append(f"Question:\n{question}")
    if answer:
        parts.append(f"Answer:\n{answer}")
    parts.extend(extras)
    if parts:
        return "\n\n".join(parts)
    return normalize_text(record)


def load_csv_documents(path: Path, source: str) -> List[Document]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [
            Document(source=source, doc_id=str(row.get("id") or row.get("problem_id") or idx), text=document_text_from_record(row))
            for idx, row in enumerate(reader, 1)
        ]


def normalize_hf_item(dataset_name: str, item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    record = dict(item)
    record.setdefault("id", item.get("question_id", item.get("id", idx + 1)))

    if dataset_name.startswith("gsm8k"):
        solution = normalize_text(item.get("answer") or item.get("solution") or "")
        if "####" in solution:
            record["solution"] = solution.split("####")[0].strip()
            record["answer"] = solution.split("####")[-1].strip()
    elif not (dataset_name.startswith("gpqa") or dataset_name.startswith("livecodebench")):
        problem = item.get("problem") or item.get("question", "")
        solution = item.get("solution", "")
        answer = item.get("answer", "")
        record.setdefault("problem", problem)
        record.setdefault("question", problem)
        record.setdefault("solution", solution or answer or "")
        record.setdefault("answer", answer)

    return record


def load_benchmark_documents(name: str, entry: Tuple[str, str, Optional[str], Optional[str]]) -> List[Document]:
    source_type, path_or_hf_name, data_dir, split = entry
    if source_type == "hf":
        if load_dataset is None:
            raise ImportError("datasets library is required for Hugging Face benchmarks. Install with: pip install datasets")
        ds = load_dataset(path_or_hf_name, name=data_dir, split=split) if data_dir else load_dataset(path_or_hf_name, split=split)
        docs = []
        for idx, item in enumerate(ds):
            if name == "math1000" and idx >= 1000:
                break
            record = normalize_hf_item(name, dict(item), idx)
            docs.append(Document(source=name, doc_id=str(record.get("id", idx + 1)), text=document_text_from_record(record)))
        return docs

    if source_type == "csv":
        return load_csv_documents(Path(path_or_hf_name), name)

    if source_type == "json":
        with Path(path_or_hf_name).open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"JSON benchmark {path_or_hf_name} must contain a list")
        return [
            Document(source=name, doc_id=str(item.get("id", idx + 1)), text=document_text_from_record(item))
            for idx, item in enumerate(data)
            if isinstance(item, dict)
        ]

    raise ValueError(f"Unsupported source type for {name}: {source_type}")


def extract_pdf_text(path: Path, max_pages: Optional[int]) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception as exc:
        raise ImportError("PyPDF2 is required for paper PDF folders. Install with: pip install PyPDF2") from exc

    reader = PdfReader(str(path))
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    return "\n".join(page.extract_text() or "" for page in pages)


def metadata_documents(path: Path, source: str) -> List[Document]:
    metadata_path = path / "metadata.json" if path.is_dir() else path
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []

    docs = []
    for idx, item in enumerate(data, 1):
        if not isinstance(item, dict):
            continue
        content = item.get("content", item)
        if isinstance(content, dict):
            title = normalize_text(content.get("title", ""))
            abstract = normalize_text(content.get("abstract", ""))
            tldr = normalize_text(content.get("TL;DR", content.get("tldr", "")))
            doc_id = normalize_text(item.get("id") or content.get("title") or idx)
            text = "\n\n".join(part for part in [title, tldr, abstract] if part)
            if text.strip():
                docs.append(Document(source=source, doc_id=doc_id, text=text))
    return docs


def load_paper_documents(path: Path, max_pdf_pages: Optional[int]) -> List[Document]:
    source = str(path)
    docs = []
    if path.is_dir() and (path / "metadata.json").exists():
        docs.extend(metadata_documents(path, source))
        if docs:
            return docs

    files = sorted(path.iterdir()) if path.is_dir() else [path]
    for file_path in files:
        if file_path.is_dir() or file_path.name.startswith("."):
            continue
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            text = extract_pdf_text(file_path, max_pdf_pages)
        elif suffix in TEXT_EXTENSIONS:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        else:
            continue
        if text.strip():
            docs.append(Document(source=source, doc_id=file_path.name, text=text))
    return docs


def load_pinchbench_documents(pinchbench_dir: Path) -> List[Document]:
    scripts_dir = pinchbench_dir / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from lib_tasks import TaskLoader  # type: ignore

    tasks = TaskLoader(pinchbench_dir / "tasks").load_all_tasks()
    docs = []
    for task in tasks:
        criteria = "\n".join(task.grading_criteria or [])
        text = "\n\n".join(
            part
            for part in [
                task.name,
                task.prompt,
                task.expected_behavior,
                criteria,
                task.automated_checks or "",
                task.llm_judge_rubric or "",
            ]
            if part
        )
        docs.append(Document(source="pinchbench", doc_id=task.task_id, text=text))
    return docs


def load_documents(option: str, registry: Dict[str, Tuple[str, str, Optional[str], Optional[str]]], max_pdf_pages: Optional[int]) -> List[Document]:
    if option == "pinchbench":
        return load_pinchbench_documents(Path(__file__).with_name("pinchbench"))
    if option in registry:
        return load_benchmark_documents(option, registry[option])

    path = Path(option)
    if not path.exists() and option.startswith("data/paper/"):
        corrected_path = Path(option.replace("data/paper/", "data/papers/", 1))
        if corrected_path.exists():
            print(f"Document path '{option}' not found; using '{corrected_path}'")
            path = corrected_path
    if path.exists():
        return load_paper_documents(path, max_pdf_pages)

    raise ValueError(
        f"Unknown document option '{option}'. Use a benchmark name, 'pinchbench', or an existing paper/file path."
    )


def compare_documents(
    docs: Sequence[Document],
    insights: Sequence[Insight],
    max_mismatches: int,
    shingle_size: int,
    progress_label: str = "Checking documents",
) -> Tuple[int, List[Dict[str, Any]]]:
    total_duplications = 0
    per_doc = []
    doc_iter = docs
    if tqdm is not None:
        doc_iter = tqdm(docs, desc=progress_label, unit="doc")
    for doc in doc_iter:
        matches = []
        for insight in insights:
            match = compare_doc_to_reference(doc, insight, max_mismatches, shingle_size)
            if match["direct_4gram_intersection_count"] or match["sliding_window_match_count"]:
                matches.append(match)
        total_duplications += sum(match["direct_4gram_intersection_count"] for match in matches)
        per_doc.append(
            {
                "id": doc.doc_id,
                "direct_4gram_count": sum(match["direct_4gram_intersection_count"] for match in matches),
                "sliding_window_match_count": sum(match["sliding_window_match_count"] for match in matches),
                "matched_reference_count": len(matches),
                "matches": matches,
            }
        )
    return total_duplications, per_doc


def compare_corresponding_documents(
    docs: Sequence[Document],
    insights: Sequence[Insight],
    max_mismatches: int,
    shingle_size: int,
    progress_label: str = "Checking aligned pairs",
) -> Tuple[int, List[Dict[str, Any]]]:
    references_by_problem = {
        problem_num: insight
        for insight in insights
        for problem_num in [problem_number_from_name(insight.name)]
        if problem_num is not None
    }
    total_duplications = 0
    per_doc = []
    doc_iter = docs
    if tqdm is not None:
        doc_iter = tqdm(docs, desc=progress_label, unit="pair")

    for idx, doc in enumerate(doc_iter, 1):
        problem_num = idx
        insight = references_by_problem.get(problem_num)
        if insight is None:
            per_doc.append(
                {
                    "id": doc.doc_id,
                    "problem_number": problem_num,
                    "direct_4gram_count": 0,
                    "direct_4gram_jaccard": 0.0,
                    "sliding_window_match_count": 0,
                    "matched_reference_count": 0,
                    "matches": [],
                    "missing_corresponding_trace": True,
                }
            )
            continue

        match = compare_doc_to_reference(doc, insight, max_mismatches, shingle_size)
        has_match = bool(
            match["direct_4gram_intersection_count"] or match["sliding_window_match_count"]
        )
        total_duplications += match["direct_4gram_intersection_count"]
        per_doc.append(
            {
                "id": doc.doc_id,
                "problem_number": problem_num,
                "corresponding_trace": insight.name,
                "direct_4gram_count": match["direct_4gram_intersection_count"],
                "direct_4gram_jaccard": match["direct_4gram_jaccard"],
                "sliding_window_match_count": match["sliding_window_match_count"],
                "matched_reference_count": 1 if has_match else 0,
                "matches": [match] if has_match else [],
            }
        )
    return total_duplications, per_doc


def print_match_details(
    per_doc: Sequence[Dict[str, Any]],
    shingle_size: int,
    max_docs: int,
) -> None:
    printed = 0
    for item in per_doc:
        if item.get("direct_4gram_count", 0) <= 0 and item.get("sliding_window_match_count", 0) <= 0:
            continue
        if max_docs >= 0 and printed >= max_docs:
            remaining = sum(
                1
                for rest in per_doc
                if rest.get("direct_4gram_count", 0) > 0 or rest.get("sliding_window_match_count", 0) > 0
            ) - printed
            if remaining > 0:
                print(f"  ... {remaining} additional matched documents omitted from console output")
            break
        printed += 1
        print(f"  {item['id']}: Jaccard={item.get('direct_4gram_jaccard', 0.0):.6f}")
        for match in item.get("matches", []):
            print(f"    trace/reference: {match['insight']}")
            for example in match.get("direct_4gram_examples", [])[:5]:
                print(f"      overlap: {example}")
            for example in match.get("sliding_window_examples", [])[:5]:
                print(
                    "      overlap: "
                    f"doc='{example['document_window']}' | "
                    f"trace='{example['reference_window']}' | "
                    f"mismatches={example['token_mismatches']}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check direct token-shingle and sliding-window overlap between an insight library and documents."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--encyclopedia",
        help="Path to insight library, same format as checker_iclr.py --encyclopedia",
    )
    input_group.add_argument(
        "--trace_dataset",
        help="Folder containing *.json trace files; extracted reasoning traces are appended for token counting",
    )
    parser.add_argument(
        "--check_array",
        nargs="+",
        required=True,
        help="Document options to check: benchmark names, paper paths, or 'pinchbench'",
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=1,
        help=(
            "Maximum token mismatches inside each shingle/window for sliding-window matching "
            "(default: 1, so a 4-token window requires at least 3 positional token matches). "
            "Exact intersection always requires all tokens to match."
        ),
    )
    parser.add_argument("--shingle-size", type=int, default=4, help="Token shingle/window size (default: 4)")
    parser.add_argument("--max-pdf-pages", type=int, default=None, help="Optional limit for PDF extraction pages")
    parser.add_argument(
        "--print-matches",
        type=int,
        default=-1,
        help="Maximum overlapped documents to print to console per checked source; default -1 prints all.",
    )
    parser.add_argument(
        "--gemini-token-model",
        type=str,
        default=DEFAULT_GEMINI_TOKEN_MODEL,
        help=(
            "Gemini model used for exact encyclopedia token counting and context window lookup "
            f"(default: {DEFAULT_GEMINI_TOKEN_MODEL})"
        ),
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key for exact token counting (defaults to GEMINI_API_KEY from the environment)",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path for detailed report")
    args = parser.parse_args()

    registry = load_dataset_registry()
    gemini_api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if args.encyclopedia:
        reference_source = args.encyclopedia
        reference_type = "encyclopedia"
        references = load_insights(args.encyclopedia, args.shingle_size)
        token_report = count_file_tokens(args.encyclopedia, args.gemini_token_model, gemini_api_key)
        trace_metadata = None
    else:
        reference_source = args.trace_dataset
        reference_type = "trace_dataset"
        references, appended_traces, trace_metadata = load_trace_dataset(args.trace_dataset, args.shingle_size)
        token_report = count_text_tokens(appended_traces, args.gemini_token_model, gemini_api_key)

    report = {
        "reference_type": reference_type,
        "reference_source": reference_source,
        "num_references": len(references),
        "reference_tokens": token_report,
        "encyclopedia_tokens": token_report if reference_type == "encyclopedia" else None,
        "appended_trace_tokens": token_report if reference_type == "trace_dataset" else None,
        "trace_dataset": trace_metadata,
        "overlap_hyperparameters": {
            "method": "direct_shingle_intersection_and_sliding_window_matching",
            "direct_intersection": (
                "Build lowercase token 4-gram sets for the reference and checked document. "
                "Report the set intersection count and Jaccard similarity = |A∩B| / |A∪B|."
            ),
            "sliding_window_matching": (
                "Compare every document window against every reference window of the same token size. "
                "Count a match when positional token mismatches are <= distance."
            ),
            "max_token_mismatches": args.distance,
            "shingle_size_tokens": args.shingle_size,
            "tokenizer_regex": r"[A-Za-z0-9_]+",
            "lowercase": True,
            "comparison_unit": "each encyclopedia entry or each trace JSON file vs each checked document",
        },
        "distance": args.distance,
        "shingle_size": args.shingle_size,
        "documents": [],
    }

    token_count = token_report.get("exact_tokens") or token_report["estimated_tokens"]
    token_label = "exact" if token_report.get("exact_tokens") is not None else "estimated"
    window = token_report.get("input_token_limit")
    if window:
        print(
            f"{reference_type} tokens ({token_report['model']}): {token_count:,} {token_label} "
            f"/ {window:,} input-window tokens "
            f"(fits={token_report['fits_input_window']})"
        )
    else:
        print(f"{reference_type} tokens ({token_report['model']}): {token_count:,} {token_label}")
    if token_report.get("error"):
        print(f"Token count note: {token_report['error']}")
    if trace_metadata:
        print(
            f"Trace dataset: {trace_metadata['traces_loaded']} traces loaded from "
            f"{trace_metadata['json_files_found']} JSON files"
        )
    print(
        "Overlap hyperparameters: "
        f"method=direct_4gram_intersection+jaccard and sliding_window, max_token_mismatches={args.distance}, "
        f"shingle_size_tokens={args.shingle_size}, tokenizer_regex=[A-Za-z0-9_]+, lowercase=True"
    )

    for option in args.check_array:
        docs = load_documents(option, registry, args.max_pdf_pages)
        use_corresponding_pairs = (
            reference_type == "trace_dataset"
            and option == "pinchbench"
            and any(problem_number_from_name(ref.name) is not None for ref in references)
        )
        if use_corresponding_pairs:
            print(
                f"Using corresponding-pair mode for {option}: problem_000N trace vs PinchBench task N"
            )
            duplications, per_doc = compare_corresponding_documents(
                docs,
                references,
                args.distance,
                args.shingle_size,
                progress_label=f"Checking aligned {option}",
            )
        else:
            duplications, per_doc = compare_documents(
                docs,
                references,
                args.distance,
                args.shingle_size,
                progress_label=f"Checking {option}",
            )
        sliding_matches = sum(item["sliding_window_match_count"] for item in per_doc)
        jaccard_values = [
            item.get("direct_4gram_jaccard", 0.0)
            for item in per_doc
            if item.get("direct_4gram_count", 0) > 0 or item.get("sliding_window_match_count", 0) > 0
        ]
        avg_jaccard = sum(jaccard_values) / len(jaccard_values) if jaccard_values else 0.0
        max_jaccard = max(jaccard_values) if jaccard_values else 0.0
        summary = {
            "option": option,
            "comparison_mode": "corresponding_pairs" if use_corresponding_pairs else "all_pairs",
            "num_problems": len(docs),
            "num_duplications": duplications,
            "direct_4gram_intersections": duplications,
            "sliding_window_matches": sliding_matches,
            "avg_direct_4gram_jaccard_matched_docs": round(avg_jaccard, 6),
            "max_direct_4gram_jaccard": round(max_jaccard, 6),
            "documents_with_duplications": sum(
                1
                for item in per_doc
                if item["direct_4gram_count"] > 0 or item["sliding_window_match_count"] > 0
            ),
            "details": per_doc,
        }
        report["documents"].append(summary)
        print(
            f"{option}: {summary['num_problems']} problems, "
            f"{summary['documents_with_duplications']} problems with overlap, "
            f"avg matched-doc Jaccard={summary['avg_direct_4gram_jaccard_matched_docs']:.6f}, "
            f"max Jaccard={summary['max_direct_4gram_jaccard']:.6f}"
        )
        print_match_details(per_doc, args.shingle_size, args.print_matches)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote detailed report to {output_path}")


if __name__ == "__main__":
    main()
