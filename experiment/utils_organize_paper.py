#!/usr/bin/env python3
"""Organize paper PDFs by OpenReview primary research area.

Workflow:
1. Scan a folder for PDF papers.
2. Resolve each paper's OpenReview metadata using local metadata.json files
   when available, or the OpenReview API when needed.
3. Extract the paper's primary research area.
4. Sort papers by area, then rename them as 001_original.pdf,
   002_original.pdf, etc., so papers from the same area stay contiguous.

The script is intentionally conservative: it defaults to dry-run mode and
writes a manifest describing the planned or completed rename operations.

Example:
  python utils_organize_paper.py \
      --root /path/to/papers \
      --year 2023 \
      --apply

If the local folder already contains a metadata.json export from OpenReview,
the script will use that first and only fall back to the API when needed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PDF_SUFFIX = ".pdf"
DEFAULT_INVITATION_TEMPLATE = "ICLR.cc/{year}/Conference/-/Blind_Submission"
MANIFEST_NAME = "paper_organization_manifest.json"
TEMP_PREFIX = ".paper_organize_tmp_"

PRIMARY_AREA_KEYS = [
    "primary_area",
    "research_area",
    "primary research area",
    "primary_research_area",
    "Please_choose_the_closest_area_that_your_submission_falls_into",
    "Please choose the closest area that your submission falls into",
    "area",
]


@dataclass
class PaperRecord:
    path: Path
    title: str
    area: str
    forum_id: str
    source: str
    target_name: str = ""

    @property
    def sort_area(self) -> str:
        return normalize_text(self.area) or "zzzzzzzz"

    @property
    def sort_title(self) -> str:
        return normalize_text(self.title) or normalize_text(self.path.stem)


class MetadataIndex:
    def __init__(self) -> None:
        self.by_forum: Dict[str, Dict[str, Any]] = {}
        self.by_title: Dict[str, Dict[str, Any]] = {}
        self.by_paperhash: Dict[str, Dict[str, Any]] = {}

    def add(self, note: Dict[str, Any]) -> None:
        content = note.get("content", {}) or {}
        forum_id = str(note.get("forum") or note.get("id") or "").strip()
        if forum_id:
            self.by_forum[forum_id] = note

        title = extract_scalar(content, "title")
        if title:
            self.by_title[normalize_text(title)] = note

        paperhash = extract_scalar(content, "paperhash")
        if paperhash:
            self.by_paperhash[normalize_text(paperhash)] = note

    def resolve(self, forum_id: str, title_guess: str) -> Optional[Dict[str, Any]]:
        if forum_id and forum_id in self.by_forum:
            return self.by_forum[forum_id]

        title_key = normalize_text(title_guess)
        if title_key and title_key in self.by_title:
            return self.by_title[title_key]

        # Try the full normalized stem, which may contain the forum id suffix.
        if title_key:
            for candidate_key, note in self.by_title.items():
                if candidate_key == title_key:
                    return note
        return None


def note_to_dict(note: Any) -> Dict[str, Any]:
    if isinstance(note, dict):
        return note

    if hasattr(note, "to_json"):
        try:
            maybe = note.to_json()
            if isinstance(maybe, dict):
                return maybe
        except (TypeError, ValueError, AttributeError):
            pass

    result: Dict[str, Any] = {}
    for key in ("id", "forum", "content", "title"):
        if hasattr(note, key):
            try:
                result[key] = getattr(note, key)
            except AttributeError:
                continue
    return result


@dataclass
class MetadataSource:
    note: Dict[str, Any]
    source: str


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def strip_leading_index(filename: str) -> str:
    return re.sub(r"^\d{3}_", "", filename)


def extract_scalar(container: Any, key: str) -> str:
    if not isinstance(container, dict):
        return ""

    value = container.get(key)
    if value is None:
        return ""

    if isinstance(value, dict):
        value = value.get("value", "")

    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)

    return str(value).strip()


def extract_forum_id_from_name(path: Path) -> str:
    stem = strip_leading_index(path.stem)
    match = re.match(r"^(?P<title>.+)_(?P<forum>[A-Za-z0-9]+)$", stem)
    if match:
        return match.group("forum")
    return ""


def guess_title_from_filename(path: Path) -> str:
    stem = strip_leading_index(path.stem)
    match = re.match(r"^(?P<title>.+)_(?P<forum>[A-Za-z0-9]+)$", stem)
    if match:
        return match.group("title").replace("_", " ").replace("-", " ").strip()
    return stem.replace("_", " ").replace("-", " ").strip()


def find_metadata_json(root: Path) -> List[Path]:
    return sorted(root.rglob("metadata.json"))


def load_metadata_index(metadata_paths: Iterable[Path]) -> MetadataIndex:
    index = MetadataIndex()
    for metadata_path in metadata_paths:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        if raw is None:
            continue

        notes: List[Dict[str, Any]] = []
        if isinstance(raw, list):
            notes = [item for item in raw if isinstance(item, dict)]
        elif isinstance(raw, dict):
            # Support a few common export shapes.
            if isinstance(raw.get("notes"), list):
                notes = [item for item in raw["notes"] if isinstance(item, dict)]
            elif isinstance(raw.get("data"), list):
                notes = [item for item in raw["data"] if isinstance(item, dict)]
            else:
                notes = [raw]

        for note in notes:
            index.add(note)
    return index


def get_primary_area_from_note(note: Dict[str, Any]) -> str:
    content = note.get("content", {}) or {}

    for key in PRIMARY_AREA_KEYS:
        area = extract_scalar(content, key)
        if area:
            return area

    # Fallback: search for any content field that looks like an area field.
    if isinstance(content, dict):
        preferred: List[Tuple[int, str]] = []
        for key, value in content.items():
            if not isinstance(key, str):
                continue
            lowered = key.lower()
            if "venue" in lowered:
                continue
            if "area" not in lowered and "research" not in lowered:
                continue
            score = 0
            if "primary" in lowered:
                score -= 2
            if "closest" in lowered:
                score -= 1
            if "research" in lowered:
                score -= 1
            preferred.append((score, extract_scalar(content, key)))
        for _, value in sorted(preferred, key=lambda x: x[0]):
            if value:
                return value

    return "Unknown"


def get_title_from_note(note: Dict[str, Any], fallback: str = "") -> str:
    content = note.get("content", {}) or {}
    title = extract_scalar(content, "title")
    if title:
        return title
    return fallback


def build_openreview_client(username: Optional[str], password: Optional[str]):
    try:
        openreview = importlib.import_module("openreview")
    except ModuleNotFoundError:
        return None
    return openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=username,
        password=password,
    )


def fetch_openreview_index(
    year: int,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> MetadataIndex:
    client = build_openreview_client(username, password)
    index = MetadataIndex()
    if client is None:
        return index

    invitation = DEFAULT_INVITATION_TEMPLATE.format(year=year)
    notes = client.get_all_notes(invitation=invitation, details="directReplies")

    for note in notes:
        index.add(note_to_dict(note))
    return index


def resolve_note(
    pdf_path: Path,
    local_index: MetadataIndex,
    api_index: MetadataIndex,
) -> MetadataSource:
    forum_id = extract_forum_id_from_name(pdf_path)
    title_guess = guess_title_from_filename(pdf_path)

    note = local_index.resolve(forum_id, title_guess)
    if note is not None:
        return MetadataSource(note=note, source="local-metadata")

    note = api_index.resolve(forum_id, title_guess)
    if note is not None:
        return MetadataSource(note=note, source="openreview-api")

    return MetadataSource(note={}, source="unresolved")


def scan_papers(root: Path) -> List[Path]:
    return sorted(
        [p for p in root.rglob("*.pdf") if p.is_file() and not p.name.startswith(TEMP_PREFIX)],
        key=lambda p: str(p).lower(),
    )


def build_records(
    root: Path,
    local_index: MetadataIndex,
    api_index: MetadataIndex,
) -> List[PaperRecord]:
    records: List[PaperRecord] = []
    for pdf_path in scan_papers(root):
        metadata = resolve_note(pdf_path, local_index, api_index)
        note = metadata.note

        title = get_title_from_note(note, fallback=guess_title_from_filename(pdf_path))
        area = get_primary_area_from_note(note) if note else "Unknown"
        forum_id = str(note.get("forum") or note.get("id") or extract_forum_id_from_name(pdf_path))
        source = metadata.source

        records.append(
            PaperRecord(
                path=pdf_path,
                title=title,
                area=area,
                forum_id=forum_id,
                source=source,
            )
        )
    return records


def planned_target_name(index: int, path: Path) -> str:
    original_name = strip_leading_index(path.name)
    return f"{index:03d}_{original_name}"


def apply_renames(records: List[PaperRecord], dry_run: bool = True) -> List[Dict[str, Any]]:
    manifest: List[Dict[str, Any]] = []

    # Two-stage rename to avoid filename collisions.
    staged: List[Tuple[Path, Path]] = []
    for record in records:
        target = record.path.with_name(record.target_name)
        if record.path.name == record.target_name:
            manifest.append(
                {
                    "source": str(record.path),
                    "target": str(target),
                    "action": "skipped",
                    "reason": "already_named",
                    "title": record.title,
                    "area": record.area,
                    "source_type": record.source,
                }
            )
            continue
        temp = record.path.with_name(f"{TEMP_PREFIX}{os.getpid()}_{record.path.name}")
        staged.append((record.path, temp))
        manifest.append(
            {
                "source": str(record.path),
                "target": str(target),
                "temp": str(temp),
                "action": "planned" if dry_run else "renamed",
                "title": record.title,
                "area": record.area,
                "source_type": record.source,
            }
        )

    if dry_run:
        return manifest

    for src, tmp in staged:
        os.replace(src, tmp)

    for item in manifest:
        if item.get("action") == "skipped":
            continue
        src = Path(item["source"])
        tmp = Path(item["temp"])
        target = Path(item["target"])
        os.replace(tmp, target)
        item["action"] = "renamed"
        item["source"] = str(src)
        item["target"] = str(target)

    return manifest


def write_manifest(root: Path, manifest: List[Dict[str, Any]]) -> Path:
    manifest_path = root / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def print_summary(records: List[PaperRecord]) -> None:
    if not records:
        print("No PDF papers found.")
        return

    print(f"Found {len(records)} papers")
    grouped: Dict[str, List[str]] = {}
    for rec in records:
        grouped.setdefault(rec.area, []).append(rec.target_name)

    for area, file_names in grouped.items():
        print(f"{area}: {' '.join(file_names)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Organize paper PDFs by OpenReview primary research area and rename them with indices."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root folder containing the PDF papers.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="ICLR year used for the OpenReview API invitation (default: 2023).",
    )
    parser.add_argument(
        "--or-username",
        type=str,
        default=None,
        help="OpenReview account email, if authentication is needed.",
    )
    parser.add_argument(
        "--or-password",
        type=str,
        default=None,
        help="OpenReview account password, if authentication is needed.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename the files. Without this flag, the script only previews the plan.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=f"Optional manifest output path. Defaults to <root>/{MANIFEST_NAME}.",
    )

    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    metadata_paths = find_metadata_json(root)
    local_index = load_metadata_index(metadata_paths)
    api_index = fetch_openreview_index(args.year, args.or_username, args.or_password)

    records = build_records(root, local_index, api_index)
    if not records:
        print("No PDF files found under the root folder.")
        return 0

    records.sort(key=lambda r: (r.sort_area, r.sort_title, str(r.path).lower()))
    for idx, record in enumerate(records, start=1):
        record.target_name = planned_target_name(idx, record.path)

    print_summary(records)

    manifest = apply_renames(records, dry_run=not args.apply)
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else root / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    mode = "DRY RUN" if not args.apply else "APPLIED"
    print(f"\n{mode}: manifest saved to {manifest_path}")
    if not args.apply:
        print("Re-run with --apply to rename the files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
