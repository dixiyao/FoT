#!/usr/bin/env python3
"""Lint: verify that tasks/manifest.yaml and the task_*.md files are in sync.

Checks performed:
1. Every entry in the manifest has a corresponding .md file.
2. Every task_*.md file (excluding TASK_TEMPLATE.md) is listed in the manifest.
3. No duplicate entries in the manifest (across all categories).
4. Each task file's frontmatter ``id`` matches its filename (without .md).
5. No task appears in more than one category.
6. Every ``run_first`` entry exists in exactly one category.
7. Frontmatter ``category`` matches the manifest category (warning, non-blocking).

Supports both the legacy flat format (``tasks: [...]``) and the new
categorized format (``categories: {cat: [...], ...}``).

Exit code 0 on success, 1 on any error.
"""

import re
import sys
from pathlib import Path

import yaml


def _extract_task_ids_flat(manifest: dict) -> tuple[list[str], dict[str, str]]:
    """Extract task IDs from the legacy flat ``tasks`` list.

    Returns (all_task_ids, category_map) where category_map is empty for flat
    manifests.
    """
    return manifest.get("tasks", []), {}


def _extract_task_ids_categorized(
    manifest: dict,
    errors: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Extract task IDs from the categorized format.

    Returns (all_task_ids, category_map).  Populates *errors* with any issues
    found during extraction.
    """
    categories: dict[str, list[str]] = manifest.get("categories", {})
    run_first: list[str] = manifest.get("run_first", [])

    all_ids: list[str] = []
    category_map: dict[str, str] = {}  # task_id -> category

    for category, ids in categories.items():
        for task_id in ids or []:
            if task_id in category_map:
                errors.append(
                    f"Task '{task_id}' appears in multiple categories: "
                    f"'{category_map[task_id]}' and '{category}'"
                )
            else:
                category_map[task_id] = category
            all_ids.append(task_id)

    # Validate run_first entries exist in a category
    for task_id in run_first:
        if task_id not in category_map:
            errors.append(f"run_first entry '{task_id}' is not listed in any category")

    return all_ids, category_map


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    tasks_dir = root / "tasks"
    manifest_path = tasks_dir / "manifest.yaml"

    errors: list[str] = []
    warnings: list[str] = []

    # --- Load manifest ---
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}")
        return 1

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    # --- Determine format and extract task IDs ---
    if "categories" in manifest:
        manifest_ids, category_map = _extract_task_ids_categorized(manifest, errors)
    else:
        manifest_ids, category_map = _extract_task_ids_flat(manifest)

    # --- Check for duplicates ---
    seen: set[str] = set()
    for task_id in manifest_ids:
        if task_id in seen:
            # Only flag if not already caught by multi-category check
            if task_id not in category_map or manifest_ids.count(task_id) > 1:
                errors.append(f"Duplicate manifest entry: {task_id}")
        seen.add(task_id)

    # --- Discover .md files ---
    md_files = {
        p.stem: p for p in sorted(tasks_dir.glob("task_*.md")) if p.name != "TASK_TEMPLATE.md"
    }

    # --- Cross-check ---
    manifest_set = set(manifest_ids)
    file_set = set(md_files.keys())

    for task_id in sorted(manifest_set - file_set):
        errors.append(f"In manifest but missing file: {task_id}.md")

    for task_id in sorted(file_set - manifest_set):
        errors.append(f"File exists but missing from manifest: {task_id}.md")

    # --- Frontmatter checks ---
    for stem, path in md_files.items():
        content = path.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if not fm_match:
            errors.append(f"{path.name}: no YAML frontmatter found")
            continue
        try:
            fm = yaml.safe_load(fm_match.group(1))
        except yaml.YAMLError:
            errors.append(f"{path.name}: invalid YAML frontmatter")
            continue

        # Check id matches filename
        fm_id = fm.get("id", "")
        if fm_id != stem:
            errors.append(f"{path.name}: frontmatter id '{fm_id}' != expected '{stem}'")

        # Check frontmatter category matches manifest category (warning only)
        if category_map and stem in category_map:
            fm_category = (fm.get("category", "") or "").lower()
            manifest_category = category_map[stem].lower()
            if fm_category and fm_category != manifest_category:
                warnings.append(
                    f"{path.name}: frontmatter category '{fm_category}' "
                    f"differs from manifest category '{manifest_category}'"
                )

    # --- Report ---
    if warnings:
        print(f"Manifest lint: {len(warnings)} warning(s)\n")
        for warn in warnings:
            print(f"  ⚠ {warn}")
        print()

    if errors:
        print(f"Manifest lint: {len(errors)} error(s) found\n")
        for err in errors:
            print(f"  - {err}")
        return 1

    task_count = len(manifest_set)
    if category_map:
        cat_count = len(set(category_map.values()))
        print(f"Manifest lint: OK ({task_count} tasks in {cat_count} categories, all in sync)")
    else:
        print(f"Manifest lint: OK ({task_count} tasks, all in sync)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
