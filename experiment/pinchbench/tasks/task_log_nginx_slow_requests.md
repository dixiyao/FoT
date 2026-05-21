---
id: task_log_nginx_slow_requests
name: Nginx Access Log - Find Largest Responses
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "nginx_access.log"
    source: "logs/nginx_access_json.log"
---

# Nginx Access Log - Find Largest Responses

## Prompt

Analyze the Nginx JSON access log at `nginx_access.log` and identify the requests that generated the largest responses (by bytes transferred). Each line is a JSON object with fields: `time`, `remote_ip`, `remote_user`, `request`, `response`, `bytes`, `referrer`, `agent`.

Your report should include:

1. **Top 10 Largest Responses**: List the requests with the highest byte counts, including timestamp, client IP, request path, status code, and bytes
2. **Byte Distribution Summary**: Overall statistics — min, max, mean, median bytes transferred (excluding zero-byte responses)
3. **Zero-Byte Responses**: Count of zero-byte responses and which status codes produce them
4. **Large Response Analysis**: What paths and client IPs are associated with the largest transfers?
5. **Efficiency Assessment**: What percentage of requests result in actual data transfer vs cache hits (304)?

Write the report to `large_responses_report.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should parse all 1000 JSON log entries and produce:

**Top Largest Responses:**
- Maximum bytes observed: ~3318 bytes
- Largest responses are 200 OK responses for `/downloads/product_1` and `/downloads/product_2`
- Top byte values include: 3318, 3316, 3301, 2582, 2578, etc.

**Zero-Byte Analysis:**
- 304 Not Modified responses all have 0 bytes (274 entries)
- 404 responses have small byte counts (300-340 range typically)

**Efficiency:**
- ~274 out of 1000 requests are 304 (cache hits) — 27.4%
- 200 OK with data: ~35 requests — 3.5%
- 404 errors: 688 requests — these transfer small error pages

Acceptable variations:
- Exact byte values are deterministic from the log
- Assessment wording will vary
- Top 10 vs top 20 is fine

---

## Grading Criteria

- [ ] `large_responses_report.md` is created in the workspace
- [ ] Top largest responses are listed with byte counts
- [ ] Zero-byte / 304 responses are analyzed separately
- [ ] Distribution statistics (min, max, mean or median) are provided
- [ ] Paths associated with largest responses are identified

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the Nginx largest responses task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "large_responses_report.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "top_responses_listed": 0.0,
            "zero_byte_analysis": 0.0,
            "distribution_stats": 0.0,
            "paths_identified": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Top responses listed with byte counts
    has_max_bytes = any(b in content for b in ["3318", "3316", "3301"])
    has_ranking = any(kw in content for kw in ["top", "largest", "biggest", "highest"])
    scores["top_responses_listed"] = (
        1.0 if has_max_bytes and has_ranking else
        0.5 if has_max_bytes else 0.0
    )

    # Check 2: Zero-byte / 304 analysis
    has_zero = "0 byte" in content or "zero byte" in content or "zero-byte" in content or "no data" in content
    has_304 = "304" in content
    scores["zero_byte_analysis"] = (
        1.0 if has_304 and has_zero else
        0.5 if has_304 else 0.0
    )

    # Check 3: Distribution statistics
    stat_keywords = ["min", "max", "mean", "median", "average", "total bytes",
                     "distribution", "range"]
    scores["distribution_stats"] = (
        1.0 if sum(1 for kw in stat_keywords if kw in content) >= 3 else
        0.5 if sum(1 for kw in stat_keywords if kw in content) >= 1 else 0.0
    )

    # Check 4: Paths identified
    has_product_1 = "product_1" in content
    has_product_2 = "product_2" in content
    has_downloads = "downloads" in content or "/download" in content
    scores["paths_identified"] = (
        1.0 if has_product_1 and has_product_2 else
        0.5 if has_downloads else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- Maximum response size is only ~3318 bytes — this is a lightweight download server
- The vast majority of responses are either 304 (0 bytes) or 404 (small error page)
- Only ~35 requests return 200 with actual content
- All requests target just two paths: `/downloads/product_1` and `/downloads/product_2`

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
