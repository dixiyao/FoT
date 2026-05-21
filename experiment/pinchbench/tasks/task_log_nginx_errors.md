---
id: task_log_nginx_errors
name: Nginx Access Log - Error Pattern Analysis
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "nginx_access.log"
    source: "logs/nginx_access_json.log"
---

# Nginx Access Log - Error Pattern Analysis

## Prompt

Analyze the Nginx JSON access log at `nginx_access.log` and produce a detailed report on error patterns (4xx and 5xx responses). Each line is a JSON object with fields: `time`, `remote_ip`, `remote_user`, `request`, `response`, `bytes`, `referrer`, `agent`.

Your report should include:

1. **Error Overview**: Total errors, error rate (as percentage of all requests), breakdown by status code
2. **404 Analysis**: Which paths are returning 404? Are these legitimate missing resources or misconfigured routes?
3. **403 Analysis**: What's being forbidden and from which IPs?
4. **Error by Client IP**: Which IPs generate the most errors? Top 10 with counts
5. **Error by Path**: Which request paths generate the most errors? Top 10 with counts
6. **Temporal Pattern**: Are errors concentrated at certain times or spread evenly?
7. **Remediation Recommendations**: Based on the error patterns, suggest 3 specific fixes

Write the report to `error_analysis.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should parse all 1000 JSON log entries and produce:

**Error Overview:**
- Total errors: 690 (69.0% of all requests)
- 404: 688 errors
- 403: 2 errors
- No 5xx errors observed

**404 Analysis:**
- All 404s target `/downloads/product_1` and `/downloads/product_2`
- These same paths also return 200 and 304 at other times
- This suggests intermittent resource availability, not permanently missing files

**403 Analysis:**
- 2 forbidden requests — identify the IPs and paths

**Top Error IPs:**
- 80.91.33.133 is the highest-volume IP overall and likely the top error generator
- Other high-frequency IPs: 5.83.131.103, 202.143.95.26, 50.57.209.92

**Key insight:**
- The extremely high 404 rate (68.8%) on a package download server is unusual
- Package managers retry automatically, which amplifies the error count
- The root cause is likely transient unavailability of download resources

Acceptable variations:
- Exact counts are deterministic
- Remediation suggestions will vary
- Assessment depth may differ

---

## Grading Criteria

- [ ] `error_analysis.md` is created in the workspace
- [ ] Error rate and status code breakdown are provided (690 errors, 69%, 404/403 split)
- [ ] 404 errors are analyzed by path (/downloads/product_1, /downloads/product_2)
- [ ] Top error-generating IPs are listed
- [ ] At least 2 remediation recommendations are provided

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the Nginx error pattern analysis task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "error_analysis.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "error_rate_breakdown": 0.0,
            "path_analysis": 0.0,
            "top_error_ips": 0.0,
            "recommendations": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Error rate and breakdown
    has_total = any(n in content for n in ["690", "688", "69%", "68.8%", "69.0%"])
    has_404 = "404" in content
    has_403 = "403" in content
    scores["error_rate_breakdown"] = (
        1.0 if has_total and has_404 and has_403 else
        0.5 if has_404 and has_total else 0.0
    )

    # Check 2: Path analysis
    has_product_1 = "product_1" in content
    has_product_2 = "product_2" in content
    scores["path_analysis"] = (
        1.0 if has_product_1 and has_product_2 else
        0.5 if has_product_1 or has_product_2 else 0.0
    )

    # Check 3: Top error IPs
    top_ips = ["80.91.33.133", "5.83.131.103", "202.143.95.26", "50.57.209.92"]
    ips_found = sum(1 for ip in top_ips if ip in content)
    scores["top_error_ips"] = (
        1.0 if ips_found >= 3 else
        0.5 if ips_found >= 1 else 0.0
    )

    # Check 4: Recommendations provided
    rec_keywords = ["recommend", "suggestion", "fix", "should", "consider",
                    "implement", "configure", "add", "improve"]
    lines = content.split("\n")
    rec_lines = [l for l in lines if any(kw in l for kw in rec_keywords)]
    scores["recommendations"] = (
        1.0 if len(rec_lines) >= 2 else
        0.5 if len(rec_lines) >= 1 else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- 690 out of 1000 requests are errors — almost entirely 404s
- Both `/downloads/product_1` and `/downloads/product_2` return a mix of 200, 304, and 404
- This pattern is consistent with a package repository where files are being updated/rotated
- Only 2 entries with 403 Forbidden
- Zero 5xx errors — the server itself is healthy

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
