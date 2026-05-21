---
id: task_log_nginx_status_codes
name: Nginx Access Log - HTTP Status Code Distribution
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "nginx_access.log"
    source: "logs/nginx_access_json.log"
---

# Nginx Access Log - HTTP Status Code Distribution

## Prompt

Analyze the Nginx JSON access log at `nginx_access.log` and produce a report on HTTP status code distribution. Each line is a JSON object with fields: `time`, `remote_ip`, `remote_user`, `request`, `response`, `bytes`, `referrer`, `agent`.

Your report should include:

1. **Total Requests**: Total number of log entries
2. **Status Code Breakdown**: Count and percentage for each HTTP status code observed
3. **Status Code Categories**: Group by category (2xx success, 3xx redirect, 4xx client error, 5xx server error) with totals
4. **Top Offenders**: For 4xx and 5xx errors, list the top 5 client IPs generating the most errors
5. **Requested Paths**: For each status code, show the top 3 most-requested paths
6. **Assessment**: Brief assessment of server health based on the status code distribution

Write the report to `status_code_report.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should parse all 1000 JSON log entries and produce:

**Total Requests:** 1000

**Status Code Breakdown:**
- 200: 35 (3.5%)
- 206: 1 (0.1%)
- 304: 274 (27.4%)
- 403: 2 (0.2%)
- 404: 688 (68.8%)

**Status Code Categories:**
- 2xx: 36 (3.6%)
- 3xx: 274 (27.4%)
- 4xx: 690 (69.0%)

**Key observations:**
- The log covers May 17, 2015, approximately 08:05–16:05 UTC
- 404 errors dominate — nearly 69% of all requests
- Paths are primarily `/downloads/product_1` and `/downloads/product_2`
- Most traffic comes from Debian APT package manager clients
- 80.91.33.133 is the most active IP with ~210 requests

Acceptable variations:
- Exact percentages may differ by rounding
- Assessment wording will vary
- Additional analysis is welcome

---

## Grading Criteria

- [ ] `status_code_report.md` is created in the workspace
- [ ] Total request count is reported (1000)
- [ ] All observed status codes are listed with counts (200, 206, 304, 403, 404)
- [ ] Status codes are grouped by category (2xx, 3xx, 4xx)
- [ ] Top error-generating IPs are identified

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the Nginx status code distribution task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "status_code_report.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "total_count": 0.0,
            "status_codes_listed": 0.0,
            "categories_grouped": 0.0,
            "top_error_ips": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Total request count
    scores["total_count"] = 1.0 if "1000" in content or "1,000" in content else 0.0

    # Check 2: All status codes listed
    has_200 = "200" in content
    has_304 = "304" in content
    has_404 = "404" in content
    has_403 = "403" in content
    codes_found = sum([has_200, has_304, has_404, has_403])
    scores["status_codes_listed"] = (
        1.0 if codes_found >= 4 else
        0.5 if codes_found >= 3 else 0.0
    )

    # Check 3: Categories grouped
    has_2xx = "2xx" in content or "success" in content
    has_3xx = "3xx" in content or "redirect" in content
    has_4xx = "4xx" in content or "client error" in content
    scores["categories_grouped"] = (
        1.0 if sum([has_2xx, has_3xx, has_4xx]) >= 3 else
        0.5 if sum([has_2xx, has_3xx, has_4xx]) >= 2 else 0.0
    )

    # Check 4: Top error IPs identified
    top_ips = ["80.91.33.133", "5.83.131.103", "202.143.95.26", "50.57.209.92"]
    ips_found = sum(1 for ip in top_ips if ip in content)
    scores["top_error_ips"] = (
        1.0 if ips_found >= 2 else
        0.5 if ips_found >= 1 else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- 1000 JSON entries, single day: May 17, 2015 (08:05–16:05 UTC)
- Extremely high 404 rate (68.8%) suggests missing resources or misconfigured download paths
- Traffic is predominantly automated (Debian APT package managers)
- Only 2 main paths: `/downloads/product_1` and `/downloads/product_2`
- No 5xx server errors observed

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
