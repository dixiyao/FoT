---
id: task_log_nginx_traffic
name: Nginx Access Log - Traffic Patterns by Time
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "nginx_access.log"
    source: "logs/nginx_access_json.log"
---

# Nginx Access Log - Traffic Patterns by Time

## Prompt

Analyze the Nginx JSON access log at `nginx_access.log` and produce a report on traffic patterns over time. Each line is a JSON object with fields: `time`, `remote_ip`, `remote_user`, `request`, `response`, `bytes`, `referrer`, `agent`.

Your report should include:

1. **Time Range**: The full date/time range covered by the log
2. **Hourly Traffic Breakdown**: Number of requests per hour
3. **Peak and Low Traffic**: Identify the busiest and quietest hours
4. **Bandwidth Over Time**: Total bytes transferred per hour
5. **Request Rate Trends**: Are requests steady, bursty, or showing a trend?
6. **Per-IP Activity Over Time**: Identify IPs that appear across multiple hours vs those that appear in bursts

Write the report to `traffic_report.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should parse all 1000 JSON log entries and produce:

**Time Range:** May 17, 2015, 08:05:01 to 16:05:10 UTC (approximately 8 hours)

**Hourly Breakdown (approximate):**
- 08:xx — the log starts mid-hour
- Traffic is distributed across the 8-hour window
- The log contains entries timestamped between 08:05 and 16:05

**Key observations:**
- 73 unique client IPs over the period
- 80.91.33.133 is the most persistent client (~210 requests spread across the time range)
- Most traffic is Debian APT package manager traffic (automated updates)
- Bytes transferred vary — 304 (Not Modified) responses have 0 bytes; 200 responses range up to ~3318 bytes
- The server appears to be a software download/repository mirror

Acceptable variations:
- Exact hourly counts may vary depending on parsing approach
- Any reasonable binning approach (hourly, 30-min, etc.) is acceptable
- Trend analysis wording will vary

---

## Grading Criteria

- [ ] `traffic_report.md` is created in the workspace
- [ ] Time range is identified (May 17, 2015; approximately 08:05–16:05 UTC)
- [ ] Traffic is broken down by time period (hourly or similar)
- [ ] Peak/busiest periods are identified
- [ ] Bandwidth or bytes transferred is analyzed

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the Nginx traffic patterns task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "traffic_report.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "time_range": 0.0,
            "hourly_breakdown": 0.0,
            "peak_identified": 0.0,
            "bandwidth_analysis": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Time range identified
    has_date = any(d in content for d in ["may 17", "2015-05-17", "17/may/2015", "may 2015"])
    has_range = any(t in content for t in ["08:05", "16:05", "8 hour", "eight hour"])
    scores["time_range"] = (
        1.0 if has_date and has_range else
        0.5 if has_date else 0.0
    )

    # Check 2: Traffic broken down by time period
    time_keywords = ["hour", "period", "interval", "08:", "09:", "10:", "11:", "12:",
                     "13:", "14:", "15:", "16:"]
    time_sections = sum(1 for kw in time_keywords if kw in content)
    scores["hourly_breakdown"] = (
        1.0 if time_sections >= 4 else
        0.5 if time_sections >= 2 else 0.0
    )

    # Check 3: Peak/busiest periods identified
    peak_keywords = ["peak", "busiest", "highest", "most active", "maximum",
                     "lowest", "quietest", "least"]
    scores["peak_identified"] = (
        1.0 if sum(1 for kw in peak_keywords if kw in content) >= 2 else
        0.5 if sum(1 for kw in peak_keywords if kw in content) >= 1 else 0.0
    )

    # Check 4: Bandwidth/bytes analysis
    bandwidth_keywords = ["bytes", "bandwidth", "transfer", "data", "0 bytes",
                          "304", "not modified"]
    scores["bandwidth_analysis"] = (
        1.0 if sum(1 for kw in bandwidth_keywords if kw in content) >= 2 else
        0.5 if sum(1 for kw in bandwidth_keywords if kw in content) >= 1 else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- 1000 JSON entries over ~8 hours on May 17, 2015
- Timestamps are in format `17/May/2015:HH:MM:SS +0000`
- The log is a download server serving APT package repositories
- Most bytes transferred are small (max ~3318 bytes for 200 responses)
- 304 Not Modified responses carry 0 bytes — important for bandwidth analysis

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
