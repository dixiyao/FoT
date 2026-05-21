---
id: task_log_ssh_unusual_times
name: SSH Auth Log - Unusual Hour Login Detection
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "auth.log"
    source: "logs/openssh_auth.log"
---

# SSH Auth Log - Unusual Hour Login Detection

## Prompt

Analyze the OpenSSH authentication log at `auth.log` and identify login activity occurring at unusual hours. Assume normal business hours are 08:00–18:00 local server time.

Your report should include:

1. **Hourly Distribution**: Count of authentication events per hour
2. **Off-Hours Activity**: All authentication events occurring before 08:00 or after 18:00
3. **Early Morning Analysis**: Detailed breakdown of events between 06:00–08:00 (the earliest in the log)
4. **Successful Logins Timing**: When did the successful login(s) occur? During business hours or not?
5. **Attack Timing Patterns**: Do attackers prefer certain hours? Is there a pattern?
6. **Temporal Risk Assessment**: Based on timing patterns, what times should be monitored most closely?

Write the report to `unusual_hours_report.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should parse the log and produce:

**Hourly Distribution:**
- 06:xx — 7 entries (log starts at 06:55)
- 07:xx — 169 entries
- 08:xx — 118 entries
- 09:xx — 676 entries (peak hour)
- 10:xx — 530 entries (log ends at 10:59)

**Off-Hours Activity:**
- 7 entries before 07:00 (at 06:55–06:56)
- All pre-08:00 entries are attack traffic (failed logins, BREAK-IN warnings)

**Successful Login Timing:**
- User fztu logged in at 09:32:20 — during business hours

**Attack Timing Patterns:**
- Attacks escalate sharply from 07:xx to 09:xx
- Peak attack volume is 09:xx with 676 entries
- This could indicate attackers in a different timezone where 09:00 LabSZ time is their working hours
- The early morning entries (06:55) represent the tail end or start of an attack campaign

Acceptable variations:
- "Unusual hours" definition may vary
- Timezone assumptions may differ
- Assessment language will vary

---

## Grading Criteria

- [ ] `unusual_hours_report.md` is created in the workspace
- [ ] Hourly distribution of events is provided
- [ ] Off-hours (pre-08:00) events are separately identified
- [ ] The successful login timing is noted (09:32:20, during business hours)
- [ ] Attack timing patterns are analyzed

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the SSH unusual hours detection task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "unusual_hours_report.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "hourly_distribution": 0.0,
            "off_hours_identified": 0.0,
            "successful_timing": 0.0,
            "timing_patterns": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Hourly distribution provided
    hour_markers = ["06:", "07:", "08:", "09:", "10:"]
    alt_markers = ["6 am", "7 am", "8 am", "9 am", "10 am", "6:00", "7:00",
                   "8:00", "9:00", "10:00"]
    all_markers = hour_markers + alt_markers
    hours_found = sum(1 for m in all_markers if m in content)
    scores["hourly_distribution"] = (
        1.0 if hours_found >= 4 else
        0.5 if hours_found >= 2 else 0.0
    )

    # Check 2: Off-hours events identified
    off_hours_keywords = ["before 08", "before 8:00", "early morning",
                          "06:55", "pre-business", "off-hour", "off hour",
                          "unusual hour", "outside business"]
    scores["off_hours_identified"] = (
        1.0 if sum(1 for kw in off_hours_keywords if kw in content) >= 1 else 0.0
    )

    # Check 3: Successful login timing noted
    has_fztu = "fztu" in content
    has_time = "09:32" in content or "9:32" in content
    has_business = any(kw in content for kw in ["business hour", "normal hour",
                                                  "working hour", "during"])
    scores["successful_timing"] = (
        1.0 if has_fztu and (has_time or has_business) else
        0.5 if has_fztu else 0.0
    )

    # Check 4: Timing patterns analyzed
    pattern_keywords = ["peak", "escalat", "increas", "pattern", "trend",
                        "676", "530", "busiest", "most active", "concentrated"]
    scores["timing_patterns"] = (
        1.0 if sum(1 for kw in pattern_keywords if kw in content) >= 2 else
        0.5 if sum(1 for kw in pattern_keywords if kw in content) >= 1 else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- Log spans ~4 hours: 06:55 to 10:59 on December 10
- Attack intensity increases dramatically over time: 7 → 169 → 118 → 676 → 530 per hour
- The 09:xx hour is the busiest with 676 entries
- Pre-08:00 entries are entirely attack traffic
- The single successful login (fztu at 09:32) occurs during peak attack activity

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
