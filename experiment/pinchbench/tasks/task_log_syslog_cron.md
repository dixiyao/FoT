---
id: task_log_syslog_cron
name: Linux Syslog - Cron Job Execution Analysis
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "syslog.log"
    source: "logs/linux_syslog.log"
---

# Linux Syslog - Cron Job Execution Analysis

## Prompt

Analyze the Linux syslog at `syslog.log` and produce a report on cron job and scheduled task activity. Look for crond, anacron, logrotate, and any other scheduled execution evidence.

Your report should include:

1. **Cron Service Status**: When does crond start? How many startup events are there?
2. **Anacron Activity**: Document all anacron startup events and their timing
3. **Logrotate Activity**: Identify logrotate executions and what services they affect
4. **su Session Patterns**: The `su(pam_unix)` entries often indicate cron executing tasks as different users — analyze these patterns
5. **Scheduled Task Timeline**: Create a timeline of all scheduled/periodic activity
6. **Recurring Patterns**: Identify any regular patterns (daily, weekly) in scheduled executions

Write the report to `cron_analysis.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should identify:

**Cron/Anacron Startups:**
- crond startup: Jun 9 06:06:49, Jun 10 11:32:10, Jul 27 14:42:23 (3 events, aligned with system boots)
- anacron startup: Jun 9 06:06:51, Jun 10 11:32:12, Jul 27 14:42:25 (follows crond immediately)

**Logrotate Activity:**
- 97 logrotate entries in the log
- Logrotate triggers service restarts (especially CUPS)
- Runs periodically — likely daily via anacron/cron

**su(pam_unix) Patterns:**
- 394 su session entries
- Sessions opened for users like: htt, cyrus, news
- Pattern: "session opened for user X by (uid=0)" → "session closed for user X"
- These are cron jobs running as specific service users

**Key scheduled users:**
- htt (web server) — regular su sessions
- cyrus (mail) — regular su sessions
- news — periodic sessions

**Recurring Patterns:**
- Daily: logrotate runs, su sessions for service users
- At boot: crond → anacron startup sequence
- Weekly: CUPS restart pattern (shutdown + startup)

Acceptable variations:
- Pattern detection approaches may differ
- Timeline granularity may vary
- Some scheduled patterns require inference from the su session data

---

## Grading Criteria

- [ ] `cron_analysis.md` is created in the workspace
- [ ] Crond and anacron startup events are documented
- [ ] Logrotate activity is identified (97 entries)
- [ ] su(pam_unix) sessions are linked to scheduled tasks
- [ ] Recurring patterns are identified (daily, at boot, etc.)

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the Linux syslog cron job analysis task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "cron_analysis.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "cron_anacron": 0.0,
            "logrotate": 0.0,
            "su_sessions": 0.0,
            "recurring_patterns": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Crond and anacron documented
    has_crond = "crond" in content or "cron daemon" in content
    has_anacron = "anacron" in content
    scores["cron_anacron"] = (
        1.0 if has_crond and has_anacron else
        0.5 if has_crond else 0.0
    )

    # Check 2: Logrotate identified
    has_logrotate = "logrotate" in content
    has_count = "97" in content or any(kw in content for kw in
        ["frequent", "numerous", "many logrotate"])
    scores["logrotate"] = (
        1.0 if has_logrotate and has_count else
        0.5 if has_logrotate else 0.0
    )

    # Check 3: su sessions analyzed
    has_su = "su(" in content or "su(pam" in content or "su session" in content
    has_users = sum(1 for u in ["htt", "cyrus", "news"] if u in content)
    scores["su_sessions"] = (
        1.0 if has_su and has_users >= 2 else
        0.5 if has_su else 0.0
    )

    # Check 4: Recurring patterns identified
    pattern_keywords = ["daily", "weekly", "periodic", "regular", "recurring",
                        "schedule", "pattern", "at boot", "every"]
    scores["recurring_patterns"] = (
        1.0 if sum(1 for kw in pattern_keywords if kw in content) >= 2 else
        0.5 if sum(1 for kw in pattern_keywords if kw in content) >= 1 else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- crond and anacron always start together at boot, 2 seconds apart
- logrotate is the most active periodic process (97 entries)
- su sessions (394 entries) are the main indicator of scheduled task execution
- Common cron user pattern: uid=0 opens session for service user, then closes it
- The server runs mail (cyrus), web (htt), and news services — all with cron maintenance

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
