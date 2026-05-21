---
id: task_log_syslog_auth_failures
name: Linux Syslog - Authentication Failure Summary
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "syslog.log"
    source: "logs/linux_syslog.log"
---

# Linux Syslog - Authentication Failure Summary

## Prompt

Analyze the Linux syslog at `syslog.log` and produce a comprehensive summary of all authentication failures. The log contains PAM authentication events from multiple services.

Your report should include:

1. **Total Auth Failures**: Count all authentication failure entries across all services
2. **Failures by Service**: Break down failures by service (sshd, ftpd, login, su, etc.)
3. **Failures by Source**: Top 10 source hosts/IPs generating the most failures
4. **Targeted Users**: Which user accounts are being targeted in failed auth attempts?
5. **Temporal Distribution**: When do most authentication failures occur? Any spikes?
6. **FTP vs SSH Analysis**: Compare the authentication attack patterns across FTP and SSH — are the same sources attacking both?
7. **Recommendations**: Based on the failure patterns, recommend specific security improvements

Write the report to `auth_failures_report.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should parse the ~2000+ PAM-related entries and produce:

**Total Auth Failures:**
- Over 2000 authentication-related PAM entries
- Primary sources: sshd(pam_unix) (~1610 entries), ftpd connections (~1655 entries)

**Failures by Service:**
- sshd(pam_unix) — the dominant source of auth failure messages
- ftpd — heavy connection volume (ftpd logs connections, not always explicit failures)
- su(pam_unix) — 394 entries (mostly legitimate — session opens/closes for cron)
- login(pam_unix) — 14 entries
- klogind — 46 entries (Kerberos login daemon)

**Failures by Source:**
- SSH attacks come from various remote hosts (rhost= in pam entries)
- FTP connections concentrated from specific IPs (e.g., 209.184.7.130)
- Some hosts appear in both SSH and FTP failure logs

**Targeted Users:**
- root — primary target for SSH brute force
- Various invalid usernames tried via SSH

**Temporal Distribution:**
- Log spans Jun 9 to Sep 14
- Attack spikes visible on specific dates
- SSH brute force tends to cluster in time

Acceptable variations:
- Exact counts depend on how "authentication failure" is defined
- FTP entries may or may not be classified as auth failures
- Temporal analysis granularity may vary

---

## Grading Criteria

- [ ] `auth_failures_report.md` is created in the workspace
- [ ] Authentication failures are counted (2000+ pam-related entries)
- [ ] Failures are broken down by service (sshd, ftpd, su, etc.)
- [ ] Top source hosts are listed
- [ ] Recommendations for security improvement are provided

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the Linux syslog authentication failure summary task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "auth_failures_report.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "failures_counted": 0.0,
            "by_service": 0.0,
            "source_hosts": 0.0,
            "recommendations": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Failures counted
    has_count = any(kw in content for kw in ["2000", "2,000", "1610", "1,610",
                                               "1655", "1,655", "thousand"])
    has_failure = "authentication failure" in content or "auth fail" in content or "failed" in content
    scores["failures_counted"] = (
        1.0 if has_count and has_failure else
        0.5 if has_failure else 0.0
    )

    # Check 2: Broken down by service
    services = ["sshd", "ftpd", "ftp", "su(pam", "su ", "login", "klogin", "pam_unix"]
    services_found = sum(1 for s in services if s in content)
    scores["by_service"] = (
        1.0 if services_found >= 3 else
        0.5 if services_found >= 2 else 0.0
    )

    # Check 3: Source hosts listed
    host_indicators = ["rhost", "source", "remote", "ip", "host", "209.184",
                       "sagonet", "iasi", "astral"]
    scores["source_hosts"] = (
        1.0 if sum(1 for kw in host_indicators if kw in content) >= 3 else
        0.5 if sum(1 for kw in host_indicators if kw in content) >= 1 else 0.0
    )

    # Check 4: Recommendations provided
    rec_keywords = ["recommend", "should", "implement", "consider", "disable",
                    "block", "firewall", "fail2ban", "key-based", "rate limit"]
    rec_lines = [l for l in content.split("\n") if any(kw in l for kw in rec_keywords)]
    scores["recommendations"] = (
        1.0 if len(rec_lines) >= 2 else
        0.5 if len(rec_lines) >= 1 else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- Server "combo" has multiple authentication surfaces: SSH, FTP, telnet (klogind), local login, su
- sshd(pam_unix) is the most frequent auth service (1610 entries)
- ftpd has 1655 connection entries — the heaviest service by volume
- su(pam_unix) sessions (394) are mostly legitimate (cron jobs, uid=0 running as service users)
- login(pam_unix) has 14 entries — console/terminal logins
- klogind has 46 entries — Kerberos remote login
- The system is a 2005-era Fedora server with many exposed services — a security hardening candidate

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
