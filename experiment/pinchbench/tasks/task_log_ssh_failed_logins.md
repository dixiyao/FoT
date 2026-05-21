---
id: task_log_ssh_failed_logins
name: SSH Auth Log - Failed Login Analysis
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "auth.log"
    source: "logs/openssh_auth.log"
---

# SSH Auth Log - Failed Login Analysis

## Prompt

Analyze the OpenSSH authentication log at `auth.log` and produce a detailed report on failed login attempts. The log is from a server named "LabSZ" and covers SSH authentication events.

Your report should include:

1. **Overview**: Total log entries, date range, total failed login attempts
2. **Failed Password Attempts**: Count of "Failed password" entries, broken down by source IP
3. **Invalid User Attempts**: Count of attempts using non-existent usernames, with a list of the top 10 most-tried usernames
4. **Top Attacking IPs**: Top 10 source IPs by number of failed attempts, with counts
5. **Authentication Methods**: What authentication methods are being attempted (password, publickey, etc.)?
6. **Reverse DNS Failures**: How many entries show "POSSIBLE BREAK-IN ATTEMPT" warnings?
7. **Summary Assessment**: Is this server under active attack? What does the pattern suggest?

Write the report to `failed_login_report.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should parse the 1500 log entries and produce:

**Overview:**
- 1500 log entries
- Date: December 10 (times range from ~06:55 to ~10:59)
- ~366 "Failed password" entries
- ~100 "Invalid user" entries

**Top Attacking IPs:**
- 183.62.140.253 — ~307 entries (dominant attacker)
- 187.141.143.180 — ~189 entries
- 103.99.0.122 — ~83 entries
- 112.95.230.3 — ~54 entries
- 5.188.10.180 — ~30 entries

**Top Invalid Usernames:**
- admin (18), oracle (6), support (5), test (4), inspur (3), 0 (3), matlab (3), webmaster (2), guest (2), 1234 (2)

**Key observations:**
- Only 1 successful login in the entire log (user "fztu" from 119.137.62.142)
- 85 "POSSIBLE BREAK-IN ATTEMPT" warnings from reverse DNS failures
- The server is clearly under active brute-force attack
- Attack comes from a small number of IPs generating hundreds of attempts each

Acceptable variations:
- Exact counts may differ by ±5 depending on parsing
- Assessment language will vary

---

## Grading Criteria

- [ ] `failed_login_report.md` is created in the workspace
- [ ] Total failed attempts are counted (approximately 366 failed passwords)
- [ ] Top attacking IPs are identified (183.62.140.253 as the top attacker)
- [ ] Invalid usernames are listed (admin, oracle, support as top targets)
- [ ] The server is assessed as being under brute-force attack

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the SSH failed login analysis task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "failed_login_report.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "failed_count": 0.0,
            "top_ips": 0.0,
            "invalid_usernames": 0.0,
            "attack_assessment": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Failed attempt count
    has_failed_count = any(n in content for n in ["366", "365", "367", "failed password"])
    scores["failed_count"] = (
        1.0 if has_failed_count else
        0.5 if "failed" in content and any(c.isdigit() for c in content) else 0.0
    )

    # Check 2: Top attacking IPs identified
    top_ips = ["183.62.140.253", "187.141.143.180", "103.99.0.122", "112.95.230.3"]
    ips_found = sum(1 for ip in top_ips if ip in content)
    scores["top_ips"] = (
        1.0 if ips_found >= 3 else
        0.5 if ips_found >= 1 else 0.0
    )

    # Check 3: Invalid usernames listed
    usernames = ["admin", "oracle", "support", "test", "webmaster", "guest"]
    users_found = sum(1 for u in usernames if u in content)
    scores["invalid_usernames"] = (
        1.0 if users_found >= 3 else
        0.5 if users_found >= 1 else 0.0
    )

    # Check 4: Attack assessment
    attack_keywords = ["brute force", "brute-force", "attack", "compromise",
                       "malicious", "automated", "scanning", "dictionary"]
    scores["attack_assessment"] = (
        1.0 if sum(1 for kw in attack_keywords if kw in content) >= 2 else
        0.5 if sum(1 for kw in attack_keywords if kw in content) >= 1 else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- Server: LabSZ, running OpenSSH with PAM authentication
- The attack window is approximately 4 hours (06:55–10:59 on December 10)
- 183.62.140.253 alone accounts for ~307 log entries — a clear brute-force source
- Reverse DNS failures trigger "POSSIBLE BREAK-IN ATTEMPT" warnings (85 occurrences)
- Only 1 successful login in the entire log (user fztu)

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
