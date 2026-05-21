---
id: task_log_ssh_user_activity
name: SSH Auth Log - User Login Activity Report
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "auth.log"
    source: "logs/openssh_auth.log"
---

# SSH Auth Log - User Login Activity Report

## Prompt

Analyze the OpenSSH authentication log at `auth.log` and produce a user-focused activity report. For every username mentioned in the log (both valid and invalid), summarize their authentication activity.

Your report should include:

1. **All Usernames Attempted**: List every username that appears in the log (both valid system users and invalid/non-existent users)
2. **Valid vs Invalid Users**: Classify each username as valid (accepted by the system) or invalid (rejected as non-existent)
3. **Per-User Summary**: For each username, show: number of attempts, source IPs, success/failure, first and last attempt timestamp
4. **Most Targeted Users**: Rank usernames by number of failed attempts
5. **Username Patterns**: Are attackers using a dictionary? Common patterns (admin, root, test, service accounts)?
6. **User Risk Assessment**: Which usernames, if they existed, would pose the greatest security risk?

Write the report to `user_activity_report.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should identify:

**Invalid Users (top by frequency):**
- admin (18 attempts), oracle (6), support (5), test (4), inspur (3), 0 (3), matlab (3), webmaster (2), guest (2), 1234 (2), and others

**Valid Users:**
- fztu — the only user with a successful login
- root — likely a valid user that's targeted (check for "Failed password for root" vs "Failed password for invalid user root")

**Username Patterns:**
- Common service accounts: admin, oracle, support, webmaster
- Default credentials: test, guest, 1234, 0
- Application-specific: matlab, inspur
- This is clearly a dictionary attack using common username lists

**Risk Assessment:**
- "admin" and "root" are highest risk — if compromised, full system access
- "oracle" suggests attackers know this is likely a Linux server running databases
- Numeric usernames like "0" and "1234" indicate automated/scripted attacks

Acceptable variations:
- The distinction between valid and invalid users depends on parsing "invalid user" messages
- Some usernames may be ambiguous
- Risk assessment language will vary

---

## Grading Criteria

- [ ] `user_activity_report.md` is created in the workspace
- [ ] Both valid and invalid usernames are listed
- [ ] The most-targeted username (admin) is identified
- [ ] Username patterns are analyzed (dictionary attack, common defaults)
- [ ] A risk assessment is provided for the most dangerous usernames

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the SSH user activity report task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "user_activity_report.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "usernames_listed": 0.0,
            "admin_targeted": 0.0,
            "patterns_analyzed": 0.0,
            "risk_assessment": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Both valid and invalid usernames listed
    invalid_users = ["admin", "oracle", "support", "test", "webmaster", "guest"]
    valid_users = ["fztu"]
    invalid_found = sum(1 for u in invalid_users if u in content)
    valid_found = sum(1 for u in valid_users if u in content)
    scores["usernames_listed"] = (
        1.0 if invalid_found >= 3 and valid_found >= 1 else
        0.5 if invalid_found >= 2 else 0.0
    )

    # Check 2: Admin identified as most targeted
    scores["admin_targeted"] = (
        1.0 if "admin" in content and any(kw in content for kw in
            ["most", "top", "highest", "18", "target"]) else
        0.5 if "admin" in content else 0.0
    )

    # Check 3: Username patterns analyzed
    pattern_keywords = ["dictionary", "common", "default", "service account",
                        "automated", "wordlist", "brute", "pattern"]
    scores["patterns_analyzed"] = (
        1.0 if sum(1 for kw in pattern_keywords if kw in content) >= 2 else
        0.5 if sum(1 for kw in pattern_keywords if kw in content) >= 1 else 0.0
    )

    # Check 4: Risk assessment
    risk_keywords = ["risk", "danger", "critical", "compromise", "privilege",
                     "escalat", "root access", "full access"]
    scores["risk_assessment"] = (
        1.0 if sum(1 for kw in risk_keywords if kw in content) >= 2 else
        0.5 if sum(1 for kw in risk_keywords if kw in content) >= 1 else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- ~100 "Invalid user" entries with various usernames
- "admin" is tried 18 times — the most popular target
- User "fztu" is the only confirmed valid user (successful login)
- "root" appears in failed password attempts but NOT as "invalid user" — suggesting root is a real account
- The username list reads like a standard SSH brute-force dictionary

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
