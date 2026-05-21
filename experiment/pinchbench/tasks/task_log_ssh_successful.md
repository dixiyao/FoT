---
id: task_log_ssh_successful
name: SSH Auth Log - Successful Authentication Summary
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "auth.log"
    source: "logs/openssh_auth.log"
---

# SSH Auth Log - Successful Authentication Summary

## Prompt

Analyze the OpenSSH authentication log at `auth.log` and produce a report focused on successful authentications. Among the noise of failed attempts, identify all legitimate access.

Your report should include:

1. **Successful Logins**: List every successful authentication with timestamp, username, source IP, port, and authentication method
2. **Success vs Failure Ratio**: What percentage of all authentication attempts succeeded?
3. **Legitimate User Profile**: For each successfully authenticated user, describe their access pattern
4. **Session Activity**: Any evidence of what happened after login (session opened/closed events)?
5. **Source IP Validation**: Is the successful login IP associated with any failed attempts as well?
6. **Anomaly Check**: Does the successful login appear legitimate, or does it look suspicious (e.g., coming from an IP that was also brute-forcing)?

Write the report to `successful_auth_report.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should identify:

**Successful Logins:**
- Only 1 successful login in the entire log:
  - Time: Dec 10 09:32:20
  - User: fztu
  - Source: 119.137.62.142
  - Port: 49116
  - Method: password (ssh2)
  - Entry: "Accepted password for fztu from 119.137.62.142 port 49116 ssh2"

**Success vs Failure Ratio:**
- 1 success out of hundreds of attempts — extremely low success rate
- This reinforces that the log captures a server under brute-force attack

**Anomaly Check:**
- 119.137.62.142 (the successful login IP) should be checked against the failed attempt list
- If it appears only in the successful entry, it's likely a legitimate user
- If it also has failed attempts, it could be a compromised credential

**Session Activity:**
- Look for corresponding "session opened" / "session closed" events for user fztu

Acceptable variations:
- Analysis depth may vary
- Some agents may find additional session-related entries
- Anomaly assessment wording will vary

---

## Grading Criteria

- [ ] `successful_auth_report.md` is created in the workspace
- [ ] The single successful login is identified (user fztu, IP 119.137.62.142)
- [ ] Success/failure ratio is calculated (1 success vs hundreds of failures)
- [ ] The successful login IP is checked against failed attempt sources
- [ ] An assessment of whether the login appears legitimate is provided

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the SSH successful authentication summary task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "successful_auth_report.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "login_identified": 0.0,
            "ratio_calculated": 0.0,
            "ip_checked": 0.0,
            "legitimacy_assessed": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Successful login identified
    has_user = "fztu" in content
    has_ip = "119.137.62.142" in content
    has_accepted = "accepted" in content or "successful" in content
    scores["login_identified"] = (
        1.0 if has_user and has_ip else
        0.5 if has_user or has_ip else 0.0
    )

    # Check 2: Ratio calculated
    ratio_keywords = ["ratio", "percent", "1 success", "1 out of", "only 1",
                      "single success", "one success", "0."]
    scores["ratio_calculated"] = (
        1.0 if sum(1 for kw in ratio_keywords if kw in content) >= 1 else 0.0
    )

    # Check 3: IP checked against failed attempts
    check_keywords = ["119.137.62.142", "not associated", "not found",
                      "does not appear", "no failed", "legitimate",
                      "only successful", "no other"]
    scores["ip_checked"] = (
        1.0 if has_ip and sum(1 for kw in check_keywords if kw in content) >= 2 else
        0.5 if has_ip else 0.0
    )

    # Check 4: Legitimacy assessment
    legit_keywords = ["legitimate", "authorized", "valid", "genuine",
                      "suspicious", "anomal", "normal", "expected"]
    scores["legitimacy_assessed"] = (
        1.0 if sum(1 for kw in legit_keywords if kw in content) >= 1 else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- Only 1 successful login among 1500 entries
- User "fztu" authenticated via password from 119.137.62.142:49116 at 09:32:20
- 119.137.62.142 does NOT appear as a source of any failed attempts — this is a legitimate user
- The massive imbalance between failed and successful attempts is characteristic of a brute-force target
- PAM session events (opened/closed) may provide additional context

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
