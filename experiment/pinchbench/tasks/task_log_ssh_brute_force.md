---
id: task_log_ssh_brute_force
name: SSH Auth Log - Brute Force Detection
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "auth.log"
    source: "logs/openssh_auth.log"
---

# SSH Auth Log - Brute Force Detection

## Prompt

You are a security analyst reviewing the OpenSSH authentication log at `auth.log`. Your job is to detect brute-force attack patterns and produce a threat assessment.

Define a brute-force attack as: **more than 10 failed authentication attempts from a single IP address within the log period**.

Your report should include:

1. **Brute Force Sources**: List all IPs that meet the brute-force threshold, with total failed attempts per IP
2. **Attack Intensity**: For each brute-force source, calculate the approximate rate of attempts (attempts per minute)
3. **Username Patterns**: For each attacking IP, what usernames are they trying? Is it a dictionary attack (many usernames) or targeted (few usernames)?
4. **Attack Timeline**: When did each attack start and stop? Any overlap between attackers?
5. **Reverse DNS Analysis**: Which attacking IPs triggered "POSSIBLE BREAK-IN ATTEMPT" warnings?
6. **Risk Assessment**: Rate the overall threat level and recommend specific countermeasures

Write the report to `brute_force_report.json` as a JSON document with the following structure:

```json
{
  "summary": "Brief summary",
  "brute_force_sources": [
    {
      "ip": "x.x.x.x",
      "total_attempts": 100,
      "first_seen": "Dec 10 HH:MM:SS",
      "last_seen": "Dec 10 HH:MM:SS",
      "usernames_tried": ["user1", "user2"],
      "attack_type": "dictionary|targeted",
      "reverse_dns_warning": true
    }
  ],
  "risk_level": "critical|high|medium|low",
  "recommendations": ["rec1", "rec2"]
}
```

---

## Expected Behavior

The agent should identify these brute-force sources:

**Primary Attackers:**
- **183.62.140.253** — ~307 entries, heaviest attacker, likely dictionary attack
- **187.141.143.180** — ~189 entries, sustained attack
- **103.99.0.122** — ~83 entries
- **112.95.230.3** — ~54 entries
- **5.188.10.180** — ~30 entries
- **185.190.58.151** — ~26 entries

**Key findings:**
- Multiple concurrent brute-force attacks from different IPs
- Attacks span approximately 4 hours (06:55–10:59)
- Username patterns include common defaults (admin, root, test, oracle, support)
- 85 "POSSIBLE BREAK-IN ATTEMPT" warnings indicate spoofed/misconfigured reverse DNS
- Risk level should be assessed as high or critical

Acceptable variations:
- Threshold for brute-force detection may vary
- Rate calculations depend on how first/last timestamps are determined
- Recommendation specifics will vary

---

## Grading Criteria

- [ ] `brute_force_report.json` is created in the workspace
- [ ] At least 3 brute-force source IPs are identified
- [ ] 183.62.140.253 is identified as the top attacker
- [ ] Attack type (dictionary vs targeted) is classified for each source
- [ ] Recommendations for countermeasures are provided

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the SSH brute force detection task."""
    from pathlib import Path
    import json

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "brute_force_report.json"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "sources_identified": 0.0,
            "top_attacker": 0.0,
            "attack_classified": 0.0,
            "recommendations": 0.0,
        }

    scores["output_created"] = 1.0

    try:
        data = json.loads(report_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, Exception):
        return {
            "output_created": 1.0,
            "sources_identified": 0.0,
            "top_attacker": 0.0,
            "attack_classified": 0.0,
            "recommendations": 0.0,
        }

    full_text = json.dumps(data).lower()

    # Check 1: At least 3 brute-force sources identified
    sources = data.get("brute_force_sources", [])
    if not isinstance(sources, list):
        sources = []
    scores["sources_identified"] = (
        1.0 if len(sources) >= 3 else
        0.5 if len(sources) >= 1 else 0.0
    )

    # Check 2: Top attacker identified
    scores["top_attacker"] = 1.0 if "183.62.140.253" in full_text else 0.0

    # Check 3: Attack type classified
    has_classification = "dictionary" in full_text or "targeted" in full_text or "attack_type" in full_text
    scores["attack_classified"] = 1.0 if has_classification else 0.0

    # Check 4: Recommendations provided
    recs = data.get("recommendations", [])
    if not isinstance(recs, list):
        recs = []
    has_recs = len(recs) >= 2 or any(kw in full_text for kw in
        ["fail2ban", "rate limit", "firewall", "block", "key-based",
         "disable password", "allowlist", "whitelist", "deny"])
    scores["recommendations"] = 1.0 if has_recs else 0.5 if len(recs) >= 1 else 0.0

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- Server: LabSZ, running OpenSSH with PAM
- Attack window: Dec 10, 06:55 to 10:59 (~4 hours)
- Multiple simultaneous attackers — suggests the server IP is on a known scan list
- 183.62.140.253 generates about 75 attempts per hour on average
- The single successful login (user fztu from 119.137.62.142) is NOT from an attacking IP

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
