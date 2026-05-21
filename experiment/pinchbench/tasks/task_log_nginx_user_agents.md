---
id: task_log_nginx_user_agents
name: Nginx Access Log - User Agent Analysis
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "nginx_access.log"
    source: "logs/nginx_access_json.log"
---

# Nginx Access Log - User Agent Analysis

## Prompt

Analyze the Nginx JSON access log at `nginx_access.log` and produce a comprehensive user agent analysis. Each line is a JSON object with fields: `time`, `remote_ip`, `remote_user`, `request`, `response`, `bytes`, `referrer`, `agent`.

Your report should include:

1. **Unique User Agents**: Total count of distinct user agent strings
2. **User Agent Ranking**: List all user agents sorted by request count, with count and percentage
3. **Client Type Classification**: Categorize agents into types (package managers, web browsers, bots/crawlers, command-line tools, unknown/empty)
4. **Agent-to-IP Mapping**: For each user agent, how many unique IPs use it?
5. **Success vs Error Rate by Agent**: For each agent, what percentage of requests result in errors (4xx/5xx)?
6. **Conclusions**: What type of server is this based on the user agent profile?

Write the report to `user_agent_report.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should parse all 1000 JSON log entries and produce:

**Unique User Agents:** 14 distinct agent strings (including "-" for empty)

**Top User Agents:**
- `Debian APT-HTTP/1.3 (0.9.7.9)` — 370 requests (37.0%)
- `Debian APT-HTTP/1.3 (0.8.16~exp12ubuntu10.16)` — 177 (17.7%)
- `Debian APT-HTTP/1.3 (0.8.16~exp12ubuntu10.22)` — 118 (11.8%)
- `Debian APT-HTTP/1.3 (1.0.1ubuntu2)` — 116 (11.6%)
- `Debian APT-HTTP/1.3 (0.8.16~exp12ubuntu10.21)` — 64 (6.4%)
- Additional APT variants and a few others (Go 1.1 package http, urlgrabber, etc.)

**Classification:**
- Package managers (Debian APT): vast majority (~95%+)
- Other automated tools: Go HTTP client, urlgrabber
- Empty/missing agent ("-"): small number

**Conclusions:**
- This is clearly a Debian/Ubuntu package repository or software download mirror
- Multiple APT versions indicate clients running different Ubuntu/Debian releases
- Very little if any human browser traffic

Acceptable variations:
- Exact counts are deterministic from the log
- Classification categories may use different names
- Assessment language will vary

---

## Grading Criteria

- [ ] `user_agent_report.md` is created in the workspace
- [ ] All user agents are listed with counts
- [ ] Agents are classified by type (package manager, bot, etc.)
- [ ] The dominant agent (Debian APT) is identified as the primary client
- [ ] Server purpose is correctly inferred (package repository/download mirror)

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the Nginx user agent analysis task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "user_agent_report.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "agents_listed": 0.0,
            "agents_classified": 0.0,
            "apt_dominant": 0.0,
            "server_purpose": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: User agents listed with counts
    has_apt = "apt-http" in content or "apt http" in content or "debian apt" in content
    has_counts = any(str(c) in content for c in ["370", "177", "118", "116"])
    scores["agents_listed"] = (
        1.0 if has_apt and has_counts else
        0.5 if has_apt else 0.0
    )

    # Check 2: Agents classified by type
    type_keywords = ["package manager", "bot", "crawler", "automated", "tool",
                     "browser", "command line", "cli", "client type", "categor"]
    scores["agents_classified"] = (
        1.0 if sum(1 for kw in type_keywords if kw in content) >= 2 else
        0.5 if sum(1 for kw in type_keywords if kw in content) >= 1 else 0.0
    )

    # Check 3: APT identified as dominant
    dominant_keywords = ["dominant", "majority", "most common", "primary",
                         "most frequent", "largest", "overwhelming"]
    has_dominant = any(kw in content for kw in dominant_keywords)
    scores["apt_dominant"] = (
        1.0 if has_apt and has_dominant else
        0.5 if has_apt else 0.0
    )

    # Check 4: Server purpose inferred
    purpose_keywords = ["repository", "mirror", "download", "package",
                        "software", "debian", "ubuntu", "apt"]
    scores["server_purpose"] = (
        1.0 if sum(1 for kw in purpose_keywords if kw in content) >= 3 else
        0.5 if sum(1 for kw in purpose_keywords if kw in content) >= 1 else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- 14 unique user agent strings
- Over 95% of traffic is from Debian APT package managers
- Multiple APT versions correspond to different Ubuntu/Debian releases
- The presence of `Go 1.1 package http` and `urlgrabber` agents indicates some non-APT automated traffic
- Some entries have agent "-" (empty/missing user agent)

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
