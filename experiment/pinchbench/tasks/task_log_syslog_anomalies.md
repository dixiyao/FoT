---
id: task_log_syslog_anomalies
name: Linux Syslog - Anomaly Detection
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "syslog.log"
    source: "logs/linux_syslog.log"
---

# Linux Syslog - Anomaly Detection

## Prompt

Analyze the Linux syslog at `syslog.log` and identify anomalous or suspicious entries. The log is from a server named "combo" running a Linux 2.6 kernel, covering several months of activity.

Your report should include:

1. **Log Overview**: Total entries, date range, top services by volume
2. **Security Anomalies**: Entries that indicate potential attacks, exploits, or unauthorized access attempts
3. **Format String Attack Detection**: Look for entries with unusual binary content or exploit payloads in service input
4. **FTP Anomalies**: The log has heavy FTP traffic — identify any suspicious FTP connection patterns (bursts, unusual sources)
5. **rpc.statd Exploitation**: Check for rpc.statd gethostbyname errors with malformed hostnames (buffer overflow attempts)
6. **Anomaly Summary**: Rank the top 5 most concerning anomalies with severity and evidence

Write the report to `syslog_anomalies.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should parse 5000 entries and identify:

**Log Overview:**
- 5000 entries, June 9 to September 14 (2005, based on kernel version)
- Top services: ftpd (1655), sshd/pam_unix (1610), kernel (545), su/pam_unix (394)

**Security Anomalies:**
1. **rpc.statd format string attack** (~9 entries on Jun 13):
   - `gethostbyname error for ^X...%8x%8x...%hn%51859x%hn` — this is a buffer overflow/format string exploitation attempt against rpc.statd
   - The payload contains format string specifiers (%x, %hn) which are classic exploit patterns
2. **SSH brute force** — heavy sshd(pam_unix) authentication failure volume
3. **FTP flood** — 1655 FTP connection entries, with bursts (e.g., 209.184.7.130 with multiple simultaneous connections)
4. **Authentication failures** — 2000+ pam_unix auth failure entries across SSH and other services

**rpc.statd Exploitation:**
- 9 entries at Jun 13 11:55:04–11:55:09
- Malformed hostname contains NOP sled (\220\220\220\220) and format string payload
- This is an attempted remote code execution exploit

Acceptable variations:
- Anomaly ranking may differ
- Additional anomalies beyond the expected ones are welcome
- Severity assessments will vary

---

## Grading Criteria

- [ ] `syslog_anomalies.md` is created in the workspace
- [ ] Log overview with date range and service breakdown is provided
- [ ] rpc.statd format string attack is identified as a security anomaly
- [ ] FTP connection patterns are analyzed
- [ ] SSH authentication failures are flagged

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the Linux syslog anomaly detection task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "syslog_anomalies.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "log_overview": 0.0,
            "rpc_statd_attack": 0.0,
            "ftp_analysis": 0.0,
            "ssh_failures": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Log overview
    has_count = any(n in content for n in ["5000", "5,000"])
    has_date = any(d in content for d in ["jun", "june", "sep", "september"])
    scores["log_overview"] = (
        1.0 if has_count and has_date else
        0.5 if has_count or has_date else 0.0
    )

    # Check 2: rpc.statd attack identified
    rpc_keywords = ["rpc.statd", "rpc statd", "gethostbyname", "format string",
                    "buffer overflow", "exploit", "%hn", "nop sled", "\\220"]
    scores["rpc_statd_attack"] = (
        1.0 if sum(1 for kw in rpc_keywords if kw in content) >= 2 else
        0.5 if sum(1 for kw in rpc_keywords if kw in content) >= 1 else 0.0
    )

    # Check 3: FTP analysis
    ftp_keywords = ["ftpd", "ftp", "1655", "209.184", "connection flood",
                    "burst", "ftp connection"]
    scores["ftp_analysis"] = (
        1.0 if sum(1 for kw in ftp_keywords if kw in content) >= 2 else
        0.5 if sum(1 for kw in ftp_keywords if kw in content) >= 1 else 0.0
    )

    # Check 4: SSH failures flagged
    ssh_keywords = ["sshd", "ssh", "authentication failure", "brute force",
                    "failed", "pam_unix"]
    scores["ssh_failures"] = (
        1.0 if sum(1 for kw in ssh_keywords if kw in content) >= 2 else
        0.5 if sum(1 for kw in ssh_keywords if kw in content) >= 1 else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- Server: "combo", Linux 2.6.5-1.358, Fedora/Red Hat based
- Date range: Jun 9 to Sep 14 (year 2005, based on kernel build date)
- The rpc.statd attack on Jun 13 is the most serious anomaly — a real exploit attempt
- 1655 FTP connections, heavily concentrated from certain IPs (209.184.7.130)
- The su(pam_unix) entries (394) show regular privilege escalation — likely legitimate cron jobs
- Some entries contain non-UTF-8 bytes (binary exploit payloads)

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
