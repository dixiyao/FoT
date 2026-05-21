---
id: task_log_hdfs_failures
name: HDFS DataNode Log - Block and Replication Failure Analysis
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "hdfs_datanode.log"
    source: "logs/hdfs_datanode.log"
---

# HDFS DataNode Log - Block and Replication Failure Analysis

## Prompt

Analyze the HDFS DataNode log at `hdfs_datanode.log` and identify any block operation failures, replication issues, or error conditions. The log is from an HDFS cluster and contains DataNode, FSNamesystem, and PacketResponder entries.

Your report should include:

1. **Log Overview**: Total entries, date/time range, log level distribution (INFO, WARN, ERROR)
2. **Block Operation Summary**: Count of block receives, allocations, stored block confirmations, and replications
3. **Error and Warning Analysis**: List any WARN or ERROR level entries with details
4. **Replication Activity**: Detail all replication requests — which blocks are being replicated, from where to where?
5. **Failed or Incomplete Operations**: Are there any blocks where receive started but confirmation was never logged?
6. **Health Assessment**: Based on the log, is the HDFS cluster operating normally?

Write the report to `hdfs_failure_report.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should parse 2000 log entries and produce:

**Log Overview:**
- 2000 entries, all from November 9, 2008 (081109), covering ~28 seconds (203518–203546)
- All entries are INFO level — no WARN or ERROR entries

**Block Operations:**
- Receiving block: ~1149 entries
- Allocate block: ~385 entries
- Received block (confirmed): ~19 entries
- addStoredBlock: ~19 entries
- PacketResponder: ~12 entries
- Replication requests: 4

**Replication Details:**
- Block blk_-1608999687919862906 has 4 replication requests:
  - 10.250.14.224 → 10.251.215.16
  - 10.251.215.16 → 10.251.74.79
  - 10.251.107.19 → 10.251.31.5
  - 10.251.31.5 → 10.251.90.64

**Health Assessment:**
- No errors or warnings — the cluster appears healthy
- The large number of "Receiving block" entries (1149) with relatively few confirmations (19) suggests high concurrency
- Replication activity for a single block across multiple nodes is normal HDFS behavior

Acceptable variations:
- Exact counts may differ slightly depending on parsing approach
- Assessment language will vary
- The distinction between "no failures" and "potential incomplete operations" is valid

---

## Grading Criteria

- [ ] `hdfs_failure_report.md` is created in the workspace
- [ ] Log overview with entry count and time range is provided
- [ ] Block operations are categorized and counted (receive, allocate, replicate)
- [ ] The absence of WARN/ERROR entries is noted (or any found are detailed)
- [ ] A health assessment is provided

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the HDFS failure analysis task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "hdfs_failure_report.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "log_overview": 0.0,
            "operations_counted": 0.0,
            "error_status": 0.0,
            "health_assessment": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Log overview
    has_count = any(n in content for n in ["2000", "2,000"])
    has_date = any(d in content for d in ["081109", "november 9", "nov 9", "2008-11-09", "nov 2008"])
    scores["log_overview"] = (
        1.0 if has_count and has_date else
        0.5 if has_count or has_date else 0.0
    )

    # Check 2: Operations categorized
    op_keywords = ["receiving", "allocate", "replicate", "addstored",
                   "packetresponder", "block operation"]
    ops_found = sum(1 for kw in op_keywords if kw in content)
    scores["operations_counted"] = (
        1.0 if ops_found >= 3 else
        0.5 if ops_found >= 2 else 0.0
    )

    # Check 3: Error status noted
    error_keywords = ["no error", "no warn", "all info", "no failures",
                      "0 error", "0 warn", "no warning", "entirely info"]
    scores["error_status"] = (
        1.0 if sum(1 for kw in error_keywords if kw in content) >= 1 else
        0.5 if "info" in content else 0.0
    )

    # Check 4: Health assessment
    health_keywords = ["healthy", "normal", "operating correctly", "no issues",
                       "good health", "stable", "functioning"]
    scores["health_assessment"] = (
        1.0 if sum(1 for kw in health_keywords if kw in content) >= 1 else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- Format: `YYMMDD HHMMSS threadID LEVEL component: message`
- Date: November 9, 2008 (081109), times 203518–203546 (~28 seconds of activity)
- 202 unique IP addresses in the cluster
- 390 unique block IDs
- Block sizes range from 11,971 to 233,217 bytes
- This is a burst of HDFS activity — likely a MapReduce job starting (job_200811092030_0001 visible in paths)

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
