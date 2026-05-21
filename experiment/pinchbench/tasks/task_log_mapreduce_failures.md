---
id: task_log_mapreduce_failures
name: MapReduce Log - Failed Task Analysis
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "mapreduce.log"
    source: "logs/hadoop_mapreduce.log"
---

# MapReduce Log - Failed Task Analysis

## Prompt

Analyze the Hadoop MapReduce application log at `mapreduce.log` and identify any task failures, errors, or anomalies. Focus on anything that went wrong during execution.

Your report should include:

1. **Error and Warning Entries**: List all WARN and ERROR level log entries with full context
2. **Task Retries**: Identify any tasks that required multiple attempts (look for attempt numbers > 0)
3. **Root Cause Analysis**: For each error, explain the likely cause
4. **I/O Errors**: Detail any IOException or network-related failures
5. **Impact Assessment**: Did any failures impact the overall job result?
6. **Failure Prevention**: Recommend changes to prevent these failures in future runs

Write the report to `mapreduce_failures.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should identify:

**WARN Entries (4 total):**
1. ResponseProcessor for block BP-1347369012-10.190.173.170-1444972147527:blk_1073742514_1708 — related to I/O issue
2. DataStreamer for file /tmp/hadoop-yarn/staging/msrabi/.staging/job_1445062781478_0011/job — write pipeline issue
3. CommitterEvent Processor — FileOutputCommitter recovery task count issue

**ERROR Entries (1 total):**
1. java.io.IOException: Bad response ERROR for block BP-1347369012-10.190.173.170-1444972147527:blk_1073742514_1708 from datanode — an I/O error during block write

**Task Retries:**
- attempt_1445062781478_0011_m_000006_1 (retry of m_000006_0)
- attempt_1445062781478_0011_m_000007_1 (retry of m_000007_0)
- Two map tasks needed a second attempt, suggesting transient failures

**Root Cause:**
- The IOException and WARN entries relate to HDFS block write failures
- A DataNode returned a "Bad response ERROR" for block blk_1073742514_1708
- This is a transient HDFS I/O error, likely caused by a DataNode being unavailable or overloaded

**Impact:**
- Despite the errors, the overall job SUCCEEDED
- YARN's retry mechanism handled the transient failures transparently
- 2 out of 10 map tasks needed retries — 20% retry rate

Acceptable variations:
- Root cause analysis depth may vary
- Prevention recommendations will differ
- Some agents may find additional context around the errors

---

## Grading Criteria

- [ ] `mapreduce_failures.md` is created in the workspace
- [ ] WARN and ERROR entries are listed (4 WARN, 1 ERROR)
- [ ] Task retries are identified (m_000006 and m_000007 retried)
- [ ] The IOException / bad response error is analyzed
- [ ] Impact assessment notes the job still succeeded

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the MapReduce failed task analysis."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "mapreduce_failures.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "warn_error_listed": 0.0,
            "retries_identified": 0.0,
            "ioexception_analyzed": 0.0,
            "impact_assessed": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: WARN/ERROR entries listed
    has_warn = "warn" in content
    has_error = "error" in content
    has_counts = any(n in content for n in ["4 warn", "4 warning", "1 error"])
    scores["warn_error_listed"] = (
        1.0 if has_warn and has_error else
        0.5 if has_warn or has_error else 0.0
    )

    # Check 2: Task retries identified
    has_006 = "m_000006" in content or "000006" in content
    has_007 = "m_000007" in content or "000007" in content
    has_retry = any(kw in content for kw in ["retry", "reattempt", "second attempt",
                                               "_1", "attempt 1"])
    scores["retries_identified"] = (
        1.0 if (has_006 or has_007) and has_retry else
        0.5 if has_retry else 0.0
    )

    # Check 3: IOException analyzed
    io_keywords = ["ioexception", "io exception", "bad response", "block write",
                   "datanode", "datastreamer", "blk_1073742514"]
    scores["ioexception_analyzed"] = (
        1.0 if sum(1 for kw in io_keywords if kw in content) >= 2 else
        0.5 if sum(1 for kw in io_keywords if kw in content) >= 1 else 0.0
    )

    # Check 4: Impact assessment
    impact_keywords = ["succeeded", "success", "still completed", "job completed",
                       "transparent", "handled", "recovered", "despite"]
    scores["impact_assessed"] = (
        1.0 if sum(1 for kw in impact_keywords if kw in content) >= 1 else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- 1206 INFO, 4 WARN, 1 ERROR entries out of 1282 total
- The ERROR is a Java IOException wrapped in a WARN-level DataStreamer message
- Block BP-1347369012-10.190.173.170-1444972147527:blk_1073742514_1708 had a write failure
- FileOutputCommitter also logged a WARN about recovery task count
- Despite these issues, all 10 map tasks and 1 reduce task eventually completed

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
