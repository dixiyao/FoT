---
id: task_log_hdfs_slow_ops
name: HDFS DataNode Log - Slow Operation Detection
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "hdfs_datanode.log"
    source: "logs/hdfs_datanode.log"
---

# HDFS DataNode Log - Slow Operation Detection

## Prompt

Analyze the HDFS DataNode log at `hdfs_datanode.log` and identify operations that took longer than expected. The log records block receives, allocations, and replications with timestamps.

Your report should include:

1. **Block Lifecycle Timing**: For blocks where both "Receiving block" and "Received block" entries exist, calculate the elapsed time
2. **Allocation-to-Receive Timing**: For blocks where both "allocateBlock" and first "Receiving block" entries exist, calculate the delay
3. **Replication Timing**: How quickly are replication requests issued after block allocation?
4. **Slowest Operations**: Rank the top 5 slowest block operations by elapsed time
5. **Block Size vs Time Correlation**: Do larger blocks take longer? Correlate block size with transfer time where both are available
6. **Performance Summary**: Overall assessment of cluster performance during this period

Write the report to `hdfs_slow_ops_report.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should parse timestamps from the log format `YYMMDD HHMMSS` and calculate:

**Block Lifecycle:**
- The log covers only ~28 seconds (203518 to 203546)
- Most block operations complete within 1-3 seconds
- Confirmed block receives with sizes: 91178 bytes, 233217 bytes, 11971 bytes, 11977 bytes

**Key observations:**
- blk_-1608999687919862906 (91178 bytes): Allocated at 203518, first receive at 203518, confirmed at 203519 (~1 second)
- blk_7503483334202473044 (233217 bytes): Allocated at 203520, confirmed at 203521 (~1 second)
- blk_-3544583377289625738 (11971 bytes): Confirmed at 203522-203523
- Block operations are very fast — consistent with a healthy cluster under normal load

**Replication:**
- 4 replication requests for blk_-1608999687919862906, issued at 203521, 203524, 203527, 203530
- ~3 second intervals between replication hops

**Performance:**
- All operations complete in 1-3 seconds — no slow operations detected
- The cluster is performing well during this snapshot

Acceptable variations:
- Timestamp resolution is 1 second, so some timing analysis will be approximate
- Different approaches to matching start/end events are valid
- Block size correlation may show insufficient data for meaningful analysis

---

## Grading Criteria

- [ ] `hdfs_slow_ops_report.md` is created in the workspace
- [ ] Block lifecycle timing is calculated for at least one block
- [ ] Block sizes are correlated with operation times where data is available
- [ ] The time range of the log is correctly identified (~28 seconds)
- [ ] A performance assessment is provided

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the HDFS slow operation detection task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "hdfs_slow_ops_report.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "lifecycle_timing": 0.0,
            "size_correlation": 0.0,
            "time_range": 0.0,
            "performance_assessment": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Block lifecycle timing calculated
    has_timing = any(kw in content for kw in ["1 second", "2 second", "3 second",
                                                "elapsed", "duration", "latency",
                                                "took", "completed in"])
    has_block = "blk_" in content or "blk-" in content or "block" in content
    scores["lifecycle_timing"] = (
        1.0 if has_timing and has_block else
        0.5 if has_timing or has_block else 0.0
    )

    # Check 2: Block sizes mentioned
    sizes = ["91178", "233217", "11971", "11977"]
    sizes_found = sum(1 for s in sizes if s in content)
    has_correlation = any(kw in content for kw in ["size", "bytes", "larger", "smaller"])
    scores["size_correlation"] = (
        1.0 if sizes_found >= 2 and has_correlation else
        0.5 if sizes_found >= 1 else 0.0
    )

    # Check 3: Time range identified
    has_28s = any(kw in content for kw in ["28 second", "~28", "30 second",
                                            "half a minute", "less than a minute"])
    has_timestamps = "203518" in content or "20:35:18" in content or "20:35" in content
    scores["time_range"] = (
        1.0 if has_28s or has_timestamps else
        0.5 if any(kw in content for kw in ["short", "brief", "seconds"]) else 0.0
    )

    # Check 4: Performance assessment
    perf_keywords = ["healthy", "normal", "fast", "no slow", "performing well",
                     "efficient", "optimal", "good performance"]
    scores["performance_assessment"] = (
        1.0 if sum(1 for kw in perf_keywords if kw in content) >= 1 else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- Timestamp format: YYMMDD HHMMSS (e.g., 081109 203518 = 2008-11-09 20:35:18)
- Resolution: 1 second — so sub-second timing is not available
- 390 unique blocks, but only ~19 have confirmed "Received" entries with sizes
- The cluster is in a burst of activity (job startup), so performance is under load
- No errors or warnings suggest all operations completed successfully

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
