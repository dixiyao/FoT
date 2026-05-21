---
id: task_log_mapreduce_slow_tasks
name: MapReduce Log - Slow Task Identification
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "mapreduce.log"
    source: "logs/hadoop_mapreduce.log"
---

# MapReduce Log - Slow Task Identification

## Prompt

Analyze the Hadoop MapReduce application log at `mapreduce.log` and identify which map and reduce tasks were slowest. Compare task completion times to find stragglers.

Your report should include:

1. **Task Completion Times**: For each completed task, calculate the time from container assignment to task completion
2. **Fastest vs Slowest**: Identify the fastest and slowest map tasks, and the reduce task timing
3. **Straggler Analysis**: Are there any tasks that took significantly longer than average? Quantify the deviation
4. **Retry Impact**: For tasks that were retried (attempt > 0), how did the retry time compare to the original?
5. **Reduce Phase Timing**: When did the reduce task start relative to map completions? How long did it take?
6. **Bottleneck Identification**: What was the critical path? Which task(s) determined the overall job duration?

Write the report to `slow_tasks_report.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should extract task completion timestamps and calculate:

**Map Task Completions (in order):**
1. m_000009: completed 15:39:24 (first)
2. m_000005: completed 15:40:28
3. m_000003: completed 15:40:32
4. m_000000: completed 15:40:34
5. m_000001: completed 15:40:50
6. m_000002: completed 15:40:50
7. m_000004: completed 15:40:50
8. m_000008: completed 15:40:52
9. m_000007: completed 15:41:12 (retry — _1 attempt)
10. m_000006: completed 15:41:25 (retry — _1 attempt, slowest/last)

**Reduce Task:**
- r_000000: completed 15:42:46

**Key findings:**
- m_000009 completed first at 15:39:24 — ~1.5 minutes after job start
- m_000006 completed last at 15:41:25 — ~3.5 minutes after job start (it's a retry)
- The retried tasks (m_000006, m_000007) were the slowest because they had to restart
- Spread between first and last map: ~2 minutes
- Reduce started after enough maps completed and finished about 1.3 minutes later

Acceptable variations:
- Exact durations depend on which timestamps are used as start reference
- Different definitions of "task start" are acceptable
- Straggler threshold may vary

---

## Grading Criteria

- [ ] `slow_tasks_report.md` is created in the workspace
- [ ] Individual task completion times are listed
- [ ] Fastest and slowest map tasks are identified
- [ ] Retried tasks (m_000006, m_000007) are flagged as slower
- [ ] The reduce task timing is analyzed separately

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the MapReduce slow task identification task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "slow_tasks_report.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "completion_times": 0.0,
            "fastest_slowest": 0.0,
            "retries_flagged": 0.0,
            "reduce_timing": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Task completion times listed
    task_ids = ["m_000009", "m_000005", "m_000003", "m_000000", "m_000006"]
    tasks_found = sum(1 for t in task_ids if t in content)
    scores["completion_times"] = (
        1.0 if tasks_found >= 4 else
        0.5 if tasks_found >= 2 else 0.0
    )

    # Check 2: Fastest and slowest identified
    has_fastest = any(kw in content for kw in ["fastest", "first to complete",
                                                 "earliest", "quickest"])
    has_slowest = any(kw in content for kw in ["slowest", "last to complete",
                                                 "longest", "straggler"])
    scores["fastest_slowest"] = (
        1.0 if has_fastest and has_slowest else
        0.5 if has_fastest or has_slowest else 0.0
    )

    # Check 3: Retried tasks flagged
    has_retry = any(kw in content for kw in ["retry", "retried", "reattempt",
                                               "second attempt", "_1"])
    has_slow_retry = any(kw in content for kw in ["m_000006", "m_000007"])
    scores["retries_flagged"] = (
        1.0 if has_retry and has_slow_retry else
        0.5 if has_retry else 0.0
    )

    # Check 4: Reduce timing analyzed
    has_reduce = "r_000000" in content or "reduce" in content
    has_reduce_time = any(t in content for t in ["15:42", "42:46"])
    scores["reduce_timing"] = (
        1.0 if has_reduce and has_reduce_time else
        0.5 if has_reduce else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- Job started at 15:37:56, ended at ~15:42:47
- Map tasks were assigned containers starting around 15:38:00
- Reduce slow start threshold was not met until enough maps completed
- Two tasks (m_000006, m_000007) failed on first attempt and succeeded on retry
- The retries added ~30-55 seconds to total map phase time
- The critical path runs through the last map completion (m_000006 at 15:41:25) plus the reduce phase

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
