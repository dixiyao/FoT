---
id: task_log_mapreduce_jobs
name: MapReduce Log - Job Completion Summary
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "mapreduce.log"
    source: "logs/hadoop_mapreduce.log"
---

# MapReduce Log - Job Completion Summary

## Prompt

Analyze the Hadoop MapReduce application log at `mapreduce.log` and produce a comprehensive job completion summary. The log is from a MapReduce v2 (YARN) application.

Your report should include:

1. **Job Identification**: Job ID, application attempt ID, and job name/type
2. **Job Configuration**: OutputCommitter type, file system, and any other configuration details
3. **Task Summary**: Total map tasks, total reduce tasks, how many of each completed successfully
4. **Task Completion Timeline**: When did each task complete? Create a timeline showing the order of task completions with timestamps
5. **Job Duration**: Total job runtime from start to finish
6. **Final Status**: Did the job succeed or fail? What was the final transition?

Write the report to `job_completion_report.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should parse 1282 log entries and produce:

**Job Identification:**
- Job ID: job_1445062781478_0011
- Application Attempt: appattempt_1445062781478_0011_000001
- Job type: pagerank (visible in history file path)
- User: msrabi

**Configuration:**
- OutputCommitter: FileOutputCommitter
- File system: hdfs://msra-sa-41:9000
- API: mapred newApiCommitter

**Task Summary:**
- 10 map tasks (m_000000 through m_000009), plus 2 retries (m_000006_1, m_000007_1)
- 1 reduce task (r_000000)
- All 11 tasks completed successfully (10 map + 1 reduce)
- Total of 12 map task attempts, 1 reduce task attempt

**Timeline:**
- Job start: 15:37:56
- First map completion: 15:39:24 (m_000009)
- Last map completion: 15:41:25 (m_000006)
- Reduce completion: 15:42:46 (r_000000)
- Job finish: ~15:42:47

**Duration:** ~5 minutes (15:37:56 to 15:42:47)

**Final Status:** SUCCEEDED

Acceptable variations:
- Timeline formatting may differ
- Duration calculation approach may vary
- Task numbering notation may differ

---

## Grading Criteria

- [ ] `job_completion_report.md` is created in the workspace
- [ ] Job ID (job_1445062781478_0011) is identified
- [ ] Map and reduce task counts are correct (10 map tasks, 1 reduce task)
- [ ] Job duration is calculated (~5 minutes)
- [ ] Final status is identified as SUCCEEDED

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the MapReduce job completion summary task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "job_completion_report.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "job_id": 0.0,
            "task_counts": 0.0,
            "duration": 0.0,
            "final_status": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Job ID identified
    scores["job_id"] = (
        1.0 if "job_1445062781478_0011" in content or "1445062781478_0011" in content else 0.0
    )

    # Check 2: Task counts correct
    has_10_map = any(kw in content for kw in ["10 map", "ten map", "10 mapper"])
    has_1_reduce = any(kw in content for kw in ["1 reduce", "one reduce", "single reduce",
                                                  "1 reducer"])
    scores["task_counts"] = (
        1.0 if has_10_map and has_1_reduce else
        0.5 if has_10_map or has_1_reduce else 0.0
    )

    # Check 3: Duration calculated
    duration_keywords = ["5 minute", "~5 min", "4 minute", "4:51", "4:50",
                         "approximately 5", "about 5", "15:37", "15:42",
                         "nearly 5"]
    scores["duration"] = (
        1.0 if sum(1 for kw in duration_keywords if kw in content) >= 1 else 0.0
    )

    # Check 4: Final status
    scores["final_status"] = (
        1.0 if "succeeded" in content else
        0.5 if "success" in content or "completed" in content else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- Single MapReduce job: a pagerank computation by user "msrabi"
- YARN-based (v2 MapReduce) on a cluster with namenode msra-sa-41
- 12 map task attempts for 10 map tasks (m_000006 and m_000007 each had retries)
- The job history file confirms SUCCEEDED status with 10 maps and 1 reduce
- October 17, 2015
- Container allocation visible: 13 unique containers used

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
