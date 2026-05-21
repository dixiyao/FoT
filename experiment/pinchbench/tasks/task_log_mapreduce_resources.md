---
id: task_log_mapreduce_resources
name: MapReduce Log - Resource Utilization Analysis
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "mapreduce.log"
    source: "logs/hadoop_mapreduce.log"
---

# MapReduce Log - Resource Utilization Analysis

## Prompt

Analyze the Hadoop MapReduce application log at `mapreduce.log` and produce a resource utilization report. Focus on container allocation, scheduling, and resource usage patterns.

Your report should include:

1. **Container Inventory**: List all containers allocated for this job, with their IDs
2. **Container Allocation Timeline**: When was each container requested and assigned?
3. **Scheduling Analysis**: Track pending maps and reduces over time from the RMContainerAllocator entries
4. **Reduce Scheduling**: When did the reduce slow start threshold get met? What was the completion percentage?
5. **Container Reuse**: Were any containers completed and then reused?
6. **Resource Efficiency**: Based on container allocation vs task completion patterns, assess resource efficiency

Write the report to `mapreduce_resources.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should parse RMContainerAllocator entries and produce:

**Container Inventory:**
- 13 unique containers used (container_1445062781478_0011_01_000001 through 000013)
- Application attempt: 01

**Scheduling Progression:**
- Initial state: 10 pending maps, 1 pending reduce
- Reduce slow start threshold repeatedly "not met" from 15:38:00 to 15:39:24
- First map completes at 15:39:24 (10% complete) — still not enough for reduce
- Reduce scheduling begins when completedMapPercent reaches sufficient threshold
- "completedMapPercent 0.1 totalResources 2" logged at 15:39:24

**Container Lifecycle:**
- Containers allocated around 15:38:00–15:38:15
- Containers released as tasks complete
- "Received completed container" entries track when containers finish

**Key observations:**
- The reduce task had to wait for enough maps to complete (slow start)
- Map tasks had varying completion times (1.5 to 3.5 minutes)
- Container turnover: some containers were released and their resources freed quickly

Acceptable variations:
- Container ID enumeration approach may vary
- Timeline granularity may differ
- Resource efficiency assessment is subjective

---

## Grading Criteria

- [ ] `mapreduce_resources.md` is created in the workspace
- [ ] Containers are listed (13 containers identified)
- [ ] Scheduling progression is tracked (pending maps/reduces over time)
- [ ] Reduce slow start threshold discussion is included
- [ ] Container completion events are analyzed

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the MapReduce resource utilization analysis task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "mapreduce_resources.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "containers_listed": 0.0,
            "scheduling_tracked": 0.0,
            "slow_start": 0.0,
            "container_completion": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Containers listed
    has_container = "container_" in content or "container" in content
    has_count = any(n in content for n in ["13 container", "13 unique", "thirteen"])
    scores["containers_listed"] = (
        1.0 if has_container and has_count else
        0.5 if has_container else 0.0
    )

    # Check 2: Scheduling tracked
    sched_keywords = ["pending", "scheduled", "pendingreds", "pendingmaps",
                      "scheduledmaps", "scheduling"]
    scores["scheduling_tracked"] = (
        1.0 if sum(1 for kw in sched_keywords if kw in content) >= 2 else
        0.5 if sum(1 for kw in sched_keywords if kw in content) >= 1 else 0.0
    )

    # Check 3: Reduce slow start discussed
    slow_start_keywords = ["slow start", "slowstart", "threshold", "reduce.*wait",
                           "completedmappercent", "not met"]
    scores["slow_start"] = (
        1.0 if sum(1 for kw in slow_start_keywords if kw in content) >= 2 else
        0.5 if sum(1 for kw in slow_start_keywords if kw in content) >= 1 else 0.0
    )

    # Check 4: Container completion analyzed
    completion_keywords = ["completed container", "container released", "received completed",
                           "container finish", "freed"]
    scores["container_completion"] = (
        1.0 if sum(1 for kw in completion_keywords if kw in content) >= 1 else
        0.5 if "complet" in content else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- YARN-based MapReduce (v2) on cluster with RM
- 13 containers for 10 map tasks + 1 reduce + 1 AM container + retries
- The reduce slow start threshold is a standard Hadoop optimization
- "Before Scheduling" / "After Scheduling" entries provide scheduling state snapshots
- Final stats: "PendingReds:0 ScheduledMaps:0" — all resources freed

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
