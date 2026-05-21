---
id: task_log_mapreduce_timeline
name: MapReduce Log - Job Timeline Visualization
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "mapreduce.log"
    source: "logs/hadoop_mapreduce.log"
---

# MapReduce Log - Job Timeline Visualization

## Prompt

Analyze the Hadoop MapReduce application log at `mapreduce.log` and create a detailed timeline visualization of the entire job execution. Show all major events in chronological order.

Your output should include:

1. **Event Timeline**: A chronological list of every significant event with timestamp, including:
   - Job initialization events
   - Container allocations
   - Task starts and completions
   - Errors and warnings
   - Reduce phase start
   - Job completion
2. **Phase Diagram**: Divide the job into phases (initialization, map phase, shuffle, reduce phase, cleanup) with start/end times and durations
3. **Gantt-Style Task View**: Show each task (m_000000 through m_000009, r_000000) with approximate start and end times in a text-based timeline
4. **Critical Events**: Highlight the most impactful events (errors, retries, job state transitions)
5. **Concurrency Analysis**: At each point in time, how many tasks were running in parallel?

Write the report to `mapreduce_timeline.md` as a well-structured markdown document with ASCII/text-based visualizations.

---

## Expected Behavior

The agent should produce a timeline like:

**Phase Breakdown:**
| Phase | Start | End | Duration |
|---|---|---|---|
| Initialization | 15:37:56 | 15:38:00 | ~4s |
| Map Phase | 15:38:00 | 15:41:25 | ~3m 25s |
| Reduce Phase | 15:39:24 | 15:42:46 | ~3m 22s |
| Cleanup | 15:42:46 | 15:42:47 | ~1s |
| **Total** | **15:37:56** | **15:42:47** | **~4m 51s** |

**Key Events:**
- 15:37:56 — MRAppMaster created
- 15:37:57 — OutputCommitter set (FileOutputCommitter)
- 15:38:00 — Container allocation begins (10 maps pending, 1 reduce pending)
- 15:39:24 — First map completes (m_000009), Num completed: 1
- 15:40:28–15:40:52 — Rapid map completions (tasks 2-8)
- 15:40:45 — WARN: Block I/O error (ResponseProcessor, DataStreamer)
- 15:41:12 — m_000007 completes (retry attempt)
- 15:41:25 — m_000006 completes (retry attempt, last map)
- 15:42:46 — r_000000 completes, Num completed: 11
- 15:42:46 — Job transitions to SUCCEEDED
- 15:42:47 — Final stats logged

**Concurrency:**
- Peak: up to 10 map tasks running simultaneously
- After 15:39:24, concurrency decreases as maps complete

Acceptable variations:
- ASCII visualization style will vary
- Not every log entry needs to be in the timeline — major events are sufficient
- Phase definitions may differ slightly

---

## Grading Criteria

- [ ] `mapreduce_timeline.md` is created in the workspace
- [ ] Events are listed chronologically with timestamps
- [ ] Phases are identified (init, map, reduce, completion)
- [ ] A visual or structured timeline/gantt is attempted
- [ ] Key events (first map completion, errors, job success) are highlighted

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the MapReduce timeline visualization task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "mapreduce_timeline.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "chronological_events": 0.0,
            "phases_identified": 0.0,
            "visual_timeline": 0.0,
            "key_events_highlighted": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Chronological events with timestamps
    timestamps = ["15:37", "15:38", "15:39", "15:40", "15:41", "15:42"]
    ts_found = sum(1 for ts in timestamps if ts in content)
    scores["chronological_events"] = (
        1.0 if ts_found >= 5 else
        0.5 if ts_found >= 3 else 0.0
    )

    # Check 2: Phases identified
    phase_keywords = ["initialization", "init", "map phase", "reduce phase",
                      "shuffle", "cleanup", "completion", "startup"]
    phases_found = sum(1 for kw in phase_keywords if kw in content)
    scores["phases_identified"] = (
        1.0 if phases_found >= 3 else
        0.5 if phases_found >= 2 else 0.0
    )

    # Check 3: Visual/structured timeline attempted
    visual_keywords = ["timeline", "gantt", "---", "===", "|||", "phase",
                       "diagram", "chart", "|", "─", "-"]
    # Check for table-like structures or ASCII art
    lines = content.split("\n")
    table_lines = [l for l in lines if l.count("|") >= 2]
    ascii_lines = [l for l in lines if any(c in l for c in ["─", "━", "═", "▓", "█", "░"])]
    scores["visual_timeline"] = (
        1.0 if len(table_lines) >= 5 or len(ascii_lines) >= 3 else
        0.5 if len(table_lines) >= 2 else 0.0
    )

    # Check 4: Key events highlighted
    key_events = ["mrappmaster", "first map", "succeeded", "warn", "error",
                  "m_000009", "r_000000", "retry", "completed"]
    events_found = sum(1 for kw in key_events if kw in content)
    scores["key_events_highlighted"] = (
        1.0 if events_found >= 4 else
        0.5 if events_found >= 2 else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- Total job duration: ~4 minutes 51 seconds
- Map phase and reduce phase overlap — reduce starts while maps are still running
- The reduce "slow start" threshold meant the reduce task didn't get scheduled immediately
- Two map task retries (m_000006, m_000007) extended the map phase by about 35 seconds
- 1282 log entries total, but only ~50 represent major state transitions

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
