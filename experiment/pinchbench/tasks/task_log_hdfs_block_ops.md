---
id: task_log_hdfs_block_ops
name: HDFS DataNode Log - Block Operations Summary
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "hdfs_datanode.log"
    source: "logs/hdfs_datanode.log"
---

# HDFS DataNode Log - Block Operations Summary

## Prompt

Analyze the HDFS DataNode log at `hdfs_datanode.log` and produce a comprehensive summary of all block operations. The log comes from an HDFS cluster and tracks block lifecycle events.

Your report should include:

1. **Block Inventory**: Total unique block IDs in the log, with a full list
2. **Operation Types**: For each operation type (allocateBlock, Receiving, Received, addStoredBlock, replicate, PacketResponder), count total occurrences
3. **Block Lifecycle Tracking**: For each block that has a complete lifecycle (allocate → receive → stored), document the full chain
4. **Replication Chain**: For blocks with replication events, trace the replication path across nodes
5. **Associated Jobs**: Identify the MapReduce jobs that triggered these block operations (visible in file paths)
6. **Per-Block Detail Table**: Create a table with columns: Block ID, Size (if known), Allocated Path, Nodes Involved, Replication Count

Write the report to `hdfs_block_ops_report.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should parse 2000 log entries and produce:

**Block Inventory:**
- ~390 unique block IDs

**Operation Counts:**
- Receiving block: ~1149
- allocateBlock: ~385
- Received block: ~19
- addStoredBlock: ~19
- PacketResponder: ~12
- Replicate: 4

**Complete Block Lifecycles (blocks with full data):**
- blk_-1608999687919862906: 91178 bytes, allocated for job_200811092030_0001/job.jar
- blk_7503483334202473044: 233217 bytes, allocated for job_200811092030_0001/job.split
- blk_-3544583377289625738: 11971 bytes
- blk_-9073992586687739851: 11977 bytes

**Replication Chain:**
- blk_-1608999687919862906 was replicated 4 times across the cluster:
  10.250.14.224 → 10.251.215.16 → 10.251.74.79 → 10.251.31.5 → 10.251.90.64

**Associated Job:**
- job_200811092030_0001 — MapReduce job, files: job.jar, job.split

Acceptable variations:
- Block ID lists may be truncated
- Not all 390 blocks need full detail — just those with complete lifecycle data
- Table format may vary

---

## Grading Criteria

- [ ] `hdfs_block_ops_report.md` is created in the workspace
- [ ] Unique block count is provided (~390)
- [ ] Operation types are counted (receiving, allocate, replicate, etc.)
- [ ] At least one block lifecycle is fully traced (allocate → receive → stored)
- [ ] The associated MapReduce job is identified (job_200811092030_0001)

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the HDFS block operations summary task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "hdfs_block_ops_report.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "block_count": 0.0,
            "operations_counted": 0.0,
            "lifecycle_traced": 0.0,
            "job_identified": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Block count
    has_count = any(n in content for n in ["390", "~390", "385", "~385", "380", "~400"])
    scores["block_count"] = (
        1.0 if has_count else
        0.5 if any(kw in content for kw in ["hundred", "unique block"]) else 0.0
    )

    # Check 2: Operations counted
    op_keywords = ["receiving", "allocate", "replicate", "addstored",
                   "packetresponder", "received"]
    ops_found = sum(1 for kw in op_keywords if kw in content)
    scores["operations_counted"] = (
        1.0 if ops_found >= 4 else
        0.5 if ops_found >= 2 else 0.0
    )

    # Check 3: Block lifecycle traced
    lifecycle_keywords = ["91178", "233217", "blk_-1608999687919862906",
                          "blk_7503483334202473044", "lifecycle", "job.jar", "job.split"]
    scores["lifecycle_traced"] = (
        1.0 if sum(1 for kw in lifecycle_keywords if kw in content) >= 2 else
        0.5 if sum(1 for kw in lifecycle_keywords if kw in content) >= 1 else 0.0
    )

    # Check 4: MapReduce job identified
    has_job = "job_200811092030_0001" in content or "200811092030" in content
    has_mapreduce = "mapreduce" in content or "mapred" in content or "map reduce" in content
    scores["job_identified"] = (
        1.0 if has_job else
        0.5 if has_mapreduce else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- Most blocks only have "Receiving" and "allocateBlock" entries — the cluster was mid-operation
- Only ~19 blocks have complete lifecycle data with confirmed sizes
- The 390 block IDs represent a MapReduce job's data being distributed across the cluster
- Replication is only logged for blk_-1608999687919862906, which is replicated 4 times
- File paths show this is related to a MapReduce job: `/mnt/hadoop/mapred/system/job_200811092030_0001/`

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
