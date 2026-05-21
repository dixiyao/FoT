---
id: task_log_hdfs_storage
name: HDFS DataNode Log - Storage and Capacity Analysis
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "hdfs_datanode.log"
    source: "logs/hdfs_datanode.log"
---

# HDFS DataNode Log - Storage and Capacity Analysis

## Prompt

Analyze the HDFS DataNode log at `hdfs_datanode.log` and produce a storage-focused analysis. Examine block sizes, data distribution across nodes, and storage patterns.

Your report should include:

1. **Data Volume**: Total bytes stored across all confirmed block receives (where size is known)
2. **Block Size Distribution**: List all known block sizes, calculate min/max/mean/median
3. **Data Distribution by Node**: For each node that confirmed receiving blocks (PacketResponder "Received" entries), total the bytes stored
4. **Storage Path Analysis**: What storage paths are being used? (Extract from allocateBlock file paths)
5. **Replication Factor**: Based on how many nodes receive the same block, what is the effective replication factor?
6. **Capacity Planning**: Based on the data ingestion rate observed, estimate the storage needed for 1 hour of similar activity

Write the report to `hdfs_storage_report.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should parse the log and calculate:

**Confirmed Block Sizes:**
- blk_-1608999687919862906: 91,178 bytes (received by 3+ nodes)
- blk_7503483334202473044: 233,217 bytes (received by 3 nodes)
- blk_-3544583377289625738: 11,971 bytes (received by 3 nodes)
- blk_-9073992586687739851: 11,977 bytes (received by 3 nodes)

**Block Size Statistics:**
- Min: 11,971 bytes (~12 KB)
- Max: 233,217 bytes (~228 KB)
- Mean: ~87,086 bytes (~85 KB)
- Total confirmed data: ~348,343 bytes per replica

**Replication Factor:**
- Each confirmed block is received by 3 nodes → replication factor of 3
- This is standard HDFS default replication

**Storage Path:**
- `/mnt/hadoop/mapred/system/job_200811092030_0001/` — MapReduce job staging directory
- Files: job.jar, job.split

**Capacity Planning:**
- ~28 seconds of activity produced ~390 block allocations
- If each block averages ~85 KB with replication factor 3, that's ~100 MB/minute raw storage
- 1 hour estimate: ~6 GB (rough)

Acceptable variations:
- Capacity estimates will be very rough given limited confirmed sizes
- Approaches to extrapolation will differ
- Statistics should be based on confirmed sizes only

---

## Grading Criteria

- [ ] `hdfs_storage_report.md` is created in the workspace
- [ ] Known block sizes are listed (91178, 233217, 11971, 11977)
- [ ] Block size statistics are calculated (min, max, mean)
- [ ] Replication factor is identified (3)
- [ ] Storage paths are extracted from the log

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the HDFS storage and capacity analysis task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "hdfs_storage_report.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "block_sizes_listed": 0.0,
            "statistics_calculated": 0.0,
            "replication_factor": 0.0,
            "storage_paths": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: Block sizes listed
    sizes = ["91178", "233217", "11971", "11977"]
    sizes_found = sum(1 for s in sizes if s in content)
    scores["block_sizes_listed"] = (
        1.0 if sizes_found >= 3 else
        0.5 if sizes_found >= 1 else 0.0
    )

    # Check 2: Statistics calculated
    stat_keywords = ["min", "max", "mean", "median", "average", "total",
                     "distribution", "range"]
    scores["statistics_calculated"] = (
        1.0 if sum(1 for kw in stat_keywords if kw in content) >= 3 else
        0.5 if sum(1 for kw in stat_keywords if kw in content) >= 1 else 0.0
    )

    # Check 3: Replication factor identified
    has_replication = any(kw in content for kw in ["replication factor",
                                                     "replication of 3",
                                                     "factor of 3",
                                                     "3 replicas",
                                                     "three replicas",
                                                     "3 copies",
                                                     "three copies",
                                                     "replicated 3",
                                                     "3 nodes"])
    scores["replication_factor"] = 1.0 if has_replication else 0.0

    # Check 4: Storage paths extracted
    has_path = any(p in content for p in ["/mnt/hadoop", "job_200811092030",
                                           "mapred/system", "job.jar", "job.split"])
    scores["storage_paths"] = 1.0 if has_path else 0.0

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- Only 4 unique block sizes are confirmed in the log (from PacketResponder "Received" entries)
- 390 blocks were allocated but only ~19 receive confirmations appear in the log window
- Standard HDFS replication factor is 3, which matches the 3 receive confirmations per block
- The addStoredBlock entries (19) update the NameSystem's block map
- Storage is under `/mnt/hadoop/mapred/system/` — standard MapReduce staging directory

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
