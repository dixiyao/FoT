---
id: task_log_hdfs_connections
name: HDFS DataNode Log - Connection Pattern Analysis
category: log_analysis
grading_type: hybrid
timeout_seconds: 180
workspace_files:
  - dest: "hdfs_datanode.log"
    source: "logs/hdfs_datanode.log"
---

# HDFS DataNode Log - Connection Pattern Analysis

## Prompt

Analyze the HDFS DataNode log at `hdfs_datanode.log` and produce a report on connection and communication patterns between nodes. The log contains entries from DataNode, FSNamesystem, and PacketResponder components.

Your report should include:

1. **Network Topology**: List all unique IP addresses that appear in the log, categorized by their role (source, destination, or both)
2. **Subnet Analysis**: Group IPs by subnet (e.g., 10.250.x.x vs 10.251.x.x). How many nodes are in each subnet?
3. **Most Active Nodes**: Top 10 IPs by frequency of appearance (as source or destination)
4. **Communication Patterns**: Which pairs of nodes communicate most frequently?
5. **DataNode vs NameSystem**: Separate the activity — what comes from DataNode operations vs FSNamesystem operations?
6. **Cluster Size Estimate**: Based on the IPs observed, estimate the cluster size

Write the report to `hdfs_connections_report.md` as a well-structured markdown document.

---

## Expected Behavior

The agent should parse 2000 log entries and produce:

**Network Topology:**
- 202 unique IP addresses observed
- IPs fall in the 10.250.x.x and 10.251.x.x ranges (private network)
- All nodes use port 50010 (HDFS DataNode data transfer port)

**Subnet Analysis:**
- 10.250.x.x subnet — contains some of the most active nodes
- 10.251.x.x subnet — contains additional DataNode cluster members
- The split suggests a multi-rack HDFS deployment

**Most Active Nodes:**
- 10.250.19.102 — extremely active (appears as source in many block transfers)
- 10.250.10.6, 10.251.215.16, 10.250.14.224 — also very active

**Component Activity:**
- DataNode$DataXceiver: Block receive operations (~1149 entries)
- FSNamesystem: Block allocation and storage tracking (~400+ entries)
- DataNode$PacketResponder: Block receive confirmations with sizes

Acceptable variations:
- Exact IP counts and rankings may vary by parsing approach
- Subnet grouping granularity may differ
- Cluster size estimates will be approximate

---

## Grading Criteria

- [ ] `hdfs_connections_report.md` is created in the workspace
- [ ] Unique IPs are listed or counted (~202)
- [ ] IPs are grouped by subnet (10.250.x.x vs 10.251.x.x)
- [ ] Most active nodes are identified
- [ ] DataNode vs FSNamesystem activity is distinguished

---

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """Grade the HDFS connection pattern analysis task."""
    from pathlib import Path

    scores = {}
    workspace = Path(workspace_path)
    report_file = workspace / "hdfs_connections_report.md"

    if not report_file.exists():
        return {
            "output_created": 0.0,
            "ips_listed": 0.0,
            "subnets_grouped": 0.0,
            "active_nodes": 0.0,
            "components_separated": 0.0,
        }

    scores["output_created"] = 1.0
    content = report_file.read_text(encoding="utf-8").lower()

    # Check 1: IPs listed/counted
    has_count = any(n in content for n in ["202", "200", "~200", "over 200"])
    has_ips = "10.250" in content and "10.251" in content
    scores["ips_listed"] = (
        1.0 if has_count and has_ips else
        0.5 if has_ips else 0.0
    )

    # Check 2: Subnets grouped
    subnet_keywords = ["subnet", "10.250", "10.251", "rack", "network segment",
                       "address range", "ip range"]
    scores["subnets_grouped"] = (
        1.0 if "10.250" in content and "10.251" in content and
              sum(1 for kw in subnet_keywords if kw in content) >= 2 else
        0.5 if "10.250" in content and "10.251" in content else 0.0
    )

    # Check 3: Active nodes identified
    active_ips = ["10.250.19.102", "10.251.215.16", "10.250.14.224", "10.250.10.6"]
    ips_found = sum(1 for ip in active_ips if ip in content)
    scores["active_nodes"] = (
        1.0 if ips_found >= 2 else
        0.5 if ips_found >= 1 else 0.0
    )

    # Check 4: Components separated
    component_keywords = ["dataxceiver", "dataxeceiver", "fsnamesystem",
                          "packetresponder", "namenode", "datanode"]
    scores["components_separated"] = (
        1.0 if sum(1 for kw in component_keywords if kw in content) >= 2 else
        0.5 if sum(1 for kw in component_keywords if kw in content) >= 1 else 0.0
    )

    return scores
```

---

## Additional Notes

**Key facts from the log:**

- 202 unique IPs — this is a large HDFS cluster
- Two main subnets: 10.250.x.x and 10.251.x.x
- Port 50010 is used throughout — standard HDFS DataNode port
- 10.250.19.102 appears as source in a disproportionate number of entries
- The log captures a burst of activity related to job_200811092030_0001

**Grading weights (equal):** Each of the five criteria contributes 0.2 to the final score.
