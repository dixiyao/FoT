---
id: task_descriptive_name
name: Task Display Name
category: category_name # must match the manifest.yaml category this task is listed under
grading_type: automated # automated | llm_judge | hybrid
timeout_seconds: 120
workspace_files: []
# workspace_files:
#   - source: assets/sample.txt
#     dest: input.txt
#   - source: assets/config.json
#     dest: config.json
# multi_session: true  # Optional: set to true for multi-turn tasks
# sessions:            # Optional: list of sequential user prompts
#   - id: step_1
#     prompt: |
#       First prompt sent to the agent.
#   - id: step_2
#     prompt: |
#       Second prompt, sent in the same conversation context.
#   - id: fresh_start
#     new_session: true  # Start a new session (no conversation history)
#     prompt: |
#       Third prompt in a fresh session — agent has no memory of prior turns.
---

# Task Template

This template defines the structure for PinchBench task specifications. Each task file in the `tasks/` directory must follow this format.

---

## Prompt

{The exact message that will be sent to the agent. This should be a complete, unambiguous instruction that represents a real-world task an OpenClaw agent might receive.}

**Example:**

> Schedule a meeting for next Tuesday at 3pm with john@example.com. Title it "Project Sync" and add a note about discussing the Q1 roadmap.

**Guidelines:**

- Be specific and actionable
- Include all necessary details
- Avoid ambiguous language
- Represent realistic user requests

---

## Expected Behavior

{Detailed description of what the agent should do to successfully complete the task. This section serves as documentation for task authors and provides context for the LLM judge.}

**Example:**

> The agent should use calendar/scheduling tools or create an ICS file to represent the meeting event. It should correctly parse the relative date "next Tuesday" and set the meeting for 3:00 PM in the user's local timezone.

**Include:**

- Primary approach(es) the agent might take
- Acceptable alternative solutions
- Key decisions the agent must make
- Any expected tool usage

---

## Grading Criteria

{Checklist of success criteria that will be evaluated. Each criterion should be atomic and independently verifiable.}

**Format as a checklist:**

- [ ] Criterion 1 description
- [ ] Criterion 2 description
- [ ] Criterion 3 description

**Example:**

- [ ] Event created with correct date (next Tuesday)
- [ ] Time is 3:00 PM
- [ ] Attendee john@example.com included
- [ ] Title matches "Project Sync"
- [ ] Note/description mentions roadmap

**Guidelines:**

- Each criterion maps to a score (0.0 to 1.0)
- Keep criteria independent when possible
- Weight important criteria by splitting into multiple items
- Avoid subjective language in automated tasks

---

## Automated Checks

{Python grading function for automated scoring. Required if `grading_type` is `automated` or `hybrid`.}

```python
def grade(transcript: list, workspace_path: str) -> dict:
    """
    Grade the task based on transcript and workspace state.

    Args:
        transcript: Parsed JSONL transcript as list of dicts. Each dict represents
                   an event with structure:
                   {
                       "type": "message",
                       "message": {
                           "role": "assistant" | "user" | "toolResult",
                           "content": [...]
                       }
                   }
        workspace_path: Path to the task's isolated workspace directory.
                       Files created by the agent will be here.

    Returns:
        Dict mapping criterion names to scores (0.0 to 1.0).
        Keys should match the Grading Criteria checklist.

    Example return:
        {
            "date_correct": 1.0,
            "time_correct": 0.5,  # Partial credit allowed
            "attendee_present": 0.0
        }
    """
    from pathlib import Path
    import re
    import json

    scores = {}

    # Check workspace for expected outputs
    workspace = Path(workspace_path)

    # Example: Check if a file was created
    if (workspace / "output.txt").exists():
        scores["file_created"] = 1.0
    else:
        scores["file_created"] = 0.0

    # Example: Check transcript for specific tool usage
    for event in transcript:
        if event.get("type") != "message":
            continue
        msg = event.get("message", {})
        if msg.get("role") == "assistant":
            for item in msg.get("content", []):
                if item.get("type") == "toolCall":
                    tool_name = item.get("name", "")
                    # Check for expected tool calls
                    if tool_name == "expected_tool":
                        scores["tool_used"] = 1.0

    # Example: Validate file content
    output_file = workspace / "output.txt"
    if output_file.exists():
        content = output_file.read_text()
        if "expected_pattern" in content:
            scores["content_correct"] = 1.0
        else:
            scores["content_correct"] = 0.0

    return scores
```

**Implementation Guidelines:**

1. **File Checks**: Use `pathlib.Path` for all filesystem operations
2. **Transcript Parsing**: Iterate through events looking for tool calls and results
3. **Partial Credit**: Return values between 0.0 and 1.0 for partial success
4. **Error Handling**: Handle missing files/data gracefully (return 0.0)
5. **No External Dependencies**: Avoid imports beyond stdlib + `pathlib`

---

## LLM Judge Rubric

{Detailed rubric for Claude Opus to use when scoring. Required if `grading_type` is `llm_judge` or `hybrid`.}

**Format:**

```markdown
### Criterion 1: [Name] (Weight: X%)

**Score 1.0**: [Description of perfect performance]
**Score 0.75**: [Description of good performance with minor issues]
**Score 0.5**: [Description of acceptable performance with notable gaps]
**Score 0.25**: [Description of poor performance with major issues]
**Score 0.0**: [Description of failure or non-attempt]

### Criterion 2: [Name] (Weight: Y%)

[Same structure...]
```

**Example:**

```markdown
### Criterion 1: Content Quality (Weight: 40%)

**Score 1.0**: Content is well-structured, comprehensive, and directly addresses all requirements. Information is accurate and well-organized.
**Score 0.75**: Content addresses most requirements with good structure. Minor gaps in coverage or organization.
**Score 0.5**: Content partially addresses requirements. Some structural issues or notable gaps.
**Score 0.25**: Content is poorly organized or misses significant requirements.
**Score 0.0**: Content is missing, irrelevant, or completely fails to address the task.

### Criterion 2: Tool Usage Appropriateness (Weight: 30%)

**Score 1.0**: Agent selected the most appropriate tools and used them efficiently. No unnecessary tool calls.
**Score 0.75**: Agent used appropriate tools with minor inefficiencies.
**Score 0.5**: Agent used somewhat appropriate tools but with notable inefficiencies or misuse.
**Score 0.25**: Agent used inappropriate tools or struggled significantly with tool selection.
**Score 0.0**: Agent failed to use tools or used completely wrong tools.

### Criterion 3: Task Completion (Weight: 30%)

**Score 1.0**: Task fully completed with all requirements met.
**Score 0.75**: Task mostly completed with minor omissions.
**Score 0.5**: Task partially completed with significant gaps.
**Score 0.25**: Task barely attempted or mostly incomplete.
**Score 0.0**: Task not completed or results unusable.
```

**Guidelines for LLM Judge Rubrics:**

1. **Explicit Criteria**: Be specific about what constitutes each score level
2. **Weights**: Ensure weights sum to 100%
3. **Examples**: Include concrete examples where helpful
4. **Objectivity**: Frame criteria to minimize subjective interpretation
5. **Edge Cases**: Address common edge cases in score descriptions

---

## Workspace Files

{Optional: List of files to pre-populate in the task workspace before execution.}

**YAML Frontmatter Format:**

```yaml
workspace_files:
  - source: assets/input_data.csv
    dest: data.csv
  - source: assets/config_template.json
    dest: config.json
```

**Guidelines:**

- `source`: Path relative to the skill's `assets/` directory
- `dest`: Path relative to the task workspace root
- Keep fixture files minimal and focused on the task
- Document any required file formats in Expected Behavior

---

## Multi-Session Tasks

{Optional: Define a sequence of prompts to test multi-turn conversation, iterative refinement, or cross-session memory.}

**YAML Frontmatter Format:**

```yaml
multi_session: true
sessions:
  - id: first_turn
    prompt: |
      First message sent to the agent.
  - id: follow_up
    prompt: |
      Follow-up message in the same conversation context.
      The agent retains all prior conversation history.
  - id: fresh_context
    new_session: true
    prompt: |
      Message sent in a brand-new session. The agent has no
      conversation history from prior turns, but workspace
      files created earlier are still accessible.
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for this session step |
| `prompt` | string | The user message sent to the agent |
| `new_session` | bool | If `true`, start a fresh OpenClaw session with no conversation history. Workspace files are preserved. Default: `false` |

**How It Works:**

1. Each session entry's `prompt` is sent to the agent in order
2. By default, all prompts share the same conversation context (the agent remembers previous turns)
3. When `new_session: true` is set, the agent's session is reset before sending the prompt — simulating a user returning after closing and reopening the agent
4. The workspace directory (including any files the agent created) is **never** reset between sessions
5. All session transcripts are merged for grading

**When to Use Multi-Session:**

- **Iterative refinement**: User asks the agent to create something, then asks for modifications
- **Cross-session memory**: Test if the agent can persist and recall information across separate conversations
- **Follow-up instructions**: Test if the agent correctly applies new constraints to prior work
- **Realistic workflows**: Simulate a user who interacts with the agent over multiple conversations

**Guidelines:**

- Use `new_session: true` when you want to test the agent's ability to recover context from files (not conversation history)
- Keep the total number of sessions reasonable (2-5) to avoid excessive timeout
- Increase `timeout_seconds` proportionally to the number of sessions
- Ensure the Prompt section notes that this is a multi-session task

---

## Additional Notes

{Optional: Any additional context, edge cases, or implementation notes for task authors or developers.}

**May include:**

- Known limitations or edge cases
- Rationale for specific grading choices
- Links to relevant documentation
- Versioning or compatibility notes

---

# Checklist for Task Authors

Before submitting a new task, verify:

- [ ] YAML frontmatter is complete and valid
- [ ] `id` follows naming convention: `task_descriptive_name` (must match filename without `.md`)
- [ ] `grading_type` matches the sections provided
- [ ] Prompt is clear and unambiguous
- [ ] Expected behavior describes acceptable solutions
- [ ] Grading criteria are atomic and verifiable
- [ ] Automated checks function runs without errors (if applicable)
- [ ] LLM judge rubric has explicit score levels (if applicable)
- [ ] Weights in rubric sum to 100% (if applicable)
- [ ] Timeout is reasonable for the task complexity
- [ ] Workspace files are included in `assets/` (if needed)
- [ ] Multi-session prompts are in the `sessions` field (if applicable)
- [ ] `new_session: true` is set for sessions that should start fresh (if applicable)
- [ ] `timeout_seconds` accounts for multiple sessions (if applicable)
