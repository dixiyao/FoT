---
id: task_oss_alternative_research
name: Open Source Alternatives Research
category: research
grading_type: llm_judge
timeout_seconds: 300
workspace_files: []
---

## Prompt

A startup team is looking for **open source alternatives to Notion** for their internal knowledge base and project documentation. They want to self-host for data sovereignty reasons.

Research and compile a report covering at least 5 open source alternatives. For each alternative, document:

1. **Project name and URL** (GitHub/GitLab repo link)
2. **License** (MIT, AGPL, Apache, etc.)
3. **Tech stack** (what it's built with)
4. **Deployment options** (Docker, Kubernetes, manual install)
5. **Key features** — specifically how they compare to Notion (blocks editor, databases, real-time collaboration, API, integrations)
6. **Community health** — GitHub stars, recent commit activity, number of contributors, release cadence
7. **Limitations** — what's missing compared to Notion, known pain points
8. **Self-hosting complexity** — how hard is it to deploy and maintain?

Save the report to `oss_alternatives.md`. Include:
- A summary comparison table at the top
- Detailed profiles for each alternative
- A recommendation section that ranks the alternatives for the startup's use case

## Expected Behavior

The agent should:

1. Use web search to find current open source Notion alternatives
2. Visit GitHub/GitLab repos to check community health metrics
3. Review documentation for deployment and feature details
4. Compile a structured comparison report
5. Provide an opinionated ranking with reasoning
6. Save to `oss_alternatives.md`

Likely alternatives include: AppFlowy, AFFiNE, Outline, BookStack, Wiki.js, Docmost, Trilium Notes, Focalboard, and others.

## Grading Criteria

- [ ] File `oss_alternatives.md` created
- [ ] At least 5 alternatives documented
- [ ] Each alternative has license information
- [ ] GitHub/GitLab URLs provided
- [ ] Deployment options described
- [ ] Features compared to Notion
- [ ] Community health metrics included
- [ ] Limitations noted for each
- [ ] Comparison table present
- [ ] Recommendations with ranking

## LLM Judge Rubric

### Criterion 1: Research Quality (Weight: 30%)

**Score 1.0**: Report includes specific, verifiable details — exact GitHub star counts, recent release dates, specific version numbers, concrete feature lists from actual documentation. Information clearly comes from checking actual repos and docs, not just general knowledge.
**Score 0.75**: Good specific details for most alternatives. A few entries may rely on general knowledge.
**Score 0.5**: Mix of specific and generic information. Some alternatives well-researched, others surface-level.
**Score 0.25**: Mostly generic descriptions without specific evidence of research.
**Score 0.0**: Report is missing or contains clearly inaccurate information.

### Criterion 2: Practical Usefulness (Weight: 25%)

**Score 1.0**: Report would genuinely help a startup make a decision. Includes self-hosting complexity assessment, migration considerations, and honest evaluation of whether each tool actually replaces Notion's key features. Recommendations are specific to the stated use case (startup, knowledge base, data sovereignty).
**Score 0.75**: Mostly useful with good practical details, but some areas lack specificity.
**Score 0.5**: Provides information but doesn't strongly address the startup's specific needs.
**Score 0.25**: Generic overviews that wouldn't help with an actual decision.
**Score 0.0**: Not useful for decision-making.

### Criterion 3: Coverage Breadth (Weight: 20%)

**Score 1.0**: 5+ alternatives covered with all 8 requested dimensions documented for each. Includes a mix of mature and emerging options. No major well-known alternatives omitted.
**Score 0.75**: 5+ alternatives with most dimensions covered. One or two gaps.
**Score 0.5**: 4-5 alternatives with inconsistent coverage across dimensions.
**Score 0.25**: Fewer than 4 alternatives or very thin coverage.
**Score 0.0**: Fewer than 2 alternatives documented.

### Criterion 4: Comparison Table and Structure (Weight: 15%)

**Score 1.0**: Clean comparison table with key dimensions (license, stars, key features, self-hosting difficulty, Notion parity). Report is well-organized with consistent sections for each alternative. Easy to scan.
**Score 0.75**: Good table and structure with minor gaps.
**Score 0.5**: Table or structure exists but incomplete.
**Score 0.25**: Poor organization.
**Score 0.0**: No table, no structure.

### Criterion 5: Honest Assessment of Limitations (Weight: 10%)

**Score 1.0**: Each alternative has an honest, specific limitations section. Doesn't oversell any option. Notes real pain points from community feedback or documentation gaps. Acknowledges that no open source option fully replaces Notion.
**Score 0.75**: Most alternatives have limitations noted, with some specificity.
**Score 0.5**: Limitations mentioned but generic ("fewer features than Notion").
**Score 0.25**: Limitations barely touched on.
**Score 0.0**: No limitations discussed — reads like marketing.

## Additional Notes

- Notion alternatives were chosen because this is a common real-world research task and the landscape changes frequently.
- The self-hosting requirement adds a practical filter that tests whether the agent actually checks deployment documentation.
- Community health metrics (stars, commits, contributors) require visiting actual GitHub/GitLab repositories.
- This task tests the agent's ability to produce actionable research for a specific decision-maker profile.
