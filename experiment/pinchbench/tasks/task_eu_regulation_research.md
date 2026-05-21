---
id: task_eu_regulation_research
name: EU AI Act Compliance Research
category: research
grading_type: llm_judge
timeout_seconds: 300
workspace_files: []
---

## Prompt

A SaaS company building AI-powered developer tools needs to understand their obligations under the **EU AI Act (Regulation 2024/1689)**. They sell to European customers and deploy models via API.

Research the EU AI Act and create a compliance briefing. Your report should cover:

1. **Risk classification**: How does the EU AI Act classify AI systems by risk level? Where would an AI coding assistant likely fall?
2. **Obligations by risk tier**: What are the specific requirements for each risk category (prohibited, high-risk, limited risk, minimal risk)?
3. **Timeline**: When do different provisions take effect? What are the key compliance deadlines?
4. **General-purpose AI (GPAI)**: What are the specific rules for general-purpose AI models and foundation models? How do these affect companies using third-party models (e.g., via API)?
5. **Transparency requirements**: What disclosures are required? When must users be told they're interacting with AI?
6. **Penalties**: What are the fines for non-compliance?
7. **Practical steps**: What should the company do now to prepare for compliance?

Save the report to `eu_ai_act_briefing.md`. Include citations to specific articles or recitals of the regulation where possible.

## Expected Behavior

The agent should:

1. Research the EU AI Act using web search and/or official EU sources
2. Find the specific regulation text and key provisions
3. Accurately classify risk levels and associated obligations
4. Identify GPAI-specific provisions (relevant for companies using LLMs)
5. Create an actionable compliance briefing
6. Save to `eu_ai_act_briefing.md`

## Grading Criteria

- [ ] File `eu_ai_act_briefing.md` created
- [ ] Risk classification system accurately described
- [ ] Four risk tiers identified (prohibited, high, limited, minimal)
- [ ] GPAI/foundation model provisions discussed
- [ ] Compliance timeline with dates
- [ ] Penalty amounts mentioned
- [ ] Transparency requirements described
- [ ] Practical compliance steps recommended
- [ ] Citations to specific articles
- [ ] Report is well-structured and actionable

## LLM Judge Rubric

### Criterion 1: Legal Accuracy (Weight: 30%)

**Score 1.0**: Risk classification is accurately described with correct tier definitions. Article numbers are cited correctly. GPAI provisions (Articles 51-56) are accurately summarized. Penalty amounts match the regulation (up to €35M or 7% of turnover for prohibited practices). Timeline is accurate.
**Score 0.75**: Mostly accurate with correct general framework. Minor inaccuracies in article numbers or specific thresholds.
**Score 0.5**: General framework is correct but multiple specific details are wrong or missing. Key provisions like GPAI rules are thin.
**Score 0.25**: Significant inaccuracies or confuses the EU AI Act with other regulations.
**Score 0.0**: Report is missing or fundamentally inaccurate.

### Criterion 2: Practical Relevance (Weight: 25%)

**Score 1.0**: Report is clearly tailored to the company's situation (AI developer tools, API deployment, EU customers). Risk classification analysis specifically considers where coding assistants fall. GPAI rules are explained in the context of using third-party models. Practical steps are specific and actionable.
**Score 0.75**: Good relevance to the company's situation. Practical steps are useful but could be more specific.
**Score 0.5**: Covers the regulation but doesn't strongly connect to the company's specific situation.
**Score 0.25**: Generic regulatory overview with no tailoring.
**Score 0.0**: Not relevant to the stated scenario.

### Criterion 3: GPAI Coverage (Weight: 20%)

**Score 1.0**: Thoroughly covers general-purpose AI provisions — classification as GPAI, systemic risk thresholds, obligations for providers vs. deployers, model cards, technical documentation requirements, and downstream provider obligations. Explains how API-based model usage affects compliance.
**Score 0.75**: Good GPAI coverage. Addresses provider vs. deployer distinction.
**Score 0.5**: Mentions GPAI rules but lacks depth on provider/deployer dynamics.
**Score 0.25**: Minimal GPAI discussion.
**Score 0.0**: GPAI provisions not addressed.

### Criterion 4: Timeline and Structure (Weight: 15%)

**Score 1.0**: Clear timeline with key dates (e.g., February 2025 prohibited practices, August 2025 GPAI rules, August 2026 full application). Report is well-organized with clear sections, easy to navigate. Would be useful as a reference document.
**Score 0.75**: Good timeline and structure. Minor organizational issues.
**Score 0.5**: Timeline present but incomplete. Adequate structure.
**Score 0.25**: Vague timeline. Poor structure.
**Score 0.0**: No timeline, poor structure.

### Criterion 5: Citation Quality (Weight: 10%)

**Score 1.0**: References specific articles and recitals of the regulation (e.g., "Article 6 — Classification rules for high-risk AI systems"). May cite official EUR-Lex publication or EU Commission guidance documents. Citations add credibility and traceability.
**Score 0.75**: Several article references. Some citations to official sources.
**Score 0.5**: A few article numbers mentioned. Limited sourcing.
**Score 0.25**: Minimal specific references.
**Score 0.0**: No citations or article references.

## Additional Notes

- The EU AI Act (Regulation 2024/1689) was published in the Official Journal on July 12, 2024 and entered into force on August 1, 2024.
- This task tests the agent's ability to research complex regulatory content and translate it into practical business guidance.
- The AI coding assistant angle is intentionally chosen — it creates an interesting classification question (likely minimal/limited risk, but the GPAI provisions are highly relevant for the underlying models).
- Agents with web search should be able to find the actual regulation text and key analysis from law firms and consultancies.
