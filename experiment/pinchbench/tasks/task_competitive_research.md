---
id: task_competitive_research
name: Competitive Product Comparison
category: research
grading_type: llm_judge
timeout_seconds: 300
workspace_files: []
---

## Prompt

Compare three competing products in the **AI code assistant** space:

1. **GitHub Copilot**
2. **Cursor**
3. **Kilo Code** (open source VS Code extension)

For each product, research and document:

- **Pricing**: All tiers with specific prices. Free options, monthly/annual pricing, enterprise plans.
- **Features**: Key capabilities — code completion, chat, multi-file editing, agent mode, tool use, MCP support, model flexibility.
- **Model support**: Which AI models does each support? Can users bring their own API keys (BYOK)?
- **Privacy/data**: What data is collected? Can it run without sending code to the cloud?
- **Open source**: Is the code open source? What license? Can it be self-hosted?

Save your analysis to `competitive_analysis.md`. Include:
- A comparison table at the top summarizing key differences
- Detailed sections for each product
- A "Bottom Line" section with recommendations for different user profiles (hobbyist, startup developer, enterprise team)

## Expected Behavior

The agent should:

1. Research each product using web search, visiting official websites and documentation
2. Find specific, current pricing information
3. Compare features systematically
4. Create a structured comparison document
5. Provide an opinionated but fair recommendation
6. Save to `competitive_analysis.md`

## Grading Criteria

- [ ] File `competitive_analysis.md` created
- [ ] All three products covered
- [ ] Pricing details included for each product
- [ ] Features comparison is specific (not generic)
- [ ] Comparison table present
- [ ] Model support details included
- [ ] Privacy/data policies discussed
- [ ] Open source status addressed
- [ ] Recommendations for different user profiles
- [ ] Web search was used for current information

## LLM Judge Rubric

### Criterion 1: Information Accuracy and Currency (Weight: 30%)

**Score 1.0**: Pricing is specific and current (dollar amounts, tier names). Feature descriptions match actual current product capabilities. No significant errors or outdated information. Evidence of checking official sources.
**Score 0.75**: Mostly accurate with current information. Minor details may be slightly off or a few months old.
**Score 0.5**: Generally correct direction but some pricing or feature details are inaccurate or clearly outdated.
**Score 0.25**: Multiple significant inaccuracies or heavily outdated information.
**Score 0.0**: Information is mostly wrong or missing.

### Criterion 2: Comparison Depth (Weight: 25%)

**Score 1.0**: Goes beyond surface features to discuss nuances — BYOK flexibility, model provider options, extension ecosystem, CLI support, offline capability, privacy implications. Identifies meaningful differentiators that would actually matter to a buyer.
**Score 0.75**: Good depth on most dimensions but misses some important distinctions.
**Score 0.5**: Adequate comparison but stays at a surface level. Lists features without exploring implications.
**Score 0.25**: Shallow comparison that mostly states the obvious.
**Score 0.0**: No meaningful comparison.

### Criterion 3: Comparison Table Quality (Weight: 15%)

**Score 1.0**: Clean, well-formatted Markdown table covering all key dimensions. Easy to scan and extract key differences at a glance. Includes pricing, key features, model support, and licensing.
**Score 0.75**: Good table with most key dimensions, minor formatting or coverage gaps.
**Score 0.5**: Table exists but is incomplete or poorly organized.
**Score 0.25**: Minimal table attempt.
**Score 0.0**: No comparison table.

### Criterion 4: Recommendation Quality (Weight: 20%)

**Score 1.0**: Recommendations are specific, actionable, and tailored to the three user profiles. Shows understanding of what different users value (cost sensitivity, privacy, customization, enterprise compliance). Takes a clear stance.
**Score 0.75**: Good recommendations with reasonable differentiation by user type.
**Score 0.5**: Generic recommendations that don't strongly differentiate by user profile.
**Score 0.25**: Vague or wishy-washy recommendations.
**Score 0.0**: No recommendations.

### Criterion 5: Fairness and Balance (Weight: 10%)

**Score 1.0**: Each product is treated fairly with both strengths and weaknesses identified. No obvious bias. Acknowledges uncertainty where information was hard to find.
**Score 0.75**: Generally balanced with minor tendencies.
**Score 0.5**: Noticeable bias toward or against one product.
**Score 0.25**: Clearly biased or unfair treatment.
**Score 0.0**: Completely one-sided.

## Additional Notes

- AI code assistants were chosen because agents should have strong knowledge of this space, and current pricing/features are easily verifiable.
- Kilo Code is included specifically because it's an open source project with a BYOK model, creating interesting differentiation.
- This task tests the agent's ability to produce a practical buyer's guide, not just a feature list.
- Agents with web search should score higher on pricing accuracy and feature currency.
