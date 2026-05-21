---
id: task_it_procurement
name: IT Procurement Research
category: research
grading_type: llm_judge
timeout_seconds: 300
workspace_files: []
---

## Prompt

A growing startup (50 engineers) needs to purchase **developer laptops** for new hires. Help them research options and make a recommendation.

**Requirements:**
- 32 GB RAM minimum
- 512 GB SSD minimum
- Modern CPU (within last 2 generations)
- Good build quality (daily professional use)
- Budget: $1,500–$2,500 per unit
- Mix of macOS and Linux compatibility needed (some devs prefer each)

Research at least 5 specific laptop models and create a procurement recommendation. For each model, document:

1. **Exact model name and configuration** (CPU, RAM, storage, display)
2. **Current price** (and where to buy — direct, CDW, etc.)
3. **Repairability and warranty** — standard warranty length, extended warranty options, ease of repair
4. **Developer-relevant specs** — keyboard quality, port selection, display quality, battery life
5. **Linux compatibility** (for non-Mac options) — known issues with WiFi, Bluetooth, suspend, etc.
6. **Bulk purchase options** — volume discounts, business/education pricing, fleet management tools

Save your report to `laptop_procurement.md`. Include:
- A comparison table
- Separate recommendations for macOS and Linux users
- A "fleet recommendation" suggesting the ideal mix for a 50-person team
- Total estimated budget

## Expected Behavior

The agent should:

1. Research current laptop models meeting the specifications
2. Find specific configurations with prices
3. Check Linux compatibility for non-Mac options
4. Consider enterprise/bulk purchasing factors
5. Produce a practical procurement document
6. Save to `laptop_procurement.md`

## Grading Criteria

- [ ] File `laptop_procurement.md` created
- [ ] At least 5 specific laptop models documented
- [ ] Prices included for each model
- [ ] Specifications meet the stated requirements
- [ ] Linux compatibility discussed for non-Mac options
- [ ] Comparison table present
- [ ] Separate macOS and Linux recommendations
- [ ] Fleet recommendation with estimated total budget
- [ ] Bulk purchase / business pricing mentioned
- [ ] Models are current (not discontinued)

## LLM Judge Rubric

### Criterion 1: Product Research Quality (Weight: 30%)

**Score 1.0**: Specific model numbers with exact configurations (e.g., "ThinkPad T14s Gen 5 — AMD Ryzen 7 PRO 8840U, 32GB, 512GB, 14\" 2.8K"). Current pricing from identifiable sources. Models are all currently available for purchase.
**Score 0.75**: Good specificity for most models. A few may have approximate configs or pricing.
**Score 0.5**: Models named but configurations or prices are vague or potentially outdated.
**Score 0.25**: Generic model lines without specific configurations.
**Score 0.0**: No specific products researched.

### Criterion 2: Practical Procurement Value (Weight: 25%)

**Score 1.0**: Report is actionable for an IT manager. Addresses fleet management, warranty, repair, bulk pricing, and deployment considerations. Includes vendor recommendations (direct, CDW, SHI, etc.). Total budget calculation is realistic.
**Score 0.75**: Good procurement detail. Minor gaps in vendor or fleet considerations.
**Score 0.5**: Addresses some procurement concerns but reads more like a consumer review.
**Score 0.25**: Minimal procurement-specific content.
**Score 0.0**: No procurement considerations.

### Criterion 3: Linux Compatibility Assessment (Weight: 20%)

**Score 1.0**: Specific Linux compatibility notes for each non-Mac option. Mentions distro-specific support (Ubuntu certified, Fedora tested), known hardware issues, and driver status. References actual Linux hardware databases or community reports.
**Score 0.75**: Good Linux compatibility coverage with some specific details.
**Score 0.5**: Mentions Linux compatibility but lacks specifics. Generic statements like "works well with Linux."
**Score 0.25**: Barely mentions Linux.
**Score 0.0**: No Linux compatibility discussion.

### Criterion 4: Fleet Recommendation (Weight: 15%)

**Score 1.0**: Provides a specific fleet mix (e.g., "30× MacBook Pro 14\", 20× ThinkPad T14s") with reasoning based on the team composition. Calculates total budget. Considers standardization benefits vs. developer choice.
**Score 0.75**: Good fleet recommendation with budget estimate. Minor gaps in reasoning.
**Score 0.5**: General recommendation without specific fleet composition or budget.
**Score 0.25**: Vague fleet guidance.
**Score 0.0**: No fleet recommendation.

### Criterion 5: Comparison Quality (Weight: 10%)

**Score 1.0**: Clean comparison table covering specs, price, OS, key differentiators. Easy to use for quick decision-making.
**Score 0.75**: Good table with minor gaps.
**Score 0.5**: Table exists but incomplete.
**Score 0.25**: Poor table.
**Score 0.0**: No comparison table.

## Additional Notes

- This task simulates a real IT procurement scenario that startup ops teams regularly face.
- The budget range ($1,500–$2,500) is realistic and should yield multiple viable options from Apple, Lenovo, Dell, Framework, and others.
- Linux compatibility is an intentional differentiator — some agents will just list specs without checking actual Linux support.
- The fleet recommendation component tests the agent's ability to synthesize individual product research into a holistic recommendation.
