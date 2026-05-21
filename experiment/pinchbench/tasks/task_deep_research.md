---
id: task_deep_research
name: Deep Research with Citations
category: research
grading_type: llm_judge
timeout_seconds: 300
workspace_files: []
---

## Prompt

Research the following topic and produce a comprehensive report with primary source citations:

**Topic: The current state of WebAssembly (Wasm) adoption outside the browser — specifically in server-side, edge computing, and plugin systems.**

Your report should cover:

1. **Current adoption**: Which major platforms and companies are using Wasm outside the browser today? Provide specific examples.
2. **Key runtimes**: Compare the major Wasm runtimes (Wasmtime, Wasmer, WasmEdge, etc.) — their focus areas, sponsors, and maturity.
3. **WASI status**: What is the current state of the WebAssembly System Interface (WASI)? What proposals are stable vs. in progress?
4. **Use cases**: Document at least 5 concrete use cases with specific companies or projects.
5. **Limitations**: What are the current technical limitations preventing broader adoption?

**Requirements:**
- Every factual claim must have a citation (URL, paper, or official documentation)
- Save the report to `wasm_research.md`
- Include a References section at the end with all sources
- Minimum 1000 words

## Expected Behavior

The agent should:

1. Use web search to find current information about WebAssembly outside the browser
2. Visit primary sources (official project sites, blog posts, documentation)
3. Synthesize findings into a well-structured research report
4. Include inline citations and a references section
5. Cover all five requested areas with specific, verifiable details
6. Save the report to `wasm_research.md`

## Grading Criteria

- [ ] File `wasm_research.md` created
- [ ] Report covers WebAssembly adoption outside the browser
- [ ] Multiple Wasm runtimes are compared
- [ ] WASI is discussed with specific proposal details
- [ ] At least 5 concrete use cases with named companies/projects
- [ ] Citations are present (URLs or specific source references)
- [ ] References section exists at the end
- [ ] Report is at least 1000 words
- [ ] Information appears current (not outdated)
- [ ] Web search tools were used

## LLM Judge Rubric

### Criterion 1: Research Depth and Accuracy (Weight: 30%)

**Score 1.0**: Report contains specific, verifiable information from primary sources. Demonstrates genuine research — mentions specific version numbers, dates, project names, and details that could only come from actual investigation. No significant inaccuracies.
**Score 0.75**: Report has good depth with mostly specific information. A few claims may be generic or could be from training data rather than fresh research.
**Score 0.5**: Report covers the topic adequately but relies heavily on general knowledge. Some specific details but many claims are unsourced or generic.
**Score 0.25**: Report is mostly generic knowledge with minimal evidence of actual research.
**Score 0.0**: Report is missing or contains clearly outdated/inaccurate information.

### Criterion 2: Citation Quality (Weight: 25%)

**Score 1.0**: Every major factual claim has an inline citation. References section contains 10+ URLs to primary sources (official docs, blog posts, papers). Citations are relevant and would actually support the claims they're attached to.
**Score 0.75**: Most claims are cited. References section has 5-10 sources. A few claims lack citations.
**Score 0.5**: Some citations present but inconsistent. References section exists but is thin (3-5 sources).
**Score 0.25**: Minimal citations. Only 1-2 sources referenced.
**Score 0.0**: No citations or references section.

### Criterion 3: Coverage Completeness (Weight: 20%)

**Score 1.0**: All five requested areas are thoroughly covered. At least 5 concrete use cases with named entities. Runtimes are meaningfully compared (not just listed). WASI discussion includes specific proposals.
**Score 0.75**: All areas covered with good detail, but one or two areas are thinner than others.
**Score 0.5**: Most areas covered but one is missing or very superficial. Fewer than 5 concrete use cases.
**Score 0.25**: Only 2-3 of the 5 areas addressed with any depth.
**Score 0.0**: Report doesn't address the requested topics.

### Criterion 4: Writing Quality and Structure (Weight: 15%)

**Score 1.0**: Professional research report structure with clear headings, logical flow, and engaging writing. Easy to read and extract key findings. Executive summary or introduction sets context well.
**Score 0.75**: Well-structured with good writing. Minor organizational issues.
**Score 0.5**: Adequate structure but writing could be more polished or organization improved.
**Score 0.25**: Poorly organized or difficult to follow.
**Score 0.0**: No discernible structure.

### Criterion 5: Critical Analysis (Weight: 10%)

**Score 1.0**: Report goes beyond listing facts to provide analysis — identifies trends, draws comparisons, notes tensions (e.g., standardization vs. fragmentation), and offers a nuanced view of the landscape. The limitations section is thoughtful and specific.
**Score 0.75**: Some analytical content beyond pure reporting.
**Score 0.5**: Mostly descriptive with occasional analytical observations.
**Score 0.25**: Pure information listing without analysis.
**Score 0.0**: No analytical content.

## Additional Notes

- WebAssembly outside the browser was chosen because it's a rapidly evolving space where web search adds significant value over training data.
- The topic is technical enough to test research depth but accessible enough that most AI agents should be able to produce something.
- The citation requirement is critical — this task specifically tests the agent's ability to find and reference primary sources, not just regurgitate knowledge.
- Agents without web search capability will score lower on citation quality and research depth, which is intentional — this task tests the full research workflow.
