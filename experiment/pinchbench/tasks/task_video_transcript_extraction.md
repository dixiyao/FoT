---
id: task_video_transcript_extraction
name: Video Transcript Extraction and Summary
category: coding
grading_type: llm_judge
timeout_seconds: 300
workspace_files: []
---

## Prompt

Get the transcript from this YouTube video and create a structured summary:

**Video:** https://www.youtube.com/watch?v=dQw4w9WgXcQ

Extract or download the transcript/subtitles from this video. Then create:

1. **Metadata**: Title, channel, duration, upload date
2. **Full transcript**: Save the complete transcript to `transcript.txt`
3. **Summary**: A 200-300 word summary of the video's content saved to `video_summary.md`
4. **Key points**: A bullet-point list of the main topics or takeaways (in the summary file)
5. **Timestamps**: Notable moments with their timestamps (in the summary file)

Save the full transcript to `transcript.txt` and the structured summary to `video_summary.md`.

## Expected Behavior

The agent should:

1. Access the YouTube video to extract its transcript (via yt-dlp, YouTube API, web scraping, or a transcript service)
2. Parse the transcript into clean, readable text
3. Create a well-structured summary document
4. Identify key points and notable timestamps
5. Save both the raw transcript and the summary

The agent may use various approaches:
- `yt-dlp --write-auto-sub` to download subtitles
- YouTube transcript APIs or services
- Web fetch to access transcript data
- Any other method that produces the transcript text

## Grading Criteria

- [ ] `transcript.txt` created with transcript content
- [ ] `video_summary.md` created with summary
- [ ] Video metadata included (title, channel)
- [ ] Summary is 200-300 words
- [ ] Key points / takeaways listed
- [ ] Timestamps referenced
- [ ] Transcript text is readable (not raw SRT with timing codes)

## LLM Judge Rubric

### Criterion 1: Transcript Extraction (Weight: 30%)

**Score 1.0**: Full transcript extracted and saved as clean, readable text. Timing codes removed or cleanly formatted. Text flows naturally and captures the video's spoken content accurately.
**Score 0.75**: Transcript extracted with minor formatting issues. Most content captured.
**Score 0.5**: Partial transcript obtained or significant formatting problems (raw SRT data, duplicate lines).
**Score 0.25**: Minimal transcript content extracted.
**Score 0.0**: No transcript obtained.

### Criterion 2: Summary Quality (Weight: 25%)

**Score 1.0**: Summary accurately captures the video's main content in 200-300 words. Well-written, informative, and would tell a reader what the video is about without watching it.
**Score 0.75**: Good summary that covers main points. May be slightly outside word count range.
**Score 0.5**: Adequate summary but misses key points or is too brief/verbose.
**Score 0.25**: Very thin summary or mostly inaccurate.
**Score 0.0**: No summary created.

### Criterion 3: Structure and Metadata (Weight: 20%)

**Score 1.0**: Summary file is well-structured with clear sections for metadata, summary, key points, and timestamps. Video title, channel, and duration are correctly identified.
**Score 0.75**: Good structure with most metadata. Minor organizational issues.
**Score 0.5**: Basic structure present but missing some metadata or sections.
**Score 0.25**: Poorly organized with minimal metadata.
**Score 0.0**: No structure.

### Criterion 4: Key Points and Timestamps (Weight: 15%)

**Score 1.0**: Key points are insightful and accurately reflect the video's content. Timestamps correspond to actual moments in the video and are useful for navigation.
**Score 0.75**: Good key points with mostly accurate timestamps.
**Score 0.5**: Key points listed but generic or timestamps inaccurate.
**Score 0.25**: Minimal key points, no timestamps.
**Score 0.0**: Neither key points nor timestamps provided.

### Criterion 5: Technical Execution (Weight: 10%)

**Score 1.0**: Agent efficiently extracted the transcript using appropriate tools. Clean execution without unnecessary retries or failures.
**Score 0.75**: Transcript obtained with minor difficulties but ultimately successful.
**Score 0.5**: Required multiple attempts or workarounds but eventually succeeded.
**Score 0.25**: Significant difficulties with partial success.
**Score 0.0**: Failed to extract transcript.

## Additional Notes

- This video URL (Rick Astley - Never Gonna Give You Up) was chosen because it's one of the most well-known YouTube videos, ensuring long-term availability and having subtitles/transcripts available.
- The task tests the agent's ability to interact with external services (YouTube) and process media content.
- Agents may use various tools: yt-dlp, web fetch, YouTube APIs, or browser automation. All approaches are valid.
- The summary quality matters more than the extraction method — agents that can only get partial transcripts but write excellent summaries should still score well overall.
