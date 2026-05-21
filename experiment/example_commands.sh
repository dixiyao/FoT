# Annotated command examples.
# This file is organized by workflow stage and redacts credentials as xxx_xxx.

# -----------------------------------------------------------------------------
# 1. Paper Reading: Scrape, Read, Aggregate
# -----------------------------------------------------------------------------

# Scrape 10 ICLR 2023 top-5 papers into a local paper folder.
# python scraper.py -n 10 -o data/papers/iclr23_top5

# Read the scraped papers and extract answers to the target research question.
python client.py -t "Find out the solutions to the proposed research questions in the paper." -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B -p data/papers/iclr23_top5 -n 10 -d cuda

# Aggregate paper-level insight JSON files into one encyclopedia.
python server.py -i output -f output/paper_*.json -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B -d cuda -o encyclopedia --r1 0.95 --r2 0.4

# Scrape diffusion-related ICLR 2023 papers across top-5, top-25, and poster subsets.
python scraper.py -f diffusion -o data/papers/iclr23_diffusion --top5 --top25 --poster

# Read diffusion papers and extract solution-oriented insights.
python client.py -t "Resolve a critical question in diffusion model and related problems according to the paper" -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B -p data/papers/iclr23_diffusion -d cuda -n 10

# Scrape ICLR 2023 top-25 papers.
python scraper.py -o data/papers/iclr23_top25 --top25

# Scrape accepted ICLR 2024 oral, spotlight, and poster papers.
python scraper.py -o data/papers/iclr24_accept_oral_spotlight --accept-oral --accept-spotlight --accept-poster

# Read ICLR 2024 papers and summarize each paper's main non-incremental contribution.
python client.py -t "Summarize the main non-incremental novel contribution of the paper" -p data/papers/iclr24_accept_oral_spotlight -d cuda --output iclr24_accept_oral_spotlight --papers-dir data/papers/iclr24_accept_oral_spotlight --use-api --api-provider gemini --api-key xxx_xxx

# Aggregate the ICLR 2024 paper-reading outputs into an encyclopedia.
python server_text.py -i iclr24_accept_oral_spotlight --use-api --api-provider gemini --api-key xxx_xxx -d cuda -o encyclopedia_iclr24_accept_oral_spotlight

# Scrape arXiv papers from multiple subject areas into one folder.
python scraper.py --arxiv-subject physics -n 100 -o data/papers/arxiv_papers
python scraper.py --arxiv-subject cs -n 100 -o data/papers/arxiv_papers
python scraper.py --arxiv-subject math -n 100 -o data/papers/arxiv_papers
python scraper.py --arxiv-subject chemistry -n 100 -o data/papers/arxiv_papers

# Run paper insight reading on a sorted ICLR 2024 paper folder with 25 papers per agent read.
python task_paper_insight_reading.py -p data/papers/iclr24_accept_oral_spotlight_sortby_area -d cuda -o iclr24_accept_oral_spotlight_sortby_area_num25 --mode text --use-api --api-provider gemini --api-key xxx_xxx --file-pattern *.pdf --model gemini-2.0-flash --agent-read-num 25

# -----------------------------------------------------------------------------
# 2. Insight Library Aggregation Variants
# -----------------------------------------------------------------------------

# Build a 10-insight Gemini library from ICLR 2024 accepted paper traces.
python server_text.py --use-api --api-provider gemini --api-key xxx_xxx -i iclr24_accept_oral_spotlight -o library_iclr24_accept_oral_spotlight_10 --num-insights 10

# Build a Gemini Flash library from the TRT ICLR 2024 output folder.
python server_text.py --use-api --api-provider gemini --api-key xxx_xxx -i iclr24_accept_oral_spotlight_trt -o iclr24_accept_oral_spotlight_trt --model gemini-2.0-flash

# Aggregate hard 2025 LiveMathBench Gemini outputs.
python server_text.py --use-api --api-provider gemini --api-key xxx_xxx -i math_output_gemini/livemathbench_hard_2025 -o math_output_gemini/livemathbench_hard_2025

# Aggregate Gemini-collected, DeepSeek-served math traces with a local CUDA model.
python server_text.py -i math_output_Cgemini_Sdeepseek -o math_output_Cgemini_Sdeepseek --device cuda --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -o math_metacognitive_deepseek_gemini

# Re-aggregate ICLR 2024 outputs with Gemini 3 Pro preview.
python server_text.py -i iclr24_accept_oral_spotlight --use-api --api-provider gemini --api-key xxx_xxx -d cuda -o iclr24_accept_oral_spotlight_client2_server3 --model gemini-3-pro-preview

# Aggregate a large mixed-domain output folder with chunking.
python server_text_chunk.py -i mix3_output_gemini -o encycloepedia_mix3_deepseek -d cuda -m deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --token-limit 16384

# Build a naive appended library from Gemini math outputs.
python server_naive_append.py -i math_output_gemini -o library_native_append

# Compact ICLR 2024 traces with the Claude-style compacting server.
python server_claude_compact.py -i iclr24_accept_oral_spotlight -o library_claude_compact_iclr24_accept_oral_spotlight_chunk500000 --gemini-api-key xxx_xxx --gemini-model gemini-2.0-flash --chunk-size 500000

# Build a chain-of-density-style library for ICLR 2024 traces.
python server_cod.py -i iclr24_accept_oral_spotlight -o library_cod_iclr24_accept_oral_spotlight_5000000 --gemini-api-key xxx_xxx --gemini-model gemini-2.0-flash --chunk-size 5000000

# Aggregate HLE multimodal traces.
python server_text.py -i hle_multimodal --use-api --api-provider gemini --api-key xxx_xxx -d cuda -o hle_multimodal

# -----------------------------------------------------------------------------
# 3. Domain Benchmarks: Math, Mixed, Multimodal, and TTS
# -----------------------------------------------------------------------------

# Resume math-domain benchmark from the encyclopedia step with a local DeepSeek model.
python task_benchmark_domain.py --num-iterations 2 --datasets aime24 aime25 livemathbench_amc livemathbench_ccee livemathbench_cnmo livemathbench_wlpmc livemathbench_hard_2024 livemathbench_hard_2025 --mode text --device cuda --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -o math_metacognitive_deepseek --resume_from_encyclopedia_step

# Run a split FoT math benchmark with Gemini 3.1 Pro preview.
python task_benchmark_domain.py --num-iterations 2 --datasets aime24 aime25 livemathbench_amc livemathbench_ccee livemathbench_cnmo livemathbench_wlpmc livemathbench_hard_2024 livemathbench_hard_2025 --mode text --device cuda --use-api --api-provider gemini --api-key xxx_xxx --output math_gemini31pro_fot_run2 --split 0.5 --seed 24 --model gemini-3.1-pro-preview

# Run a split FoT math benchmark with Gemini 2.5 Flash Lite.
python task_benchmark_domain.py --num-iterations 2 --datasets aime24 aime25 livemathbench_amc livemathbench_ccee livemathbench_cnmo livemathbench_wlpmc livemathbench_hard_2024 livemathbench_hard_2025 --mode text --device cuda --use-api --api-provider gemini --api-key xxx_xxx -o math_gemini25flashlite_fot_run2 --model gemini-2.5-flash-lite --split 0.5 --seed 24

# Run isolated evaluation on a mixed math, science, and coding dataset.
python task_benchmark_domain.py --num-iterations 1 --datasets aime24 aime25 gpqa_diamond gpqa livecodebench_lite --mode text --device cuda --use-api --api-provider gemini --api-key xxx_xxx -o mix3_gemini25flashlite_isolated_run2 --model gemini-2.5-flash-lite --split 0.5 --seed 24 --eval-only

# Evaluate DeepSeek on math tasks with a pre-existing encyclopedia.
python task_benchmark_domain.py --num-iterations 1 --datasets aime24 aime25 livemathbench_amc livemathbench_ccee livemathbench_cnmo livemathbench_wlpmc livemathbench_hard_2024 livemathbench_hard_2025 --mode text --device cuda --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -o math_output_Cdeepseek_Sgemini --eval-only --encyclopedia math_output_Cdeepseek_Sgemini/encyclopedia.json

# Run five mixed-domain split iterations with Gemini 3.1 Pro preview.
python task_benchmark_domain.py --num-iterations 5 --datasets aime24 aime25 gpqa_diamond gpqa livecodebench_lite --mode text --device cuda --use-api --api-provider gemini --api-key xxx_xxx -o mix3_gemini_split05 --model gemini-3.1-pro-preview --split 0.5 --seed 42 --mode text

# Run a domain-insight benchmark over mixed datasets.
python task_benchmark_domain_insight.py --datasets aime24 aime25 gpqa_diamond gpqa livecodebench_lite --device cuda --use-api --api-provider gemini --api-key xxx_xxx

# Run task_benchmark_domain with the metacognitive client for explicit reflection-style insight extraction.
python task_benchmark_domain.py --num-iterations 2 --datasets aime24 aime25 gpqa_diamond --mode text --device cuda --use-api --api-provider gemini --api-key xxx_xxx -o mix3_metacognitive_gemini --model gemini-3.1-pro-preview --split 0.5 --seed 42 --client metacognitive

# Run task_benchmark_domain with the TRT client for trace/reasoning-transfer style insight extraction.
python task_benchmark_domain.py --num-iterations 2 --datasets aime24 aime25 gpqa_diamond --mode text --device cuda --use-api --api-provider gemini --api-key xxx_xxx -o mix3_trt_gemini --model gemini-3.1-pro-preview --split 0.5 --seed 42 --client trt

# Run task_benchmark_domain with the HyperAgents client for hyperagent-style insight extraction.
python task_benchmark_domain.py --num-iterations 2 --datasets aime24 aime25 gpqa_diamond --mode text --device cuda --use-api --api-provider gemini --api-key xxx_xxx -o mix3_hyperagents_gemini --model gemini-3.1-pro-preview --split 0.5 --seed 42 --client hyperagents

# Run task_benchmark_domain with the EvolvePrompt client for evolved-prompt insight extraction.
python task_benchmark_domain.py --num-iterations 2 --datasets aime24 aime25 gpqa_diamond --mode text --device cuda --use-api --api-provider gemini --api-key xxx_xxx -o mix3_evolveprompt_gemini --model gemini-3.1-pro-preview --split 0.5 --seed 42 --client evolveprompt

# Run task_benchmark_domain with the ACE client. The CLI supports --client ace; there is no --client ave option in task_benchmark_domain.py.
python task_benchmark_domain.py --num-iterations 2 --datasets aime24 aime25 gpqa_diamond --mode text --device cuda --use-api --api-provider gemini --api-key xxx_xxx -o mix3_ace_gemini --model gemini-3.1-pro-preview --split 0.5 --seed 42 --client ace

# Run TTS evaluation without encyclopedia guidance.
python task_benchmark_domain_tts.py --datasets aime24 aime25 gpqa_diamond gpqa livecodebench_lite --device cuda --eval-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -o mix3_output_deepseek/tts_single_model/ --eval_only

# Run TTS evaluation with encyclopedia guidance.
python task_benchmark_domain_tts.py --datasets aime24 aime25 gpqa_diamond gpqa livecodebench_lite --device cuda --eval-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -o mix3_output_deepseek/tts_with_encyclopedia/ --eval_only --encyclopedia mix3_output_gemini/encyclopedia_all.json

# Run a small HLE multimodal benchmark sample.
python task_benchmark_domain_multimodal.py --num-iterations 3 --datasets hle --mode text --device cuda --use-api --api-provider gemini --api-key xxx_xxx -o hle_multimodal_problem10 --max-problems 10

# Run multimodal evaluation across HLE, GPQA Diamond, and LiveMathBench with an encyclopedia.
python task_benchmark_domain_multimodal.py --num-iterations 2 --datasets hle gpqa_diamond livemathbench_hard_2025 --mode text --device cuda --use-api --api-provider gemini --api-key xxx_xxx -o hle_multimodal --encyclopedia hle_multimodal/encyclopedia.json

# Run trace-appending math benchmark using prior hyperagent traces.
python task_benchmark_domain_traceappending.py --trace-folder math_gemini_hyperagents --datasets aime24 aime25 livemathbench_amc livemathbench_ccee livemathbench_cnmo livemathbench_wlpmc livemathbench_hard_2024 livemathbench_hard_2025 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --output-dir traceappending_output_math_deepseek --device cuda --api-key xxx_xxx

# -----------------------------------------------------------------------------
# 4. Training and Tuning Baselines
# -----------------------------------------------------------------------------

# Tune or evaluate a DeepSeek baseline from mixed-domain traces.
python task_benchmark_baseline_tune.py --datasets aime24 aime25 gpqa_diamond gpqa livecodebench_lite --device cuda --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --output-dir mix3_output_deepseek/tune_with_traces --skip-step --trace-root mix3_output_gemini/ --num-iterations 1 --eval-checkpoint mix3_output_deepseek/tune_with_traces/iter_01 --eval-mode

# Tune a DeepSeek baseline from math traces for two iterations.
python task_benchmark_baseline_tune.py --datasets aime24 aime25 livemathbench_amc livemathbench_ccee livemathbench_cnmo livemathbench_wlpmc livemathbench_hard_2024 livemathbench_hard_2025 --device cuda --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --output-dir math_output_2/tune_with_traces --skip-step --trace-root math_output_2/ --num-iterations 2

# Evaluate a tuned DeepSeek baseline checkpoint on math tasks.
python task_benchmark_baseline_tune.py --datasets aime24 aime25 livemathbench_amc livemathbench_ccee livemathbench_cnmo livemathbench_wlpmc livemathbench_hard_2024 livemathbench_hard_2025 --device cuda --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --output-dir math_output_2/tune_with_traces --skip-step --trace-root math_output_2/ --num-iterations 1 --eval-checkpoint math_output_2/tune_with_traces/iter_01 --eval-mode

# -----------------------------------------------------------------------------
# 5. ICLR Acceptance and Baseline Checkers
# -----------------------------------------------------------------------------

# Check ICLR 2025 acceptance with an ICLR 2024 insight library and Gemini judge.
python checker_iclr.py --gemini-key xxx_xxx --encyclopedia iclr24_accept_oral_spotlight_trt/encyclopedia.json --year 2025 --output guided_accept_results_library_iclr24_accept_oral_spotlight_trt.json --accept-oral --accept-spotlight --accept-poster --sleep 0 --gemini-model gemini-2.0-flash --or-username dixi.yao@mail.utoronto.ca --or-password xxx_xxx

# Check ICLR 2025 acceptance using a local DeepSeek model and prior insights.
python checker_iclr.py --encyclopedia insight_paper_reading_2024 --year 2025 --output guided_accept_results_deepseek_r1_qwen7b.json --accept-oral --accept-spotlight --accept-poster --sleep 0 --device cuda --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# Run an ICLR acceptance baseline generated from ICLR 2024 keywords.
python checker_iclr_baseline.py --gemini-key xxx_xxx --year 2025 --accept-oral --accept-spotlight --accept-poster --generate-mode iclr2024_keywords

# Run an ICLR acceptance baseline from general knowledge with 10 skills.
python checker_iclr_baseline.py --gemini-key xxx_xxx --year 2025 --accept-oral --accept-spotlight --accept-poster --generate-mode general_knowledge --num-skills 10 --gemini-model gemini-2.0-flash

# Check a pre-generated 200-skill general-knowledge baseline on ICLR 2024.
python checker_iclr_baseline.py --gemini-key xxx_xxx --year 2024 --accept-oral --accept-spotlight --accept-poster --generate-mode general_knowledge --num-skills 200 --phase check --skills-file gemini_baseline_skills_200.json

# Run an ICLR acceptance baseline with RAG over ICLR 2024 papers.
python checker_iclr_baseline.py --gemini-key xxx_xxx --year 2025 --accept-oral --accept-spotlight --accept-poster --generate-mode rag --num-skills 10 --rag-papers-dir data/papers/iclr24_accept_oral_spotlight --gemini-model gemini-2.0-flash

# Evaluate acceptance predictions on a fixed sampled-paper file.
python checker_iclr_givensample.py --sample sampled_papers_2024.json --encyclopedia insight_paper_reading_2024/encyclopedia_iclr24.json --api-type gemini --key xxx_xxx --api-model gemini-2.5-flash-lite --output results_sample_2024_50samples_gemini25flashlite.json

# -----------------------------------------------------------------------------
# 6. OpenClaw PinchBench and ClawEval
# -----------------------------------------------------------------------------

# Run 15 OpenClaw PinchBench FoT iterations with Gemini 3.1 Pro preview and medium thinking.
python task_openclaw_pinchbench.py --iterations 15 --use-api --api-provider gemini --api-key xxx_xxx --output-dir pinchbench_openclaw_gemini_10iter_meduim --model gemini-3.1-pro-preview --api-model gemini-3.1-pro-preview --thinking-level medium

# Run one OpenClaw PinchBench iteration with predefined OpenClaw skills installed.
python task_openclaw_pinchbench.py --iterations 1 --use-api --api-provider gemini --api-key xxx_xxx --output-dir pinchbench_openclaw_gemini_wskills --model gemini-3.1-pro-preview --api-model gemini-3.1-pro-preview --thinking-level high --openclaw-skill

# Run one OpenClaw PinchBench iteration intended for RAG-trace output.
python task_openclaw_pinchbench.py --iterations 1 --use-api --api-provider gemini --api-key xxx_xxx --output-dir pinchbench_openclaw_gemini_ragtrace --model gemini-3.1-pro-preview --api-model gemini-3.1-pro-preview --thinking-level high

# Evaluate Gemini 2.5 Flash Lite on non-V1 PinchBench tasks with an existing FoT encyclopedia.
python task_openclaw_pinchbench.py --iterations 1 --use-api --api-provider gemini --api-key xxx_xxx --output-dir pinchbench_openclaw_gemini-2.5-flash-lite_53task_fot_eval_run2 --model gemini-2.5-flash-lite --api-model gemini-2.5-flash-lite --encyclopedia pinchbench_openclaw_gemini-2.5-flash-lite_53task_fot_eval/encyclopedia.json --eval-only --judge google/gemini-3.1-flash-lite-preview --timeout-multiplier 3 --exclude-V1

# Evaluate Gemini 2.5 Flash Lite on non-V1 PinchBench tasks without an encyclopedia.
python task_openclaw_pinchbench.py --iterations 1 --use-api --api-provider gemini --api-key xxx_xxx --output-dir pinchbench_openclaw_gemini-2.5-flash-lite_53task_isolated_eval_run2 --model gemini-2.5-flash-lite --api-model gemini-2.5-flash-lite --eval-only --judge google/gemini-3.1-flash-lite-preview --timeout-multiplier 3 --exclude-V1

# Evaluate Gemini 3.1 Pro preview on non-V1 PinchBench tasks with an existing FoT encyclopedia.
python task_openclaw_pinchbench.py --iterations 1 --use-api --api-provider gemini --api-key xxx_xxx --output-dir pinchbench_openclaw_gemini-3.1-pro-preview_53task_fot_eval_run2 --model gemini-3.1-pro-preview --api-model gemini-3.1-pro-preview --encyclopedia pinchbench_openclaw_gemini-3.1-pro-preview_53task_fot_eval/encyclopedia.json --eval-only --judge google/gemini-3.1-flash-lite-preview --timeout-multiplier 3 --exclude-V1

# Evaluate Gemini 3.1 Pro preview on non-V1 PinchBench tasks without an encyclopedia.
python task_openclaw_pinchbench.py --iterations 1 --use-api --api-provider gemini --api-key xxx_xxx --output-dir pinchbench_openclaw_gemini-3.1-pro-preview_53task_isolated_eval_run2 --model gemini-3.1-pro-preview --api-model gemini-3.1-pro-preview --eval-only --judge google/gemini-3.1-flash-lite-preview --timeout-multiplier 3 --exclude-V1

# Run ClawEval with PinchBench-to-ClawEval transfer using Docker.
CONTAINER_ENGINE=docker python task_openclaw_claweval.py --iterations 1 --use-api --api-provider gemini --api-key xxx_xxx --output-dir claweval_pinch2clawval_gemini25flashlite --model gemini-2.5-flash-lite --api-model gemini-2.5-flash-lite --encyclopedia claweval_pinch2clawval_gemini25flashlite/encyclopedia.json

# Run a same-task multi-agent comparison on LiveMathBench hard 2025.
python task_benchmark_sametask.py --datasets livemathbench_hard_2025 --agent-models gemini-3.1-pro-preview gemini-2.5-flash gemini-2.5-flash-lite gemini-2.5-pro --use-api --api-provider gemini --api-key xxx_xxx --num-iterations 3 -o sametask_agents_output

# Run PinchBench with the EXPEL-style RAG store.
python task_openclaw_pinchbench_expel.py --iterations 1 --api-key xxx_xxx --output-dir pinchbench_openclaw_gemini_ragtrace --api-model gemini-3.1-pro-preview --judge google/gemini-3.1-pro-preview --thinking-level high --suite all --rag-store corpus_expel

# -----------------------------------------------------------------------------
# 7. Privacy and Reverse-Prompt Checks
# -----------------------------------------------------------------------------

# Check whether HLE multimodal encyclopedia content leaks target benchmark items.
python checker_privacy.py --encyclopedia hle_multimodal/encyclopedia.json --check_array hle gpqa_diamond gpqa livemathbench_hard_2025 --gemini-token-model gemini-3.1-pro-preview --gemini-api-key xxx_xxx --print-matches 1

# Check whether Gemini math traces overlap with PinchBench items.
python checker_privacy.py --trace_dataset math_output_gemini --check_array pinchbench --gemini-token-model gemini-3.1-pro-preview --gemini-api-key xxx_xxx

# Check whether ICLR 2024 hyperagent traces overlap with ICLR 2023 top-5 papers.
python checker_privacy.py --trace_dataset iclr24_accept_oral_spotlight_hyperagents --check_array data/papers/iclr23_top5/ --gemini-token-model gemini-3.1-pro-preview --gemini-api-key xxx_xxx

# Reverse-prompt check for Gemini 3.1 PinchBench outputs.
python checker_reverse_prompt.py --input-folder pinchbench_openclaw_gemini/iter_04 --benchmark pinchbench --tasks-dir pinchbench/tasks --use-api --api-provider gemini --api-key xxx_xxx --api-model gemini-2.5-pro --output reverse_prompt_repor_gemini31.json

# Reverse-prompt check for Gemini 2.5 Flash Lite PinchBench outputs.
python checker_reverse_prompt.py --input-folder pinchbench_openclaw_gemini-2.5-flash-lite/iter_02 --benchmark pinchbench --tasks-dir pinchbench/tasks --use-api --api-provider gemini --api-key xxx_xxx --api-model gemini-2.5-pro --output reverse_prompt_repor_gemini25.json

# -----------------------------------------------------------------------------
# 8. External Provider and Authentication Examples
# -----------------------------------------------------------------------------

# Log in to Hugging Face CLI with a redacted token.
hf auth login --api-key xxx_xxx

# Run a mixed-domain benchmark through OpenRouter.
python task_benchmark_domain.py --num-iterations 2 --datasets aime24 aime25 gpqa_diamond --mode text --use-api --api-provider openrouter --api-key xxx_xxx --model anthropic/claude-sonnet-4 -o mix_output_openrouter

# Export an OpenRouter API key for commands that read OPENROUTER_API_KEY.
export OPENROUTER_API_KEY=xxx_xxx
