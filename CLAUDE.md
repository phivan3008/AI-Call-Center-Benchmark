# AI Voice Benchmark Project
## Mission
Build a production-grade benchmark framework for evaluating Native Speech-to-Speech (Speech-to-Speech / Audio-to-Audio) AI models for Japanese AI Call Center deployment.
This repository is focused on benchmarking, evaluation, reporting, and leaderboard generation.
The objective is to identify the most suitable model for Japanese call center operations rather than finding the strongest research model.
---
# Target Models
## OSS Candidates
- Qwen3-Omni-30B-A3B-FP8
- MiniCPM-o 4.5
- StepAudio 2.5 Realtime
- GLM-4-Voice-9B
- LLaMA-Omni 2
- Baichuan-Omni-1.5
## Commercial Baseline
- GPT-Realtime 2.1
---
# Infrastructure
## Environment A
Developer Machine
Capabilities:
- Claude Code
- Git
- GitHub
Limitations:
- Cannot run GPU benchmarks
- Cannot host large models
- Cannot execute latency benchmarks
Responsibilities:
- Source code generation
- Architecture design
- Test framework generation
- Analysis of benchmark results provided later by user
---
## Environment B
Transfer Machine
Capabilities:
- GitHub access
- Upload source code to GPU server
Responsibilities:
- Download repository
- Upload repository to benchmark server
---
## Environment C
GPU Benchmark Server
Specifications:
- Linux
- 1x H100 80GB
- 200GB SSD
Responsibilities:
- Execute benchmark workloads
- Collect raw metrics
- Generate benchmark artifacts
Limitations:
- Not directly controlled by Claude
---
# Human-In-The-Loop Requirement
IMPORTANT:
Claude never performs real GPU benchmarking.
Claude never invents benchmark results.
Claude never estimates benchmark results without evidence.
When benchmark execution is required:
1. Generate code.
2. Stop.
3. Ask user to execute benchmark.
4. Wait for benchmark outputs.
5. Analyze outputs.
6. Improve framework.
Accepted benchmark inputs:
- csv
- json
- parquet
- html
- screenshots
- benchmark logs
- profiler outputs
---
# Benchmark Goal
Evaluate models for Japanese AI Call Center deployment.
Primary dimensions:
1. Japanese language capability
2. Conversational realtime performance
3. Tool calling reliability
4. Voice quality
5. Infrastructure cost
---
# Benchmark Layers
## Layer 1
Japanese Capability Benchmark
Evaluate:
- ASR
- CER
- WER
- Intent Classification
- Slot Extraction
- Japanese Understanding
- Keigo Compliance
- Long Context Memory
---
## Layer 2
Realtime Voice Benchmark
Evaluate:
- Time To First Audio (TTFA)
- Interrupt Latency
- Barge-In
- Turn Taking
- Duplex Interaction
- Long Conversation Stability
---
## Layer 3
Tool Calling Benchmark
Evaluate:
- Booking
- FAQ
- CRM Lookup
- Call Transfer
- Structured Output
Metrics:
- Tool Success Rate
- JSON Accuracy
- Hallucinated Tool Rate
- Task Completion Rate
---
## Layer 4
Voice Quality Benchmark
Evaluate:
- MOS
- Naturalness
- Accent Quality
- Pronunciation
- Emotional Expression
- Consistency
Human reviewers may be required.
---
## Layer 5
Infrastructure Benchmark
Evaluate:
- VRAM Usage
- Throughput
- Concurrent Calls
- GPU Utilization
- Cost Per Call
- Cost Per Minute
- TCO
---
# Repository Rules
Always prioritize:
1. Reproducibility
2. Automation
3. Traceability
4. Measurability
Every benchmark run must be reproducible.
Every metric must explain:
- how measured
- source data
- timestamp
- model version
---
# Development Principles
Before implementing:
1. Design architecture
2. Create implementation plan
3. Create benchmark spec
4. Implement incrementally
Do not write large amounts of code without architecture approval.
---
# Technology Stack
Preferred:
- Python 3.11+
- Typer
- Pydantic
- FastAPI
- Pytest
- Pandas
- Polars
- Plotly
- DuckDB
Optional:
- Streamlit
- Gradio
Avoid:
- Jupyter-only solutions
- hardcoded credentials
- hardcoded paths
---
# Directory Structure
artifacts/
    raw/
    processed/
    reports/
    dashboards/
datasets/
benchmark/
tests/
docs/
configs/
scripts/
---
# Required Outputs
Every benchmark execution must generate:
- CSV
- JSON
- HTML report
Required artifacts:
leaderboard.csv
leaderboard.json
leaderboard.html
benchmark_summary.json
---
# Ranking Philosophy
Never rank models by a single metric.
Business Score:
35% Japanese Quality
25% Task Completion
15% Barge-In
15% Cost
10% Voice Quality
Final ranking must be derived exclusively from measured benchmark results.
---
# Documentation Requirements
Maintain:
docs/
including:
- ARCHITECTURE.md
- IMPLEMENTATION_ROADMAP.md
- DATASET_SPEC.md
- METRIC_DEFINITIONS.md
- DEPLOYMENT_GUIDE.md
---
# Dataset Principles
Never fabricate benchmark labels.
Synthetic datasets must be clearly marked.
Human-reviewed datasets must be stored separately.
Japanese benchmark datasets must be versioned.
---
# Coding Standards
Use:
- type hints
- pydantic models
- unit tests
Target:
>= 80% test coverage
Logging:
structured logging only
No print-debugging in production code.
---
# Benchmark Execution Rule
If benchmark execution requires GPU resources:
Generate code.
Generate commands.
Generate deployment instructions.
STOP.
Wait for benchmark results from user.
`