# Japanese AI Call Center Benchmark
## Overview
This repository contains a comprehensive benchmark framework for evaluating Native Speech-to-Speech AI models for Japanese AI Call Center deployment.
The benchmark focuses on real-world business requirements rather than purely academic performance.
Primary objective:
Identify the best model for production deployment in Japanese customer service and appointment booking scenarios.
---
# Candidate Models
## Open Source Models
- Qwen3-Omni-30B-A3B-FP8
- MiniCPM-o 4.5
- StepAudio 2.5 Realtime
- GLM-4-Voice-9B
- LLaMA-Omni 2
- Baichuan-Omni-1.5
## Commercial Baseline
- GPT-Realtime 2.1
---
# Benchmark Layers
## Layer 1: Japanese Capability Benchmark
Measure:
- CER
- WER
- Japanese Understanding
- Intent Classification
- Slot Extraction
- Keigo Compliance
- Long Context Memory
Deliverables:
- japanese_score.json
- japanese_report.html
---
## Layer 2: Realtime Voice Benchmark
Measure:
- TTFA P50
- TTFA P95
- TTFA P99
- Interrupt Latency
- Barge-In Performance
- Turn Taking Quality
- Long Call Stability
Deliverables:
- latency_report.json
- realtime_report.html
---
## Layer 3: Tool Calling Benchmark
Measure:
- Booking Success Rate
- FAQ Accuracy
- CRM Lookup Success
- Call Transfer Accuracy
- JSON Exact Match
- Task Completion Rate
Deliverables:
- toolcalling_report.json
- task_completion_report.html
---
## Layer 4: Voice Quality Benchmark
Measure:
- MOS
- Naturalness
- Pronunciation
- Japanese Accent Quality
- Emotional Expression
Deliverables:
- voice_quality_report.json
- mos_summary.csv
---
## Layer 5: Infrastructure & TCO Benchmark
Measure:
- GPU Utilization
- VRAM Usage
- Throughput
- Concurrent Calls
- Cost Per Call
- Cost Per Minute
- Total Cost of Ownership
Deliverables:
- infra_report.json
- tco_report.html
---
# Scoring Methodology
Final Business Score:
```text
35% Japanese Quality
25% Task Completion
15% Barge-In
15% Cost
10% Voice Quality
```
Final ranking is generated from measured benchmark metrics only.
No estimated scores are allowed.
---
# Repository Structure
```text
.
├── benchmark/
│   ├── layer1_japanese/
│   ├── layer2_realtime/
│   ├── layer3_toolcalling/
│   ├── layer4_voice/
│   └── layer5_infra/
│
├── datasets/
│
├── configs/
│
├── scripts/
│
├── artifacts/
│   ├── raw/
│   ├── processed/
│   ├── reports/
│   └── dashboards/
│
├── docs/
│
├── tests/
│
├── CLAUDE.md
│
└── README.md
```
---
# Execution Workflow
## Step 1
Claude Code generates source code.
## Step 2
Source code is committed to GitHub.
## Step 3
Repository is downloaded to transfer machine.
## Step 4
Repository is uploaded to GPU server.
## Step 5
Benchmark executes on GPU server.
## Step 6
Artifacts are collected.
## Step 7
Artifacts are provided back to Claude.
## Step 8
Claude analyzes results and improves framework.
---
# Environment Topology
Developer PC
```text
Claude Code
    ↓
GitHub
```
Transfer Machine
```text
GitHub
    ↓
GPU Server
```
GPU Benchmark Server
```text
Linux
1x H100 80GB
200GB SSD
```
---
# Ground Rules
1. Never fabricate benchmark results.
2. Never assume model performance.
3. Every metric must be reproducible.
4. Every benchmark run must generate artifacts.
5. Human-provided benchmark outputs are the source of truth.
6. Model rankings must come from measured results only.
---
# Deliverables
Required final outputs:
- leaderboard.csv
- leaderboard.json
- leaderboard.html
- benchmark_summary.json
Required reports:
- Layer 1 Japanese Report
- Layer 2 Realtime Report
- Layer 3 Tool Calling Report
- Layer 4 Voice Quality Report
- Layer 5 Infrastructure Report
---
# Success Criteria
The benchmark framework should allow reproducible comparisons between all target models and generate a final production-focused ranking suitable for AI Call Center deployment.