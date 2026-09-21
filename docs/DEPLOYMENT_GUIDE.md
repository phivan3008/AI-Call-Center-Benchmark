# Deployment Guide — GPU Benchmark Server

| Field | Value |
|---|---|
| Status | **APPROVED** (2026-09-22) |
| Version | 0.1.0 |
| Date | 2026-09-21 |
| Audience | Operator running benchmarks on Env C (Linux, 1x H100 80GB, 200GB SSD) |

> Commands shown here are the **planned** CLI (ARCHITECTURE §13). Each section notes the phase in which it becomes available. Until then, only the commands of already-delivered phases work.

---

## 1. Server Facts (from review, 2026-09-21)

| Item | Value |
|---|---|
| Docker | **Not available.** Models run as local processes in per-model `uv` venvs. |
| Serving engine | vLLM by default; official inference code as documented fallback per model. |
| Hugging Face | Reachable directly from the server. |
| OpenAI API | Reachable directly from the server (GPT-Realtime baseline runs here). |

---

## 2. Prerequisites (one-time)

| Requirement | Check command | Notes |
|---|---|---|
| NVIDIA driver + CUDA compatible with the pinned vLLM/torch | `nvidia-smi` | Exact minimum versions are fixed in Phase 1 from `vbench env check` output. |
| `uv` | `uv --version` | Installs Python 3.11+ per venv; no system Python changes needed. |
| `git` | `git --version` | |
| `ffmpeg` | `ffmpeg -version` | Audio conversion, telephone-channel simulation. |
| `libsndfile` | `ldconfig -p \| grep sndfile` | FLAC I/O. |
| `zstd` | `zstd --version` | Result bundles. |
| Free disk | `df -h $VBENCH_HOME` | See §5 disk budget. |

No root access is required by the framework itself beyond installing the system packages above.

---

## 3. Directory Layout on the Server

Choose one base directory on the benchmark SSD and export it as `VBENCH_HOME`. **All** paths derive from it; nothing is hardcoded.

```text
$VBENCH_HOME/
  repo/                         # this git repository (uploaded from Env B)
  hf_cache/                     # HF_HOME: model weights
  runtimes/<model>/.venv        # one uv venv per model (created by `vbench model prepare`)
  datasets_audio/               # built dataset audio (git-ignored)
  artifacts/                    # raw / processed / reports / dashboards
  bundles/                      # result archives to return
```

---

## 4. Environment Variables

Create `$VBENCH_HOME/.env` (never commit it; `chmod 600`):

```bash
VBENCH_HOME=/path/to/benchmark_ssd/vbench
HF_HOME=${VBENCH_HOME}/hf_cache
HF_TOKEN=...                 # only if a gated model requires it
OPENAI_API_KEY=...           # GPT-Realtime baseline and LLM judge (if an OpenAI model is chosen as judge)
VBENCH_LOG_LEVEL=INFO
```

Security notes:

- Keys are read from the environment only. The framework never writes key values to logs, manifests, or bundles; `env check` reports only "present / missing".
- Before returning a bundle, `vbench bundle create` scans it for strings matching configured secret patterns and refuses to create it if any are found.

---

## 5. Disk Budget Procedure

The 200GB SSD cannot be assumed to hold all models at once. Per-model weight and venv sizes are **measured** at Phase 2/9 (not estimated) and recorded in `configs/models/<model>.yaml: disk_measured_gb`.

Procedure for each model:

1. `vbench env check` — shows free space and the measured size of the next model (once known).
2. `vbench model prepare --model <id>` — refuses to start if free space < measured size + `disk_safety_margin_gb` (config).
3. Run the benchmark.
4. `vbench bundle create <run_id>` — archive results first.
5. `vbench model evict --model <id>` — removes weights from `hf_cache` and the model's venv; keeps artifacts.

Raw audio artifacts can be large; the `full` profile stores output audio as FLAC and the bundle step can exclude audio (`--no-audio`) when only metrics and timelines need to be returned. Layer 4 (voice quality) needs output audio, so L4 bundles always include it.

---

## 6. Standard Operating Procedure

### 6.1 Update code (every checkpoint)

On Env B: `git pull` the branch named in the checkpoint instructions, then upload `repo/` to `$VBENCH_HOME/repo/`. On the server:

```bash
cd $VBENCH_HOME/repo
git log -1 --oneline          # confirm the commit named in the checkpoint instructions
uv sync                       # harness environment (not model venvs)
```

### 6.2 Preflight and mock round trip (available from Phase 1)

```bash
set -a; source $VBENCH_HOME/.env; set +a
uv run vbench env check --output $VBENCH_HOME/bundles/env_report.json
uv run vbench run --model mock --profile mock_smoke      # CPU-only pipeline check, prints RUN_ID
uv run vbench bundle create <RUN_ID>
```

`env check` exits non-zero if a required check fails. Stop and report back in that case. Mock runs are `is_mock=true` and can never be ranked.

### 6.3 Per-model run (available from Phase 2)

```bash
MODEL=qwen3-omni-30b-a3b-fp8
uv run vbench data prepare --profile smoke
uv run vbench model prepare --model $MODEL
uv run vbench model serve   --model $MODEL        # starts vLLM/official server + gateway, waits for health
uv run vbench run --model $MODEL --profile smoke  # prints RUN_ID
uv run vbench evaluate $RUN_ID --evaluators gpu   # stops the model server first, then loads evaluator models
uv run vbench bundle create $RUN_ID
uv run vbench model evict --model $MODEL          # only when done with this model
```

`scripts/run_all.sh` (Phase 10) wraps this loop for all models and is resumable.

### 6.4 Resume after failure

```bash
uv run vbench run --resume $RUN_ID
```

Completed samples are skipped; failed samples are retried once, then recorded with `status=error`.

### 6.5 GPU exclusivity

Only one GPU workload at a time: `vbench` takes a lock file (`$VBENCH_HOME/.gpu.lock`) while a model server or GPU evaluator is running. Other GPU jobs on the server during a run invalidate latency and infrastructure results — the NVML sampler records foreign processes and the run is flagged `gpu_contended=true`.

---

## 7. Returning Results

Return to Claude (via the transfer machine or attachment):

| What | Where |
|---|---|
| Result bundle(s) | `$VBENCH_HOME/bundles/<run_id>.tar.zst` + `<run_id>.SHA256SUMS` |
| Env report (when requested) | `$VBENCH_HOME/bundles/env_report.json` |
| On failure: logs | included in the bundle (`logs/run.jsonl`); if bundling itself failed, send `artifacts/raw/<run_id>/logs/` and the terminal output |

Also accepted: CSV/JSON/parquet/HTML reports, screenshots, profiler output. Claude verifies checksums first (`vbench bundle verify`) and analyzes only verified data.

---

## 8. Troubleshooting (grows with each phase)

| Symptom | First step |
|---|---|
| `model serve` health check times out | Check `artifacts/raw/<run_id>/logs/runtime_<model>.log`; send it back. |
| CUDA OOM at load | Confirm no other GPU processes (`nvidia-smi`); send `env_report.json` and the runtime log. |
| vLLM refuses the model architecture | Expected for some models; the model YAML's `runtime.engine` should be `official`. Send the log so the runtime can be fixed. |
| OpenAI connection errors | Check `OPENAI_API_KEY` present and outbound HTTPS; `vbench env check` shows reachability. |
