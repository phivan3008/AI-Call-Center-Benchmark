# Metric Definitions

| Field | Value |
|---|---|
| Status | **APPROVED** (2026-09-22) |
| Version | 0.1.0 |
| Date | 2026-09-21 |
| Depends on | `docs/ARCHITECTURE.md` v0.2.0 |

This document is the single source of truth for every metric the framework reports. Each metric's `method` field in a `MetricRecord` references an entry here (`metric_id` + evaluator version). A metric that is not defined here must not appear in any report.

---

## 1. Conventions

### 1.1 Metric record fields

Every metric produces `MetricRecord` rows (ARCHITECTURE §9.2). For each metric this document specifies:

| Field | Meaning |
|---|---|
| **ID** | Stable identifier, `l<layer>.<group>.<name>` |
| **Unit** | `ratio` (0–1), `ms`, `mb`, `score_1_5`, `usd`, `count`, … |
| **Direction** | `↑` higher is better, `↓` lower is better |
| **Level** | `sample` (one row per sample) and/or `aggregate` (computed in DuckDB from sample rows) |
| **Source** | Raw artifact(s) the value is computed from |
| **Evaluator** | Module that computes it (`benchmark/evaluators/...` or `benchmark/layers/...`) |
| **Business dimension** | Which business-score dimension it feeds (or `report-only`) |

### 1.2 Status values

| Status | Meaning | Value |
|---|---|---|
| `measured` | Computed from raw artifacts | number |
| `unsupported` | The model verifiably lacks the capability (capability report) | `null`; scores 0 in the business score, tagged |
| `not_measured` | Not run, or run failed before producing the artifact | `null`; business score becomes `incomplete` |
| `error` | Sample ran but failed (timeout, crash, invalid output) | `null` at sample level; **counted as a failure** in rate metrics (e.g., task completion), never silently dropped |

### 1.3 Aggregation and uncertainty

- **Rates** are micro-averaged over samples unless stated otherwise.
- **Latency** metrics report P50, P95, P99 using the linear-interpolation percentile (`numpy.percentile(method="linear")`), plus mean and count.
- **95% confidence intervals** for every aggregate: non-parametric bootstrap, 2,000 resamples, resampling at the **sample (dialogue) level**, fixed seed recorded in the manifest. For multi-turn scenarios the resampling unit is the whole dialogue.
- **Repeats:** stochastic metrics run `N` repeats per sample (`N` set in the run profile); repeat index is stored in the record's `tags.repeat`.
- **Channel tag:** L1 text metrics are computed separately for `tags.channel = text` (model's text output, if the model has one) and `tags.channel = audio` (ASR-judge transcript of the output audio). The **audio channel is the one used for scoring**, because the caller hears audio; the text channel is report-only.

### 1.4 Japanese text normalization (`ja_norm_v1`)

Applied to both reference and hypothesis before any text comparison, implemented in `evaluators/ja_text.py`:

1. Unicode NFKC (unifies full-width/half-width alphanumerics and katakana).
2. Lowercase Latin letters.
3. Remove punctuation and symbols: Unicode categories `P*` and `S*`, plus `・`, `ー` only when standing alone (long-vowel marks inside words are kept).
4. Remove all whitespace.
5. Numbers: no conversion between kanji numerals and digits in `ja_norm_v1` (both CER variants below make this explicit).

Two CER variants are reported:

| Variant | Additional step | Purpose |
|---|---|---|
| `cer_surface` | none | Strict, what a transcript shows |
| `cer_kana` | Convert both sides to katakana reading with a pinned morphological analyzer (fugashi + UniDic, version recorded) | Removes kanji/kana orthographic variation (e.g., 御座います vs ございます), closer to "was it said correctly" |

`cer_kana` is the scored variant; `cer_surface` is report-only. Changes to normalization create `ja_norm_v2`; old results stay reproducible.

### 1.5 Silence / audio activity definition (`vad_v1`)

Used by L2 and L4. Output audio is resampled to 16 kHz mono and split into 10 ms frames. A frame is **active** if its RMS level is above `activity_threshold_dbfs` (config, default proposal −45 dBFS) — thresholds are config values recorded in the manifest, and a second, model-based VAD (Silero VAD, pinned) is run as a cross-check with disagreement reported.

### 1.6 Judges

- **ASR judge** (`asr_judge_v1`): a single pinned ASR model (candidate: Whisper large-v3 or a Japanese-specialized ASR; chosen in Phase 3, name + revision recorded). Its own error floor is measured by transcribing the **reference stimulus audio** and reported as `l1.asr.judge_floor_cer`.
- **LLM judge** (`llm_judge_v1`): a pinned API model that is **not** one of the benchmarked models, temperature 0, rubric prompts versioned in `configs/prompts/ja/judges/`. Each judge call stores prompt, raw response and parsed label in raw artifacts. Judge agreement with human labels on a calibration subset is reported as Cohen's κ (`judge.<rubric>.kappa`); a rubric with κ below the configured minimum is flagged in reports.

---

## 2. Layer 1 — Japanese Capability

| ID | Unit | Dir | Level | Definition | Source | Business dimension |
|---|---|---|---|---|---|---|
| `l1.asr.cer` | ratio | ↓ | sample + aggregate | Character error rate of the model's transcription/repetition of the stimulus. Aggregate = Σ edit distance / Σ reference characters (micro). Variants via `tags.variant ∈ {kana, surface}`. | `output.flac` → ASR judge; `events.jsonl` text | Japanese Quality |
| `l1.asr.wer` | ratio | ↓ | sample + aggregate | Word error rate on fugashi/UniDic tokens (surface forms). | same | report-only |
| `l1.asr.judge_floor_cer` | ratio | ↓ | aggregate | ASR judge CER on reference stimulus audio (evaluator quality, not model quality). | stimulus audio | report-only |
| `l1.language.ja_rate` | ratio | ↑ | aggregate | Share of responses whose ASR transcript is detected as Japanese (pinned language-ID, e.g., fastText lid, recorded). | ASR transcript | gate (see §8) |
| `l1.intent.accuracy` | ratio | ↑ | aggregate | Exact match of predicted intent label to gold. Prediction is parsed from the model's response per the intent prompt protocol; unparseable = wrong. | response text / transcript | report-only |
| `l1.intent.macro_f1` | ratio | ↑ | aggregate | Macro-averaged F1 over the closed intent set. | same | Japanese Quality |
| `l1.slot.precision` / `.recall` | ratio | ↑ | aggregate | Slot-level micro precision/recall. A slot matches if name matches and normalized values are equal (§2.1). | same | report-only |
| `l1.slot.f1` | ratio | ↑ | aggregate | Harmonic mean of the two above. | same | Japanese Quality |
| `l1.understanding.accuracy` | ratio | ↑ | aggregate | Closed-form QA: exact match after normalization. Open-form QA: LLM judge label `correct` (rubric `understanding_v1`). Reported separately by `tags.form`. | transcript + judge | Japanese Quality |
| `l1.keigo.compliance_rate` | ratio | ↑ | aggregate | Share of responses that (a) have zero violations from the rule detector **and** (b) are labelled `appropriate` by the LLM judge (rubric `keigo_v1`). | transcript + detector + judge | Japanese Quality |
| `l1.keigo.rule_violation_rate` | ratio | ↓ | aggregate | Share of responses with ≥ 1 rule violation (casual forms such as plain-form sentence endings to the customer, forbidden expressions list; rule list versioned). | transcript | report-only |
| `l1.long_context.recall_accuracy` | ratio | ↑ | aggregate | Share of probe questions answered correctly about facts stated earlier in the dialogue. Reported by `tags.distance_bucket` (turn distance: 1–3, 4–10, 11+). Overall = micro over all probes. | transcript + judge/exact | Japanese Quality |

### 2.1 Slot value normalization (`slot_norm_v1`)

| Slot type | Normalized form |
|---|---|
| date | ISO `YYYY-MM-DD`, relative dates resolved against the scenario's fixed `reference_date` |
| time | `HH:MM` 24h |
| phone | digits only |
| person name | `ja_norm_v1` + katakana reading (name readings compared, not kanji) |
| number / count | integer |
| free text | `ja_norm_v1` exact match |

---

## 3. Layer 2 — Realtime Voice

All times come from harness receipt/send timestamps (`time.monotonic_ns()`) in `events.jsonl`. OSS models are measured over localhost; GPT-Realtime includes the public network and carries `tags.network = internet` plus the measured `l2.network.rtt_ms`.

### 3.1 Reference time points

| Symbol | Definition |
|---|---|
| `t_eos` | Send timestamp of the audio frame containing the last speech sample of the user turn. The last speech sample position is stored in the stimulus manifest (`speech_end_s`, computed once by `vad_v1` on the stimulus and human-spot-checked). |
| `t_first_audio` | Receipt timestamp of the first `AudioDelta` whose content contains at least one **active** frame (§1.5). |
| `t_bargein` | Send timestamp of the frame containing the barge-in onset (`bargein_onset_s` in the scenario). |
| `t_gen_stop` | After `t_bargein`: receipt time of a `ResponseCancelled` event, or receipt time of the last `AudioDelta` containing an active frame, followed by ≥ `stop_silence_ms` (config) without active output — whichever is earlier. |

Generated audio can arrive faster than real time and be buffered by a client, so **generation-stop latency** (what the model controls) is the scored metric. Perceived stop depends on client buffer flushing and is simulated as report-only (`l2.interrupt.perceived_ms`, playback buffer simulated at real time, flushed on `t_gen_stop`).

### 3.2 Metrics

| ID | Unit | Dir | Level | Definition | Business dimension |
|---|---|---|---|---|---|
| `l2.ttfa_ms` | ms | ↓ | sample + P50/P95/P99 | `t_first_audio − t_eos`. Negative values (model starts before user ended) are kept and also counted in `l2.turn.premature_rate`. | Barge-In (P95) |
| `l2.interrupt.latency_ms` | ms | ↓ | sample + P50/P95/P99 | `t_gen_stop − t_bargein`. If output never stops within `interrupt_timeout_ms`, sample = failure, value `null`, counted in success rate. | Barge-In (P95) |
| `l2.interrupt.perceived_ms` | ms | ↓ | sample + percentiles | Simulated audible stop time − `t_bargein`. | report-only |
| `l2.bargein.success_rate` | ratio | ↑ | aggregate | Share of barge-in trials where (a) `l2.interrupt.latency_ms ≤ interrupt_slo_ms` **and** (b) the next response addresses the interrupting utterance (LLM judge, rubric `bargein_followup_v1`). | Barge-In |
| `l2.bargein.false_stop_rate` | ratio | ↓ | aggregate | Share of backchannel/noise injections (e.g., 「はい」「ええ」, cough, background noise) after which generation stops within `stop_silence_ms`. | Barge-In |
| `l2.turn.gap_ms` | ms | — | distribution | Same as TTFA but over all turns of multi-turn dialogues; reported as a histogram. | report-only |
| `l2.turn.premature_rate` | ratio | ↓ | aggregate | Share of turns where `t_first_audio < t_eos`, including stimuli with scripted mid-utterance pauses (the customer thinking). | report-only |
| `l2.stream.rtf` | ratio | ↑ | sample + P5 | Real-time factor of streaming output: seconds of audio received / wall-clock seconds from first to last delta. Values < 1 cause audible stutter. | report-only |
| `l2.stream.underrun_rate` | ratio | ↓ | aggregate | Share of responses where the simulated real-time playback buffer runs empty before the response ends. | report-only |
| `l2.duplex.supported` | bool | — | model | From capability report (`native_full_duplex = verified`). Duplex-specific metrics (e.g., backchannel production) are defined in a later version. | report-only |
| `l2.stability.ttfa_drift_ratio` | ratio | ↓ | per long call | P50 TTFA over the last 20% of turns / P50 TTFA over the first 20%. | report-only |
| `l2.stability.error_rate` | ratio | ↓ | aggregate | Share of turns in long calls ending in `error` (timeout, disconnect, empty response). | gate (see §8) |
| `l2.stability.vram_growth_mb` | mb | ↓ | per long call | NVML used memory at end − at start of the long call. | report-only |
| `l2.network.rtt_ms` | ms | — | aggregate | Measured RTT to the API endpoint (GPT-Realtime only). | report-only |

Every L2 record carries `tags.mode ∈ {native, orchestrated}` (ARCHITECTURE §6.3).

---

## 4. Layer 3 — Tool Calling

Source artifacts: `tool_trace.jsonl` (every tool call with raw arguments, schema validation result, backend response) and the toolserver's final state snapshot. Each scenario defines an expected tool trace and an expected final state (`DATASET_SPEC.md` §5).

| ID | Unit | Dir | Level | Definition | Business dimension |
|---|---|---|---|---|---|
| `l3.task.completion_rate` | ratio | ↑ | aggregate | Share of scenarios whose final toolserver state equals the expected final state (state comparator, normalized values). Errors/timeouts count as failures. By `tags.category ∈ {booking, faq, crm_lookup, transfer, structured_output}`. | Task Completion (primary) |
| `l3.tool.success_rate` | ratio | ↑ | aggregate | Share of emitted tool calls that are (a) to a defined tool, (b) schema-valid, and (c) accepted by the backend without error. | Task Completion |
| `l3.json.schema_valid_rate` | ratio | ↑ | aggregate | Share of emitted tool calls whose arguments parse as JSON and validate against the tool's JSON Schema. | report-only |
| `l3.json.arg_accuracy` | ratio | ↑ | aggregate | For expected calls that were matched to an emitted call: share of expected arguments whose normalized value equals the emitted value (`slot_norm_v1`). | Task Completion |
| `l3.tool.hallucination_rate` | ratio | ↓ | aggregate | Share of emitted tool calls that (a) name an undefined tool, or (b) contain an argument value not grounded in the dialogue or earlier tool results (grounding check: value appears, after normalization, in user turns or tool outputs; ambiguous cases go to LLM judge rubric `grounding_v1`). | Task Completion (penalty) |
| `l3.tool.missed_call_rate` | ratio | ↓ | aggregate | Share of expected calls with no matching emitted call. | report-only |
| `l3.structured.exact_match` | ratio | ↑ | aggregate | Structured-output scenarios: normalized emitted JSON equals expected JSON exactly. | report-only |

Matching emitted to expected calls: same tool name, then maximum argument overlap (Hungarian assignment). Order is ignored unless the scenario marks `ordered: true`.

`tags.tool_mode ∈ {native, prompted}` on every L3 record.

---

## 5. Layer 4 — Voice Quality

| ID | Unit | Dir | Level | Definition | Business dimension |
|---|---|---|---|---|---|
| `l4.mos.human` | score_1_5 | ↑ | per clip + aggregate | Mean of ratings on a 5-point ACR scale (1 bad – 5 excellent) of overall quality, from raters passing attention checks. 95% CI by bootstrap over clips and raters. | Voice Quality (primary) |
| `l4.naturalness.human` | score_1_5 | ↑ | same | 5-point naturalness rating. | Voice Quality |
| `l4.accent.human` | score_1_5 | ↑ | same | 5-point rating of Japanese accent / pitch-accent correctness. | Voice Quality |
| `l4.emotion.human` | score_1_5 | ↑ | same | 5-point appropriateness of emotional tone for the scripted situation (apology, empathy, cheerfulness). | Voice Quality |
| `l4.mos.proxy.<predictor>` | score_1_5 | ↑ | per clip + aggregate | Automatic MOS predictor output (predictor name + version in `method`). Pearson/Spearman correlation vs `l4.mos.human` reported as `l4.mos.proxy.<predictor>.corr_human`. | fallback only (§7.3) |
| `l4.pronunciation.cer` | ratio | ↓ | aggregate | `l1.asr.cer` (kana variant) on the read-aloud set: model instructed to say fixed text containing numbers, dates, names, business vocabulary. | Voice Quality |
| `l4.consistency.speaker_sim` | ratio | ↑ | aggregate | Mean cosine similarity of speaker embeddings (pinned ECAPA-TDNN-class model) between each response and the model's per-session voice centroid; also across sessions. | Voice Quality |
| `l4.rater.agreement` | ratio | ↑ | aggregate | Krippendorff's α of human ratings (evaluator quality). | report-only |

Human MOS protocol: blind, per-rater randomized order, model identity hidden in file names, each clip rated by ≥ `min_raters_per_clip` raters, anchor clips (high/low quality references) and attention-check clips included. Details in `DATASET_SPEC.md` §7.

---

## 6. Layer 5 — Infrastructure & TCO

NVML samples at 10 Hz (`monitoring/nvml.parquet`) with phase markers (`idle`, `single_call`, `concurrency_N`).

| ID | Unit | Dir | Level | Definition | Business dimension |
|---|---|---|---|---|---|
| `l5.vram.idle_mb` | mb | ↓ | model | Median NVML used memory in `idle` phase (model loaded, no traffic). | report-only |
| `l5.vram.peak_mb` | mb | ↓ | per phase | Max NVML used memory in the phase. | gate (fits in 80 GB) |
| `l5.gpu.util_mean` | ratio | — | per phase | Mean SM utilization. | report-only |
| `l5.throughput.audio_rtf` | ratio | ↑ | per concurrency level | Total seconds of audio generated by all sessions / wall-clock seconds. | report-only |
| `l5.concurrency.max_at_slo` | count | ↑ | model | Largest tested concurrency `N` (sweep 1, 2, 4, 8, …, then bisection between the last passing and first failing level) where, at `N` concurrent streaming calls: `l2.ttfa_ms` P95 ≤ `ttfa_p95_slo_ms`, `l2.stream.underrun_rate` ≤ `underrun_slo`, and error rate ≤ `error_rate_slo`. `0` if `N = 1` fails. | Cost (input) |
| `l5.cost.per_minute_usd` | usd | ↓ | model | OSS: `gpu_hourly_usd / (60 × l5.concurrency.max_at_slo)`. API baseline: measured billed usage (input/output audio + text tokens from API usage fields) × `pricing.yaml` prices / call minutes. `null` with `not_measured` if max concurrency is 0 or prices are missing. | Cost (primary) |
| `l5.cost.per_call_usd` | usd | ↓ | model | `l5.cost.per_minute_usd × mean call duration` of the standard scenario set (measured). | report-only |
| `l5.tco.monthly_usd` | usd | ↓ | model | Configured TCO model: GPUs needed for `target_peak_concurrent_calls` (= ceil(target / max_at_slo)) × monthly GPU cost (amortized purchase or rental) + power + ops overhead, all from `pricing.yaml`. | report-only |

Every price input carries `source` and `as_of` in `configs/cost/pricing.yaml`; reports print them next to cost figures.

---

## 7. Business Score (`business_v1`)

### 7.1 Normalization

Each input metric `m` is mapped to `s(m) ∈ [0, 1]` with a clipped linear function between two owner-defined anchors in `configs/scoring/anchors.yaml`:

```
direction ↑:  s = clip((m - worst) / (best - worst), 0, 1)
direction ↓:  s = clip((worst - m) / (worst - best), 0, 1)
```

Anchors are **business thresholds** (e.g., "TTFA P95 of X ms or better is perfect; Y ms or worse is unusable"), fixed and signed off **before** results are seen. For ratio metrics where 1.0 / 0.0 are natural, the defaults are `best = 1, worst = 0` (↑) or `best = 0, worst = 1` (↓) unless the owner overrides them.

### 7.2 Dimensions and sub-weights

Top-level weights are fixed by CLAUDE.md. Sub-weights are a **proposal** for the owner to confirm (stored in `configs/scoring/business_v1.yaml`).

| Dimension (weight) | Input metrics (proposed sub-weight) |
|---|---|
| Japanese Quality (35%) | `l1.asr.cer` kana (0.25), `l1.intent.macro_f1` (0.15), `l1.slot.f1` (0.15), `l1.understanding.accuracy` (0.15), `l1.keigo.compliance_rate` (0.20), `l1.long_context.recall_accuracy` (0.10) |
| Task Completion (25%) | `l3.task.completion_rate` (0.60), `l3.tool.success_rate` (0.15), `l3.json.arg_accuracy` (0.15), `1 − l3.tool.hallucination_rate` (0.10) |
| Barge-In (15%) | `l2.interrupt.latency_ms` P95 (0.30), `l2.bargein.success_rate` (0.30), `l2.bargein.false_stop_rate` (0.15), `l2.ttfa_ms` P95 (0.25) |
| Cost (15%) | `l5.cost.per_minute_usd` (1.00) |
| Voice Quality (10%) | `l4.mos.human` (0.40), `l4.naturalness.human` (0.15), `l4.accent.human` (0.15), `l4.pronunciation.cer` (0.15), `l4.consistency.speaker_sim` (0.10), `l4.emotion.human` (0.05) |

`dimension_score = Σ sub_weight × s(metric)`; `business_score = Σ weight × dimension_score`, reported on a 0–100 scale.

TTFA is placed in the Barge-In dimension because CLAUDE.md defines no separate latency dimension and turn responsiveness is part of conversational interaction; this placement is a proposal for review.

### 7.3 Missing inputs

- A metric with status `unsupported` contributes `s = 0` (tagged).
- A metric with status `not_measured` makes the model's business score `incomplete` (ARCHITECTURE §10.3). No re-weighting, no imputation.
- **Voice Quality fallback:** if human MOS is not yet available for any model, the report may compute a *provisional* Voice Quality score using `l4.mos.proxy.<predictor>` instead of the human metrics — only if the predictor's `corr_human` has been measured on a calibration subset — and the whole leaderboard is labelled **provisional**.

### 7.4 Ranking

Models with complete scores are ranked by `business_score`. The 95% CI of the business score comes from a joint bootstrap (resampling samples in every layer simultaneously). Adjacent models whose score difference has a CI that includes 0 are shown as tied. A sensitivity table re-ranks under ±10 percentage-point perturbations of each top-level weight (report-only, to show ranking robustness).

---

## 8. Deployment Gates (report-only, not scored)

| Gate | Pass condition |
|---|---|
| Fits hardware | `l5.vram.peak_mb` at `N = 1` ≤ 80 GB and model runs on 1x H100 |
| Speaks Japanese | `l1.language.ja_rate` ≥ `ja_rate_gate` (anchors config) |
| Stable | `l2.stability.error_rate` ≤ `stability_error_gate` |
| License | Manual review field in model YAML: `commercial_use: allowed / restricted / unknown` |

A failed gate does not change the score; the leaderboard shows the model as "not deployable as benchmarked" with the failing gate listed.
