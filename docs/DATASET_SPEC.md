# Dataset Specification

| Field | Value |
|---|---|
| Status | **APPROVED** (2026-09-22) |
| Version | 0.1.0 |
| Date | 2026-09-21 |
| Depends on | `docs/ARCHITECTURE.md` v0.2.0, `docs/METRIC_DEFINITIONS.md` v0.1.0 |

---

## 1. Principles

1. **Never fabricate labels.** Labels come from (a) the source corpus, (b) construction — the scenario author writes the text *and* its intent/slots/expected tool calls together, or (c) human annotation. Labels are never produced by a benchmarked model or by an unreviewed LLM.
2. **Synthetic data is always flagged.** Any audio produced by TTS, or text written for the benchmark rather than collected from real callers, has `synthetic: true` and a `generator` record.
3. **Human-reviewed data lives separately** under `datasets/human_reviewed/`.
4. **Versioned and immutable.** A published dataset version never changes; any change creates a new version. Each version has a `manifest.jsonl` whose SHA-256 is recorded in every run manifest.
5. **Reproducible builds.** Each dataset has a `build.py` that recreates it from its sources (download + transform, or TTS render with pinned generator/voice/seed).
6. **Audio is not committed to git.** Manifests, scripts, texts and datasheets are committed; audio is built on the GPU server with `vbench data prepare` and verified against manifest checksums.
7. **Licenses are recorded per sample** and checked before inclusion. A source whose license has not been reviewed is `status: candidate` and cannot be used in a `full` run.

---

## 2. Directory Layout

```text
datasets/
  <dataset_name>/
    <version>/                  # e.g. v1, v2
      manifest.jsonl            # one line per sample (committed)
      scenarios/*.yaml          # scenario definitions for L2/L3/L1 dialogues (committed)
      texts/                    # source texts / scripts (committed)
      DATASHEET.md              # purpose, sources, licenses, stats, known limitations (committed)
      build.py                  # reproducible builder (committed)
      audio/                    # built on server, git-ignored
  human_reviewed/
    <dataset_name>/<version>/   # human recordings, human labels, MOS ratings
  registry.yaml                 # all datasets, versions, status, manifest sha256
```

Audio format: FLAC, mono, 16-bit. Native sample rate kept per source (16 kHz or 24 kHz); adapters resample to each model's required input rate, and the resampler (library + version) is recorded.

---

## 3. Manifest Schema (`manifest.jsonl`)

Validated by a Pydantic model (`benchmark/core/schemas.py: DatasetSample`) and by CI lint.

```json
{
  "sample_id": "ja_asr_eval.v1.000123",
  "audio_path": "audio/000123.flac",
  "audio_sha256": "…",
  "duration_s": 4.21,
  "sample_rate_hz": 16000,
  "speech_start_s": 0.18,
  "speech_end_s": 3.96,
  "text": "お問い合わせありがとうございます。",
  "text_reading_kana": "オトイアワセアリガトウゴザイマス",
  "labels": { "intent": null, "slots": [] },
  "channel": "clean_16k",
  "speaker": { "id": "spk_03", "gender": "f", "age_band": "30s", "synthetic_voice": false },
  "synthetic": false,
  "generator": null,
  "source": { "corpus": "<corpus name>", "item": "<original id>", "url": "<url>" },
  "license": "<SPDX id or URL>",
  "split": "test",
  "tags": ["asr"]
}
```

| Field | Rule |
|---|---|
| `speech_start_s` / `speech_end_s` | Computed by `vad_v1` at build time; used for `t_eos` (METRIC_DEFINITIONS §3.1). A random 5% is human spot-checked; spot-check results are stored in `human_reviewed/`. |
| `text_reading_kana` | From the source corpus if present; otherwise generated with the pinned analyzer and marked `reading_source: analyzer`. |
| `generator` | Required when `synthetic: true`: `{ "type": "tts", "engine": "...", "version": "...", "voice": "...", "seed": 0, "params": {...} }`. For LLM-assisted text drafting (if ever used), the draft must be human-reviewed and `text_origin: llm_draft_human_reviewed` recorded. |
| `channel` | `clean_16k`, `clean_24k`, or `telephone_8k` (see §6). |

---

## 4. Datasets per Layer

Sizes are **design targets** for statistical usefulness, not results. `smoke` uses a fixed small subset of each.

| Dataset | Layer | Content | Label origin | Synthetic | Target size (standard / full) |
|---|---|---|---|---|---|
| `ja_asr_eval` | L1 | Read and spontaneous Japanese speech from public corpora | corpus transcripts | no | 200 / 1,000 utterances |
| `ja_callcenter_intent_slot` | L1 | Customer utterances for call center intents with slots | construction | yes (text + TTS); human-recorded subset in `human_reviewed/` | 200 / 600 utterances |
| `ja_understanding` | L1 | Spoken passages + questions (closed and open form) | construction | yes | 100 / 300 questions |
| `ja_keigo` | L1 | Customer turns designed to elicit polite business responses (complaints, requests, apologies) | construction; calibration subset human-labelled | yes | 100 / 300 turns |
| `ja_long_context` | L1 | Multi-turn dialogues with facts stated early and probed later | construction | yes | 20 / 60 dialogues |
| `ja_realtime` | L2 | Turn-taking stimuli, mid-utterance pauses, barge-in, backchannel/noise injections, long calls | construction (timing is part of the scenario) | yes | 100 turns + 50 barge-ins + 50 false-barge-ins + 1 long call / 500 + 200 + 200 + 3 long calls |
| `ja_toolcalling` | L3 | Scripted dialogues with expected tool trace and final state | construction | yes | 60 / 200 scenarios |
| `ja_read_aloud` | L4 | Fixed texts with numbers, dates, names, business vocabulary | construction | text yes, no audio input needed beyond instruction | 50 / 150 prompts |
| `ja_voice_prompts` | L4 | Situations requiring apology / empathy / cheerful tone | construction | yes | 30 / 90 prompts |
| `mos_ratings` | L4 | Human MOS ratings | human | no | `human_reviewed/` only |

Intents v1 (closed set, may be extended in v2): `booking_new`, `booking_change`, `booking_cancel`, `order_status`, `faq_hours`, `faq_price`, `faq_access`, `account_lookup`, `complaint`, `request_operator`, `other`.

### 4.1 Public corpus candidates for `ja_asr_eval`

All start as `status: candidate`. License terms must be reviewed by the project owner and recorded in `DATASHEET.md` before use in a `full` run; nothing here is a legal conclusion.

| Candidate | Why | License review |
|---|---|---|
| Common Voice (Japanese) | many speakers, read speech | pending |
| JSUT / JVS | clean read speech, single and multi-speaker | pending |
| ReazonSpeech (test subset) | broadcast-style, spontaneous | pending |
| CSJ | spontaneous, academic | pending (paid license) |

---

## 5. Scenario Schema (L1 dialogues, L2, L3)

Scenarios are YAML, validated by Pydantic. User turns are fixed audio; they never depend on model output (ARCHITECTURE §7).

```yaml
scenario_id: ja_toolcalling.v1.booking_017
category: booking                  # booking | faq | crm_lookup | transfer | structured_output
reference_date: "2026-10-01"       # fixed "today" for relative dates
synthetic: true
system_prompt_ref: prompts/ja/callcenter_agent_v1.yaml
tools_ref: tools/v1                # JSON Schemas shared by all models
initial_state_ref: states/clinic_calendar_v1.json
turns:
  - speaker: user
    text: "来週の火曜日の午後に予約を取りたいのですが。"
    audio: audio/booking_017_t1.flac
    labels: { intent: booking_new, slots: [{name: date, value: "2026-10-06"}, {name: time_range, value: "afternoon"}] }
  - speaker: user
    text: "山田太郎です。電話番号は090-1234-5678です。"
    audio: audio/booking_017_t2.flac
    after: model_done              # model_done | at_offset_s
expected:
  tool_calls:
    ordered: false
    calls:
      - tool: check_availability
        args: { date: "2026-10-06", time_range: "afternoon" }
      - tool: create_booking
        args: { date: "2026-10-06", name: "山田太郎", phone: "09012345678" }
        args_match: { time: any_available_afternoon_slot }
  final_state:
    bookings:
      - { date: "2026-10-06", name_reading: "ヤマダタロウ", phone: "09012345678", slot_in: afternoon }
```

L2 scenario extensions:

```yaml
events:
  - type: bargein                  # bargein | backchannel | noise
    audio: audio/bargein_03.flac
    onset: { relative_to: model_first_audio, offset_s: 1.5 }
    expected: { stop: true, followup_topic: "日時の変更" }
  - type: backchannel
    audio: audio/hai_02.flac
    onset: { relative_to: model_first_audio, offset_s: 2.0 }
    expected: { stop: false }
```

Placeholder names/phones in scenarios are fictitious and must not belong to real people; phone numbers use a reserved/fictional pattern documented in `DATASHEET.md`.

---

## 6. Stimulus Audio Conditions

Call center audio is narrowband. Every synthetic and human-recorded stimulus is rendered in two channels, reported separately via `channel`:

| Channel | Processing |
|---|---|
| `clean_16k` (or `clean_24k`) | as recorded/synthesized |
| `telephone_8k` | band-limit 300–3400 Hz, resample to 8 kHz, G.711 μ-law encode/decode, then resample to the model's input rate; optional additive noise at configured SNR (noise source + SNR recorded) |

Scoring uses `telephone_8k` as primary for L1–L3 (it matches deployment); `clean` is report-only. This is a proposal for review.

---

## 7. Human-Reviewed Data

| Kind | Location | Required metadata |
|---|---|---|
| Human recordings of scenarios | `human_reviewed/<dataset>/<version>/audio/` | pseudonymous speaker ID, consent record reference (stored outside git), recording device, date |
| Human labels (keigo calibration, VAD spot checks, judge calibration) | `human_reviewed/<dataset>/<version>/labels.jsonl` | pseudonymous annotator ID, date, guideline version |
| MOS ratings | `human_reviewed/mos_ratings/<round>/ratings.csv` | rater ID (pseudonymous), clip ID (blinded), scale, timestamp, attention-check results, app version |

MOS protocol: ACR 5-point scale; clip IDs are random and the model mapping is stored in a separate key file that raters never see; each rater gets an independent random order; ≥ `min_raters_per_clip` ratings per clip (proposal: 5 — owner to confirm based on rater availability); anchor clips (reference human speech, degraded speech) and attention checks are included; raters failing attention checks are excluded, and the exclusion is reported.

No personal data of real callers is used. If real call recordings are ever added, that requires a separate privacy review and is out of scope for v1.

---

## 8. Synthetic Audio Generation

Open decision (ARCHITECTURE §15 Q1): which TTS engine(s) render scenario audio. Requirements for any choice:

- License permits generating benchmark stimuli.
- Multiple Japanese voices: target ≥ 6 voices, balanced by gender, recorded as `speaker.id`.
- Deterministic or seed-controlled; engine + version + voice + params recorded in `generator`.
- **Must not be the same system as a benchmarked model** (e.g., not GPT-Realtime / OpenAI TTS, not the TTS head of any candidate), to avoid giving any model in-distribution input.
- Results are reported per stimulus source (`synthetic` vs `human_reviewed`) so any gap is visible.

---

## 9. Versioning and Registry

`datasets/registry.yaml`:

```yaml
- name: ja_toolcalling
  version: v1
  status: draft            # candidate | draft | released | deprecated
  manifest_sha256: null    # set when released
  released_at: null
  changelog: "Initial scenario set."
```

Only `released` versions can be used in `standard` and `full` runs. `vbench data prepare` refuses to build a dataset whose built audio does not match `audio_sha256`.
