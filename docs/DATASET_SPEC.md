# Đặc tả Dataset

| Trường | Giá trị |
|---|---|
| Trạng thái | **ĐÃ DUYỆT** (2026-09-22), cập nhật v0.2.0: chiến lược dữ liệu khi không có người nói tiếng Nhật (§10) |
| Phiên bản | 0.2.0 |
| Ngày | 2026-09-21 |
| Phụ thuộc | `docs/ARCHITECTURE.md` v0.2.0, `docs/METRIC_DEFINITIONS.md` v0.1.0 |

> Bản tiếng Việt. Tên trường, tên dataset, giá trị enum và ví dụ JSON/YAML giữ nguyên vì khớp với code (`benchmark/core/schemas.py`).

---

## 1. Nguyên tắc

1. **Không bao giờ bịa nhãn.** Nhãn đến từ (a) corpus nguồn, (b) quá trình soạn — người soạn kịch bản viết text *cùng lúc* với intent/slot/tool call kỳ vọng của nó, hoặc (c) người gán nhãn. Nhãn không bao giờ do model đang benchmark hay một LLM chưa được review tạo ra.
2. **Dữ liệu synthetic luôn được gắn cờ.** Mọi audio do TTS tạo ra, hoặc text được viết riêng cho benchmark chứ không thu thập từ người gọi thật, đều có `synthetic: true` và bản ghi `generator` (nếu là TTS).
3. **Dữ liệu có người review nằm riêng** trong `datasets/human_reviewed/`.
4. **Có phiên bản và bất biến.** Một phiên bản dataset đã phát hành không bao giờ thay đổi; mọi thay đổi tạo phiên bản mới. Mỗi phiên bản có `manifest.jsonl`, và SHA-256 của file này được ghi vào mọi run manifest.
5. **Build tái lập được.** Mỗi dataset có `build.py` tái tạo dataset từ nguồn (tải + biến đổi, hoặc render TTS với generator/giọng/seed đã chốt).
6. **Audio không commit vào git.** Manifest, script, text và datasheet được commit; audio được build trên GPU server bằng `vbench data prepare` và kiểm tra với checksum trong manifest.
7. **License được ghi cho từng mẫu** và kiểm tra trước khi đưa vào. Nguồn chưa được review license có `status: candidate` và không được dùng trong run `full`.

---

## 2. Cấu trúc thư mục

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

Định dạng audio: FLAC, mono, 16-bit. Giữ nguyên sample rate gốc của từng nguồn (16 kHz hoặc 24 kHz); adapter resample về sample rate đầu vào mà từng model yêu cầu, và bộ resample (thư viện + phiên bản) được ghi lại.

---

## 3. Schema manifest (`manifest.jsonl`)

Được kiểm tra bằng model Pydantic (`benchmark/core/schemas.py: DatasetSample`) và bằng lint trong CI.

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

| Trường | Quy tắc |
|---|---|
| `speech_start_s` / `speech_end_s` | Tính bằng `vad_v1` khi build; dùng để xác định `t_eos` (METRIC_DEFINITIONS §3.1). Kiểm tra ngẫu nhiên 5% bởi người; kết quả kiểm tra lưu trong `human_reviewed/`. |
| `text_reading_kana` | Lấy từ corpus nguồn nếu có; nếu không thì tạo bằng bộ phân tích chốt phiên bản và ghi `reading_source: analyzer`. |
| `generator` | Bắt buộc khi audio do TTS tạo (`speaker.synthetic_voice: true`): `{ "type": "tts", "engine": "...", "version": "...", "voice": "...", "seed": 0, "params": {...} }`. Nếu có dùng LLM hỗ trợ soạn text, bản nháp phải được người review và ghi `text_origin: llm_draft_human_reviewed`. |
| `channel` | `clean_16k`, `clean_24k`, hoặc `telephone_8k` (xem §6). |

Các quy tắc code đang kiểm tra: giọng TTS (`synthetic_voice: true`) bắt buộc `synthetic: true` và có `generator`; có `generator` thì bắt buộc `synthetic: true`; `speech_start_s ≤ speech_end_s ≤ duration_s`. Text soạn sẵn do người thật đọc là `synthetic: true` nhưng không cần `generator`.

---

## 4. Dataset theo từng lớp

Kích thước là **mục tiêu thiết kế** để đủ ý nghĩa thống kê, không phải kết quả. `smoke` dùng một tập con nhỏ cố định của mỗi dataset.

| Dataset | Lớp | Nội dung | Nguồn nhãn | Synthetic | Kích thước mục tiêu (standard / full) |
|---|---|---|---|---|---|
| `ja_asr_eval` | L1 | Tiếng Nhật đọc và nói tự nhiên từ corpus công khai | transcript của corpus | không | 200 / 1.000 câu |
| `ja_callcenter_intent_slot` | L1 | Câu nói của khách hàng theo intent tổng đài, có slot | soạn sẵn | có (text + TTS); tập con giọng người thật trong `human_reviewed/` | 200 / 600 câu |
| `ja_understanding` | L1 | Đoạn văn dạng nói + câu hỏi (dạng đóng và mở) | soạn sẵn | có | 100 / 300 câu hỏi |
| `ja_keigo` | L1 | Lượt khách hàng được thiết kế để cần câu trả lời lịch sự kiểu doanh nghiệp (phàn nàn, yêu cầu, xin lỗi) | soạn sẵn; tập con hiệu chỉnh do người gán nhãn | có | 100 / 300 lượt |
| `ja_long_context` | L1 | Hội thoại nhiều lượt có thông tin nêu sớm và hỏi lại sau | soạn sẵn | có | 20 / 60 hội thoại |
| `ja_realtime` | L2 | Kích thích luân phiên lượt, ngập ngừng giữa câu, barge-in, chèn tiếng đệm/nhiễu, cuộc gọi dài | soạn sẵn (thời điểm là một phần của kịch bản) | có | 100 lượt + 50 barge-in + 50 barge-in sai + 1 cuộc gọi dài / 500 + 200 + 200 + 3 cuộc gọi dài |
| `ja_toolcalling` | L3 | Hội thoại soạn sẵn có trace tool kỳ vọng và trạng thái cuối | soạn sẵn | có | 60 / 200 kịch bản |
| `ja_read_aloud` | L4 | Câu cố định có số, ngày, tên, từ vựng nghiệp vụ | soạn sẵn | text có; không cần audio đầu vào ngoài câu lệnh | 50 / 150 prompt |
| `ja_voice_prompts` | L4 | Tình huống cần giọng xin lỗi / đồng cảm / vui vẻ | soạn sẵn | có | 30 / 90 prompt |
| `mos_ratings` | L4 | Điểm MOS của người chấm | người | không | chỉ trong `human_reviewed/` |

Intent v1 (tập đóng, có thể mở rộng ở v2): `booking_new`, `booking_change`, `booking_cancel`, `order_status`, `faq_hours`, `faq_price`, `faq_access`, `account_lookup`, `complaint`, `request_operator`, `other`.

### 4.1 Corpus công khai ứng viên cho `ja_asr_eval`

Tất cả bắt đầu với `status: candidate`. Điều khoản license phải được chủ dự án review và ghi vào `DATASHEET.md` trước khi dùng trong run `full`; tài liệu này không đưa ra kết luận pháp lý nào.

| Ứng viên | Lý do | Review license |
|---|---|---|
| Common Voice (tiếng Nhật) | nhiều người nói, giọng đọc | chưa |
| JSUT / JVS | giọng đọc sạch, một và nhiều người nói | chưa |
| ReazonSpeech (tập test) | kiểu phát thanh, nói tự nhiên | chưa |
| CSJ | nói tự nhiên, học thuật | chưa (license trả phí) |

---

## 5. Schema kịch bản (hội thoại L1, L2, L3)

Kịch bản là YAML, được kiểm tra bằng Pydantic. Lượt người dùng là audio cố định; không bao giờ phụ thuộc vào đầu ra của model (ARCHITECTURE §7).

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

Phần mở rộng cho kịch bản L2:

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

Tên/số điện thoại mẫu trong kịch bản là hư cấu và không được thuộc về người thật; số điện thoại dùng mẫu dành riêng/hư cấu được ghi trong `DATASHEET.md`.

---

## 6. Điều kiện audio kích thích

Audio tổng đài là băng hẹp. Mọi kích thích synthetic và giọng người thật được render thành hai kênh, báo cáo riêng qua trường `channel`:

| Kênh | Xử lý |
|---|---|
| `clean_16k` (hoặc `clean_24k`) | như khi thu/tổng hợp |
| `telephone_8k` | lọc dải 300–3400 Hz, resample về 8 kHz, mã hoá/giải mã G.711 μ-law, rồi resample về sample rate đầu vào của model; tuỳ chọn cộng nhiễu ở SNR cấu hình (ghi lại nguồn nhiễu + SNR) |

Chấm điểm dùng `telephone_8k` làm kênh chính cho L1–L3 (khớp môi trường triển khai); `clean` chỉ báo cáo. Đề xuất này đã được duyệt.

---

## 7. Dữ liệu có người review

| Loại | Vị trí | Metadata bắt buộc |
|---|---|---|
| Bản thu giọng người thật cho kịch bản | `human_reviewed/<dataset>/<version>/audio/` | ID người nói (ẩn danh), tham chiếu đến bản ghi đồng ý (lưu ngoài git), thiết bị thu, ngày |
| Nhãn do người gán (hiệu chỉnh keigo, kiểm tra ngẫu nhiên VAD, hiệu chỉnh judge) | `human_reviewed/<dataset>/<version>/labels.jsonl` | ID người gán nhãn (ẩn danh), ngày, phiên bản hướng dẫn gán nhãn |
| Điểm MOS | `human_reviewed/mos_ratings/<round>/ratings.csv` | ID người chấm (ẩn danh), ID clip (đã làm mù), thang điểm, thời điểm, kết quả kiểm tra sự chú ý, phiên bản ứng dụng |

Quy trình MOS: thang ACR 5 mức; ID clip là ngẫu nhiên và bảng ánh xạ sang model lưu trong một file khoá riêng mà người chấm không bao giờ thấy; mỗi người chấm có thứ tự ngẫu nhiên riêng; mỗi clip có ≥ `min_raters_per_clip` lượt chấm (đề xuất: 5 — chủ dự án xác nhận theo số người chấm có sẵn); có clip neo (giọng người tham chiếu, giọng bị làm xấu) và clip kiểm tra sự chú ý; người chấm không qua kiểm tra sự chú ý bị loại, và việc loại được báo cáo.

Không dùng dữ liệu cá nhân của người gọi thật. Nếu sau này thêm bản ghi cuộc gọi thật, cần một đợt review quyền riêng tư riêng và việc đó nằm ngoài phạm vi v1.

---

## 8. Tạo audio synthetic

Hướng đã chốt (2026-09-22): TTS mã nguồn mở chạy cục bộ, chọn ở Phase 3 bằng đo lường (xem §10). Yêu cầu với bất kỳ lựa chọn nào:

- License cho phép tạo kích thích cho benchmark.
- Nhiều giọng tiếng Nhật: mục tiêu ≥ 6 giọng, cân bằng giới tính, ghi ở `speaker.id`.
- Tất định hoặc điều khiển được bằng seed; engine + phiên bản + giọng + tham số ghi trong `generator`.
- **Không được là cùng hệ thống với một model đang benchmark** (ví dụ không dùng GPT-Realtime / OpenAI TTS, không dùng phần TTS của bất kỳ model ứng viên nào), để tránh cho model nào đó nhận đầu vào quen thuộc với phân bố của nó.
- Kết quả được báo cáo theo nguồn kích thích (`synthetic` vs `human_reviewed`) để mọi chênh lệch đều hiện rõ.

---

## 9. Phiên bản và Registry

`datasets/registry.yaml`:

```yaml
- name: ja_toolcalling
  version: v1
  status: draft            # candidate | draft | released | deprecated
  manifest_sha256: null    # set when released
  released_at: null
  changelog: "Initial scenario set."
```

Chỉ các phiên bản `released` mới được dùng trong run `standard` và `full`. `vbench data prepare` từ chối build dataset nếu audio build ra không khớp `audio_sha256`: khi repo có `datasets/<name>/<version>/manifest.jsonl`, manifest build trên server phải giống hệt file đó (từng mẫu, từng trường).

---

## 10. Chiến lược dữ liệu khi nhóm không có người nói tiếng Nhật (quyết định 2026-09-22)

Nhóm và công ty hiện không có người nói tiếng Nhật. Vì vậy không thể thu âm giọng người trong nhóm, không thể nhờ người review kịch bản, và không thể tự tổ chức chấm MOS bằng người bản ngữ. Chiến lược:

| Nhu cầu | Nguồn | Trạng thái |
|---|---|---|
| Giọng người thật, transcript đúng (ASR/CER, smoke) | **FLEURS `ja_jp`** (`google/fleurs`, CC-BY-4.0, người bản ngữ đọc, 650 câu ở split test) | Dùng từ Phase 2 (`fleurs_ja_smoke`); Lớp 1 ASR ở Phase 3 |
| Câu nói có intent/slot do người bản ngữ viết | **MASSIVE `ja-JP`** (`AmazonScience/massive`, CC-BY-4.0, do người bản ngữ bản địa hoá, có nhãn intent và slot) | Ứng viên cho Phase 4; cần kiểm tra lại license trước khi dùng |
| Hội thoại nhiều lượt kiểu đặt chỗ, có trạng thái | **JMultiWOZ** (`nu-dialogue/jmultiwoz`, CC-BY-SA-4.0, 4.246 hội thoại do người bản ngữ tạo theo phương pháp Wizard-of-Oz, có belief/book state) | Ứng viên cho Phase 4–5; cần kiểm tra lại license trước khi dùng |
| Audio cho các câu chỉ có dạng text (MASSIVE, JMultiWOZ, kịch bản tổng đài) | **TTS mã nguồn mở chạy cục bộ trên GPU server** | Chọn ở Phase 3 (xem dưới) |
| Kịch bản đặc thù tổng đài (keigo, tool riêng) mà dữ liệu công khai không có | Claude soạn, **không có người review** → `text_origin: llm_draft_unreviewed` | Chỉ dùng khi thiếu nguồn công khai; báo cáo kết quả tách riêng theo nguồn text |

### 10.1 Chọn TTS (Phase 3)

Tiêu chí bắt buộc: chạy cục bộ; license cho phép tạo dữ liệu benchmark; có giọng tiếng Nhật; tải được weights từ Hugging Face và thư viện từ PyPI (server không vào được GitHub); **không** cùng họ với model đang benchmark (loại trừ Qwen3-TTS, CosyVoice và Fish Speech vì liên quan đến Qwen/Alibaba, cũng như OpenAI TTS). Các ứng viên đã tra cứu (chưa đánh giá):

| Ứng viên | License (theo model card) | Ghi chú |
|---|---|---|
| Kokoro-82M | Apache-2.0 | Có 5 giọng tiếng Nhật, nhưng model card tự xếp hạng chất lượng tiếng Nhật chỉ C/C-/C+ |
| MioTTS (0.1B–2.6B) | LFM Open License v1.0 | Huấn luyện trên ~100k giờ Anh–Nhật; cần audio tham chiếu để nhân bản giọng; code inference nằm trên GitHub |
| Style-Bert-VITS2 (JP-Extra) | AGPL-3.0 (thư viện) | Cần biên dịch PyOpenJTalk; license từng giọng khác nhau |

Việc chọn TTS dựa trên **đo lường**: cổng kiểm tra độ rõ tự động (mục 10.2) và MOS proxy, không dựa trên cảm nhận.

### 10.2 Thay cho người nghe kiểm tra: cổng kiểm tra độ rõ tự động

Mỗi câu TTS được chép lại bằng ASR judge (không phải model đang benchmark). Chỉ giữ câu có `cer_kana` ≤ ngưỡng cấu hình; câu không đạt được sinh lại với seed khác hoặc bị loại, và tỷ lệ loại được ghi vào datasheet. Như vậy mọi câu trong dataset đều được máy xác nhận là "nói đúng text", dù không có người bản ngữ nghe.

### 10.3 Hệ quả phải ghi rõ trong báo cáo

- **Lớp 4 (MOS người chấm, 10% business score):** không có người chấm bản ngữ. Lựa chọn: (a) thuê người chấm tiếng Nhật bên ngoài (crowdsourcing/vendor), hoặc (b) chỉ dùng MOS proxy, và toàn bộ leaderboard bị ghi nhãn **provisional** (METRIC_DEFINITIONS §7.3). Cần chủ dự án quyết định trước Phase 8.
- **Hiệu chỉnh LLM judge (keigo, hiểu):** không có nhãn người để tính Cohen's κ. Thay bằng độ đồng thuận giữa ít nhất hai LLM judge khác nhau, ghi rõ là "không có hiệu chỉnh bằng người".
- Kết quả luôn được tách theo nguồn kích thích (FLEURS người thật vs TTS) để thấy chênh lệch.

