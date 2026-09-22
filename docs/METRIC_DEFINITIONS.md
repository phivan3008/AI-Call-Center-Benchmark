# Định nghĩa Metric

| Trường | Giá trị |
|---|---|
| Trạng thái | **ĐÃ DUYỆT** (2026-09-22) |
| Phiên bản | 0.1.0 |
| Ngày | 2026-09-21 |
| Phụ thuộc | `docs/ARCHITECTURE.md` v0.2.0 |

> Bản tiếng Việt. ID metric, đơn vị, tên trường và tên evaluator giữ nguyên tiếng Anh vì chúng khớp với code và file cấu hình.

Tài liệu này là nguồn tham chiếu duy nhất cho mọi metric mà framework báo cáo. Trường `method` trong mỗi `MetricRecord` tham chiếu đến một mục ở đây (`metric_id` + phiên bản evaluator). Metric nào không được định nghĩa ở đây thì không được xuất hiện trong báo cáo.

---

## 1. Quy ước

### 1.1 Các trường của metric record

Mỗi metric tạo ra các dòng `MetricRecord` (ARCHITECTURE §9.2). Với mỗi metric, tài liệu này xác định:

| Trường | Ý nghĩa |
|---|---|
| **ID** | Định danh ổn định, dạng `l<layer>.<group>.<name>` |
| **Đơn vị** | `ratio` (0–1), `ms`, `mb`, `score_1_5`, `usd`, `count`, … |
| **Chiều** | `↑` càng cao càng tốt, `↓` càng thấp càng tốt |
| **Mức** | `sample` (một dòng cho mỗi mẫu) và/hoặc `aggregate` (tính trong DuckDB từ các dòng mẫu) |
| **Nguồn** | Artifact thô dùng để tính giá trị |
| **Evaluator** | Module tính toán (`benchmark/evaluators/...` hoặc `benchmark/layers/...`) |
| **Nhóm business** | Nhóm business score mà metric đóng góp (hoặc `report-only` = chỉ báo cáo) |

### 1.2 Giá trị trạng thái

| Trạng thái | Ý nghĩa | Giá trị |
|---|---|---|
| `measured` | Đã tính từ artifact thô | số |
| `unsupported` | Model đã được xác minh là không có năng lực này (capability report) | `null`; được 0 điểm trong business score, có gắn tag |
| `not_measured` | Chưa chạy, hoặc run lỗi trước khi tạo ra artifact | `null`; business score chuyển thành `incomplete` |
| `error` | Mẫu đã chạy nhưng lỗi (timeout, crash, đầu ra không hợp lệ) | `null` ở mức mẫu; **tính là thất bại** trong các metric tỷ lệ (ví dụ task completion), không bao giờ âm thầm bị bỏ qua |

### 1.3 Tổng hợp và độ bất định

- **Tỷ lệ** được tính micro-average trên các mẫu, trừ khi ghi khác.
- **Độ trễ** báo cáo P50, P95, P99 dùng percentile nội suy tuyến tính (`numpy.percentile(method="linear")`), kèm trung bình và số lượng.
- **Khoảng tin cậy 95%** cho mọi giá trị tổng hợp: bootstrap phi tham số, 2.000 lần lấy mẫu lại, lấy mẫu lại ở **mức mẫu (hội thoại)**, seed cố định được ghi trong manifest. Với kịch bản nhiều lượt, đơn vị lấy mẫu lại là cả hội thoại.
- **Lặp lại:** metric có tính ngẫu nhiên được chạy `N` lần cho mỗi mẫu (`N` đặt trong run profile); chỉ số lần lặp lưu ở `tags.repeat` của record.
- **Tag kênh:** metric text của L1 được tính riêng cho `tags.channel = text` (đầu ra text của model, nếu có) và `tags.channel = audio` (bản chép lời audio đầu ra do ASR judge tạo). **Kênh audio là kênh được dùng để chấm điểm**, vì người gọi nghe audio; kênh text chỉ để báo cáo.

### 1.4 Chuẩn hoá text tiếng Nhật (`ja_norm_v1`)

Áp dụng cho cả tham chiếu và giả thuyết trước mọi phép so sánh text, cài đặt trong `evaluators/ja_text.py`:

1. Unicode NFKC (thống nhất chữ số/chữ Latin full-width/half-width và katakana).
2. Chuyển chữ Latin về chữ thường.
3. Bỏ dấu câu và ký hiệu: nhóm Unicode `P*` và `S*`, cộng thêm `・`, và `ー` chỉ khi đứng riêng (dấu kéo dài nguyên âm trong từ được giữ lại).
4. Bỏ mọi khoảng trắng.
5. Số: `ja_norm_v1` không chuyển đổi giữa chữ số kanji và chữ số Ả Rập (hai biến thể CER dưới đây làm rõ điều này).

Báo cáo hai biến thể CER:

| Biến thể | Bước bổ sung | Mục đích |
|---|---|---|
| `cer_surface` | không | Nghiêm ngặt, đúng như bản chép lời hiển thị |
| `cer_kana` | Chuyển cả hai phía sang cách đọc katakana bằng bộ phân tích hình thái chốt phiên bản (fugashi + UniDic, có ghi phiên bản) | Loại bỏ khác biệt chính tả kanji/kana (ví dụ 御座います với ございます), gần với "có nói đúng không" hơn |

`cer_kana` là biến thể được chấm điểm; `cer_surface` chỉ báo cáo. Thay đổi cách chuẩn hoá sẽ tạo ra `ja_norm_v2`; kết quả cũ vẫn tái lập được.

### 1.5 Định nghĩa im lặng / có tiếng (`vad_v1`)

Dùng cho L2 và L4. Audio đầu ra được resample về 16 kHz mono và chia thành frame 10 ms. Một frame là **có tiếng (active)** nếu mức RMS lớn hơn `activity_threshold_dbfs` (cấu hình, mặc định đề xuất −45 dBFS). Ngưỡng là giá trị cấu hình được ghi trong manifest; một VAD dựa trên model (Silero VAD, chốt phiên bản) được chạy song song để đối chiếu và báo cáo mức bất đồng.

### 1.6 Judge (bộ chấm)

- **ASR judge** (`asr_judge_v1`): một model ASR duy nhất chốt phiên bản (ứng viên: Whisper large-v3 hoặc một ASR chuyên tiếng Nhật; chọn ở Phase 3, ghi tên + revision). Mức lỗi nền của chính nó được đo bằng cách chép lời **audio kích thích tham chiếu** và báo cáo là `l1.asr.judge_floor_cer`.
- **LLM judge** (`llm_judge_v1`): một model API chốt phiên bản, **không** phải model đang benchmark, temperature 0, prompt rubric có phiên bản trong `configs/prompts/ja/judges/`. Mỗi lần gọi judge lưu prompt, phản hồi thô và nhãn đã parse vào artifact thô. Độ đồng thuận của judge với nhãn người trên tập con hiệu chỉnh được báo cáo bằng Cohen's κ (`judge.<rubric>.kappa`); rubric có κ thấp hơn ngưỡng tối thiểu cấu hình sẽ bị gắn cờ trong báo cáo.

---

## 2. Lớp 1 — Năng lực tiếng Nhật

| ID | Đơn vị | Chiều | Mức | Định nghĩa | Nguồn | Nhóm business |
|---|---|---|---|---|---|---|
| `l1.asr.cer` | ratio | ↓ | sample + aggregate | Tỷ lệ lỗi ký tự khi model chép lại/nhắc lại câu kích thích. Tổng hợp = Σ khoảng cách chỉnh sửa / Σ số ký tự tham chiếu (micro). Biến thể qua `tags.variant ∈ {kana, surface}`. | `output.flac` → ASR judge; text trong `events.jsonl` | Chất lượng tiếng Nhật |
| `l1.asr.wer` | ratio | ↓ | sample + aggregate | Tỷ lệ lỗi từ trên token fugashi/UniDic (dạng bề mặt). | như trên | report-only |
| `l1.asr.judge_floor_cer` | ratio | ↓ | aggregate | CER của ASR judge trên audio kích thích tham chiếu (chất lượng của evaluator, không phải của model). | audio kích thích | report-only |
| `l1.language.ja_rate` | ratio | ↑ | aggregate | Tỷ lệ câu trả lời có bản chép lời ASR được nhận diện là tiếng Nhật (bộ nhận diện ngôn ngữ chốt phiên bản, ví dụ fastText lid, có ghi lại). | bản chép lời ASR | điều kiện (xem §8) |
| `l1.intent.accuracy` | ratio | ↑ | aggregate | Nhãn intent dự đoán khớp chính xác với nhãn chuẩn. Dự đoán được parse từ câu trả lời theo giao thức prompt intent; không parse được = sai. | text trả lời / bản chép lời | report-only |
| `l1.intent.macro_f1` | ratio | ↑ | aggregate | F1 trung bình macro trên tập intent đóng. | như trên | Chất lượng tiếng Nhật |
| `l1.slot.precision` / `.recall` | ratio | ↑ | aggregate | Precision/recall micro ở mức slot. Một slot khớp nếu tên khớp và giá trị sau chuẩn hoá bằng nhau (§2.1). | như trên | report-only |
| `l1.slot.f1` | ratio | ↑ | aggregate | Trung bình điều hoà của hai giá trị trên. | như trên | Chất lượng tiếng Nhật |
| `l1.understanding.accuracy` | ratio | ↑ | aggregate | Hỏi đáp dạng đóng: khớp chính xác sau chuẩn hoá. Dạng mở: LLM judge gán nhãn `correct` (rubric `understanding_v1`). Báo cáo tách riêng theo `tags.form`. | bản chép lời + judge | Chất lượng tiếng Nhật |
| `l1.keigo.compliance_rate` | ratio | ↑ | aggregate | Tỷ lệ câu trả lời (a) không có vi phạm nào theo bộ phát hiện luật **và** (b) được LLM judge gán nhãn `appropriate` (rubric `keigo_v1`). | bản chép lời + bộ phát hiện + judge | Chất lượng tiếng Nhật |
| `l1.keigo.rule_violation_rate` | ratio | ↓ | aggregate | Tỷ lệ câu trả lời có ≥ 1 vi phạm luật (dạng thân mật như kết câu thể thường với khách hàng, danh sách cách nói bị cấm; danh sách luật có phiên bản). | bản chép lời | report-only |
| `l1.long_context.recall_accuracy` | ratio | ↑ | aggregate | Tỷ lệ câu hỏi kiểm tra được trả lời đúng về thông tin đã nêu trước đó trong hội thoại. Báo cáo theo `tags.distance_bucket` (khoảng cách lượt: 1–3, 4–10, 11+). Tổng = micro trên mọi câu hỏi kiểm tra. | bản chép lời + judge/khớp chính xác | Chất lượng tiếng Nhật |

### 2.1 Chuẩn hoá giá trị slot (`slot_norm_v1`)

| Loại slot | Dạng chuẩn hoá |
|---|---|
| date (ngày) | ISO `YYYY-MM-DD`, ngày tương đối được quy đổi theo `reference_date` cố định của kịch bản |
| time (giờ) | `HH:MM` 24h |
| phone (điện thoại) | chỉ giữ chữ số |
| person name (tên người) | `ja_norm_v1` + cách đọc katakana (so sánh cách đọc tên, không so kanji) |
| number / count (số) | số nguyên |
| free text (văn bản tự do) | khớp chính xác sau `ja_norm_v1` |

---

## 3. Lớp 2 — Giọng nói thời gian thực

Mọi thời điểm lấy từ mốc gửi/nhận của harness (`time.monotonic_ns()`) trong `events.jsonl`. Model OSS được đo qua localhost; GPT-Realtime có tính cả mạng internet công cộng, được gắn `tags.network = internet` và kèm `l2.network.rtt_ms` đo được.

### 3.1 Các mốc thời gian tham chiếu

| Ký hiệu | Định nghĩa |
|---|---|
| `t_eos` | Mốc gửi của frame audio chứa mẫu tiếng nói cuối cùng trong lượt người dùng. Vị trí mẫu tiếng nói cuối cùng lưu trong manifest kích thích (`speech_end_s`, tính một lần bằng `vad_v1` trên audio kích thích và có người kiểm tra ngẫu nhiên). |
| `t_first_audio` | Mốc nhận của `AudioDelta` đầu tiên có ít nhất một frame **có tiếng** (§1.5). |
| `t_bargein` | Mốc gửi của frame chứa điểm bắt đầu barge-in (`bargein_onset_s` trong kịch bản). |
| `t_gen_stop` | Sau `t_bargein`: mốc nhận của event `ResponseCancelled`, hoặc mốc nhận của `AudioDelta` cuối cùng có frame có tiếng, sau đó ≥ `stop_silence_ms` (cấu hình) không có đầu ra có tiếng — lấy mốc nào sớm hơn. |

Audio sinh ra có thể đến nhanh hơn thời gian thực và được client đệm lại, nên **độ trễ dừng sinh (generation-stop latency)** (phần model kiểm soát) là metric được chấm điểm. Thời điểm người nghe thực sự hết nghe thấy phụ thuộc cách client xả bộ đệm, được mô phỏng và chỉ báo cáo (`l2.interrupt.perceived_ms`; bộ đệm phát được mô phỏng theo thời gian thực, xả tại `t_gen_stop`).

### 3.2 Metric

| ID | Đơn vị | Chiều | Mức | Định nghĩa | Nhóm business |
|---|---|---|---|---|---|
| `l2.ttfa_ms` | ms | ↓ | sample + P50/P95/P99 | `t_first_audio − t_eos`. Giá trị âm (model nói trước khi người dùng nói xong) được giữ lại và cũng được tính vào `l2.turn.premature_rate`. | Barge-In (P95) |
| `l2.interrupt.latency_ms` | ms | ↓ | sample + P50/P95/P99 | `t_gen_stop − t_bargein`. Nếu đầu ra không dừng trong `interrupt_timeout_ms`, mẫu = thất bại, giá trị `null`, được tính trong tỷ lệ thành công. | Barge-In (P95) |
| `l2.interrupt.perceived_ms` | ms | ↓ | sample + percentile | Thời điểm dừng nghe thấy (mô phỏng) − `t_bargein`. | report-only |
| `l2.bargein.success_rate` | ratio | ↑ | aggregate | Tỷ lệ lần thử barge-in mà (a) `l2.interrupt.latency_ms ≤ interrupt_slo_ms` **và** (b) câu trả lời tiếp theo xử lý đúng câu ngắt lời (LLM judge, rubric `bargein_followup_v1`). | Barge-In |
| `l2.bargein.false_stop_rate` | ratio | ↓ | aggregate | Tỷ lệ lần chèn tiếng đệm/nhiễu (ví dụ 「はい」「ええ」, tiếng ho, tiếng ồn nền) mà sau đó model dừng sinh trong `stop_silence_ms`. | Barge-In |
| `l2.turn.gap_ms` | ms | — | phân bố | Giống TTFA nhưng trên mọi lượt của hội thoại nhiều lượt; báo cáo dạng histogram. | report-only |
| `l2.turn.premature_rate` | ratio | ↓ | aggregate | Tỷ lệ lượt có `t_first_audio < t_eos`, gồm cả kích thích có khoảng ngập ngừng giữa câu (khách hàng đang suy nghĩ). | report-only |
| `l2.stream.rtf` | ratio | ↑ | sample + P5 | Hệ số thời gian thực của đầu ra streaming: số giây audio nhận được / số giây đồng hồ từ delta đầu đến delta cuối. Giá trị < 1 gây giật tiếng. | report-only |
| `l2.stream.underrun_rate` | ratio | ↓ | aggregate | Tỷ lệ câu trả lời mà bộ đệm phát mô phỏng theo thời gian thực bị cạn trước khi câu trả lời kết thúc. | report-only |
| `l2.duplex.supported` | bool | — | model | Lấy từ capability report (`native_full_duplex = verified`). Metric riêng cho duplex (ví dụ tạo tiếng đệm) sẽ được định nghĩa ở phiên bản sau. | report-only |
| `l2.stability.ttfa_drift_ratio` | ratio | ↓ | mỗi cuộc gọi dài | P50 TTFA của 20% lượt cuối / P50 TTFA của 20% lượt đầu. | report-only |
| `l2.stability.error_rate` | ratio | ↓ | aggregate | Tỷ lệ lượt trong cuộc gọi dài kết thúc bằng `error` (timeout, mất kết nối, trả lời rỗng). | điều kiện (xem §8) |
| `l2.stability.vram_growth_mb` | mb | ↓ | mỗi cuộc gọi dài | Bộ nhớ đã dùng theo NVML lúc kết thúc − lúc bắt đầu cuộc gọi dài. | report-only |
| `l2.network.rtt_ms` | ms | — | aggregate | RTT đo được đến endpoint API (chỉ GPT-Realtime). | report-only |

Mọi record L2 đều có `tags.mode ∈ {native, orchestrated}` (ARCHITECTURE §6.3).

---

## 4. Lớp 3 — Tool Calling

Artifact nguồn: `tool_trace.jsonl` (mọi lời gọi tool kèm tham số thô, kết quả kiểm tra schema, phản hồi của backend) và ảnh chụp trạng thái cuối của toolserver. Mỗi kịch bản định nghĩa một trace tool kỳ vọng và một trạng thái cuối kỳ vọng (`DATASET_SPEC.md` §5).

| ID | Đơn vị | Chiều | Mức | Định nghĩa | Nhóm business |
|---|---|---|---|---|---|
| `l3.task.completion_rate` | ratio | ↑ | aggregate | Tỷ lệ kịch bản có trạng thái cuối của toolserver bằng trạng thái cuối kỳ vọng (bộ so sánh trạng thái, giá trị đã chuẩn hoá). Lỗi/timeout tính là thất bại. Theo `tags.category ∈ {booking, faq, crm_lookup, transfer, structured_output}`. | Hoàn thành tác vụ (chính) |
| `l3.tool.success_rate` | ratio | ↑ | aggregate | Tỷ lệ lời gọi tool (a) đến tool có định nghĩa, (b) hợp lệ theo schema, và (c) được backend chấp nhận không lỗi. | Hoàn thành tác vụ |
| `l3.json.schema_valid_rate` | ratio | ↑ | aggregate | Tỷ lệ lời gọi tool có tham số parse được thành JSON và hợp lệ theo JSON Schema của tool. | report-only |
| `l3.json.arg_accuracy` | ratio | ↑ | aggregate | Với các lời gọi kỳ vọng đã được ghép với một lời gọi thực tế: tỷ lệ tham số kỳ vọng có giá trị chuẩn hoá bằng giá trị thực tế (`slot_norm_v1`). | Hoàn thành tác vụ |
| `l3.tool.hallucination_rate` | ratio | ↓ | aggregate | Tỷ lệ lời gọi tool (a) gọi tên tool không tồn tại, hoặc (b) có giá trị tham số không có căn cứ trong hội thoại hay kết quả tool trước đó (kiểm tra căn cứ: giá trị, sau chuẩn hoá, có xuất hiện trong lượt người dùng hoặc đầu ra tool; trường hợp mơ hồ chuyển cho LLM judge rubric `grounding_v1`). | Hoàn thành tác vụ (trừ điểm) |
| `l3.tool.missed_call_rate` | ratio | ↓ | aggregate | Tỷ lệ lời gọi kỳ vọng không có lời gọi thực tế tương ứng. | report-only |
| `l3.structured.exact_match` | ratio | ↑ | aggregate | Kịch bản structured output: JSON thực tế sau chuẩn hoá bằng đúng JSON kỳ vọng. | report-only |

Ghép lời gọi thực tế với lời gọi kỳ vọng: cùng tên tool, sau đó chọn phương án trùng tham số nhiều nhất (thuật toán Hungarian). Bỏ qua thứ tự, trừ khi kịch bản đánh dấu `ordered: true`.

Mọi record L3 đều có `tags.tool_mode ∈ {native, prompted}`.

---

## 5. Lớp 4 — Chất lượng giọng nói

| ID | Đơn vị | Chiều | Mức | Định nghĩa | Nhóm business |
|---|---|---|---|---|---|
| `l4.mos.human` | score_1_5 | ↑ | mỗi clip + aggregate | Trung bình điểm theo thang ACR 5 mức (1 tệ – 5 xuất sắc) cho chất lượng tổng thể, từ những người chấm vượt qua kiểm tra sự chú ý. CI 95% bằng bootstrap trên clip và người chấm. | Chất lượng giọng nói (chính) |
| `l4.naturalness.human` | score_1_5 | ↑ | như trên | Điểm tự nhiên 5 mức. | Chất lượng giọng nói |
| `l4.accent.human` | score_1_5 | ↑ | như trên | Điểm 5 mức cho độ chính xác ngữ điệu / pitch-accent tiếng Nhật. | Chất lượng giọng nói |
| `l4.emotion.human` | score_1_5 | ↑ | như trên | Điểm 5 mức cho mức phù hợp của sắc thái cảm xúc với tình huống soạn sẵn (xin lỗi, đồng cảm, vui vẻ). | Chất lượng giọng nói |
| `l4.mos.proxy.<predictor>` | score_1_5 | ↑ | mỗi clip + aggregate | Đầu ra của bộ dự đoán MOS tự động (tên + phiên bản bộ dự đoán ghi trong `method`). Tương quan Pearson/Spearman với `l4.mos.human` báo cáo ở `l4.mos.proxy.<predictor>.corr_human`. | chỉ dự phòng (§7.3) |
| `l4.pronunciation.cer` | ratio | ↓ | aggregate | `l1.asr.cer` (biến thể kana) trên bộ câu đọc: model được yêu cầu nói câu cố định có số, ngày, tên, từ vựng nghiệp vụ. | Chất lượng giọng nói |
| `l4.consistency.speaker_sim` | ratio | ↑ | aggregate | Trung bình độ tương đồng cosine của speaker embedding (model loại ECAPA-TDNN, chốt phiên bản) giữa mỗi câu trả lời và tâm giọng của model trong phiên; tính cả giữa các phiên. | Chất lượng giọng nói |
| `l4.rater.agreement` | ratio | ↑ | aggregate | Krippendorff's α của điểm người chấm (chất lượng của việc chấm). | report-only |

Quy trình MOS người chấm: chấm mù, thứ tự ngẫu nhiên riêng cho từng người chấm, tên file không lộ danh tính model, mỗi clip được ≥ `min_raters_per_clip` người chấm, có clip neo (tham chiếu chất lượng cao/thấp) và clip kiểm tra sự chú ý. Chi tiết ở `DATASET_SPEC.md` §7.

---

## 6. Lớp 5 — Hạ tầng & TCO

Mẫu NVML lấy ở 10 Hz (`monitoring/nvml.parquet`) kèm mốc giai đoạn (`idle`, `single_call`, `concurrency_N`).

| ID | Đơn vị | Chiều | Mức | Định nghĩa | Nhóm business |
|---|---|---|---|---|---|
| `l5.vram.idle_mb` | mb | ↓ | model | Trung vị bộ nhớ đã dùng theo NVML ở giai đoạn `idle` (model đã nạp, không có lưu lượng). | report-only |
| `l5.vram.peak_mb` | mb | ↓ | mỗi giai đoạn | Bộ nhớ đã dùng lớn nhất theo NVML trong giai đoạn. | điều kiện (vừa 80 GB) |
| `l5.gpu.util_mean` | ratio | — | mỗi giai đoạn | SM utilization trung bình. | report-only |
| `l5.throughput.audio_rtf` | ratio | ↑ | mỗi mức đồng thời | Tổng số giây audio mọi phiên sinh ra / số giây đồng hồ. | report-only |
| `l5.concurrency.max_at_slo` | count | ↑ | model | Mức đồng thời `N` lớn nhất đã thử (quét 1, 2, 4, 8, …, rồi chia đôi giữa mức đạt cuối cùng và mức trượt đầu tiên) mà tại `N` cuộc gọi streaming đồng thời: `l2.ttfa_ms` P95 ≤ `ttfa_p95_slo_ms`, `l2.stream.underrun_rate` ≤ `underrun_slo`, và tỷ lệ lỗi ≤ `error_rate_slo`. Bằng `0` nếu `N = 1` đã trượt. | Chi phí (đầu vào) |
| `l5.cost.per_minute_usd` | usd | ↓ | model | OSS: `gpu_hourly_usd / (60 × l5.concurrency.max_at_slo)`. Baseline API: lượng sử dụng tính phí đo được (token audio/text vào/ra từ trường usage của API) × giá trong `pricing.yaml` / số phút gọi. `null` với `not_measured` nếu mức đồng thời tối đa là 0 hoặc thiếu giá. | Chi phí (chính) |
| `l5.cost.per_call_usd` | usd | ↓ | model | `l5.cost.per_minute_usd × thời lượng cuộc gọi trung bình` của bộ kịch bản chuẩn (đo được). | report-only |
| `l5.tco.monthly_usd` | usd | ↓ | model | Mô hình TCO cấu hình được: số GPU cần cho `target_peak_concurrent_calls` (= ceil(target / max_at_slo)) × chi phí GPU hàng tháng (khấu hao mua hoặc thuê) + điện + chi phí vận hành, tất cả từ `pricing.yaml`. | report-only |

Mọi giá đầu vào có `source` và `as_of` trong `configs/cost/pricing.yaml`; báo cáo in các giá trị này cạnh số liệu chi phí.

---

## 7. Business Score (`business_v1`)

### 7.1 Chuẩn hoá

Mỗi metric đầu vào `m` được đưa về `s(m) ∈ [0, 1]` bằng hàm tuyến tính có cắt ngưỡng giữa hai anchor do chủ dự án đặt trong `configs/scoring/anchors.yaml`:

```
direction ↑:  s = clip((m - worst) / (best - worst), 0, 1)
direction ↓:  s = clip((worst - m) / (worst - best), 0, 1)
```

Anchor là **ngưỡng nghiệp vụ** (ví dụ "TTFA P95 từ X ms trở xuống là hoàn hảo; từ Y ms trở lên là không dùng được"), cố định và được ký duyệt **trước khi** thấy kết quả. Với metric tỷ lệ có biên tự nhiên 1.0 / 0.0, mặc định là `best = 1, worst = 0` (↑) hoặc `best = 0, worst = 1` (↓), trừ khi chủ dự án ghi đè.

### 7.2 Các nhóm và trọng số con

Trọng số cấp cao được cố định theo CLAUDE.md. Trọng số con là **đề xuất** để chủ dự án xác nhận (lưu trong `configs/scoring/business_v1.yaml`).

| Nhóm (trọng số) | Metric đầu vào (trọng số con đề xuất) |
|---|---|
| Chất lượng tiếng Nhật (35%) | `l1.asr.cer` kana (0.25), `l1.intent.macro_f1` (0.15), `l1.slot.f1` (0.15), `l1.understanding.accuracy` (0.15), `l1.keigo.compliance_rate` (0.20), `l1.long_context.recall_accuracy` (0.10) |
| Hoàn thành tác vụ (25%) | `l3.task.completion_rate` (0.60), `l3.tool.success_rate` (0.15), `l3.json.arg_accuracy` (0.15), `1 − l3.tool.hallucination_rate` (0.10) |
| Barge-In (15%) | `l2.interrupt.latency_ms` P95 (0.30), `l2.bargein.success_rate` (0.30), `l2.bargein.false_stop_rate` (0.15), `l2.ttfa_ms` P95 (0.25) |
| Chi phí (15%) | `l5.cost.per_minute_usd` (1.00) |
| Chất lượng giọng nói (10%) | `l4.mos.human` (0.40), `l4.naturalness.human` (0.15), `l4.accent.human` (0.15), `l4.pronunciation.cer` (0.15), `l4.consistency.speaker_sim` (0.10), `l4.emotion.human` (0.05) |

`dimension_score = Σ sub_weight × s(metric)`; `business_score = Σ weight × dimension_score`, báo cáo trên thang 0–100.

TTFA được xếp vào nhóm Barge-In vì CLAUDE.md không có nhóm độ trễ riêng, và khả năng phản hồi đúng lượt là một phần của tương tác hội thoại; vị trí này đã được duyệt.

### 7.3 Đầu vào bị thiếu

- Metric có trạng thái `unsupported` đóng góp `s = 0` (có gắn tag).
- Metric có trạng thái `not_measured` làm business score của model thành `incomplete` (ARCHITECTURE §10.3). Không phân bổ lại trọng số, không tự điền giá trị.
- **Dự phòng cho chất lượng giọng nói:** nếu chưa có MOS người chấm cho bất kỳ model nào, báo cáo có thể tính điểm Chất lượng giọng nói *tạm thời* bằng `l4.mos.proxy.<predictor>` thay cho metric người chấm — chỉ khi `corr_human` của bộ dự đoán đã được đo trên tập con hiệu chỉnh — và toàn bộ leaderboard được ghi nhãn **tạm thời (provisional)**.

### 7.4 Xếp hạng

Các model có điểm đầy đủ được xếp theo `business_score`. CI 95% của business score lấy từ bootstrap đồng thời (lấy mẫu lại ở mọi lớp cùng lúc). Hai model liền kề có CI của chênh lệch điểm chứa 0 được hiển thị là đồng hạng. Một bảng độ nhạy xếp hạng lại khi dịch chuyển ±10 điểm phần trăm từng trọng số cấp cao (chỉ báo cáo, để cho thấy độ vững của thứ hạng).

---

## 8. Điều kiện triển khai (chỉ báo cáo, không tính điểm)

| Điều kiện | Điều kiện đạt |
|---|---|
| Vừa phần cứng | `l5.vram.peak_mb` tại `N = 1` ≤ 80 GB và model chạy trên 1x H100 |
| Nói tiếng Nhật | `l1.language.ja_rate` ≥ `ja_rate_gate` (cấu hình anchors) |
| Ổn định | `l2.stability.error_rate` ≤ `stability_error_gate` |
| License | Trường review thủ công trong YAML của model: `commercial_use: allowed / restricted / unknown` |

Không đạt điều kiện không làm thay đổi điểm; leaderboard hiển thị model là "không triển khai được như cấu hình đã benchmark" kèm điều kiện không đạt.
