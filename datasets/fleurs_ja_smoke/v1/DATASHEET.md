# Datasheet — `fleurs_ja_smoke` v1

| Trường | Giá trị |
|---|---|
| Mục đích | Smoke test Phase 2: kiểm tra pipeline và năng lực model (turn, tool call, huỷ phản hồi). **Không dùng để chấm điểm.** |
| Nguồn | [`google/fleurs`](https://huggingface.co/datasets/google/fleurs), cấu hình `ja_jp`, split `test` |
| Revision | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` (chốt cứng trong `benchmark/datasets/fleurs.py`) |
| License | CC-BY-4.0 (theo metadata dataset card trên Hugging Face). Khi dùng lại cần ghi nguồn FLEURS. |
| Loại dữ liệu | Giọng đọc thật của người bản ngữ tiếng Nhật; `synthetic: false`, `text_origin: corpus` |
| Nhãn | Transcript gốc của corpus (`raw_transcription`), không chỉnh sửa |
| Số mẫu | 12 |
| Cách chọn | Các dòng có độ dài 2–10 giây, sắp theo tên file, lấy 12 dòng đầu (tất định) |
| Định dạng | WAV PCM16 mono 16 kHz (FLEURS gốc là WAV float32, được chuyển về PCM16) |
| Ranh giới tiếng nói | `stim_vad_v1`: ngưỡng năng lượng thích ứng theo từng clip (METRIC_DEFINITIONS §1.5). Chưa có người nghe kiểm tra; chỉ gần đúng. |
| File trong repo | `manifest.jsonl` (tham chiếu, gồm SHA-256 từng file audio) và `dataset_info.json`. Audio **không** nằm trong git. |
| Cách build | `uv run vbench data prepare --dataset fleurs_ja_smoke`. Kết quả build phải khớp `manifest.jsonl` trong repo, nếu không lệnh sẽ báo lỗi. |

## Hạn chế đã biết

- Nội dung là câu đọc từ Wikipedia (FLEURS), **không phải hội thoại tổng đài**. Chỉ dùng để kiểm tra pipeline, không phản ánh tình huống nghiệp vụ.
- Mức âm lượng và tiếng nền giữa các clip rất khác nhau (tiếng nền từ khoảng -95 đến -30 dBFS).
- Nhóm không có người nói tiếng Nhật, nên chưa ai nghe kiểm tra ranh giới tiếng nói hay nội dung.
