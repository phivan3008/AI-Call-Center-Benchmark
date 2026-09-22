# Giao thức Realtime (harness ↔ model)

| Trường | Giá trị |
|---|---|
| Trạng thái | Phase 2 — đã cài đặt trong `benchmark/adapters/realtime.py` |
| Phiên bản | 0.1.0 |
| Ngày | 2026-09-22 |
| Thay thế | "Model Gateway Protocol" trong ARCHITECTURE §6.1 (bản dự kiến `GATEWAY_PROTOCOL.md`) |

> Bản tiếng Việt. Tên event, tên trường JSON và code giữ nguyên tiếng Anh.

## 1. Vì sao dùng giao thức kiểu OpenAI Realtime

vLLM-Omni 0.28 có sẵn WebSocket `/v1/realtime` theo event của OpenAI Realtime API cho **Qwen3-Omni** và **MiniCPM-o 4.5** ([tài liệu](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/realtime_duplex_api/)). Baseline GPT-Realtime dùng chính OpenAI Realtime API. Vì vậy harness chỉ cần **một adapter** (`RealtimeAdapter`) với hai "dialect":

| Dialect | Dùng cho | Audio vào | Audio ra | Xác thực |
|---|---|---|---|---|
| `vllm_omni` | Qwen3-Omni, MiniCPM-o 4.5 (server cục bộ) | PCM16 16 kHz | PCM16 24 kHz | không |
| `openai` | GPT-Realtime | PCM16 24 kHz | PCM16 24 kHz | `Authorization: Bearer $OPENAI_API_KEY` |

Các model mà vLLM-Omni không hỗ trợ (StepAudio 2.5, GLM-4-Voice, LLaMA-Omni 2, Baichuan-Omni-1.5) sẽ cần một gateway nhỏ nói cùng tập event này; việc đó thuộc Phase 9.

Harness tự resample audio đầu vào về sample rate của dialect (thư viện `soxr`, chế độ HQ).

## 2. Event harness gửi đi

| Event | Khi nào | Ghi chú |
|---|---|---|
| `session.update` | ngay sau khi kết nối | `instructions` = system prompt; `turn_detection: null` (harness tự quyết định lượt nói); khai báo `tools` nếu có |
| `input_audio_buffer.append` | mỗi frame 20 ms | `audio` = base64 PCM16; dialect `vllm_omni` kèm `sample_rate_hz` |
| `input_audio_buffer.commit` + `response.create` | người dùng nói xong | `vllm_omni`: `response.modalities = ["audio", "text"]` |
| `response.cancel` | huỷ phản hồi (kiểm tra barge-in kiểu orchestrated) | kèm `response_id` nếu đã biết |
| `conversation.item.create` (`function_call_output`) + `response.create` | trả kết quả tool | `output` = JSON |

Kết nối tới vLLM-Omni dùng `?duplex=0` để chọn handler theo lượt (turn-based). Chế độ full-duplex (`?duplex=1`) được kiểm tra ở Phase 6.

## 3. Event từ server và cách harness diễn giải

| Event server (chấp nhận cả tên mới và tên cũ) | Event harness |
|---|---|
| `response.output_audio.delta`, `response.audio.delta` | `AudioDelta` |
| `response.output_audio_transcript.delta`, `response.audio_transcript.delta`, `response.output_text.delta`, `response.text.delta` | `TextDelta` |
| `response.function_call_arguments.done`, `response.output_item.done` (item `function_call`) | `ToolCall` (bỏ trùng theo `call_id`) |
| `response.done` với `status` = `completed` | `ResponseDone` |
| `response.done` với `status` = `cancelled` | `ResponseCancelled` |
| `response.done` với `status` = `failed` / `incomplete` | `ErrorEvent` |
| `error` | `ErrorEvent` |
| mất kết nối | `ErrorEvent(code="connection_closed")` |

Handler theo lượt của Qwen3-Omni dùng tên cũ (`response.audio.delta`), còn handler duplex dùng tên mới; adapter chấp nhận cả hai.

**Mốc thời gian:** mọi event được harness gán `time.monotonic_ns()` lúc nhận (ARCHITECTURE §6.2), không dùng thời gian do server báo.

## 4. Kiểm thử không cần GPU

`benchmark/testing/fake_realtime.py` là một server WebSocket giả lập tập event trên (kể cả tool call, huỷ, lỗi, từ chối session, tên event cũ). Test chạy toàn bộ adapter, runner và capability report với server này trên CPU. Đầu ra của server giả lập không bao giờ là dữ liệu benchmark.
