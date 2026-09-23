# Giao thức Realtime (harness ↔ model)

| Trường | Giá trị |
|---|---|
| Trạng thái | Phase 2 — `benchmark/adapters/chat.py` (theo lượt) và `benchmark/adapters/realtime.py` (thời gian thực) |
| Phiên bản | 0.2.0 |
| Ngày | 2026-09-22 |
| Thay thế | "Model Gateway Protocol" trong ARCHITECTURE §6.1 (bản dự kiến `GATEWAY_PROTOCOL.md`) |

> Bản tiếng Việt. Tên event, tên trường JSON và code giữ nguyên tiếng Anh.

## 0. Hai kênh giao tiếp (cập nhật 2026-09-23, sau checkpoint 2)

| Kênh | Dùng cho | Endpoint | Hỗ trợ |
|---|---|---|---|
| **chat** (`transport: chat`) | Lớp chấm điểm theo lượt: L1, L3, L4 | `POST /v1/chat/completions`, `stream: true` | system prompt, tool calling, audio vào/ra, huỷ bằng cách đóng luồng |
| **realtime** (`transport: realtime`) | Thời gian thực, barge-in (Phase 6) và GPT-Realtime | WebSocket `/v1/realtime` | streaming, huỷ phản hồi, song công (duplex) |

**Vì sao tách ra.** Ở checkpoint 2, mọi mẫu đều trả về event `error` rỗng. Đọc mã nguồn cho thấy `/v1/realtime?duplex=0` của vLLM-Omni chính là **API chép lời (speech-to-text) của vLLM upstream**: nó chỉ nhận ba event `session.update {model}`, `input_audio_buffer.append {audio}`, `input_audio_buffer.commit {final}`. Không có chỗ cho system prompt, không có tool, không có lệnh huỷ, và event lỗi có trường `error` là **chuỗi** chứ không phải object (nên thông báo lỗi của chúng ta rỗng). Handler duplex mới dùng bộ event kiểu OpenAI như mô tả ở các mục dưới.

Vì các lớp chấm điểm cần system prompt và tool, chế độ theo lượt chuyển sang `/v1/chat/completions`.

### 0.1 Kênh chat

Request (vLLM-Omni):

```json
{
  "model": "openbmb/MiniCPM-o-4_5",
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "<system prompt>"}]},
    {"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}}]}
  ],
  "modalities": ["text", "audio"],
  "chat_template_kwargs": {"use_tts_template": true},
  "tools": [{"type": "function", "function": {...}}],
  "stream": true
}
```

Mỗi chunk SSE có `modality` ở cấp cao nhất; khi `modality == "audio"` thì `choices[].delta.content` là base64 của audio (WAV hoặc PCM16 thô, adapter xử lý cả hai), ngược lại là text. Tool call theo chuẩn OpenAI (`choices[].delta.tool_calls`), kết quả tool gửi lại bằng message `role: tool`.

**Tool calling.** Có hai chế độ, đặt bằng `chat.tool_protocol` trong cấu hình model:

| Chế độ | Cách làm | Khi nào dùng |
|---|---|---|
| `native` | Gửi `tools` + `tool_choice: auto` theo chuẩn OpenAI | Server đã bật `--enable-auto-tool-choice` và `--tool-call-parser <parser>` |
| `prompted` | Mô tả tool trong system prompt, model trả về đúng một object JSON `{"tool": ..., "arguments": {...}}`; harness tự parse và gửi kết quả tool lại bằng một message người dùng | Server chưa có parser phù hợp (checkpoint 2: MiniCPM-o trả HTTP 400 với `tool_choice: auto`) |

Kết quả luôn được gắn tag `tool_mode` để không trộn hai chế độ khi so sánh. Tên tool do model bịa ra vẫn được ghi lại, để đánh giá được tỷ lệ gọi tool ảo.

Huỷ phản hồi = đóng luồng HTTP; server dừng sinh. Đây là "huỷ kiểu orchestrated", được gắn tag như vậy trong kết quả.

Cài đặt: `benchmark/adapters/chat.py`; server giả lập để test: `benchmark/testing/fake_chat.py`.

---

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
