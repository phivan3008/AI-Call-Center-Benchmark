# Hướng dẫn triển khai — GPU Benchmark Server

| Trường | Giá trị |
|---|---|
| Trạng thái | **ĐÃ DUYỆT** (2026-09-22), cập nhật v0.2.0 cho bối cảnh không có GitHub trên server |
| Phiên bản | 0.2.0 |
| Ngày | 2026-09-22 |
| Đối tượng | Người vận hành chạy benchmark trên Env C (Linux, 1x H100 80GB, SSD 200GB) |

> Bản tiếng Việt. Các lệnh giữ nguyên, copy chạy trực tiếp được. Mỗi mục ghi rõ phase bắt đầu có lệnh đó; trước phase đó chỉ các lệnh của những phase đã bàn giao mới chạy được.

---

## 1. Bối cảnh vận hành

| Mục | Giá trị |
|---|---|
| GitHub | **GPU server không kết nối được GitHub.** Code chỉ đến server bằng cách copy file từ máy công ty. |
| Máy công ty (Env B) | Truy cập được GitHub; copy/paste được mọi loại file và thư mục lên server và từ server về. |
| Internet trên server | Tải được từ PyPI (thư viện Python), Hugging Face (model) và gọi được OpenAI API. Chỉ không truy cập được GitHub. |
| Docker | **Không có.** Model chạy như process cục bộ trong `uv` venv riêng cho từng model. |
| Engine phục vụ model | Mặc định vLLM; dự phòng bằng code inference chính thức (có tài liệu) cho từng model. |

### 1.1 Luồng đi của code và kết quả

```text
Claude (máy dev) --push + tag--> GitHub --tạo Release--> [file .zip + .zip.sha256]
                                                               |
                               máy công ty tải 2 file từ trang GitHub Releases
                                                               |
                                               copy 2 file lên GPU server
                                                               |
            GPU server: kiểm tra sha256 -> giải nén -> vbench release verify -> chạy benchmark
                                                               |
                          bundle kết quả (.tar.zst + .SHA256SUMS) copy về máy công ty
                                                               |
                                       gửi cho Claude để phân tích
```

**Vì sao dùng gói release thay vì copy thư mục repo:** server không có `.git`, nên framework không tự biết code đang chạy là commit nào. Gói release (`.zip`) chứa file `BUILD_INFO.json` ghi commit và SHA-256 của từng file. Lệnh `vbench release verify` dùng file này để chứng minh code trên server giống hệt commit đó. Mọi run đều ghi lại tên release và commit, nên kết quả vẫn truy vết được và được xếp hạng. Nếu code bị sửa, thiếu file, hoặc bị đổi ký tự xuống dòng khi copy qua Windows, lệnh verify sẽ phát hiện, và run bị đánh dấu `dirty` (không được xếp hạng).

**Luôn copy nguyên file `.zip`**, không giải nén trên máy công ty rồi copy thư mục. Copy thư mục qua Windows có thể đổi ký tự xuống dòng (LF → CRLF) hoặc bỏ sót file.

---

## 2. Yêu cầu cài đặt trên server (một lần)

| Yêu cầu | Lệnh kiểm tra | Ghi chú |
|---|---|---|
| NVIDIA driver + CUDA tương thích với vLLM/torch đã chốt | `nvidia-smi` | Phiên bản tối thiểu chính xác được chốt sau checkpoint 1 từ output của `vbench env check`. |
| `uv` | `uv --version` | Cài: `curl -LsSf https://astral.sh/uv/install.sh \| sh` (hoặc `pip install uv`). `uv` tự cài Python 3.11+ cho từng venv. |
| `python3` hoặc `unzip` | `python3 --version` | Để giải nén gói release (`python3 -m zipfile -e ...`). |
| `sha256sum` | `sha256sum --version` | Kiểm tra file `.zip` sau khi copy (có sẵn trên Linux). |
| Thư viện hệ thống cho model runtime | `ldconfig -p \| grep -E "libGL\\.so\\.1\|libglib-2.0\|libsndfile"` | **Bắt buộc từ Phase 2.** Tiến trình con của vLLM-Omni import OpenCV (`cv2`), cần `libGL.so.1` và `libglib2.0`. Cài một lần bằng quyền root:<br>`apt-get update && apt-get install -y libgl1 libglib2.0-0 libsndfile1`<br>(ảnh cũ dùng tên `libgl1-mesa-glx`) |
| `ffmpeg` | `ffmpeg -version` | Chuyển đổi audio, mô phỏng kênh điện thoại (cần từ Phase 3). |
| Dung lượng đĩa trống | `df -h $VBENCH_HOME` | Xem §5. |

Server **không cần** `git`. Bản thân framework không cần quyền root, ngoài việc cài các gói hệ thống ở trên.

---

## 3. Cấu trúc thư mục trên server

Chọn một thư mục gốc trên SSD benchmark, gọi là `VBENCH_HOME`. Mọi đường dẫn đều suy ra từ thư mục này; không có đường dẫn nào bị hardcode.

```text
$VBENCH_HOME/
  .env                          # biến môi trường (xem §4), chmod 600
  releases/
    ai-callcenter-benchmark-0.1.1-448a82fa.zip          # gói release đã copy lên
    ai-callcenter-benchmark-0.1.1-448a82fa.zip.sha256
    ai-callcenter-benchmark-0.1.1-448a82fa/             # thư mục code đã giải nén
  hf_cache/                     # HF_HOME: model weights
  runtimes/<model>/.venv        # one uv venv per model (created by `vbench model prepare`)
  datasets_audio/               # built dataset audio
  artifacts/                    # raw / processed / reports / dashboards (kết quả run)
  bundles/                      # file kết quả để copy về
```

Mỗi release giải nén vào một thư mục riêng; không ghi đè lên release cũ. Kết quả run nằm trong `$VBENCH_HOME/artifacts/`, **bên ngoài** thư mục code, nên xoá hay thay release không làm mất kết quả. `vbench env check` báo `fail` (`home_outside_repo`) nếu `VBENCH_HOME` trỏ vào chính thư mục code.

Có thể xoá thư mục release cũ khi không còn cần (chỉ xoá thư mục trong `releases/`, không xoá `artifacts/` hay `bundles/`).

---

## 4. Biến môi trường

Tạo file `$VBENCH_HOME/.env` một lần (không bao giờ đưa file này vào repo; đặt quyền `chmod 600`):

```bash
VBENCH_HOME=/path/to/benchmark_ssd/vbench
HF_HOME=${VBENCH_HOME}/hf_cache
HF_TOKEN=...                 # only if a gated model requires it
OPENAI_API_KEY=...           # GPT-Realtime baseline and LLM judge (if an OpenAI model is chosen as judge)
VBENCH_LOG_LEVEL=INFO
VBENCH_DISK_BUDGET_GB=200    # usable share of the 300 GB SSD
```

Thay `/path/to/benchmark_ssd/vbench` bằng đường dẫn thật trên SSD. `HF_TOKEN` chỉ cần khi một model yêu cầu đăng nhập (gated model).

Lưu ý bảo mật:

- Key chỉ được đọc từ biến môi trường. Framework không bao giờ ghi giá trị key vào log, manifest hay bundle; `env check` chỉ báo "present" (có) / "missing" (thiếu).
- Trước khi tạo bundle, `vbench bundle create` quét các file text để tìm chuỗi giống secret (giá trị thật của `OPENAI_API_KEY`/`HF_TOKEN` và các mẫu key phổ biến) và từ chối tạo bundle nếu tìm thấy.

---

## 5. Quy trình quản lý dung lượng đĩa

Không thể giả định SSD 200GB chứa được tất cả model cùng lúc. Dung lượng weights và venv của từng model được **đo thực tế** ở Phase 2/9 (không ước lượng) và ghi vào `configs/models/<model>.yaml: disk_measured_gb`.

Quy trình cho mỗi model:

1. `vbench env check` — hiển thị dung lượng trống và dung lượng đo được của model tiếp theo (khi đã biết).
2. `vbench model prepare --model <id>` — từ chối chạy nếu dung lượng trống < dung lượng đo được + `disk_safety_margin_gb` (cấu hình).
3. Chạy benchmark.
4. `vbench bundle create <run_id>` — đóng gói kết quả trước.
5. `vbench model evict --model <id>` — xoá weights khỏi `hf_cache` và xoá venv của model; giữ nguyên artifact.

Artifact audio thô có thể lớn. Profile `full` lưu audio đầu ra dạng FLAC, và bước đóng gói có thể bỏ audio (`--no-audio`) khi chỉ cần gửi metric và timeline. Lớp 4 (chất lượng giọng nói) cần audio đầu ra, nên bundle L4 luôn kèm audio.

---

## 6. Quy trình vận hành chuẩn

### 6.1 Đưa code mới lên server (mỗi checkpoint)

Mỗi hướng dẫn checkpoint của Claude ghi rõ **tên release** (ví dụ `v0.1.1`) và **tên file** (ví dụ `ai-callcenter-benchmark-0.1.1-448a82fa.zip`).

**Trên máy công ty:**

1. Mở trang Releases của repo trên GitHub: `https://github.com/phivan3008/AI-Call-Center-Benchmark/releases`.
2. Mở release được nêu trong hướng dẫn (ví dụ `v0.1.1`), tải **cả hai** file trong mục Assets:
   - `ai-callcenter-benchmark-<phiên bản>-<commit>.zip`
   - `ai-callcenter-benchmark-<phiên bản>-<commit>.zip.sha256`
3. Copy nguyên hai file (không giải nén) lên server, vào thư mục `$VBENCH_HOME/releases/`.

**Trên GPU server:**

```bash
export VBENCH_HOME=/path/to/benchmark_ssd/vbench         # đường dẫn thật, giống trong .env
set -a; source $VBENCH_HOME/.env; set +a                  # nạp biến môi trường

REL=ai-callcenter-benchmark-0.1.1-448a82fa               # tên file trong hướng dẫn checkpoint, bỏ đuôi .zip
cd $VBENCH_HOME/releases
sha256sum -c $REL.zip.sha256                              # phải in ra: <tên file>.zip: OK
python3 -m zipfile -e $REL.zip .                          # hoặc: unzip -q $REL.zip
cd $VBENCH_HOME/releases/$REL

uv sync                                                   # cài thư viện của harness (tải từ PyPI)
uv run vbench release verify                              # phải in ra "ok": true
```

Nếu `sha256sum -c` báo `FAILED` hoặc `release verify` báo `"ok": false`: file bị hỏng hoặc sai khi copy. Copy lại hai file từ máy công ty và làm lại. Nếu vẫn lỗi, gửi output về cho Claude.

Mọi lệnh ở các mục sau đều chạy **từ thư mục release** (`$VBENCH_HOME/releases/$REL`), trong terminal đã nạp `.env`.

### 6.2 Preflight và mock round trip (có từ Phase 1)

```bash
uv run vbench env check --output $VBENCH_HOME/bundles/env_report.json
uv run vbench run --model mock --profile mock_smoke      # CPU-only pipeline check, prints RUN_ID
uv run vbench bundle create <RUN_ID>
```

Thay `<RUN_ID>` bằng giá trị in ra ở dòng `RUN_ID=...` của lệnh trước.

`env check` thoát với mã khác 0 nếu một mục bắt buộc bị fail. Khi đó hãy dừng lại và gửi kết quả về. Ý nghĩa các trạng thái: `pass` = đạt, `warn` = cảnh báo (không bắt buộc), `fail` = không đạt mục bắt buộc. Run mock có `is_mock=true` và không bao giờ được xếp hạng.

Với `--role server` (mặc định), các mục bắt buộc gồm:

| Mục | Ý nghĩa |
|---|---|
| `python_version` | Python ≥ 3.11 |
| `gpu`, `nvidia_driver` | Thấy GPU NVIDIA và driver qua NVML |
| `home_writable` | Ghi được vào `VBENCH_HOME` |
| `home_outside_repo` | `VBENCH_HOME` nằm ngoài thư mục code |
| `source_integrity` | Code khớp đúng một commit (release đã verify, hoặc git checkout sạch) |
| `tool:uv`, `tool:nvidia-smi` | Có các công cụ này |
| `network:huggingface`, `network:openai`, `network:pypi` | Kết nối ra ngoài được |
| `env:OPENAI_API_KEY` | Có API key (không in giá trị) |

`ffmpeg`, `zstd`, `git`, `libsndfile`, `HF_TOKEN` chỉ là cảnh báo.

### 6.3 Chạy cho từng model (có từ Phase 2)

Các lệnh quản lý model:

| Lệnh | Việc làm |
|---|---|
| `uv run vbench model list` | Liệt kê model đã cấu hình |
| `uv run vbench data prepare --dataset <tên>` | Tải dữ liệu (revision chốt cứng) và build audio; kết quả phải khớp manifest trong repo |
| `uv run vbench model prepare --model <id>` | Tạo venv runtime (tải thư viện từ PyPI) và tải weights (Hugging Face) đúng revision; kiểm tra giới hạn đĩa 200 GB trước khi tải; ghi `prepare_report.json` với dung lượng **đo thực tế** |
| `uv run vbench model serve --model <id>` | Chạy server model ở nền, chờ đến khi `/health` trả lời; ghi log vào `$VBENCH_HOME/logs/runtime_<id>.log`; khoá GPU |
| `uv run vbench model check --model <id>` | Import thử các module trong venv của runtime (bắt lỗi thiếu thư viện hệ thống) |
| `uv run vbench model status --model <id>` | Đã prepare chưa, có đang chạy và khoẻ không |
| `uv run vbench model stop --model <id>` | Dừng server, mở khoá GPU |
| `uv run vbench model evict --model <id>` | Dừng server và xoá weights để giải phóng đĩa (`--with-runtime` xoá cả venv dùng chung) |
| `uv run vbench run --model <id> --profile smoke` | Chạy benchmark; model phải đang được serve (trừ GPT-Realtime) |

Quy trình cho một model:

```bash
MODEL=minicpm-o-4_5
uv run vbench model prepare --model $MODEL        # lâu: tải thư viện + weights
uv run vbench model serve   --model $MODEL        # chờ đến khi server sẵn sàng
uv run vbench run --model $MODEL --profile smoke  # in ra RUN_ID
uv run vbench model stop    --model $MODEL        # giải phóng GPU trước model tiếp theo
uv run vbench bundle create <RUN_ID>              # đính kèm cả log server và prepare_report
```

Lưu ý:

- `model prepare` kết thúc bằng bước import thử `vllm`, `vllm_omni`, `cv2`, `soundfile` trong venv. Nếu thiếu thư viện hệ thống, lệnh báo lỗi ngay kèm cách khắc phục, thay vì để `model serve` treo 10 phút rồi timeout.
- `model prepare` có thể chạy rất lâu (tải vài GB thư viện và hàng chục GB weights). Nên chạy trong `tmux` hoặc `screen` để không bị ngắt khi mất kết nối.
- Chỉ một model được serve tại một thời điểm (khoá GPU `$VBENCH_HOME/.gpu.lock`). Luôn `model stop` trước khi serve model khác.
- `HF_HOME` phải nằm trong `VBENCH_HOME` (như mẫu `.env` ở §4) để việc kiểm tra giới hạn đĩa tính cả weights.
- Nếu `model serve` báo lỗi, server sẽ tự dừng; gửi file `$VBENCH_HOME/logs/runtime_<id>.log` về cho Claude (khi đó chưa có RUN_ID nên không có bundle).
- Mỗi run tạo `capability_report.json`: ghi lại những gì **quan sát được** (streaming audio, kênh text, tool call, huỷ phản hồi). Báo cáo này chỉ để review, không tự động sửa cấu hình model.
- `gpt-realtime` chỉ chạy được khi đã điền `realtime.api_model` trong `configs/models/gpt-realtime.yaml` và có `OPENAI_API_KEY`; nếu thiếu, lệnh `run` từ chối chạy.

`scripts/run_all.sh` (Phase 10) sẽ bọc vòng lặp này cho mọi model.

### 6.4 Tiếp tục sau khi lỗi

```bash
uv run vbench run --resume $RUN_ID
```

Mẫu đã hoàn thành được bỏ qua; mẫu lỗi được thử lại một lần, rồi được ghi với `status=error`. (Có từ Phase 2.)

### 6.5 Độc quyền GPU

Mỗi thời điểm chỉ một workload GPU: `vbench` giữ một file khoá (`$VBENCH_HOME/.gpu.lock`) khi model server hoặc evaluator GPU đang chạy. Job GPU khác chạy trên server trong lúc benchmark sẽ làm kết quả độ trễ và hạ tầng mất giá trị — NVML sampler ghi lại process lạ và run bị gắn cờ `gpu_contended=true`.

### 6.6 Không sửa code trên server

Không sửa file trong thư mục release trên server. Mọi thay đổi code phải đi qua Claude → GitHub → release mới. Nếu sửa tại chỗ, `release verify` báo `modified file`, run bị đánh dấu `dirty` và không được xếp hạng. Cấu hình riêng của server (đường dẫn, key) chỉ đặt trong `$VBENCH_HOME/.env`.

---

## 7. Gửi kết quả về

**Trên GPU server**, các file cần gửi nằm trong `$VBENCH_HOME/bundles/`:

| Nội dung | File |
|---|---|
| Bundle kết quả | `<run_id>.tar.zst` + `<run_id>.SHA256SUMS` (luôn gửi cả hai) |
| Env report (khi được yêu cầu) | `env_report.json` |
| Khi lỗi: log | có sẵn trong bundle (`logs/run.jsonl`); nếu chính bước đóng gói bị lỗi, gửi thư mục `$VBENCH_HOME/artifacts/raw/<run_id>/logs/` và output terminal |

**Chuyển về:** copy các file trên từ server về máy công ty (cùng cách copy lên), rồi gửi cho Claude bằng một trong hai cách:

- đặt vào thư mục `incoming/` trong repo trên máy dev (thư mục này nằm trong `.gitignore`, không bị commit), rồi báo đường dẫn; hoặc
- đính kèm trực tiếp vào cuộc hội thoại.

Không giải nén hay sửa file bundle trước khi gửi. Claude kiểm tra checksum trước (`vbench bundle verify`) và chỉ phân tích dữ liệu đã được kiểm tra. Thiếu file `.SHA256SUMS` thì bundle không được chấp nhận.

Cũng chấp nhận: báo cáo CSV/JSON/parquet/HTML, ảnh chụp màn hình, output profiler, output terminal.

---

## 8. Xử lý sự cố (bổ sung dần theo từng phase)

| Triệu chứng | Bước đầu tiên |
|---|---|
| `sha256sum -c` báo `FAILED` | File `.zip` bị hỏng khi tải hoặc copy. Tải lại cả hai file từ GitHub Releases trên máy công ty và copy lại. |
| `sha256sum: ... no properly formatted checksum lines found` | File `.sha256` bị đổi định dạng khi copy (ví dụ thêm CRLF). Chạy `sed -i 's/\r$//' $REL.zip.sha256` rồi kiểm tra lại. |
| `release verify` báo `modified file` / `missing file` / `unexpected file` | Code trên server khác với release. Xoá thư mục `$VBENCH_HOME/releases/$REL`, giải nén lại từ file `.zip` (không copy thư mục từ Windows). |
| `env check` báo `fail` ở `source_integrity` | Đang chạy từ thư mục không phải release đã giải nén, hoặc code đã bị sửa. Làm lại §6.1. |
| `env check` báo `fail` ở `home_outside_repo` | Chưa nạp `.env` hoặc `VBENCH_HOME` trỏ vào thư mục code. Kiểm tra §4 và chạy lại `set -a; source $VBENCH_HOME/.env; set +a`. |
| `env check` báo `fail` ở `env:OPENAI_API_KEY` | Chưa nạp file `.env` trong terminal hiện tại: chạy lại `set -a; source $VBENCH_HOME/.env; set +a`. |
| `uv sync` lỗi mạng | Kiểm tra kết nối PyPI (`curl -I https://pypi.org/simple/`); gửi output về. |
| `vbench: command not found` | Dùng `uv run vbench ...` (không gọi `vbench` trực tiếp) và chạy `uv sync` trong thư mục release trước. |
| Log server báo `ImportError: There is no such class as cosyvoice2....` ở stage 2 | File `flow.yaml` trong checkpoint gọi lớp ở module cấp cao nhất `cosyvoice2`, do gói `step-audio2` cung cấp. Từ v0.2.5, `model prepare` cài cả gói này. Chạy lại `uv run vbench model prepare --model minicpm-o-4_5`. |
| Log server báo `ModuleNotFoundError: No module named 's3tokenizer'` (hoặc `stepaudio2`) ở stage 2 | Stage Token2Wav của MiniCPM-o cần gói `stepaudio2-minicpmo`, không có trong phụ thuộc mặc định của vLLM-Omni. Từ v0.2.4, `model prepare` cài gói này (kèm ràng buộc phiên bản) và kiểm tra import. Chạy lại `uv run vbench model prepare --model minicpm-o-4_5`. |
| `model prepare`/`model check` báo `ModuleNotFoundError: No module named 'cv2'` | Hai bản OpenCV (GUI và headless) từng cùng được cài và dùng chung thư mục `cv2`; gỡ một bản làm mất file của bản kia. Từ v0.2.3, `model prepare` gỡ cả hai rồi cài lại bản headless. Chạy lại `uv run vbench model prepare --model <id>`. |
| `ImportError: libGL.so.1` trong log server, hoặc `model prepare`/`model check` báo "cannot import 'cv2'" | Container thiếu thư viện đồ hoạ mà OpenCV cần. Cài bằng quyền root: `apt-get update && apt-get install -y libgl1 libglib2.0-0 libsndfile1`, rồi chạy lại `uv run vbench model prepare --model <id>`. |
| Log server có `Orchestrator did not become ready within 600s` | Đây là hệ quả, không phải nguyên nhân: một tiến trình con đã chết trước đó. Tìm dòng lỗi đầu tiên (thường là `ImportError` hoặc `CUDA out of memory`) ở phía trên trong cùng file log. |
| `model serve` báo "server exited during startup" hoặc "not healthy" | Gửi `$VBENCH_HOME/logs/runtime_<model>.log` về. Với Qwen3-Omni trên 1 GPU, lỗi hết bộ nhớ là rủi ro đã biết (cấu hình `configs/deploy/qwen3_omni_1gpu.yaml` chưa được kiểm chứng). |
| `model prepare` báo "disk budget exceeded" | Đĩa đã dùng gần 200 GB. Xoá model không cần nữa: `uv run vbench model evict --model <id>`. |
| `model serve` báo "GPU is locked" | Một model khác đang chạy: `uv run vbench model stop --model <id đó>`. |
| `run` chạy xong nhưng mọi mẫu `error`, `capability_report.json` toàn `not_tested` | Xem `events.jsonl` của một mẫu để đọc event lỗi. Từ v0.3.0 các tác vụ theo lượt đi qua `/v1/chat/completions`; nếu server trả lỗi HTTP thì nội dung lỗi nằm trong event `error`. |
| `run` báo "server is not healthy" | Chưa chạy `model serve`, hoặc server đã dừng: kiểm tra `uv run vbench model status --model <id>`. |
| CUDA OOM khi nạp model | Kiểm tra không có process GPU nào khác (`nvidia-smi`); gửi `env_report.json` và log runtime. |
| vLLM từ chối kiến trúc model | Có thể xảy ra với một số model; khi đó `runtime.engine` trong YAML của model phải là `official`. Gửi log để Claude sửa runtime. |
| Lỗi kết nối OpenAI | Kiểm tra `OPENAI_API_KEY` và HTTPS ra ngoài; `vbench env check` hiển thị khả năng kết nối. |
| `bundle create` báo "secret-like strings found" | Có file trong run chứa chuỗi giống API key. Không gửi file đó; gửi danh sách file bị báo để Claude kiểm tra. |
