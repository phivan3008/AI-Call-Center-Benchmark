# Hướng dẫn triển khai — GPU Benchmark Server

| Trường | Giá trị |
|---|---|
| Trạng thái | **ĐÃ DUYỆT** (2026-09-22) |
| Phiên bản | 0.1.0 |
| Ngày | 2026-09-21 |
| Đối tượng | Người vận hành chạy benchmark trên Env C (Linux, 1x H100 80GB, SSD 200GB) |

> Bản tiếng Việt. Các lệnh giữ nguyên, copy chạy trực tiếp được. Các lệnh ở đây là CLI **dự kiến** (ARCHITECTURE §13); mỗi mục ghi rõ phase bắt đầu có lệnh đó. Trước phase đó, chỉ các lệnh của những phase đã bàn giao mới chạy được.

---

## 1. Thông tin server (từ review, 2026-09-21)

| Mục | Giá trị |
|---|---|
| Docker | **Không có.** Model chạy như process cục bộ trong `uv` venv riêng cho từng model. |
| Engine phục vụ model | Mặc định vLLM; dự phòng bằng code inference chính thức (có tài liệu) cho từng model. |
| Hugging Face | Truy cập trực tiếp được từ server. |
| OpenAI API | Truy cập trực tiếp được từ server (baseline GPT-Realtime chạy ở đây). |

---

## 2. Yêu cầu cài đặt (một lần)

| Yêu cầu | Lệnh kiểm tra | Ghi chú |
|---|---|---|
| NVIDIA driver + CUDA tương thích với vLLM/torch đã chốt | `nvidia-smi` | Phiên bản tối thiểu chính xác được chốt ở Phase 1 từ output của `vbench env check`. |
| `uv` | `uv --version` | Tự cài Python 3.11+ cho từng venv; không cần sửa Python của hệ thống. Cài: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `git` | `git --version` | |
| `ffmpeg` | `ffmpeg -version` | Chuyển đổi audio, mô phỏng kênh điện thoại. |
| `libsndfile` | `ldconfig -p \| grep sndfile` | Đọc/ghi FLAC. |
| `zstd` | `zstd --version` | Nén bundle kết quả (tuỳ chọn; bundle do Python tạo, `zstd` chỉ để giải nén thủ công). |
| Dung lượng đĩa trống | `df -h $VBENCH_HOME` | Xem §5 về quản lý dung lượng đĩa. |

Bản thân framework không cần quyền root, ngoài việc cài các gói hệ thống ở trên.

---

## 3. Cấu trúc thư mục trên server

Chọn một thư mục gốc trên SSD benchmark và đặt vào biến `VBENCH_HOME`. **Mọi** đường dẫn đều suy ra từ thư mục này; không có đường dẫn nào bị hardcode.

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

## 4. Biến môi trường

Tạo file `$VBENCH_HOME/.env` (không bao giờ commit file này; đặt quyền `chmod 600`):

```bash
VBENCH_HOME=/path/to/benchmark_ssd/vbench
HF_HOME=${VBENCH_HOME}/hf_cache
HF_TOKEN=...                 # only if a gated model requires it
OPENAI_API_KEY=...           # GPT-Realtime baseline and LLM judge (if an OpenAI model is chosen as judge)
VBENCH_LOG_LEVEL=INFO
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

### 6.1 Cập nhật code (mỗi checkpoint)

Trước mỗi checkpoint, Claude đã merge code vào nhánh `main` trên GitHub và ghi rõ commit cần dùng.

Trên Env B (máy công ty): tải repo từ GitHub (`git clone` lần đầu, `git pull` các lần sau, nhánh `main`), rồi upload thư mục repo lên `$VBENCH_HOME/repo/` trên server. Trên server:

```bash
set -a; source /path/to/benchmark_ssd/vbench/.env; set +a   # nạp VBENCH_HOME, OPENAI_API_KEY, ...
cd $VBENCH_HOME/repo
git log -1 --oneline          # confirm the commit named in the checkpoint instructions
uv sync                       # harness environment (not model venvs)
```

Nếu repo được upload dưới dạng thư mục không có `.git`, lệnh `git log` sẽ lỗi; khi đó hãy báo lại commit đã tải về. Lưu ý: run tạo ra từ thư mục không có `.git` sẽ bị ghi `git_commit=unknown`, `git_dirty=true` và không được xếp hạng (chỉ chấp nhận với `--allow-dirty`). Vì vậy nên upload cả thư mục `.git`.

### 6.2 Preflight và mock round trip (có từ Phase 1)

```bash
uv run vbench env check --output $VBENCH_HOME/bundles/env_report.json
uv run vbench run --model mock --profile mock_smoke      # CPU-only pipeline check, prints RUN_ID
uv run vbench bundle create <RUN_ID>
```

`env check` thoát với mã khác 0 nếu một mục bắt buộc bị fail. Khi đó hãy dừng lại và gửi kết quả về. Ý nghĩa các trạng thái: `pass` = đạt, `warn` = cảnh báo (không bắt buộc), `fail` = không đạt mục bắt buộc. Run mock có `is_mock=true` và không bao giờ được xếp hạng.

Với `--role server` (mặc định), các mục bắt buộc gồm: Python ≥ 3.11, GPU + NVIDIA driver, quyền ghi vào `VBENCH_HOME`, `uv`, `git`, `nvidia-smi`, kết nối đến Hugging Face và OpenAI, và có `OPENAI_API_KEY`. `ffmpeg`, `zstd`, `libsndfile`, `HF_TOKEN` chỉ là cảnh báo.

### 6.3 Chạy cho từng model (có từ Phase 2)

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

`scripts/run_all.sh` (Phase 10) bọc vòng lặp này cho mọi model và có thể tiếp tục khi bị gián đoạn.

### 6.4 Tiếp tục sau khi lỗi

```bash
uv run vbench run --resume $RUN_ID
```

Mẫu đã hoàn thành được bỏ qua; mẫu lỗi được thử lại một lần, rồi được ghi với `status=error`. (Có từ Phase 2.)

### 6.5 Độc quyền GPU

Mỗi thời điểm chỉ một workload GPU: `vbench` giữ một file khoá (`$VBENCH_HOME/.gpu.lock`) khi model server hoặc evaluator GPU đang chạy. Job GPU khác chạy trên server trong lúc benchmark sẽ làm kết quả độ trễ và hạ tầng mất giá trị — NVML sampler ghi lại process lạ và run bị gắn cờ `gpu_contended=true`.

---

## 7. Gửi kết quả về

Gửi cho Claude (qua máy trung chuyển hoặc đính kèm):

| Nội dung | Vị trí |
|---|---|
| Bundle kết quả | `$VBENCH_HOME/bundles/<run_id>.tar.zst` + `<run_id>.SHA256SUMS` |
| Env report (khi được yêu cầu) | `$VBENCH_HOME/bundles/env_report.json` |
| Khi lỗi: log | có sẵn trong bundle (`logs/run.jsonl`); nếu chính bước đóng gói bị lỗi, gửi `artifacts/raw/<run_id>/logs/` và output terminal |

Cũng chấp nhận: báo cáo CSV/JSON/parquet/HTML, ảnh chụp màn hình, output profiler. Claude kiểm tra checksum trước (`vbench bundle verify`) và chỉ phân tích dữ liệu đã được kiểm tra.

Luôn gửi cả hai file `.tar.zst` và `.SHA256SUMS` của cùng một run; thiếu file `.SHA256SUMS` thì bundle không được chấp nhận.

---

## 8. Xử lý sự cố (bổ sung dần theo từng phase)

| Triệu chứng | Bước đầu tiên |
|---|---|
| Health check của `model serve` bị timeout | Xem `artifacts/raw/<run_id>/logs/runtime_<model>.log`; gửi file đó về. |
| CUDA OOM khi nạp model | Kiểm tra không có process GPU nào khác (`nvidia-smi`); gửi `env_report.json` và log runtime. |
| vLLM từ chối kiến trúc model | Có thể xảy ra với một số model; khi đó `runtime.engine` trong YAML của model phải là `official`. Gửi log để sửa runtime. |
| Lỗi kết nối OpenAI | Kiểm tra `OPENAI_API_KEY` đã có và HTTPS ra ngoài hoạt động; `vbench env check` hiển thị khả năng kết nối. |
| `env check` báo `fail` ở `env:OPENAI_API_KEY` | Chưa nạp file `.env`: chạy lại `set -a; source $VBENCH_HOME/.env; set +a` trong cùng terminal. |
| `vbench: command not found` | Dùng `uv run vbench ...` (không gọi `vbench` trực tiếp) và chạy `uv sync` trong thư mục repo trước. |
| `bundle create` báo "secret-like strings found" | Có file trong run chứa chuỗi giống API key. Không gửi file đó; gửi danh sách file bị báo để Claude kiểm tra. |
