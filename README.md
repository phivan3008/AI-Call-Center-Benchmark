# Benchmark Tổng đài AI tiếng Nhật

## Tổng quan

Repository này chứa một framework benchmark toàn diện để đánh giá các mô hình AI Speech-to-Speech bản địa (native) cho việc triển khai Tổng đài AI tiếng Nhật.

Benchmark tập trung vào yêu cầu kinh doanh thực tế thay vì hiệu năng thuần học thuật.

Mục tiêu chính:
Xác định model tốt nhất để triển khai production cho các tình huống chăm sóc khách hàng và đặt lịch hẹn bằng tiếng Nhật.

---

# Tài liệu

| Tài liệu | Nội dung |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Kiến trúc hệ thống, pipeline, thiết kế từng lớp, chấm điểm |
| [docs/IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md) | Lộ trình theo phase và các checkpoint cần chạy trên GPU server |
| [docs/METRIC_DEFINITIONS.md](docs/METRIC_DEFINITIONS.md) | Định nghĩa và công thức của từng metric |
| [docs/DATASET_SPEC.md](docs/DATASET_SPEC.md) | Đặc tả dataset, manifest, kịch bản |
| [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) | Hướng dẫn vận hành trên GPU server (từng bước) |

---

# Bắt đầu nhanh

Yêu cầu: Python 3.11+ và [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
uv run pytest                                          # chạy test (chỉ CPU)
uv run vbench env check --role dev                     # kiểm tra môi trường trên máy dev
uv run vbench run --model mock --profile mock_smoke    # chạy thử pipeline bằng mock adapter
uv run vbench bundle create <RUN_ID>                   # đóng gói kết quả
uv run vbench bundle verify bundles/<RUN_ID>.tar.zst   # kiểm tra bundle
```

Trên GPU server, làm theo [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md).

Trạng thái hiện tại: Phase 1 (khung cốt lõi, mock adapter) đã xong; đang chờ checkpoint 1 trên GPU server. Các model thật được thêm từ Phase 2.

---

# Các model ứng viên

## Model mã nguồn mở
- Qwen3-Omni-30B-A3B-FP8
- MiniCPM-o 4.5
- StepAudio 2.5 Realtime
- GLM-4-Voice-9B
- LLaMA-Omni 2
- Baichuan-Omni-1.5

## Baseline thương mại
- GPT-Realtime 2.1

---

# Các lớp benchmark

## Lớp 1: Benchmark năng lực tiếng Nhật
Đo:
- CER
- WER
- Hiểu tiếng Nhật
- Phân loại intent
- Trích xuất slot
- Tuân thủ keigo (kính ngữ)
- Nhớ ngữ cảnh dài

Sản phẩm:
- japanese_score.json
- japanese_report.html

---

## Lớp 2: Benchmark giọng nói thời gian thực
Đo:
- TTFA P50
- TTFA P95
- TTFA P99
- Độ trễ ngắt lời (interrupt latency)
- Hiệu năng barge-in
- Chất lượng luân phiên lượt nói
- Độ ổn định cuộc gọi dài

Sản phẩm:
- latency_report.json
- realtime_report.html

---

## Lớp 3: Benchmark gọi công cụ (tool calling)
Đo:
- Tỷ lệ đặt lịch thành công
- Độ chính xác FAQ
- Tỷ lệ tra cứu CRM thành công
- Độ chính xác chuyển cuộc gọi
- Khớp chính xác JSON
- Tỷ lệ hoàn thành tác vụ

Sản phẩm:
- toolcalling_report.json
- task_completion_report.html

---

## Lớp 4: Benchmark chất lượng giọng nói
Đo:
- MOS
- Độ tự nhiên
- Phát âm
- Chất lượng ngữ điệu tiếng Nhật
- Biểu cảm

Sản phẩm:
- voice_quality_report.json
- mos_summary.csv

---

## Lớp 5: Benchmark hạ tầng & TCO
Đo:
- GPU utilization
- Mức dùng VRAM
- Throughput
- Số cuộc gọi đồng thời
- Chi phí mỗi cuộc gọi
- Chi phí mỗi phút
- Tổng chi phí sở hữu (TCO)

Sản phẩm:
- infra_report.json
- tco_report.html

---

# Phương pháp chấm điểm

Business Score cuối cùng:

```text
35% Japanese Quality   (Chất lượng tiếng Nhật)
25% Task Completion    (Hoàn thành tác vụ)
15% Barge-In
15% Cost               (Chi phí)
10% Voice Quality      (Chất lượng giọng nói)
```

Xếp hạng cuối cùng chỉ được tạo từ metric đo thực tế.
Không cho phép điểm ước lượng.

---

# Cấu trúc repository

```text
.
├── benchmark/
│   ├── core/            # config, schemas, provenance, artifacts, logging
│   ├── adapters/        # giao diện model + mock adapter
│   ├── scoring/         # điều kiện xếp hạng, chấm điểm
│   ├── runner.py        # thu thập dữ liệu (turn mode)
│   ├── bundle.py        # đóng gói / kiểm tra kết quả
│   ├── envcheck.py      # kiểm tra môi trường
│   └── cli.py           # lệnh vbench
│   (các lớp layer1..layer5 được thêm từ Phase 3)
│
├── datasets/
│
├── configs/
│
├── scripts/
│
├── artifacts/
│   ├── raw/
│   ├── processed/
│   ├── reports/
│   └── dashboards/
│
├── docs/
│
├── tests/
│
├── CLAUDE.md
│
└── README.md
```

---

# Quy trình thực thi

## Bước 1
Claude Code sinh source code.
## Bước 2
Source code được commit, merge vào `main` và push lên GitHub.
## Bước 3
Repository được tải về máy trung chuyển (máy công ty).
## Bước 4
Repository được upload lên GPU server.
## Bước 5
Benchmark chạy trên GPU server.
## Bước 6
Thu thập artifact.
## Bước 7
Artifact được gửi lại cho Claude.
## Bước 8
Claude phân tích kết quả và cải thiện framework.

---

# Mô hình môi trường

Máy dev
```text
Claude Code
    ↓
GitHub
```
Máy trung chuyển
```text
GitHub
    ↓
GPU Server
```
GPU Benchmark Server
```text
Linux
1x H100 80GB
200GB SSD
```

---

# Nguyên tắc cơ bản

1. Không bao giờ bịa kết quả benchmark.
2. Không bao giờ giả định hiệu năng model.
3. Mọi metric phải tái lập được.
4. Mọi lần chạy benchmark phải tạo ra artifact.
5. Đầu ra benchmark do con người cung cấp là nguồn sự thật.
6. Xếp hạng model chỉ đến từ kết quả đo được.

---

# Sản phẩm bàn giao

Đầu ra cuối cùng bắt buộc:
- leaderboard.csv
- leaderboard.json
- leaderboard.html
- benchmark_summary.json

Báo cáo bắt buộc:
- Báo cáo Lớp 1 Tiếng Nhật
- Báo cáo Lớp 2 Thời gian thực
- Báo cáo Lớp 3 Tool Calling
- Báo cáo Lớp 4 Chất lượng giọng nói
- Báo cáo Lớp 5 Hạ tầng

---

# Tiêu chí thành công

Framework benchmark phải cho phép so sánh tái lập được giữa mọi model mục tiêu và tạo ra thứ hạng cuối cùng hướng đến production, phù hợp cho việc triển khai Tổng đài AI.
