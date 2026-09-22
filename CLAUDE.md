# Dự án AI Voice Benchmark
## Sứ mệnh
Xây dựng một framework benchmark cấp production để đánh giá các mô hình AI Speech-to-Speech bản địa (Speech-to-Speech / Audio-to-Audio) cho việc triển khai Tổng đài AI tiếng Nhật.
Repository này tập trung vào benchmark, đánh giá, báo cáo và tạo bảng xếp hạng.
Mục tiêu là xác định model phù hợp nhất cho vận hành tổng đài tiếng Nhật, chứ không phải tìm model nghiên cứu mạnh nhất.
---
# Các model mục tiêu
## Ứng viên OSS
- Qwen3-Omni-30B-A3B-FP8
- MiniCPM-o 4.5
- StepAudio 2.5 Realtime
- GLM-4-Voice-9B
- LLaMA-Omni 2
- Baichuan-Omni-1.5
## Baseline thương mại
- GPT-Realtime 2.1
---
# Hạ tầng
## Môi trường A
Máy dev
Khả năng:
- Claude Code
- Git
- GitHub
Giới hạn:
- Không chạy được benchmark GPU
- Không host được model lớn
- Không chạy được benchmark độ trễ
Trách nhiệm:
- Sinh source code
- Thiết kế kiến trúc
- Sinh framework test
- Phân tích kết quả benchmark do người dùng cung cấp sau này
---
## Môi trường B
Máy trung chuyển (máy công ty)
Khả năng:
- Truy cập GitHub
- Upload source code lên GPU server
Trách nhiệm:
- Tải repository về
- Upload repository lên benchmark server
---
## Môi trường C
GPU Benchmark Server
Cấu hình:
- Linux
- 1x H100 80GB
- SSD 200GB
- Không có Docker (model chạy bằng vLLM trong `uv` venv riêng cho từng model)
- Tải model trực tiếp từ Hugging Face được
- Gọi OpenAI API được
Trách nhiệm:
- Chạy workload benchmark
- Thu thập metric thô
- Tạo artifact benchmark
Giới hạn:
- Claude không điều khiển trực tiếp
---
# Yêu cầu có con người tham gia (Human-In-The-Loop)
QUAN TRỌNG:
Claude không bao giờ tự chạy benchmark GPU thật.
Claude không bao giờ bịa kết quả benchmark.
Claude không bao giờ ước lượng kết quả benchmark khi không có bằng chứng.
Khi cần chạy benchmark:
1. Sinh code.
2. Merge vào `main` và push lên GitHub (người dùng chỉ lấy code bằng cách tải từ GitHub về máy công ty).
3. Dừng lại.
4. Yêu cầu người dùng chạy benchmark.
5. Chờ đầu ra benchmark.
6. Phân tích đầu ra.
7. Cải thiện framework.
Mỗi lần cần người dùng kiểm thử thực tế trên máy công ty hoặc GPU server, Claude phải đưa source code và mọi thành phần cần review lên GitHub trước, và ghi rõ commit cần dùng.
Đầu vào benchmark được chấp nhận:
- csv
- json
- parquet
- html
- ảnh chụp màn hình
- log benchmark
- output profiler
---
# Mục tiêu benchmark
Đánh giá các model cho việc triển khai Tổng đài AI tiếng Nhật.
Các chiều chính:
1. Năng lực tiếng Nhật
2. Hiệu năng hội thoại thời gian thực
3. Độ tin cậy khi gọi công cụ (tool calling)
4. Chất lượng giọng nói
5. Chi phí hạ tầng
---
# Các lớp benchmark
## Lớp 1
Benchmark năng lực tiếng Nhật
Đánh giá:
- ASR
- CER
- WER
- Phân loại intent
- Trích xuất slot
- Hiểu tiếng Nhật
- Tuân thủ keigo
- Nhớ ngữ cảnh dài
---
## Lớp 2
Benchmark giọng nói thời gian thực
Đánh giá:
- Time To First Audio (TTFA)
- Độ trễ ngắt lời (Interrupt Latency)
- Barge-In
- Luân phiên lượt nói (Turn Taking)
- Tương tác song công (Duplex Interaction)
- Độ ổn định hội thoại dài
---
## Lớp 3
Benchmark gọi công cụ
Đánh giá:
- Đặt lịch (Booking)
- FAQ
- Tra cứu CRM
- Chuyển cuộc gọi
- Structured Output
Metric:
- Tool Success Rate
- JSON Accuracy
- Hallucinated Tool Rate
- Task Completion Rate
---
## Lớp 4
Benchmark chất lượng giọng nói
Đánh giá:
- MOS
- Độ tự nhiên
- Chất lượng ngữ điệu (accent)
- Phát âm
- Biểu cảm
- Tính nhất quán
Có thể cần người chấm.
---
## Lớp 5
Benchmark hạ tầng
Đánh giá:
- Mức dùng VRAM
- Throughput
- Số cuộc gọi đồng thời
- GPU Utilization
- Chi phí mỗi cuộc gọi
- Chi phí mỗi phút
- TCO
---
# Quy tắc repository
Luôn ưu tiên:
1. Khả năng tái lập
2. Tự động hoá
3. Khả năng truy vết
4. Khả năng đo lường
Mọi lần chạy benchmark phải tái lập được.
Mọi metric phải giải thích được:
- đo như thế nào
- dữ liệu nguồn
- thời điểm
- phiên bản model
---
# Nguyên tắc phát triển
Trước khi triển khai:
1. Thiết kế kiến trúc
2. Lập kế hoạch triển khai
3. Viết đặc tả benchmark
4. Triển khai từng bước
Không viết lượng code lớn khi kiến trúc chưa được duyệt.
---
# Công nghệ
Ưu tiên:
- Python 3.11+
- Typer
- Pydantic
- FastAPI
- Pytest
- Pandas
- Polars
- Plotly
- DuckDB
Tuỳ chọn:
- Streamlit
- Gradio
Tránh:
- Giải pháp chỉ chạy trên Jupyter
- Hardcode thông tin xác thực
- Hardcode đường dẫn
---
# Cấu trúc thư mục
artifacts/
    raw/
    processed/
    reports/
    dashboards/
datasets/
benchmark/
tests/
docs/
configs/
scripts/
---
# Đầu ra bắt buộc
Mọi lần chạy benchmark phải tạo ra:
- CSV
- JSON
- Báo cáo HTML
Artifact bắt buộc:
leaderboard.csv
leaderboard.json
leaderboard.html
benchmark_summary.json
---
# Triết lý xếp hạng
Không bao giờ xếp hạng model chỉ bằng một metric.
Business Score:
35% Chất lượng tiếng Nhật
25% Hoàn thành tác vụ
15% Barge-In
15% Chi phí
10% Chất lượng giọng nói
Thứ hạng cuối cùng phải được suy ra hoàn toàn từ kết quả benchmark đo được.
---
# Yêu cầu tài liệu
Duy trì:
docs/
bao gồm:
- ARCHITECTURE.md
- IMPLEMENTATION_ROADMAP.md
- DATASET_SPEC.md
- METRIC_DEFINITIONS.md
- DEPLOYMENT_GUIDE.md
Ngôn ngữ tài liệu: tiếng Việt. Giữ nguyên tiếng Anh cho lệnh, tên file, tên metric, tên trường cấu hình và code. Source code (kể cả comment, docstring, log) viết bằng tiếng Anh.
---
# Nguyên tắc dataset
Không bao giờ bịa nhãn benchmark.
Dataset synthetic phải được đánh dấu rõ ràng.
Dataset có người review phải lưu riêng.
Dataset benchmark tiếng Nhật phải có phiên bản.
---
# Chuẩn code
Dùng:
- type hint
- model Pydantic
- unit test
Mục tiêu:
>= 80% độ phủ test
Logging:
chỉ dùng structured logging
Không dùng print để debug trong code production.
---
# Quy tắc chạy benchmark
Nếu benchmark cần tài nguyên GPU:
Sinh code.
Sinh lệnh.
Sinh hướng dẫn triển khai.
Merge vào `main` và push lên GitHub.
DỪNG LẠI.
Chờ kết quả benchmark từ người dùng.
