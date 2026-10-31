# DataMate — Qwen3.5 trên Kaggle

Pipeline inference cho dữ liệu kinh doanh nhiều cấu trúc. Qwen3.5 đọc README/rulebook, lựa chọn tool, tính bằng SQL và xuất JSON. Giữ nguyên Excel/CSV/SQLite; bảng trung gian được tạo trong RAM theo từng câu hỏi. Không fine-tune, không cập nhật trọng số, không tính training loss.

## Chạy trên Kaggle

Bật Internet, chọn GPU **T4 ×2**, gắn dataset có cấu trúc:

```text
<dataset>/train/
  questions.jsonl
  answers.jsonl
  rulebook.md
  calendar.csv
  data/biz_101/...
```

Dataset không nằm trong Git (`data/` đã được ignore). Upload/gắn riêng vào Kaggle. Không đưa file đáp án vào thư mục con của doanh nghiệp.

Mở [notebooks/kaggle_inference.ipynb](notebooks/kaggle_inference.ipynb), sửa URL repo và DATA_DIR, rồi chạy lần lượt các cell. Hoặc chạy trong notebook:

```python
!git clone <REPO_URL> /kaggle/working/zAIC
%cd /kaggle/working/zAIC
%pip install -r requirements.txt
```

Nếu pip thay các thư viện đang được import, restart kernel trước khi nạp model. Giữ PyTorch CUDA có sẵn của Kaggle nếu đã đáp ứng requirements. Không cần API key cho checkpoint Qwen công khai. Lần đầu cần Internet để tải trọng số.

Kiểm tra chọn mẫu, chưa tải model:

```bash
python inference.py --data-dir /kaggle/input/<dataset>/train --limit 5 --dry-run
```

Chạy 5 câu trước để kiểm tra model và VRAM, rồi dùng đường dẫn output mới để chạy 20 câu:

```bash
python inference.py \
  --data-dir /kaggle/input/<dataset>/train \
  --model Qwen/Qwen3.5-9B --quantization 4bit \
  --device-map split --limit 5 --sampling stratified \
  --output /kaggle/working/smoke/predictions.jsonl
```

Mặc định: `--device-map split` chia các decoder layer nguyên khối qua tất cả GPU đang thấy. Với Qwen3.5-9B và hai T4, 16 layer đầu đặt trên GPU 0 và 16 layer sau trên GPU 1; embedding ở GPU 0, norm và output head ở GPU cuối (output head ở GPU 0 nếu buộc dùng chung trọng số embedding). Chế độ này cần ít nhất hai GPU. Đây là model parallelism, không phải hai worker chạy hai câu đồng thời hoặc tensor parallelism; không đảm bảo nhanh gấp đôi.

Các cấu hình còn lại: FP16 compute, NF4 4-bit, context 16.384 token gồm input và output, 2.048 token đầu ra mỗi lượt, 16 lượt/câu. Chỉ tính logits của token cuối (`logits_to_keep=1`) để tránh tạo logits cho toàn bộ prompt. Code in device map và bộ nhớ từng GPU sau khi nạp; các layer được chia đều nhưng tổng VRAM có thể lệch vì embedding, head, cache và tensor tạm.

`--device-map auto`, `balanced`, `balanced_low_0`, `sequential` vẫn được hỗ trợ. `--gpu-memory-gib 13` là ngân sách đặt trọng số cho các chế độ tự ánh xạ này, không phải hard limit tổng VRAM. Nó không áp dụng cho map `split` tường minh. `--device-map cuda:0` dùng một card. `--quantization none` dùng FP16 và cần nhiều VRAM hơn. Hybrid attention có thể dùng đường PyTorch chậm nếu không có kernel tăng tốc phù hợp; không cài FlashAttention tùy tiện trên T4.

Nếu dùng uv trên Kaggle, cập nhật repo rồi chạy trong cell:

```python
%cd /kaggle/working/zAIC
!git pull --ff-only
!uv run python inference.py --data-dir "<đường dẫn train thực tế>" --device-map split --quantization 4bit --limit 5 --output /kaggle/working/split/predictions.jsonl
```

Dừng tiến trình inference cũ trước khi chạy; nếu model còn trong kernel notebook, restart session/kernel để giải phóng GPU. Khi đổi device map hãy dùng output mới thay vì resume đợt chạy cũ. Split giúp phân bổ trọng số nhưng vẫn có thể OOM do context dài hoặc tiến trình khác chiếm VRAM; chưa benchmark bản sửa trên T4 thật.


Nếu vượt context, chương trình ghi lỗi rồi abstain, không âm thầm cắt mất rulebook. Hãy xem trace, tăng `--context-tokens` nếu VRAM cho phép, hoặc giảm độ dài tool output/chuỗi xử lý. `--thinking` bật suy luận nội bộ, có thể cần tăng `--max-new-tokens`; mặc định tắt để đo baseline nhanh hơn. Generation mặc định greedy để dễ tái lập, không phải sampling preset tối ưu của nhà phát hành.

### Chấm kết quả

```bash
python evaluate.py \
  --predictions /kaggle/working/smoke/predictions.jsonl \
  --answers /kaggle/input/<dataset>/train/answers.jsonl \
  --questions /kaggle/input/<dataset>/train/questions.jsonl
```

Mặc định chấm đúng các ID trong `.run.json` của đợt inference; câu đã chọn nhưng chưa chạy được tính abstain. Không lấy 5 dự đoán chia cho toàn bộ 3.000 câu. `--scope predictions` chấm các ID hiện có nếu dùng dự đoán bên ngoài, `--scope all` chấm toàn bộ nhãn.

Báo cáo: accuracy trên toàn mẫu, coverage, accuracy của câu có trả lời, điểm **đúng +1 / sai −1 / abstain 0**, phân nhóm theo type và doanh nghiệp. Đây là bộ chấm cục bộ dựa trên rulebook và alias phát hành, không phải scorer chính thức. Biến thể tên hợp lệ có trong nguồn nhưng chưa có trong alias có thể cần review. Reason của `unanswerable` phải có nội dung nhưng không được LLM chấm.

### Chạy tiếp / chạy toàn bộ

Dùng lại chính lệnh inference ban đầu và thêm `--resume`. Cấu hình và danh sách mẫu phải giống nhau. Chỉ ID đã lưu dự đoán mới được bỏ qua. Lỗi thực thi được lưu thành abstain; để thử lại các câu lỗi, dùng một đợt chạy mới. Khi đổi limit/model/cấu hình hãy dùng output mới.

```bash
python inference.py \
  --data-dir /kaggle/input/<dataset>/train --limit 0 \
  --output /kaggle/working/full/predictions.jsonl
```

`--limit 0` chạy toàn bộ. Lấy mẫu mặc định round robin qua các shop đã shuffle, không đảm bảo đại diện mọi nhóm nghiệp vụ khi mẫu nhỏ. Thời gian ngoại suy 3.000 câu chỉ là ước lượng thô từ mẫu hiện tại, không gồm tải model; session Kaggle có thể bị ngắt. Save Version/output notebook để giữ artifact.

### Nhập câu hỏi riêng

```bash
python inference.py \
  --data-dir /kaggle/input/<dataset>/train \
  --business-id biz_123 \
  --question "Tổng doanh thu tháng 5/2026 là bao nhiêu?" \
  --as-of-date 2026-06-28 \
  --output /kaggle/working/custom/predictions.jsonl
```

### File đầu ra

- `predictions.jsonl`: chỉ ID, type, answer/reason; đúng cấu trúc submission.
- `predictions.run.json`: cấu hình, mẫu câu, phiên bản thư viện, device map, thời gian nạp model.
- `predictions.traces.jsonl`: phản hồi model, lời gọi tool, kết quả/lỗi.
- `predictions.metrics.jsonl`: thời gian/token/VRAM theo câu, không có toàn bộ trace.
- `predictions.summary.json`: thống kê thời gian cho phần chạy trong session hiện tại.
- `predictions.evaluation.json`: kết quả chấm.

Lưu sau từng câu để tiếp tục được. Nếu tiến trình dừng giữa lúc ghi một dòng JSONL, sửa/bỏ dòng cuối không hoàn chỉnh trước khi resume; parser báo lỗi thay vì bỏ qua dữ liệu hỏng. Trace có thể chứa thông tin kinh doanh và khách hàng, cần giữ cùng phạm vi truy cập với dữ liệu nguồn.

## Tool và ranh giới dữ liệu

- `list_files`, `read_document`, `inspect_table`: khám phá dữ liệu đúng shop.
- `load_table`: đọc cả CSV/sheet Excel, chọn dòng header; không tự fill hay deduplicate.
- `run_sql`: DuckDB SELECT trên bảng đã nạp, chặn truy cập file/network ngoài.
- `query_sqlite`: kết nối read-only, authorizer chỉ cho đọc, có timeout.

Đường dẫn được giới hạn trong shop hiện tại, cộng `rulebook.md`/`calendar.csv` của split. Agent không có tool đọc JSONL hoặc đáp án. Chỉ bộ chấm đọc `answers.jsonl`. Công cụ trả tối đa 100 dòng, không giới hạn phạm vi tính tổng; kết quả lớn được báo truncated. Model dùng **giao thức action JSON tự định nghĩa**, không phải parser native tool-call của Qwen. Có retry sửa lỗi protocol trong giới hạn lượt.

Chưa cung cấp thực thi Python tùy ý; các phép chuẩn hóa dùng SQL và tham số đọc bảng. Một số shop đặc thù có thể cần bổ sung tool có hàm xử lý xác định. Hiện giới hạn thời gian áp dụng cho truy vấn SQL; nạp workbook và generation chỉ bị giới hạn bởi tài nguyên, context và số token, chưa có hard timeout cho cả câu.

## Kiểm thử không cần GPU

```bash
pip install pandas openpyxl duckdb pytest ruff
python -m pytest -q
ruff check zaic inference.py evaluate.py tests
```

Đã chạy qua 12 test và Ruff tại môi trường phát triển (Python 3.13, pandas 3.1.0, DuckDB 2.0.0, OpenPyXL 3.1.5).

Các test dùng nguồn giả lập và model phản hồi theo kịch bản để kiểm tra đầy đủ vòng lặp → tool → đáp án → chấm, file isolation, SQL read-only, alias, danh sách/bảng, lấy mẫu và format. Không thay thế kiểm thử Qwen thật trên Kaggle.

## Tài liệu model

- [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [BitsAndBytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes)

Python >=3.10, cần Transformers hỗ trợ `Qwen3_5ForConditionalGeneration`. Sau khi chạy thành công trên Kaggle, lưu `pip freeze` cùng notebook để khóa môi trường đã kiểm chứng.

Chiến lược device map tham khảo: [Accelerate Big Model Inference](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference).
