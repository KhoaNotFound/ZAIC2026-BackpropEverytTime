# DataMate — Qwen3.5 trên Kaggle

Pipeline inference cho dữ liệu kinh doanh nhiều cấu trúc. Qwen3.5 đọc README/rulebook, lựa chọn tool, tính bằng SQL và xuất JSON. Giữ nguyên Excel/CSV/SQLite; bảng trung gian được tạo trong RAM theo từng câu hỏi. Không fine-tune, không cập nhật trọng số, không tính training loss.

## Chạy trên Kaggle (benchmark tập train)

Mục tiêu: đo Qwen3.5-9B pretrained (chưa fine-tune) với harness hiện tại trả lời đúng được bao nhiêu câu trong tập **train**. Mọi lần chạy đều dùng `train` vì chỉ tập này có `answers.jsonl` để chấm.

### 1. Chuẩn bị notebook

1. Tạo notebook mới trên Kaggle. Ở panel bên phải, mục **Session options**:
   - **Accelerator**: chọn **GPU T4 x2** (bắt buộc 2 GPU).
   - **Internet**: bật **On** (cần để clone repo và tải trọng số Qwen từ Hugging Face).
2. Gắn dataset: **Add Input** → tab **Your Work** / **Datasets** → chọn dataset bạn đã upload → **Add**.
3. Dataset phải có thư mục train với cấu trúc:

```text
.../train/
  questions.jsonl
  answers.jsonl
  rulebook.md
  calendar.csv
  data/biz_101/README.md ...
```

Không đưa `answers.jsonl` vào thư mục con của doanh nghiệp. Dataset không nằm trong Git (`data/` đã được ignore).

### 2. Tìm đường dẫn dataset

Kaggle gắn dataset chỉ đọc dưới `/kaggle/input/`. Tên thư mục là slug của dataset, và có thể có thêm cấp thư mục tuỳ cách bạn upload (ví dụ file zip giải nén ra `data/train`). Notebook **tự tìm** thư mục có đủ `questions.jsonl`, `answers.jsonl`, `rulebook.md` và `data/` khi để `DATA_DIR = ""`. Muốn tự xem, chạy trong một cell:

```python
!ls /kaggle/input
!find /kaggle/input -maxdepth 4 -name questions.jsonl
```

Đường dẫn train là thư mục chứa `questions.jsonl`, ví dụ `/kaggle/input/<slug>/train`. Cũng có thể bấm vào thư mục ở panel **Input** bên phải → biểu tượng copy để lấy đường dẫn.

### 3. Lấy notebook từ repo

Tải file [notebooks/kaggle_inference.ipynb](notebooks/kaggle_inference.ipynb) từ GitHub, rồi trên Kaggle chọn **File → Import Notebook** và upload file đó. Hoặc tự tạo các cell như dưới đây (giống nội dung notebook).

### 4. Các cell, chạy lần lượt

**Cell 1: cấu hình.** Chỉ cần sửa `LIMIT` (số câu chạy thử; `0` = toàn bộ). Để `DATA_DIR = ""` để tự tìm. `OUTPUT` tự đổi theo `LIMIT`/`SEED`/`CONTEXT_TOKENS` nên mỗi cấu hình có thư mục riêng, không bị lỗi "Output exists".

```python
REPO_URL = "https://github.com/KhoaNotFound/ZAIC2026-BackpropEverytTime.git"
DATA_DIR = ""  # để trống: tự tìm thư mục train trong /kaggle/input
LIMIT = 1      # số câu chạy thử; 0 = toàn bộ
SEED = 42
CONTEXT_TOKENS = 32768  # input + output mỗi bước
RESUME = False
# ... hàm find_train_dirs() tự tìm DATA_DIR (xem notebook), rồi:
OUTPUT = f"/kaggle/working/qwen_train_limit{LIMIT}_seed{SEED}_ctx{CONTEXT_TOKENS}/predictions.jsonl"
```

**Cell 2: clone/pull repo và cài thư viện.** Lần đầu clone, các lần sau `git pull` để lấy code mới nhất.

```python
import subprocess
from pathlib import Path
repo = Path("/kaggle/working/zAIC")
if not repo.exists():
    subprocess.run(["git", "clone", REPO_URL, str(repo)], check=True)
else:
    subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=True)
%cd /kaggle/working/zAIC
%pip install -r requirements.txt
```

Nếu pip nâng cấp thư viện đang được import (thường là `transformers`), **Restart kernel** rồi chạy lại Cell 1 và đi tiếp từ Cell 3 (không cần chạy lại Cell 2). Giữ PyTorch CUDA có sẵn của Kaggle. Không cần API key cho checkpoint Qwen công khai.

**Cell 3: kiểm tra GPU.** Phải thấy 2 dòng `Tesla T4`, khoảng 14.7 GiB mỗi card.

```python
import torch
print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(i, props.name, round(props.total_memory / 2**30, 2), "GiB")
```

**Cell 4: dry run.** Kiểm tra dữ liệu và in danh sách câu được chọn, chưa tải model.

```python
import subprocess, sys, os
os.chdir("/kaggle/working/zAIC")
assert (Path(DATA_DIR) / "answers.jsonl").is_file(), "DATA_DIR phải là tập train có answers.jsonl"
base = [sys.executable, "inference.py", "--data-dir", DATA_DIR, "--limit", str(LIMIT),
        "--seed", str(SEED), "--output", OUTPUT]
subprocess.run(base + ["--dry-run"], check=True)
```

**Cell 5: inference trên 2 GPU.** `--device-map split` đặt 16 decoder layer lên GPU 0 và 16 layer lên GPU 1. Log in ra device map và VRAM mỗi GPU sau khi nạp, rồi một dòng cho mỗi câu (`[i/n] id status type thời_gian`). Lần đầu tải trọng số mất vài phút.

```python
import torch
assert torch.cuda.device_count() >= 2, "Cần 2 GPU: chọn Accelerator GPU T4 x2"
command = base + ["--model", "Qwen/Qwen3.5-9B", "--quantization", "4bit", "--device-map", "split",
                   "--context-tokens", str(CONTEXT_TOKENS)]
if RESUME:
    command += ["--resume"]
subprocess.run(command, check=True)
```

**Cell 6: chấm điểm và in benchmark.**

```python
subprocess.run([sys.executable, "evaluate.py", "--predictions", OUTPUT,
                "--answers", str(Path(DATA_DIR) / "answers.jsonl"),
                "--questions", str(Path(DATA_DIR) / "questions.jsonl")], check=True)
```

Benchmark in ra gồm: số câu đúng/sai/abstain, accuracy, coverage, attempted accuracy, score (đúng +1, sai −1, abstain 0), kết quả theo loại đáp án, trạng thái kết thúc (`ok`, `stalled`, `max_steps`, `error`) và kết quả theo trạng thái, thời gian/số bước/tool call/token mỗi câu, VRAM đỉnh từng GPU, thời gian nạp model. Bản đầy đủ (thêm theo cửa hàng và từng câu) nằm trong `predictions.evaluation.json`.

**Cell 7: liệt kê file kết quả.**

```python
summary = Path(OUTPUT).with_suffix(".summary.json")
if summary.exists():
    print(summary.read_text())
print("Artifacts:", [p.name for p in Path(OUTPUT).parent.glob("predictions.*")])
```

### 5. Quy trình gợi ý

1. `LIMIT = 1`: kiểm tra model nạp được, xem VRAM đỉnh từng GPU trong benchmark.
2. `LIMIT = 20` (hoặc 50, 100): đo độ chính xác sơ bộ. Chỉ cần sửa Cell 1 rồi chạy lại Cell 1, 4, 5, 6.
3. `LIMIT = 0`: toàn bộ tập train. Thời gian ước lượng cho 3.000 câu nằm trong `predictions.summary.json`; một session GPU Kaggle tối đa khoảng 12 giờ và có hạn mức GPU theo tuần, nên chạy toàn bộ có thể cần nhiều session kèm `RESUME = True`.
4. Session bị ngắt giữa chừng: giữ nguyên Cell 1, đặt `RESUME = True`, chạy lại. Câu đã có dự đoán sẽ được bỏ qua.
5. **Save Version** (hoặc tải thư mục `/kaggle/working/...`) để giữ kết quả; `/kaggle/working` mất khi session kết thúc.

**Context:** system prompt + `rulebook.md` (~36 KB) + README cửa hàng đã khoảng 12-13 nghìn token trước khi gọi tool, nên notebook dùng `CONTEXT_TOKENS = 32768` thay vì mặc định 16384 của `inference.py`. Câu nào vẫn vượt giới hạn sẽ abstain với trạng thái `context_limit` và chạy tiếp câu sau, không dừng cả đợt. Nếu benchmark báo nhiều `context_limit`, tăng lên 49152 hoặc 65536 và theo dõi VRAM đỉnh; context dài hơn cũng làm mỗi bước chậm hơn vì toàn bộ prompt được xử lý lại.

Nếu chạy dừng do lỗi (ví dụ CUDA out of memory), chương trình lưu traceback vào `predictions.traces.jsonl`, dừng và thoát mã 1. Restart kernel để giải phóng GPU trước khi chạy lại với `LIMIT`/output mới.

### Chạy bằng dòng lệnh (không dùng notebook)

Kiểm tra chọn mẫu, chưa tải model:

```bash
python inference.py --data-dir /kaggle/input/<dataset>/train --limit 5 --dry-run
```

Chạy 1 câu trước để kiểm tra model và VRAM, rồi dùng đường dẫn output mới để chạy 20 câu:

```bash
python inference.py \
  --data-dir /kaggle/input/<dataset>/train \
  --model Qwen/Qwen3.5-9B --quantization 4bit \
  --device-map split --limit 1 --sampling stratified \
  --output /kaggle/working/smoke/predictions.jsonl
```

Mặc định: `--device-map split` chia các decoder layer nguyên khối qua tất cả GPU đang thấy. Với Qwen3.5-9B và hai T4, 16 layer đầu đặt trên GPU 0 và 16 layer sau trên GPU 1; embedding ở GPU 0, norm ở GPU cuối, output head luôn ở GPU 0 cùng input. Chế độ này cần ít nhất hai GPU. Đây là model parallelism, không phải hai worker chạy hai câu đồng thời hoặc tensor parallelism; không đảm bảo nhanh gấp đôi.

Các cấu hình còn lại: FP16 compute, NF4 4-bit, context 16.384 token gồm input và output, 2.048 token đầu ra mỗi lượt, 16 lượt/câu. Nạp lớp text-only `Qwen3_5ForCausalLM`, bỏ vision tower không dùng. Chỉ tính logits của token cuối (`logits_to_keep=1`) và mặc định xử lý prompt theo chunk 512 token (`--prefill-chunk-size 512`, đặt 0 để tắt). Chunking không cắt bỏ prompt, nhưng cần Transformers >=5.13 để giữ đúng trạng thái hybrid attention qua các chunk. Dùng dynamic cache mới cho mỗi lần sinh. Code in device map và bộ nhớ từng GPU sau khi nạp; các layer được chia đều nhưng tổng VRAM có thể lệch vì embedding, head, cache và tensor tạm.

`--device-map auto`, `balanced`, `balanced_low_0`, `sequential` vẫn được hỗ trợ. `--gpu-memory-gib 13` là ngân sách đặt trọng số cho các chế độ tự ánh xạ này, không phải hard limit tổng VRAM. Nó không áp dụng cho map `split` tường minh. `--device-map cuda:0` dùng một card. `--quantization none` dùng FP16 và cần nhiều VRAM hơn. Hybrid attention có thể dùng đường PyTorch chậm nếu không có kernel tăng tốc phù hợp; không cài FlashAttention tùy tiện trên T4.

Nếu dùng uv trên Kaggle, cập nhật repo rồi chạy trong cell:

```python
%cd /kaggle/working/zAIC
!git pull --ff-only
!uv sync --upgrade-package transformers
!uv run python inference.py --data-dir "<đường dẫn train thực tế>" --device-map split --quantization 4bit --limit 5 --output /kaggle/working/split/predictions.jsonl
```

Dừng tiến trình inference cũ trước khi chạy; nếu model còn trong kernel notebook, restart session/kernel để giải phóng GPU. Khi đổi device map hãy dùng output mới thay vì resume đợt chạy cũ. Split giúp phân bổ trọng số nhưng vẫn có thể OOM do context dài hoặc tiến trình khác chiếm VRAM; chưa benchmark bản sửa trên T4 thật. Nếu còn lỗi, xem `error_traceback` trong trace/metrics để xác định phép tính lỗi.


Nếu vượt context, câu đó abstain với trạng thái `context_limit` (không âm thầm cắt mất rulebook) và đợt chạy tiếp tục với câu sau. Hãy xem trace, tăng `--context-tokens` nếu VRAM cho phép, hoặc giảm độ dài tool output/chuỗi xử lý. `--thinking` bật suy luận nội bộ, có thể cần tăng `--max-new-tokens`; mặc định tắt để đo baseline nhanh hơn. Generation mặc định greedy để dễ tái lập, không phải sampling preset tối ưu của nhà phát hành.

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

Dùng lại chính lệnh inference ban đầu và thêm `--resume`. Cấu hình và danh sách mẫu phải giống nhau. Chỉ ID đã lưu dự đoán mới được bỏ qua. Lỗi model/runtime được lưu thành abstain kèm traceback, chương trình dừng ngay và thoát mã 1; để thử lại các câu lỗi, dùng một đợt chạy mới. Không tiếp tục dùng model sau OOM và không ngoại suy thời gian khi có lỗi thực thi. Khi đổi limit/model/cấu hình hãy dùng output mới.

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
- `predictions.summary.json`: thống kê thời gian cho phần chạy trong session hiện tại; ngoại suy null nếu có lỗi thực thi.
- `predictions.evaluation.json`: benchmark do `evaluate.py` tạo: accuracy, coverage, score theo tổng/loại đáp án/cửa hàng, kết quả từng câu, và `runtime` (trạng thái ok/stalled/max_steps/error, kết quả theo trạng thái, thời gian, số bước, tool call, token, VRAM đỉnh mỗi GPU, thời gian nạp model).

Lưu sau từng câu để tiếp tục được. Nếu tiến trình dừng giữa lúc ghi một dòng JSONL, sửa/bỏ dòng cuối không hoàn chỉnh trước khi resume; parser báo lỗi thay vì bỏ qua dữ liệu hỏng. Trace có thể chứa thông tin kinh doanh và khách hàng, cần giữ cùng phạm vi truy cập với dữ liệu nguồn.

## Tool và ranh giới dữ liệu

- `list_files`, `read_document`, `inspect_table`: khám phá dữ liệu đúng shop.
- `load_table`: đọc cả CSV/sheet Excel, chọn dòng header; không tự fill hay deduplicate.
- `run_sql`: DuckDB SELECT trên bảng đã nạp, chặn truy cập file/network ngoài.
- `query_sqlite`: kết nối read-only, authorizer chỉ cho đọc, có timeout.

Đường dẫn được giới hạn trong shop hiện tại, cộng `rulebook.md`/`calendar.csv` của split. Agent không có tool đọc JSONL hoặc đáp án. Chỉ bộ chấm đọc `answers.jsonl`. Công cụ trả tối đa 100 dòng, không giới hạn phạm vi tính tổng; kết quả lớn được báo truncated. Model dùng **giao thức action JSON tự định nghĩa**, không phải parser native tool-call của Qwen. Nhận cả dạng bọc `{"action":"tool","name":"load_table","arguments":{...}}` lẫn dạng phẳng `{"action":"load_table","path":"...","name":"sales"}`; chỉ tên tool trong whitelist được nhận. Với `list_files`, path dư được bỏ qua vì tool luôn giới hạn ở shop hiện tại. Trace giữ phản hồi gốc và `normalized_action`. Có ví dụ gọi tool với tên file nguồn thực tế và retry sửa lỗi protocol. Ba lời gọi giống hệt liên tiếp hoặc ba phản hồi lỗi giống hệt dẫn đến `stalled`/abstain, không chạy hết lượt một cách vô ích.

Chưa cung cấp thực thi Python tùy ý; các phép chuẩn hóa dùng SQL và tham số đọc bảng. Một số shop đặc thù có thể cần bổ sung tool có hàm xử lý xác định. Hiện giới hạn thời gian áp dụng cho truy vấn SQL; nạp workbook và generation chỉ bị giới hạn bởi tài nguyên, context và số token, chưa có hard timeout cho cả câu.

## Kiểm thử không cần GPU

```bash
pip install pandas openpyxl duckdb pytest ruff
python -m pytest -q
ruff check zaic inference.py evaluate.py tests
```

Đã chạy qua 25 test CPU; một test GPU được skip khi không có PyTorch/hai GPU CUDA. Ruff đã qua. Test GPU nhỏ không tải checkpoint, so generation single-GPU với split + chunked prefill: `uv run pytest -q tests/test_gpu_smoke.py`.

Các kiểm tra dữ liệu ban đầu được thực hiện tại môi trường phát triển (Python 3.13, pandas 3.1.0, DuckDB 2.0.0, OpenPyXL 3.1.5).

Các test dùng nguồn giả lập và model phản hồi theo kịch bản để kiểm tra đầy đủ vòng lặp → tool → đáp án → chấm, file isolation, SQL read-only, alias, danh sách/bảng, lấy mẫu và format. Không thay thế kiểm thử Qwen thật trên Kaggle.

## Tài liệu model

- [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [BitsAndBytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes)

Python >=3.10, cần Transformers >=5.13 hỗ trợ `Qwen3_5ForCausalLM` và chunked prefill của hybrid attention. Sau khi chạy thành công trên Kaggle, lưu `pip freeze` cùng notebook để khóa môi trường đã kiểm chứng.

Chiến lược device map tham khảo: [Accelerate Big Model Inference](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference).

Chunked prefill: [Transformers generation](https://huggingface.co/docs/transformers/main_classes/text_generation). Text-only Qwen loader: [Qwen3.5 docs](https://huggingface.co/docs/transformers/v5.13.0/model_doc/qwen3_5).
