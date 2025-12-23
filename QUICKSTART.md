# Quick Start Guide

## Cài đặt

### Bước 1: Cài đặt dependencies

```bash
cd /Users/luungoc/Qualcomm/Project/Reasoning/persistent_sampling

# Cài đặt dependencies cơ bản
pip install numpy tqdm pyyaml pandas datasets

# Cài đặt vLLM (yêu cầu GPU)
pip install vllm

# Hoặc nếu muốn build từ source
pip install git+https://github.com/vllm-project/vllm.git
```

### Bước 2: Kiểm tra installation

```bash
# Chạy basic tests (không cần GPU)
python scripts/test_basic.py
```

## Sử dụng nhanh

### 1. Test với sample data (không cần dataset lớn)

```bash
# Chạy trên sample MATH problems
python scripts/run_math500.py \
    --use_sample \
    --N 8 \
    --max_steps 30 \
    --output_name sample_test.json
```

### 2. Chạy demo đơn giản

```bash
# Demo với 1 problem
python scripts/demo.py
```

### 3. Python API

```python
import sys
sys.path.append('src')

from persistent_smc import PersistentSMC
from vllm_wrapper import vLLMGenerator

# Khởi tạo vLLM
llm = vLLMGenerator(
    model_name="Qwen/Qwen2.5-Math-7B-Instruct"
)

# Khởi tạo Persistent SMC
solver = PersistentSMC(
    llm_generator=llm,
    N=16,                    # Số particles
    k_max=10,                # Window size
    annealing_method="ess_targeted"
)

# Giải bài toán
problem = "What is 15% of 80?"
prompt = llm.format_math_prompt(problem)
solutions = solver.solve(prompt, max_steps=30)

# Lấy kết quả
from collections import Counter
from dataset_loaders import extract_final_answer_math

answers = [extract_final_answer_math(s.text) for s in solutions]
answers = [a for a in answers if a is not None]
final_answer = Counter(answers).most_common(1)[0][0] if answers else None

print(f"Answer: {final_answer}")
```

## Chạy evaluation trên datasets

### MATH500

```bash
# Download MATH dataset từ HuggingFace (tự động)
python scripts/run_math500.py \
    --model Qwen/Qwen2.5-Math-7B-Instruct \
    --N 16 \
    --max_steps 50 \
    --num_problems 100 \
    --output_name math500_100.json
```

### AIME24

```bash
# Cần prepare AIME24 dataset trước (hoặc dùng sample)
python scripts/run_aime24.py \
    --data_path data/aime24_sample.json \
    --model Qwen/Qwen2.5-Math-7B-Instruct \
    --N 24 \
    --max_steps 80 \
    --output_name aime24_test.json
```

## Hyperparameter tuning

### Cho easy problems (MATH Level 1-2):
```bash
python scripts/run_math500.py \
    --N 8 \
    --k_max 5 \
    --max_steps 30 \
    --temperature 0.7
```

### Cho hard problems (MATH Level 4-5, AIME):
```bash
python scripts/run_math500.py \
    --N 24 \
    --k_max 15 \
    --max_steps 80 \
    --temperature 0.8 \
    --difficulty "Level 5"
```

## Troubleshooting

### Lỗi: ModuleNotFoundError: No module named 'vllm'

```bash
# Cài đặt vLLM
pip install vllm

# Nếu gặp lỗi với CUDA
pip install vllm --no-build-isolation
```

### Lỗi: CUDA out of memory

```bash
# Giảm số particles
python scripts/run_math500.py --N 8 --k_max 5

# Hoặc sử dụng model nhỏ hơn
python scripts/run_math500.py \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct
```

### Lỗi: Dataset not found

```bash
# Sử dụng sample data
python scripts/run_math500.py --use_sample

# Hoặc download từ HuggingFace (tự động nếu có datasets library)
pip install datasets
```

## Monitoring

Trong quá trình chạy, bạn sẽ thấy:

```
=== Step 5 ===
Alive particles: 16
ESS: 12.34 / 16 (77.13%)
Beta: 0.2341
SC scores: mean=0.123, std=0.045
```

**Chỉ số quan trọng:**
- **ESS**: Nên > 30% của N_alive
- **Beta**: Tăng dần từ 0 → 1
- **SC scores**: Đo độ tự tin của model

## Kết quả

Kết quả được lưu trong `results/` dưới dạng JSON:

```json
{
  "accuracy": 0.85,
  "correct_count": 85,
  "total": 100,
  "pass_at_k": {
    "pass@1": 0.82,
    "pass@8": 0.91
  },
  "results": [...]
}
```

## Tài liệu chi tiết

- **Lý thuyết đầy đủ**: `PERSISTENT_SMC_FOR_LLM_REASONING.md`
- **Thuật toán chi tiết**: `algorithms.tex`
- **Code reference**: Comments trong các file `.py`

## Performance Tips

1. **Sử dụng ESS-targeted annealing**: Tốt nhất cho hầu hết cases
2. **Enable prefix caching**: Đã mặc định trong vLLM wrapper
3. **Batch generation**: Tự động trong implementation
4. **Monitor ESS**: Nếu ESS thấp, tăng `target_ess_ratio`

## Next Steps

1. **Thử với model khác**:
   - DeepSeek-Math
   - WizardMath
   - MetaMath

2. **Tùy chỉnh hyperparameters**:
   - Chỉnh `N` và `k_max` theo độ khó
   - Thử các annealing methods khác
   - Điều chỉnh temperature

3. **Thêm datasets**:
   - GSM8K
   - SVAMP
   - MathQA

4. **Integrate Process Reward Model**:
   - Thay thế self-certainty bằng PRM trained

Chúc bạn thành công với research! 🚀
