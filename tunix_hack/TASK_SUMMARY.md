# Google Tunix Hack - Competition Summary

---

## 🎯 Task

Fine-tune Google's Gemma model to output reasoning traces in the format:
```
<reasoning>step-by-step thinking</reasoning>
<answer>final answer</answer>
```

---

## ✅ Completed Work

### 1. Training Script (`train_a100.py`)
- Full training pipeline for A100 GPU
- Same parameters as Kaggle notebook
- Handles HuggingFace authentication

### 2. Datasets Used (~570k samples)
| Dataset | Samples | Source |
|---------|---------|--------|
| GSM8K | 7,473 | Math word problems |
| OpenThoughts-114k | 113,957 | R1 distilled reasoning |
| Bespoke-Stratos-17k | 16,710 | High quality R1 |
| Medical-O1 (en) | 19,704 | Medical reasoning |
| Medical-O1 (en_mix) | 24,887 | Medical reasoning |
| MetaMathQA | 395,000 | Augmented math |
| **Total** | **577,731** | After filtering: ~570,699 |

### 3. Model Configuration
- **Base Model:** google/gemma-2-2b-it
- **Method:** LoRA (Low-Rank Adaptation)
- **LoRA Rank:** 32
- **LoRA Alpha:** 64
- **Target Modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Quantization:** 4-bit NF4 with double quantization

### 4. Training Configuration
- **Batch Size:** 4
- **Gradient Accumulation:** 8 (effective batch = 32)
- **Learning Rate:** 2e-4
- **Epochs:** 1
- **Max Sequence Length:** 1024
- **Optimizer:** paged_adamw_32bit
- **Scheduler:** cosine

### 5. Training Results
- **Total Steps:** 17,835
- **Training Time:** ~6 hours on A100 40GB
- **Adapter Size:** ~83MB
- **Checkpoints:** checkpoint-17500, checkpoint-17835

---

## 📁 Project Structure

```
tunix_hack/
├── notebooks/
│   ├── train_a100.py           # Main training script (local A100)
│   ├── train_kaggle_fixed.py   # Kaggle P100 version (OOM fixed)
│   ├── tunixhack.ipynb         # Original Kaggle notebook
│   └── gemma-reasoning/        # Trained model adapter
│       ├── adapter_config.json
│       ├── adapter_model.safetensors  # 83MB LoRA weights
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── special_tokens_map.json
│       ├── chat_template.jinja
│       ├── training_args.bin
│       ├── README.md
│       ├── checkpoint-17500/
│       └── checkpoint-17835/
├── writeup_draft.md            # Competition writeup
├── video_script.md             # 3-minute video script
└── TASK_SUMMARY.md             # This file
```

---

## 🧪 Model Test Results

```
📝 Question: What is 125 + 347?
🤖 Response:
<reasoning>
To solve this problem, we need to add the two numbers together.
125 + 347 = 472.
</reasoning>
<answer>472</answer>

📝 Question: Solve: 2x + 5 = 13
🤖 Response:
<reasoning>
To solve for x, we want to isolate x on one side of the equation.
Subtracting 5 from both sides: 2x = 8
Dividing both sides by 2: x = 4
</reasoning>
<answer>4</answer>

📝 Question: A train travels 240 km in 4 hours. What is its speed?
🤖 Response:
<reasoning>
The speed is the distance traveled divided by the time taken.
Speed = 240 km / 4 hours = 60 km/hour.
</reasoning>
<answer>60</answer>
```

---

## 🚀 Next Steps for Submission

### 1. Upload Model to Kaggle
```bash
# Install kaggle CLI
pip install kaggle

# Upload model as Kaggle dataset/model
kaggle datasets create -p ./gemma-reasoning
```

### 2. Create Inference Notebook
Create a Kaggle notebook that:
- Loads the fine-tuned adapter
- Generates predictions with `<reasoning>` and `<answer>` tags
- Formats output for submission

### 3. Submit to Competition
- Upload inference notebook to Kaggle
- Run on competition test data
- Submit predictions

### 4. Complete Writeup
- Update `writeup_draft.md` with final results
- Add training metrics and examples

### 5. Create Video (3 minutes)
- Follow `video_script.md` outline
- Record demo of model reasoning

---

## 💻 How to Use the Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Load with quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    quantization_config=bnb_config,
    device_map="auto",
)

# Load tokenizer and adapter
tokenizer = AutoTokenizer.from_pretrained("./gemma-reasoning")
model = PeftModel.from_pretrained(base_model, "./gemma-reasoning")

# Generate
def generate(question):
    prompt = f"<start_of_turn>user\n{question}\n<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
print(generate("What is 2 + 2?"))
```

---

## 🔗 Resources

- **Competition Page:** [Kaggle Tunix Hack](https://www.kaggle.com/competitions/google-tunix-hack)
- **Base Model:** [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)
- **Datasets:**
  - [GSM8K](https://huggingface.co/datasets/gsm8k)
  - [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)
  - [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k)
  - [Medical-O1](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
  - [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)
