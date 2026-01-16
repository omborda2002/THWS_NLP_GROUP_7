#!/usr/bin/env python3
"""
Google Tunix Hack - Reasoning Model Training
Optimized for A100 40GB GPU

Author: Om Borda (omborda2002)
Output Format: <reasoning>...</reasoning><answer>...</answer>
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
import random
import re
import os

print("=" * 60)
print("🧠 TUNIX HACK - REASONING MODEL TRAINING")
print("   Same config as Kaggle notebook - for A100 GPU")
print("=" * 60)

# ============================================
# CONFIGURATION - SAME AS KAGGLE NOTEBOOK
# ============================================
CONFIG = {
    # Model - will download from HuggingFace
    "model_name": "google/gemma-2-2b-it",
    "max_seq_length": 1024,
    
    # LoRA - same as Kaggle
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    
    # Training - same as Kaggle
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "num_epochs": 1,
    "warmup_ratio": 0.03,
    
    # Output
    "output_dir": "./gemma-reasoning",
}

# Dataset limits - ALL DATA (same as Kaggle notebook)
DATASET_LIMITS = {
    "gsm8k": None,           # All 7.4k
    "openthoughts": None,    # All 114k
    "stratos": None,         # All 17k
    "medical_o1": None,      # All ~44k
    "metamath": None,        # All 395k
}

print(f"\n📊 Configuration:")
print(f"   Model: {CONFIG['model_name']}")
print(f"   Batch size: {CONFIG['batch_size']} x {CONFIG['gradient_accumulation_steps']} = {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
print(f"   Seq length: {CONFIG['max_seq_length']}")
print(f"   LoRA rank: {CONFIG['lora_r']}")

# ============================================
# DATA FORMATTERS
# ============================================

def extract_think_answer(text):
    """Extract thinking from <think> tags."""
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    if '</think>' in text:
        answer = text.split('</think>')[-1].strip()
    else:
        answer = text
    return thinking, answer

def format_gsm8k(example):
    """Format GSM8K math problems."""
    question = example.get('question', '')
    answer_text = example.get('answer', '')
    
    if '####' in answer_text:
        reasoning = answer_text.split('####')[0].strip()
        final = answer_text.split('####')[1].strip()
    else:
        reasoning = answer_text
        final = answer_text.split('\n')[-1]
    
    return {
        "instruction": question,
        "response": f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>{final}</answer>"
    }

def format_openthoughts(example):
    """Format OpenThoughts-114k."""
    try:
        conversations = example.get('conversations', [])
        question, answer = "", ""
        
        for conv in conversations:
            role = conv.get('from', '')
            if role in ['human', 'user']:
                question = conv.get('value', '')
            elif role in ['gpt', 'assistant']:
                answer = conv.get('value', '')
        
        if not question or not answer:
            return None
        
        thinking, final = extract_think_answer(answer)
        if thinking:
            response = f"<reasoning>\n{thinking}\n</reasoning>\n<answer>{final}</answer>"
        else:
            response = f"<reasoning>\n{answer[:1500]}\n</reasoning>\n<answer>{answer[-300:]}</answer>"
        
        return {"instruction": question, "response": response}
    except:
        return None

def format_stratos(example):
    """Format Bespoke-Stratos-17k."""
    try:
        conversations = example.get('conversations', [])
        question, answer = "", ""
        
        for conv in conversations:
            role = conv.get('from', '')
            if role in ['human', 'user']:
                question = conv.get('value', '')
            elif role in ['gpt', 'assistant']:
                answer = conv.get('value', '')
        
        if not question or not answer:
            return None
        
        thinking, final = extract_think_answer(answer)
        if thinking:
            response = f"<reasoning>\n{thinking}\n</reasoning>\n<answer>{final}</answer>"
        else:
            response = f"<reasoning>\n{answer[:1500]}\n</reasoning>\n<answer>{answer[-300:]}</answer>"
        
        return {"instruction": question, "response": response}
    except:
        return None

def format_medical_o1(example):
    """Format Medical O1 reasoning."""
    try:
        question = example.get('Question', '')
        cot = example.get('Complex_CoT', '')
        response_text = example.get('Response', '')
        
        if not question:
            return None
        
        if cot:
            response = f"<reasoning>\n{cot[:1500]}\n</reasoning>\n<answer>{response_text}</answer>"
        else:
            response = f"<reasoning>\nAnalyzing medical question.\n</reasoning>\n<answer>{response_text}</answer>"
        
        return {"instruction": question, "response": response}
    except:
        return None

def format_metamath(example):
    """Format MetaMathQA."""
    query = example.get('query', '')
    response = example.get('response', '')
    
    if 'The answer is' in response:
        parts = response.split('The answer is')
        reasoning = parts[0].strip()
        final = parts[1].strip().rstrip('.')
    else:
        reasoning = response
        final = response.split('\n')[-1]
    
    return {
        "instruction": query,
        "response": f"<reasoning>\n{reasoning[:1500]}\n</reasoning>\n<answer>{final}</answer>"
    }

print("\n✓ Formatters ready")

# ============================================
# GEMMA CHAT TEMPLATE
# ============================================

def create_prompt(instruction: str) -> str:
    """Gemma chat format."""
    return f"<start_of_turn>user\n{instruction}\n<end_of_turn>\n<start_of_turn>model\n"

def format_for_training(example: dict) -> dict:
    """Final training format."""
    if example is None:
        return None
    prompt = create_prompt(example["instruction"])
    return {"text": prompt + example["response"] + "<end_of_turn>"}

# ============================================
# LOAD DATASETS
# ============================================

def load_and_format_dataset(name, config, formatter, limit, desc):
    """Load and format a single dataset."""
    print(f"\n📊 Loading {desc}...")
    try:
        if config:
            ds = load_dataset(name, config, split="train")
        else:
            ds = load_dataset(name, split="train")
        
        if limit and len(ds) > limit:
            ds = ds.shuffle(seed=42).select(range(limit))
        
        formatted = [formatter(ex) for ex in ds]
        formatted = [f for f in formatted if f is not None]
        
        print(f"   ✓ {len(formatted):,} examples")
        return formatted
    except Exception as e:
        print(f"   ✗ Failed: {str(e)[:80]}")
        return []

print("\n" + "="*60)
print("📥 LOADING DATASETS")
print("="*60)

all_examples = []

all_examples += load_and_format_dataset("gsm8k", "main", format_gsm8k, DATASET_LIMITS["gsm8k"], "GSM8K")
all_examples += load_and_format_dataset("open-thoughts/OpenThoughts-114k", None, format_openthoughts, DATASET_LIMITS["openthoughts"], "OpenThoughts")
all_examples += load_and_format_dataset("bespokelabs/Bespoke-Stratos-17k", None, format_stratos, DATASET_LIMITS["stratos"], "Stratos")
all_examples += load_and_format_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", format_medical_o1, DATASET_LIMITS["medical_o1"], "Medical-O1-en")
all_examples += load_and_format_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en_mix", format_medical_o1, DATASET_LIMITS["medical_o1"], "Medical-O1-mix")
all_examples += load_and_format_dataset("meta-math/MetaMathQA", None, format_metamath, DATASET_LIMITS["metamath"], "MetaMath")

print(f"\n📊 Total collected: {len(all_examples):,}")

# ============================================
# PREPARE FINAL DATASET
# ============================================

print("\n" + "="*60)
print("🔧 PREPARING DATASET")
print("="*60)

random.seed(42)
random.shuffle(all_examples)

# Filter valid examples
valid = []
for ex in all_examples:
    if ex and len(ex.get("instruction", "")) > 10 and len(ex.get("response", "")) > 30:
        if "<reasoning>" in ex["response"] and "<answer>" in ex["response"]:
            valid.append(ex)

# Format for training
final_data = [format_for_training(ex) for ex in valid]
final_data = [f for f in final_data if f and len(f["text"]) < 4000]  # Skip very long

dataset = Dataset.from_list(final_data)
print(f"\n✅ Final training dataset: {len(dataset):,} samples")

# Preview
print("\n📝 Sample training example:")
print("="*60)
print(dataset[0]["text"][:1000])
print("="*60)

# ============================================
# LOAD MODEL
# ============================================

print("\n" + "="*60)
print("🤖 LOADING MODEL")
print("="*60)

# Check GPU
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Quantization config for A100
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Get HF token from environment
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not hf_token:
    print("⚠️  No HF_TOKEN found. Set with: export HF_TOKEN=your_token")
    print("   Get token from: https://huggingface.co/settings/tokens")
    hf_token = input("Enter HuggingFace token (or press Enter to try without): ").strip() or None

print(f"\nLoading tokenizer from {CONFIG['model_name']}...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("✓ Tokenizer loaded")

print(f"Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_name"],
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=hf_token,
)

model = prepare_model_for_kbit_training(model)
print("✓ Model loaded")

# ============================================
# APPLY LoRA
# ============================================

print("\n" + "="*60)
print("🔧 APPLYING LoRA")
print("="*60)

lora_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================
# TRAINING
# ============================================

print("\n" + "="*60)
print("🚀 SETTING UP TRAINING")
print("="*60)

training_args = SFTConfig(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    warmup_ratio=CONFIG["warmup_ratio"],
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    bf16=True,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    report_to="none",
    gradient_checkpointing=True,
    max_length=CONFIG["max_seq_length"],
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_args,
)

steps = len(dataset) // (CONFIG["batch_size"] * CONFIG["gradient_accumulation_steps"])
print(f"\n🚀 Training Plan:")
print(f"   Samples: {len(dataset):,}")
print(f"   Steps: ~{steps:,}")
print(f"   Estimated time: ~4-6 hours")

# Train!
print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60)

trainer.train()

# ============================================
# SAVE MODEL
# ============================================

print("\n" + "="*60)
print("💾 SAVING MODEL")
print("="*60)

trainer.save_model()
tokenizer.save_pretrained(CONFIG["output_dir"])
print(f"✅ Model saved to {CONFIG['output_dir']}")

# ============================================
# TEST MODEL
# ============================================

print("\n" + "="*60)
print("🧪 TESTING MODEL")
print("="*60)

def generate_response(question, max_tokens=400):
    """Generate a response with reasoning."""
    prompt = create_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<start_of_turn>model" in response:
        response = response.split("<start_of_turn>model")[-1]
    return response.strip()

test_questions = [
    "What is 125 + 347?",
    "Solve: 2x + 5 = 13",
    "A train travels 240 km in 4 hours. What is its speed?",
    "What is the probability of rolling a 6 on a fair die?",
    "What are the symptoms of diabetes?",
]

for q in test_questions:
    print(f"\n📝 Question: {q}")
    print(f"🤖 Response:\n{generate_response(q)}")
    print("-"*60)

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"\nModel saved to: {CONFIG['output_dir']}")
