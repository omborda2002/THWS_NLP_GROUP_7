"""
🧠 Kaggle Reasoning Model Training
Author: Om Borda (omborda2002)
Competition: Google Tunix Hack

Model: Gemma 2B (from Kaggle hub)
Output Format: <reasoning>...</reasoning><answer>...</answer>

Optimized for Kaggle:
  - Uses Gemma from Kaggle Models (no download)
  - ~250k samples (fits in 20GB disk)
  - Training time: ~4-6 hours on TPU
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
import random
import re
import os

print("=" * 60)
print("🧠 KAGGLE REASONING MODEL TRAINING")
print("   Model: Gemma 2B | Target: ~250k samples")
print("=" * 60)

# ============================================
# CONFIGURATION
# ============================================
# Detect if running on Kaggle
IS_KAGGLE = os.path.exists('/kaggle')

CONFIG = {
    # Model - Use Kaggle path if on Kaggle, else HuggingFace
    "model_name": "/kaggle/input/gemma-2/transformers/gemma-2-2b-it/1" if IS_KAGGLE else "google/gemma-2-2b-it",
    "max_seq_length": 1024,
    
    # LoRA
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    
    # Training - optimized for Kaggle TPU/GPU
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "num_epochs": 1,
    "warmup_ratio": 0.03,
    
    # Output
    "output_dir": "/kaggle/working/gemma-reasoning" if IS_KAGGLE else "./outputs/gemma-reasoning",
    
    # Data limits for Kaggle
    "samples_per_dataset": {
        "gsm8k": None,  # Use all 7.4k
        "openthoughts": 100000,
        "stratos": None,  # Use all 17k  
        "medical_o1": None,  # Use all ~44k
        "metamath": 80000,
    },
}

# ============================================
# FORMATTING FUNCTIONS
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
        question = ""
        answer = ""
        
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
            # Truncate long responses
            response = f"<reasoning>\n{answer[:1500]}\n</reasoning>\n<answer>{answer[-300:]}</answer>"
        
        return {"instruction": question, "response": response}
    except:
        return None

def format_stratos(example):
    """Format Bespoke-Stratos-17k (highest quality)."""
    try:
        conversations = example.get('conversations', [])
        question = ""
        answer = ""
        
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
# DATA LOADING
# ============================================
def load_kaggle_datasets():
    """Load datasets optimized for Kaggle."""
    
    all_examples = []
    
    datasets_config = [
        ("gsm8k", "main", format_gsm8k, CONFIG["samples_per_dataset"]["gsm8k"], "GSM8K"),
        ("open-thoughts/OpenThoughts-114k", None, format_openthoughts, CONFIG["samples_per_dataset"]["openthoughts"], "OpenThoughts"),
        ("bespokelabs/Bespoke-Stratos-17k", None, format_stratos, CONFIG["samples_per_dataset"]["stratos"], "Stratos"),
        ("FreedomIntelligence/medical-o1-reasoning-SFT", "en", format_medical_o1, CONFIG["samples_per_dataset"]["medical_o1"], "Medical-O1-en"),
        ("FreedomIntelligence/medical-o1-reasoning-SFT", "en_mix", format_medical_o1, CONFIG["samples_per_dataset"]["medical_o1"], "Medical-O1-mix"),
        ("meta-math/MetaMathQA", None, format_metamath, CONFIG["samples_per_dataset"]["metamath"], "MetaMath"),
    ]
    
    for name, config, formatter, limit, desc in datasets_config:
        print(f"\n📊 Loading {desc}...")
        try:
            if config:
                ds = load_dataset(name, config, split="train")
            else:
                ds = load_dataset(name, split="train")
            
            # Limit samples
            if limit and len(ds) > limit:
                ds = ds.shuffle(seed=42).select(range(limit))
            
            # Format
            formatted = []
            for ex in ds:
                result = formatter(ex)
                if result:
                    formatted.append(result)
            
            all_examples.extend(formatted)
            print(f"   ✓ {len(formatted):,} examples")
            
        except Exception as e:
            print(f"   ✗ Failed: {str(e)[:50]}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_examples)
    
    # Filter valid
    valid = []
    for ex in all_examples:
        if ex and len(ex.get("instruction", "")) > 10 and len(ex.get("response", "")) > 30:
            if "<reasoning>" in ex["response"] and "<answer>" in ex["response"]:
                valid.append(ex)
    
    # Format for training
    final = [format_for_training(ex) for ex in valid if ex]
    final = [f for f in final if f and len(f["text"]) < 4000]  # Skip very long
    
    print(f"\n✅ Total training samples: {len(final):,}")
    
    return Dataset.from_list(final)

# ============================================
# MODEL LOADING
# ============================================
def load_model():
    """Load Gemma with quantization."""
    print(f"\n🔧 Loading model: {CONFIG['model_name']}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Detect if model path is local (Kaggle) or HuggingFace hub
    model_path = CONFIG["model_name"]
    is_local_model = os.path.isdir(model_path)
    print(f"   Model path: {model_path}")
    print(f"   Loading from: {'local directory' if is_local_model else 'HuggingFace Hub'}")
    
    # For local paths, list contents to verify
    if is_local_model:
        print(f"   Directory contents: {os.listdir(model_path)[:10]}...")
    
    # Load tokenizer - use GemmaTokenizerFast for local Kaggle paths
    print("   Loading tokenizer...")
    if is_local_model:
        from transformers import GemmaTokenizerFast
        tokenizer = GemmaTokenizerFast.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("   ✓ Tokenizer loaded")
    
    # Load model - use Gemma2ForCausalLM for local Kaggle paths
    print("   Loading model...")
    if is_local_model:
        from transformers import Gemma2ForCausalLM
        model = Gemma2ForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    
    model = prepare_model_for_kbit_training(model)
    print("   ✓ Model loaded")
    
    return model, tokenizer

def setup_lora(model):
    """Apply LoRA."""
    print("\n🔧 Applying LoRA...")
    
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
    
    return model

# ============================================
# TRAINING
# ============================================
def train():
    """Main training loop."""
    
    # Load data
    dataset = load_kaggle_datasets()
    
    # Show sample
    print("\n📝 Sample:")
    print("-" * 50)
    print(dataset[0]["text"][:500])
    print("-" * 50)
    
    # Load model
    model, tokenizer = load_model()
    model = setup_lora(model)
    
    # Training config
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
    
    # Calculate estimates
    steps = len(dataset) // (CONFIG["batch_size"] * CONFIG["gradient_accumulation_steps"])
    
    print("\n" + "=" * 60)
    print("🚀 STARTING TRAINING")
    print("=" * 60)
    print(f"   Samples: {len(dataset):,}")
    print(f"   Steps: ~{steps:,}")
    print(f"   Estimated time: ~4-6 hours on Kaggle TPU")
    print("=" * 60)
    
    trainer.train()
    
    # Save
    print("\n💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(CONFIG["output_dir"])
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print(f"   Model saved to: {CONFIG['output_dir']}")
    print("=" * 60)
    
    return model, tokenizer

# ============================================
# INFERENCE
# ============================================
def test_model(model, tokenizer):
    """Test the trained model."""
    
    questions = [
        "What is 125 + 347?",
        "Solve: 2x + 5 = 13",
        "A train travels 240 km in 4 hours. What is its speed?",
        "What is the probability of rolling a 6 on a fair die?",
    ]
    
    print("\n" + "=" * 60)
    print("🧪 TESTING MODEL")
    print("=" * 60)
    
    model.eval()
    
    for q in questions:
        prompt = create_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1]
        
        print(f"\n📝 Q: {q}")
        print(f"🤖 A: {response.strip()[:400]}")
        print("-" * 50)

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print(f"\n🌐 Running on: {'Kaggle' if IS_KAGGLE else 'Local'}")
    model, tokenizer = train()
    test_model(model, tokenizer)
