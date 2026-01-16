#!/usr/bin/env python3
"""
Google Tunix Hack - Model Inference Script
Test the fine-tuned Gemma reasoning model

Author: Om Borda (omborda2002)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os

print("=" * 60)
print("🧠 TUNIX HACK - MODEL INFERENCE")
print("=" * 60)

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "./gemma-reasoning"  # Path to LoRA adapter
BASE_MODEL = "google/gemma-2-2b-it"

# ============================================
# LOAD MODEL
# ============================================
print("\n📥 Loading model...")

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Get HF token
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# Load base model
print("   Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=hf_token,
)

# Load tokenizer
print("   Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# Load LoRA adapter
print("   Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

print("✓ Model loaded!\n")

# ============================================
# INFERENCE FUNCTION
# ============================================
def generate_response(question, max_tokens=500, temperature=0.7):
    """Generate a response with reasoning."""
    prompt = f"<start_of_turn>user\n{question}\n<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract model response
    if "<start_of_turn>model" in response:
        response = response.split("<start_of_turn>model")[-1].strip()
    
    return response

# ============================================
# SAMPLE TEST QUESTIONS
# ============================================
SAMPLE_QUESTIONS = [
    # Math - Basic
    "What is 125 + 347?",
    "What is 15 × 12?",
    "What is 144 ÷ 12?",
    
    # Math - Algebra
    "Solve: 2x + 5 = 13",
    "Solve for y: 3y - 7 = 14",
    "If 5x = 45, what is x?",
    
    # Math - Word Problems
    "A train travels 240 km in 4 hours. What is its speed?",
    "Sarah has 24 apples. She gives 1/3 to her friend. How many apples does she have left?",
    "A rectangle has length 8 cm and width 5 cm. What is its area?",
    
    # Probability
    "What is the probability of rolling a 6 on a fair die?",
    "If you flip a coin twice, what is the probability of getting two heads?",
    
    # Medical
    "What are the common symptoms of diabetes?",
    "What causes high blood pressure?",
    
    # Logic
    "If all cats are animals, and Whiskers is a cat, what can we conclude?",
    "What comes next in the sequence: 2, 4, 8, 16, ?",
]

# ============================================
# RUN INFERENCE
# ============================================
print("=" * 60)
print("🧪 MODEL EVALUATION - SAMPLE OUTPUTS")
print("=" * 60)

results = []

for i, question in enumerate(SAMPLE_QUESTIONS, 1):
    print(f"\n{'='*60}")
    print(f"📝 Question {i}/{len(SAMPLE_QUESTIONS)}:")
    print(f"   {question}")
    print("-" * 60)
    
    response = generate_response(question)
    print(f"🤖 Response:")
    print(response)
    
    results.append({
        "question": question,
        "response": response
    })

# ============================================
# SAVE RESULTS
# ============================================
print("\n" + "=" * 60)
print("💾 SAVING RESULTS")
print("=" * 60)

# Save to file
output_file = "sample_outputs.txt"
with open(output_file, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("GOOGLE TUNIX HACK - MODEL SAMPLE OUTPUTS\n")
    f.write("Author: Om Borda (omborda2002)\n")
    f.write("Model: Gemma 2B IT + LoRA Fine-tuned\n")
    f.write("=" * 80 + "\n\n")
    
    for i, result in enumerate(results, 1):
        f.write(f"{'='*80}\n")
        f.write(f"QUESTION {i}:\n")
        f.write(f"{result['question']}\n")
        f.write(f"{'-'*80}\n")
        f.write(f"RESPONSE:\n")
        f.write(f"{result['response']}\n")
        f.write("\n")

print(f"✅ Results saved to: {output_file}")

# ============================================
# INTERACTIVE MODE
# ============================================
print("\n" + "=" * 60)
print("💬 INTERACTIVE MODE")
print("   Type your question and press Enter")
print("   Type 'quit' or 'exit' to stop")
print("=" * 60)

while True:
    try:
        user_input = input("\n📝 Your question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        print("\n🤖 Response:")
        response = generate_response(user_input)
        print(response)
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
