# Question Answering with HuggingFace - Tutorial Activity

## 📚 Overview

This project demonstrates extractive question answering by fine-tuning BERT on the SQuAD (Stanford Question Answering Dataset). The model learns to identify answer spans within a given context.

## 📓 Notebook

- [`qa.ipynb`](./qa.ipynb) - Complete QA pipeline with training and evaluation

## 🎯 Task

Given a context passage and a question, extract the exact answer span from the context.

**Example:**
```
Context: 🤗 Transformers is backed by the three most popular deep learning 
         libraries — Jax, PyTorch and TensorFlow...

Question: Which deep learning libraries back 🤗 Transformers?

Answer: Jax, PyTorch and TensorFlow
```

## 📊 Dataset

- **Dataset:** SQuAD (Stanford Question Answering Dataset)
- **Training Examples:** 87,599
- **Validation Examples:** 10,570
- **Task Type:** Extractive QA

## 🛠️ Model & Training

| Parameter | Value |
|-----------|-------|
| Base Model | bert-base-cased |
| Max Sequence Length | 384 |
| Stride (Overlap) | 128 |
| Batch Size | 8 |
| Learning Rate | 2e-5 |
| Epochs | 3 |
| Optimizer | AdamW |
| Scheduler | Linear |
| Mixed Precision | FP16 |

## 📈 Results

| Epoch | Exact Match | F1 Score |
|-------|-------------|----------|
| 0 | 79.66% | 87.50% |
| 1 | 81.30% | 88.65% |
| 2 | 81.17% | 88.66% |

## 📦 Dependencies

```bash
pip install transformers datasets evaluate accelerate
```

## 🔑 Key Concepts Implemented

1. **Tokenization with Stride** - Handling long contexts with overlapping chunks
2. **Offset Mapping** - Mapping tokens back to original character positions
3. **Start/End Position Labels** - Supervised learning for span extraction
4. **N-best Predictions** - Selecting best answer from multiple candidates
5. **Accelerate Integration** - Distributed training with mixed precision

## 💻 Inference Example

```python
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="bert-finetuned-squad-accelerate",
    device_map="auto"
)

result = qa_pipeline(
    question="Which deep learning libraries back Transformers?",
    context="Transformers is backed by Jax, PyTorch and TensorFlow."
)
print(result["answer"])  # Jax, PyTorch and TensorFlow
```

## 👥 Team Contributions

All team members contributed equally to this tutorial.
