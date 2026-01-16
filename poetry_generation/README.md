# Poetry Generation - Poetry Slam Activity

## 🎭 Overview

This project fine-tunes GPT-2 for constrained poetry generation. Given a set of 4 keywords, the model generates coherent lyrical poems that incorporate all specified words.

## 📓 Notebook

- [`poet_generation.ipynb`](./poet_generation.ipynb) - Complete training and inference pipeline

## 🎯 Task

Generate short, coherent lyrical poems using ALL provided keywords.

**Example Prompt:**
```
Words: love, moon, dream, silence
Write a short, coherent lyrical poem using ALL of these words.
Poem:
```

## 📊 Dataset

- **Source:** PoemDataset.csv
- **Poems Used:** ~4,545 samples (filtered)
- **Max Examples:** 8,000
- **Filtering Criteria:**
  - Length: 40-800 characters
  - At least 4 content words (≥4 chars)

## 🛠️ Model & Training

| Parameter | Value |
|-----------|-------|
| Base Model | GPT-2 |
| Max Sequence Length | 256 |
| Batch Size | 2 |
| Gradient Accumulation | 4 |
| Device | CUDA (GPU) |

## 📦 Dependencies

```bash
pip install transformers datasets accelerate
```

## 🚀 Usage

1. Open the notebook in Google Colab or Jupyter
2. Upload your poem dataset (PoemDataset.csv)
3. Run all cells to train the model
4. Generate poems with custom keyword prompts

## 💡 Key Features

- **Keyword Extraction:** Automatically extracts 4 content words from each poem
- **Prompt Engineering:** Structured prompts ensure keyword inclusion
- **Causal Language Modeling:** Standard autoregressive generation

## 👥 Team Contributions

All team members contributed equally to this activity.
