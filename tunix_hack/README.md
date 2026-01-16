# Google Tunix Hackathon - Competition Entry

## 🏆 Competition

**Kaggle Competition:** [Google Tunix Hackathon](https://www.kaggle.com/competitions/google-tunix-hackathon)

## 📝 Our Writeup

**Final Submission:** [View on Kaggle](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1767829089071)

---

## 🎯 Task

Fine-tune Google's Gemma model to output structured reasoning traces:

```
<reasoning>step-by-step thinking</reasoning>
<answer>final answer</answer>
```

## 🛠️ Approach

| Component | Details |
|-----------|---------|
| **Base Model** | google/gemma-2-2b-it |
| **Method** | LoRA (Low-Rank Adaptation) |
| **Quantization** | 4-bit NF4 |
| **Training Data** | ~570k samples |
| **Training Time** | ~6 hours on A100 40GB |

## 📊 Datasets Used

- GSM8K (Math word problems)
- OpenThoughts-114k (R1 distilled reasoning)
- Bespoke-Stratos-17k (High quality R1)
- Medical-O1 (Medical reasoning)
- MetaMathQA (Augmented math)

## 📁 Project Structure

```
tunix_hack/
├── README.md              # This file
├── TASK_SUMMARY.md        # Detailed task summary
└── notebooks/
    ├── train_a100.py      # Training script (A100)
    ├── train_kaggle_fixed.py
    ├── tunixhack.ipynb    # Main Kaggle notebook
    ├── inference.py       # Inference script
    └── gemma-reasoning/   # Trained LoRA adapter
```

## 👥 Team Contributions

All team members contributed equally to this competition.
