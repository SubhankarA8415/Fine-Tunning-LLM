# 📘 Fine-Tuning a Large Language Model for Academic Question Answering

This repository presents a **complete, end-to-end implementation of fine-tuning Large Language Models (LLMs)** using a **custom academic, textbook-style dataset**.  
The project focuses on correctness, stability, and reproducibility rather than raw performance.

The workflow begins with a **DistilGPT-2** baseline experiment and then transitions to a **FLAN-T5 Base instruction-tuned model** to achieve reliable and academically aligned outputs.

---

## 🎯 Objectives

- Demonstrate supervised fine-tuning of LLMs on a custom dataset
- Apply **instruction tuning** for academic-style question answering
- Ensure **numerically stable training** (no NaN / zero-loss collapse)
- Compare limitations of small causal models vs encoder–decoder models
- Produce deterministic, formal, textbook-style answers

---

## 🧠 Key Concepts Covered

- Supervised Fine-Tuning (SFT)
- Instruction tuning
- JSONL dataset validation
- Encoder–Decoder (Seq2Seq) architecture
- Safe tokenization with EOS enforcement
- Decoder label masking using `-100`
- Gradient clipping & NaN-safe training
- Controlled inference decoding

---

## 📂 Dataset Overview

| Property | Value |
|--------|------|
| Format | JSON Lines (`.jsonl`) |
| Domain | Large Language Models |
| Style | Academic / Textbook |
| Initial Samples | ~60 |
| Final Samples | ~862 |
| Structure | User Question → Assistant Answer |

### Sample Format
```json
{
  "messages": [
    { "role": "user", "content": "What is a Large Language Model?" },
    { "role": "assistant", "content": "A Large Language Model is a neural network..." }
  ]
}
```
---

## 🧪 Model Experiments

### Phase 1 – DistilGPT-2 (Exploratory)

- Used to demonstrate the fine-tuning pipeline
- Observed limitations:
  - Small model capacity
  - Shallow or collapsed outputs
  - Inconsistent academic depth

### Phase 2 – FLAN-T5 Base (Final Model)

- Instruction-tuned encoder–decoder architecture
- Stable loss behavior
- Consistent academic tone
- Deterministic and reliable inference

---

## ⚙️ Tech Stack

- Python 3.9+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Accelerate
- Matplotlib
- Google Colab (Tesla T4 GPU)

---

## 🔐 Stability & Safety Measures

- Strict JSONL validation before training
- Instruction-based input formatting
- Safe tokenization with forced EOS tokens
- Decoder label masking (`label_pad_token_id = -100`)
- Gradient clipping (max norm = 1.0)
- NaN loss detection with hard-stop
- Deterministic inference configuration

---

## 📉 Training Results (FLAN-T5)

| Epoch | Train Loss | Validation Loss |
|------|-----------|----------------|
| 1 | 3.5607 | 2.9902 |
| 2 | 3.1534 | 2.8599 |
| 3 | 3.0049 | 2.7987 |
| 4 | 2.9242 | 2.7623 |
| 5 | 2.8563 | 2.7407 |
| 6 | 2.8300 | 2.7353 |

### Observations

- Consistent loss reduction
- No NaN or zero-loss collapse
- Stable convergence by Epoch 5–6
- Good generalization behavior

---

## 🔍 Inference Capabilities

- Academic, textbook-style answers
- Deterministic decoding (no randomness)
- Beam search for improved coherence
- Repetition control
- Controlled output length

### Example

```python
generate_academic_answer_controlled(
    "What is instruction tuning in Large Language Models?"
)
```
---

## 💾 Model Saving & Reloading

The fine-tuned model and tokenizer are saved using Hugging Face format:

```python
model.save_pretrained("flan_t5_academic_final")
tokenizer.save_pretrained("flan_t5_academic_final")
```
---

## Supports:

- Reloading without retraining
- Google Drive persistence
- ZIP-based backup and restoration

---

## 📁 Project Structure

```
├── training_data_small.jsonl
├── training_data.jsonl
├── training_data_clean.jsonl
├── training_data_final.jsonl
├── fine_tuning_notebook.ipynb
├── flan_t5_academic_final/
└── README.md
```

---

## 🏁 Final Conclusion

- Fine-tuning was successful, stable, and reproducible
- Instruction-tuned FLAN-T5 significantly outperformed small causal models
- Proper data validation and label masking were critical
- The final model is suitable for academic question answering and evaluation

---

## 🚀 Future Work

- Larger and more diverse academic datasets
- Domain-specific textbooks
- Automatic evaluation (BLEU / ROUGE)
- API or chatbot deployment
- Parameter-efficient fine-tuning (LoRA / PEFT)

---

## 📜 License

This project is intended for **educational and academic purposes only**.
