# Fake News Detection: LSTM to LLM

Reimplementation and extension of a published LSTM-based fake news detection study, upgraded with Transformer and LLM-based NLP methods.

## Background

This project is based on:

> Liu, S. (2023). *Social Media Fake Information Identification Method Based on LSTM*. Highlights in Business, Economics and Management, Vol. 21, pp. 703–709. (GEFHR 2023)

The original paper achieved 99% accuracy and 100% AUC on the Kaggle Fake and Real News Dataset using an LSTM model. This project:

1. **Reproduces** the original LSTM pipeline in a clean, documented codebase
2. **Extends** it with Transformer (DistilBERT) and LLM prompting experiments
3. **Critically analyses** the dataset's known source-leakage bias and its effect on reported metrics

## Project Structure

```
fake-news-detection-lstm-to-llm/
├── data/
│   ├── raw/              # Fake.csv, True.csv (Kaggle dataset)
│   └── README.md         # Dataset description and known bias notes
├── notebooks/
│   ├── 00_eda.ipynb                    # EDA + dataset bias analysis
│   ├── 01_lstm_reimplementation.ipynb  # Paper reproduction
│   ├── 02_baseline_comparison.ipynb    # Naive Bayes, LR, RF, KNN
│   ├── 03_transformer_distilbert.ipynb # DistilBERT fine-tuning
│   ├── 04_llm_prompting.ipynb          # Zero-shot / few-shot with Claude
│   └── 05_error_analysis.ipynb         # FP/FN analysis across all models
├── src/                  # Shared utility functions
├── results/              # Saved metrics, plots, error cases
├── reports/              # Written summaries
└── paper/                # Reference to original publication
```

## Notebooks Overview

| Notebook | Purpose |
|----------|---------|
| `00_eda` | Explore data distribution; confirm subject-column and Reuters-byline leakage |
| `01_lstm_reimplementation` | Reproduce original paper (Tokenizer → Embedding → LSTM → Dense → Sigmoid) |
| `02_baseline_comparison` | TF-IDF + Naive Bayes / Logistic Regression / Random Forest / KNN |
| `03_transformer_distilbert` | Fine-tune DistilBERT without leaking features |
| `04_llm_prompting` | Claude zero-shot & few-shot on LSTM error cases; reasoning traces |
| `05_error_analysis` | Cross-model FP/FN breakdown; dataset-artifact investigation |

## Results Summary

| Model | Accuracy | AUC |
|-------|----------|-----|
| Naive Bayes | 93% | 98% |
| Logistic Regression | — | — |
| Random Forest | 86% | 94% |
| KNN | 91% | 97% |
| **LSTM (paper)** | **99%** | **100%** |
| DistilBERT | — | — |
| LLM (Claude) | — | — |

*Results will be updated as experiments complete.*

## Data

Download the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) and place `Fake.csv` and `True.csv` under `data/raw/`. See [data/README.md](data/README.md) for details on known dataset bias.

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

## Key Findings (in progress)

- The `subject` column is a near-perfect classifier on its own, suggesting the 99% LSTM accuracy partially reflects dataset artifacts rather than genuine language understanding.
- DistilBERT trained without leaking features will reveal how much accuracy is attributable to content alone.
- LLM prompting provides reasoning traces for cases where all other models fail.
