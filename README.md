# Fake News Detection: LSTM to LLM

Extension of my own published LSTM-based fake news detection study, upgraded with a corrected dataset, optimised model architecture, and Transformer-based NLP methods.

## Background

This project extends my own prior publication:

> Liu, S. (2023). *Social Media Fake Information Identification Method Based on LSTM*. Highlights in Business, Economics and Management, Vol. 21, pp. 703–709. (GEFHR 2023)

My original paper achieved 99% accuracy and 100% AUC on the Kaggle Fake and Real News Dataset using an LSTM model. This project:

1. **Reproduces** my original LSTM pipeline in a clean, documented codebase
2. **Critically re-examines** the dataset and identifies source-leakage bias that inflated the reported metrics
3. **Optimises** the model with a corrected dataset and improved architecture (Bidirectional LSTM), achieving 97.9% accuracy and 99.7% AUC on clean data
4. **Extends** the work further with Transformer (DistilBERT) and LLM prompting experiments

## Project Structure

```
fake-news-detection-lstm-to-llm/
├── data/
│   ├── raw/              # Fake.csv, True.csv (Kaggle dataset)
│   └── README.md         # Dataset description and known bias notes
├── notebooks/
│   ├── 00_eda.ipynb                    # EDA + dataset bias analysis
│   ├── 01_lstm_reimplementation.ipynb  # Reproduction of my original paper
│   ├── 02_lstm_corrected.ipynb         # Optimised LSTM on debiased data
│   ├── 03_transformer_distilbert.ipynb # DistilBERT fine-tuning
│   ├── 04_llm_prompting.ipynb          # Zero-shot / few-shot prompting
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
| `01_lstm_reimplementation` | Reproduce my original paper (Tokenizer → Embedding → LSTM → Dense → Sigmoid) |
| `02_lstm_corrected` | Optimised Bidirectional LSTM on debiased data; nine-step cleaning pipeline |
| `03_transformer_distilbert` | Fine-tune DistilBERT without leaking features |
| `04_llm_prompting` | LLM zero-shot & few-shot on LSTM error cases; reasoning traces |
| `05_error_analysis` | Cross-model FP/FN breakdown; dataset-artifact investigation |

## Results Summary

| Model | Accuracy | AUC | Notes |
|-------|----------|-----|-------|
| **LSTM (my paper)** | **99.1%** | **99.8%** | Original result — includes dataset leakage |
| **Optimised LSTM** | **97.9%** | **99.7%** | Bidirectional LSTM on debiased data |
| DistilBERT | — | — | In progress |
| LLM | — | — | In progress |

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
