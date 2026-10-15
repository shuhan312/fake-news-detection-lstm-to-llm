# Fake News Detection: LSTM to LLM

Extension of my own published LSTM-based fake news detection study, upgraded with a corrected dataset, optimised model architecture, and Transformer-based NLP methods.

## Background

This project extends my own prior publication:

> Liu, S. (2023). *Social Media Fake Information Identification Method Based on LSTM*. Highlights in Business, Economics and Management, Vol. 21, pp. 703–709. (GEFHR 2023)

My original paper achieved 99% accuracy and 100% AUC on the Kaggle Fake and Real News Dataset using an LSTM model. This project:

1. **Reproduces** my original LSTM pipeline in a clean, documented codebase
2. **Critically re-examines** the dataset and identifies source-leakage bias that inflated the reported metrics
3. **Re-evaluates** LSTM, BiLSTM and DistilBERT on a cleaned, de-duplicated dataset with 3 seeds each, both inside ISOT and on news from other sources (cross-dataset)
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
│   ├── 04_leakage_audit.ipynb          # Shortcut probes + split-integrity checks
│   ├── 05_cross_dataset.ipynb          # External test sets (WELFake minus ISOT, LIAR)
│   ├── 06_llm_prompting.ipynb          # Zero-shot / few-shot prompting
│   └── 07_error_analysis.ipynb         # FP/FN analysis across all models
├── src/
│   ├── data.py           # Shared pipeline: symmetric cleaning, de-dup, grouped split
│   ├── external.py       # External datasets: download, MD5 check, same cleaning, ISOT-overlap removal
│   ├── baselines.py      # TF-IDF + logistic regression, C selection, metrics, bootstrap CIs
│   ├── probes.py         # Leakage-audit tools (artefact rates, shortcut probes)
│   └── utils.py          # set_seed()
├── tests/                # Unit tests for the cleaning rules (python -m pytest tests/)
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
| `04_leakage_audit` | Audit of the shared pipeline: residual artefacts, duplicate overlap, shortcut probes |
| `05_cross_dataset` | Does an ISOT-trained model transfer to other sources? External sets: WELFake minus ISOT, LIAR |
| `06_llm_prompting` | LLM zero-shot & few-shot on LSTM error cases; reasoning traces |
| `07_error_analysis` | Cross-model FP/FN breakdown; dataset-artifact investigation |

## Results Summary

Accuracy on the ISOT test set (in-domain) and on WELFake minus ISOT (cross-dataset: real news from other outlets, never Reuters). Mean ± std over 3 seeds; TF-IDF is deterministic. WELFake's majority baseline (always "real") is **56.1%**.

| Model | ISOT test acc. | ISOT AUC | WELFake acc. | WELFake AUC | Notebook |
|-------|---------------:|---------:|-------------:|------------:|----------|
| LSTM, my paper (original run, leaky data) | 99.1% | 0.998 | — | — | `01` |
| TF-IDF + logistic regression (baseline) | 98.7% | 0.999 | 56.7% | **0.714** | `05` |
| LSTM, paper settings (A0) | 93.5 ± 1.0% | 0.975 | 54.9 ± 5.0% | 0.628 | `02` |
| BiLSTM, old notebook 02 settings (A4) | 94.2 ± 2.6% | 0.979 | 56.6 ± 1.2% | 0.647 | `02` |
| BiLSTM, tuned (A6) | 97.6 ± 0.4% | 0.997 | 56.4 ± 0.2% | 0.673 | `02` |
| DistilBERT | **99.0 ± 0.3%** | 0.999 | 58.6 ± 0.7% | 0.693 | `03` |
| LLM | — | — | — | — | in progress |

Every model scores 93–99% inside ISOT but barely beats the majority baseline on other sources.

## Data

Download the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) and place `Fake.csv` and `True.csv` under `data/raw/`. See [data/README.md](data/README.md) for details on known dataset bias.

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests/    # check the cleaning rules
jupyter lab
```

All notebooks should load data through the shared pipeline so every model sees the same cleaned text and the same split:

```python
from src.data import load_dataset, get_splits
train, val, test = get_splits(load_dataset())   # models read the `model_text` column
```

The split is locked in `data/processed/splits.csv` (committed); the cleaned-text cache next to it is rebuilt automatically when `src/data.py` or the raw data changes.

## Key Findings

- **ISOT's labels equal its sources** (all real news is Reuters). After removing formatting artefacts and duplicates, a bag-of-words model still reaches 94% from the title alone, by recognising writing style (`04_leakage_audit`).
- **Nothing transfers**: TF-IDF, LSTM, BiLSTM and DistilBERT all drop from 93–99% inside ISOT to 52–59% on other outlets (majority baseline 56.1%). The ISOT model calls most New York Times articles fake (`05`, `02`, `03`).
- **Architecture is not the bottleneck**: DistilBERT ties TF-IDF in-domain and transfers no better (its WELFake AUC is reliably lower). The old notebook 02 "optimisations" are mostly within seed noise; the largest single gain is keeping the start of each article (Keras keeps the end by default).
- Next: methods that reduce reliance on source style (content-only input, de-biasing, multi-source training with a held-out outlet), then evidence-based LLM fact-checking.
