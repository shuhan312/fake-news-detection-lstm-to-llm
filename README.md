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
│   ├── 02_lstm_corrected.ipynb         # LSTM / BiLSTM ablation on the shared pipeline (3 seeds)
│   ├── 03_transformer_distilbert.ipynb # DistilBERT on the shared pipeline (3 seeds)
│   ├── 04_leakage_audit.ipynb          # Shortcut probes + split-integrity checks
│   ├── 05_cross_dataset.ipynb          # External test sets (WELFake minus ISOT, LIAR)
│   ├── 06_reducing_source_reliance.ipynb # Step 4: content words, de-biasing, multi-source training
│   ├── 07_liar_benchmark.ipynb         # Fact-checked statements: the difficulty without the source shortcut
│   ├── 08_llm_prompting.ipynb          # Zero-shot / few-shot prompting (planned)
│   └── 09_error_analysis.ipynb         # FP/FN analysis across all models (planned)
├── src/
│   ├── data.py           # Shared pipeline: symmetric cleaning, de-dup, grouped split
│   ├── external.py       # External datasets: download, MD5 check, same cleaning, ISOT-overlap removal
│   ├── baselines.py      # TF-IDF + logistic regression, C selection, metrics, bootstrap CIs
│   ├── lstm_model.py     # LSTM / BiLSTM ablation ladder (A0-A5) and tuned A6
│   ├── distilbert_model.py # DistilBERT fine-tuning (optional per-article weights)
│   ├── experiments.py    # Resumable runner: one JSON + NPZ per model x config x seed
│   ├── analysis.py       # Publisher breakdown, paired bootstrap AUC tests
│   ├── step4.py          # Step 4: content-only input, de-biased reweighting, leave-outlet-out
│   ├── liar.py           # LIAR benchmark (statement text only)
│   ├── probes.py         # Leakage-audit tools (artefact rates, shortcut probes)
│   └── utils.py          # set_seed()
├── tests/                # Unit tests (python -m pytest tests/)
├── results/              # Saved metrics, plots, error cases
├── reports/              # Written summaries
└── paper/                # Reference to original publication
```

## Notebooks Overview

| Notebook | Purpose |
|----------|---------|
| `00_eda` | Explore data distribution; confirm subject-column and Reuters-byline leakage |
| `01_lstm_reimplementation` | Reproduce my original paper (Tokenizer → Embedding → LSTM → Dense → Sigmoid) |
| `02_lstm_corrected` | LSTM / BiLSTM ablation (A0 paper → A5, plus tuned A6), 3 seeds, in-domain and cross-dataset |
| `03_transformer_distilbert` | DistilBERT on the shared pipeline, 3 seeds; paired tests against TF-IDF |
| `04_leakage_audit` | Audit of the shared pipeline: residual artefacts, duplicate overlap, shortcut probes |
| `05_cross_dataset` | Does an ISOT-trained model transfer to other sources? External sets: WELFake minus ISOT, LIAR |
| `06_reducing_source_reliance` | Step 4: content words only, de-biased reweighting, multi-source training with a held-out outlet (TF-IDF, then DistilBERT) |
| `07_liar_benchmark` | LIAR fact-checked statements: how well text alone detects falsehood once the source shortcut is gone |
| `08_llm_prompting` | LLM fact-checking on LIAR (Claude Haiku 4.5): without evidence, with retrieved Wikipedia evidence (`src/evidence.py`), a random-evidence control (`src/llm_factcheck.py`), and the LLM combined with the text models (`src/ensemble.py`) |
| `09_error_analysis` | Planned: cross-model FP/FN breakdown |

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
| LLM | — | — | — | — | LIAR only (`08`) |

Every model scores 93–99% inside ISOT but barely beats the majority baseline on other sources.

**Reducing reliance on source style (`06`)** — test set: unseen New York Times real articles + fake articles.

| Training data | Model | NYT accepted as real | Fakes accepted as real | AUC |
|---------------|-------|---------------------:|-----------------------:|----:|
| ISOT only | TF-IDF | 43.2% | 9.2% | 0.790 |
| ISOT only | DistilBERT (3 seeds) | 43–47% | 10–14% | 0.76–0.78 |
| ISOT + WELFake, NYT held out | TF-IDF | 84.5% | 5.1% | 0.968 |
| ISOT + WELFake, NYT held out | DistilBERT (3 seeds) | 74–82% | 2–3% | **0.975** |

Removing style words or de-biased reweighting (ISOT only) made transfer worse.

**Fact-checked statements (`07`, LIAR test, majority baseline 56.7%)**

| Model | Accuracy | AUC |
|-------|---------:|----:|
| TF-IDF | 62.5% | 0.672 |
| DistilBERT (3 seeds) | 64.3 ± 0.8% | 0.682 ± 0.006 |
| Claude Haiku 4.5, no evidence (zero-shot) | 61.1% | 0.655 |
| Claude Haiku 4.5 + Wikipedia evidence | 53.5% | 0.647 |
| Claude Haiku 4.5 + another statement's evidence (control) | 49.9% | 0.613 |
| **DistilBERT + TF-IDF + Haiku, combined on val (3 seeds)** | **66.4–67.5%** | **0.725–0.729** |

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
- **Changing the data helps; changing the model barely does** (`06`): removing style words or de-biased reweighting made transfer worse, because source information sits in both style (style words alone: 96.6% on ISOT) and topic (content words alone: 97.8%). Training on several sources raised AUC on an unseen outlet by about 0.2, while DistilBERT added only 0.007 over TF-IDF.
- **Without the source shortcut, text alone is weak** (`07`): on fact-checked LIAR statements, TF-IDF and DistilBERT reach only AUC 0.67–0.68 (62–64% accuracy against a 56.7% baseline), and DistilBERT is not reliably better than TF-IDF.
- **An LLM's own knowledge is not enough either** (step 5a): Claude Haiku 4.5 judging LIAR statements without evidence reaches AUC 0.655, statistically indistinguishable from TF-IDF and DistilBERT. That it does not score higher also suggests it is not simply recalling PolitiFact verdicts.
- **Wikipedia evidence does not raise the ceiling** (step 5b): with three retrieved passages per statement Haiku reaches AUC 0.647, no different from 0.655 without evidence (paired difference −0.008, 95% CI −0.034 to 0.018). A control that shows each statement another statement's passages scores 0.613, so the right evidence is reliably better than unrelated evidence (+0.034, CI 0.008 to 0.060), but adding passages also makes the model sceptical: it calls only 21% of statements true (vs 57% in the data), treating "not in the passages" as "false". In a manual check only 3 of 30 passages contained the fact needed to judge the statement.
- **Combining the LLM with the text models breaks the ceiling** (step 5c): Haiku's scores are almost uncorrelated with the text models' (Spearman 0.12–0.17 on val, vs 0.73–0.77 between TF-IDF and DistilBERT). A logistic-regression combiner, chosen and fitted on LIAR val only, reaches test AUC 0.725–0.729 and 66.4–67.5% accuracy, reliably above DistilBERT for every seed (+0.042 to +0.050, 95% CIs exclude 0). Language patterns and world knowledge are complementary; a bigger model on its own is not the answer.
- Next: check on val whether a prompt that does not penalise missing evidence or a larger model improves the combination further, then error analysis (`09`).
