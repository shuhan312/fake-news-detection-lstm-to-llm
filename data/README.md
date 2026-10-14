# Data

## Source

Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)  
Originally used in: *Social Media Fake Information Identification Method Based on LSTM* (Liu, GEFHR 2023)

## Files

| File | Rows | Label |
|------|------|-------|
| `raw/Fake.csv` | 23,481 | 0 (fake) |
| `raw/True.csv` | 21,417 | 1 (real) |

## Columns

| Column | Description |
|--------|-------------|
| `title` | Headline of the article |
| `text` | Full body text |
| `subject` | Category tag (e.g. politicsNews, News) |
| `date` | Publication date |

## Known Bias

This dataset has a well-documented source-leakage issue:

- **`True.csv`**: Almost all articles originate from Reuters and contain the `(Reuters)` byline in the text body.
- **`subject` column**: Real news uses `politicsNews` / `worldNews`; fake news uses `News` / `politics` / etc. The subject column alone is nearly a perfect classifier.

Any model trained on this dataset without removing these signals may learn source style rather than content truthfulness. Experiment notebooks control for this explicitly.

## Processed data

`src/data.py` builds the cleaned dataset used by every notebook:

| File | In git | Content |
|------|--------|---------|
| `processed/splits.csv` | yes | `doc_id` → `train` / `val` / `test`. Locks the split so all models share it. |
| `processed/clean_dataset.csv` | no | Cached cleaned text; rebuilt automatically when `src/data.py` or the raw CSVs change. |

`doc_id` is `fake_<row>` / `true_<row>`, the row number in the raw CSV. The cleaning rules and their rationale are documented in `src/data.py`; `notebooks/04_leakage_audit.ipynb` measures what is left after cleaning.

## External datasets

Used to test whether models trained on ISOT transfer to other sources (`notebooks/05_cross_dataset.ipynb`). `src/external.py` downloads them into `external/` (not in git), verifies their MD5, and processes them with the same cleaning as ISOT.

| Dataset | Use | Licence | Notes |
|---------|-----|---------|-------|
| **WELFake** — Verma, Agrawal & Prodan (2021), *IEEE Transactions on Computational Social Systems*; Zenodo [10.5281/zenodo.4561253](https://doi.org/10.5281/zenodo.4561253) | Primary external test set, after removing every article linked to ISOT | CC-BY-4.0 | **Labels in the CSV are the reverse of the Zenodo description** (raw `1` = fake); `src/external.py` converts and re-verifies them against the ISOT overlap on every build. |
| **LIAR** — Wang (2017), "Liar, Liar Pants on Fire", ACL 2017 | Secondary: individually fact-checked short statements | Research use only | 6-way labels binarised (true / mostly-true / half-true → real); original labels kept in `label6`. |

