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
