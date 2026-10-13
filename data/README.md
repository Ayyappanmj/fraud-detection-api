# data/

Nothing in this folder is committed except this file (see `.gitignore`).

## Default: synthetic data (no download needed)

```bash
python -m src.data_generator --output data/synthetic_transactions.csv
python -m src.data_validation --data-path data/synthetic_transactions.csv
python -m src.train --data-path data/synthetic_transactions.csv
```

All rows are **SYNTHETIC** (`data_source = synthetic`). They contain no real cardholder data.
If you run `python -m src.train` with no `--data-path`, the same generator is used in memory.

## Optional: your own labelled CSV

Required columns: `transaction_id, timestamp, amount, hour, merchant_category, distance_from_home_km,
txn_count_24h, avg_amount_30d, is_online, is_foreign, card_age_days, is_fraud` (0/1).
Run `python -m src.data_validation` first; training refuses invalid data.

## About the Kaggle "Credit Card Fraud Detection" dataset

This project does **not** use or redistribute it. It is a popular public dataset (ULB / Worldline research
collaboration, released under the Open Database License, ODbL 1.0) whose columns are anonymised PCA
components (`V1`..`V28`), so it has a different schema from this project. Downloading it requires a free
Kaggle account and acceptance of Kaggle's terms. If you adapt the pipeline to it, **never commit the CSV**
(it is ~150 MB and covered by its own licence terms) and check the licence on the dataset page yourself.
