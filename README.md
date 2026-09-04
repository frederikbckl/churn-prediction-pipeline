# Customer Churn Prediction Pipeline

A binary classification pipeline that merges several relational CSVs on a
shared customer ID, builds a preprocessing + model pipeline, tunes a decision
threshold for balanced accuracy, and writes predictions for whichever rows
are missing a label.

## What it does

1. **Merge** — left-joins `customers`, `payment_info`, `service_options`, and
   `churn_analysis` (the label source) on `customer_id`.
2. **Split** — rows with a non-null `churn` value become the training set;
   rows with a null value are the ones to predict.
3. **Clean** — drops categorical columns that are effectively free text or
   disguised IDs (too many unique values relative to row count).
4. **Preprocess** — median-imputes + scales numeric columns, most-frequent-imputes
   + one-hot-encodes categoricals, via an `sklearn` `ColumnTransformer`.
5. **Train** — fits a configurable classifier (`LogisticRegression`,
   `RandomForestClassifier`, or `HistGradientBoostingClassifier`).
6. **Tune** — sweeps a decision-threshold grid on a held-out validation split
   and picks the one that maximizes balanced accuracy (useful when churn is
   imbalanced and 0.5 isn't the right cutoff).
7. **Predict** — retrains on all labeled data and writes `submission.csv`
   with `id, prediction` columns.

## Usage

```bash
pip install -r requirements.txt
python baml_churn_pipeline.py
```

Edit the constants at the top of `baml_churn_pipeline.py` to point at your
own files and choose a model:

```python
DATA_PATHS = ["data/churn_analysis.csv", "data/customers.csv", ...]
MODEL = "hgb"  # "logreg", "rf", or "hgb"
```

## Expected input schema

Each CSV must share a `customer_id` column. One of them — the "label" file —
must also contain a `churn` column that is `0`/`1` for known rows and empty
for the rows you want predictions on. Beyond `customer_id` and `churn`, any
number of numeric or categorical feature columns is supported; datetime-like
columns are treated as high-cardinality categoricals and dropped unless you
extend `drop_high_cardinality_categoricals`.

No sample data is included in this repo — the dataset this was built against
is coursework material and isn't mine to redistribute. Point `DATA_PATHS` at
your own CSVs matching the schema above.

## Notes

This started as a script for a business analytics / ML course assignment
and was cleaned up afterward (removed dead code, generalized file paths,
tidied variable names) for use as a general-purpose churn-prediction
starting point.
