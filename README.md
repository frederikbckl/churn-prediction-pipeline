# Customer Churn Prediction Pipeline

Customer churn prediction pipeline for TUM course "Business Analytics and Machine Learning" (midterm assignment).
Merges several relational CSVs on a shared customer ID. 
Preprocessing, classification pipeline, decision threshold on balanced accuracy.

## What it does

1. **Merge**: left-joins csv files on id
2. **Split**: rows with a non-null `churn` value are the training set. Rows with a null value are being predicted
3. **Clean**: drops categorical columns with too many unique values relative to row count
4. **Preprocess**: median-imputes + scales numeric columns, most-frequent-imputes + one-hot-encodes categoricals, via an `sklearn` `ColumnTransformer`
5. **Train**: uses a configurable classifier (`LogisticRegression`, `RandomForestClassifier`, or `HistGradientBoostingClassifier`)
6. **Tune**: decision-threshold grid on a validation split. Pick the one that maximizes balanced accuracy
7. **Predict**: retrains on all labeled data and writes `submission.csv`


Started as a script for a business analytics and machine learning course assignment and was cleaned up afterward for general use.