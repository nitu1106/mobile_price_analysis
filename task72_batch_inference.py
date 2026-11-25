"""
Task 3: Batch prediction for a CSV containing many phones.
Run: python task3_batch_inference.py phones_to_predict.csv
Output: batch_predictions.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

TRAIN_FILE = "train.csv"
BATCH_SIZE = 5000
OUTPUT_FILE = "batch_predictions.csv"

def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "phone_to_predict.csv"
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Cannot find '{input_file}'. Provide a CSV filename, for example: "
            "python task72_batch_inference.py phone_to_predict.csv"
        )

    train = pd.read_csv(TRAIN_FILE)
    if "price_range" not in train.columns:
        raise ValueError("train.csv must contain the 'price_range' column.")
    train = train.drop(columns=["id"], errors="ignore")
    X_train = train.drop(columns=["price_range"]).select_dtypes(include=np.number)
    y_train = train["price_range"]
    if X_train.empty:
        raise ValueError("No numeric feature columns found in train.csv.")

    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    feature_names = list(X_train.columns)

    # Read input in chunks so large files do not need to fit in memory at once.
    first_chunk = True
    total_rows = 0
    for chunk in pd.read_csv(input_file, chunksize=BATCH_SIZE,sep=','):
        chunk.columns = chunk.columns.str.strip()
        print("Columns found:", list(chunk.columns))
        chunk = chunk.drop(columns=["id", "price_range"], errors="ignore")
        missing = [col for col in feature_names if col not in chunk.columns]
        extra = [col for col in chunk.columns if col not in feature_names]
        if missing:
            raise ValueError(f"Input file is missing required columns: {missing}")
        if extra:
            print("Ignoring extra columns:", extra)

        X_batch = chunk[feature_names].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(X_batch.to_numpy()).all():
            raise ValueError("Input contains missing or non-finite feature values.")

        predictions = model.predict(X_batch)
        output = chunk.copy()
        output["predicted_price_range"] = predictions
        output.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )
        first_chunk = False
        total_rows += len(chunk)
        print(f"Processed {total_rows} rows...")

    print(f"Finished. Predictions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
