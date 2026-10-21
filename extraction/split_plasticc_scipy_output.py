"""
Splits the combined plasticc_raw_superphot_features.csv (from
plasticc_scipy_extraction.py) into X_train/X_test/y_train/y_test.csv,
matching the file structure the rest of the pipeline (data_loader.py,
feature_engineering.py) expects.

Uses a stratified split (by target class) so rare classes don't
accidentally end up all in one split, matching the ~6278/1570 train/test
proportions your original files had.

Run this from the "superphot features" folder after copying
plasticc_raw_superphot_features.csv into it.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "plasticc_raw_superphot_features.csv"

df = pd.read_csv(RAW_PATH)
print(f"Loaded {len(df)} objects, {df['target'].nunique()} classes")

# id/label columns that should NOT go into the feature matrix
id_cols = ["object_id", "target", "hostgal_photoz", "distmod", "mwebv"]
feature_cols = [c for c in df.columns if c not in id_cols]

X = df[feature_cols].copy()
y = df["target"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1570 / len(df), stratify=y, random_state=42
)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_frame(name="label").to_csv("y_train.csv", index=False)
y_test.to_frame(name="label").to_csv("y_test.csv", index=False)

print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
print("Saved X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print("\n*** BACK UP your existing X_train.csv/X_test.csv/y_train.csv/y_test.csv")
print("*** BEFORE running this, in case you need to roll back. ***")
