"""
Compare a re-extracted PLAsTiCC feature table with the original one used for the paper.
Usage: python compare_plasticc_features.py original.csv reextracted.csv
Reports: objects present in only one table; per-column agreement (fraction of objects whose values
agree to 1e-6 relative, and to 1 %); and whether the row order (which fixes the train/test splits) matches.
"""
import sys
import numpy as np, pandas as pd

a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
print(f"original: {len(a)} objects | re-extracted: {len(b)} objects")
only_a, only_b = set(a.object_id) - set(b.object_id), set(b.object_id) - set(a.object_id)
print(f"only in original: {len(only_a)} | only in re-extracted: {len(only_b)}")
m = a.merge(b, on="object_id", suffixes=("_o", "_n"))
rows = []
for c in [c for c in a.columns if c != "object_id" and c in b.columns]:
    x, y = m[f"{c}_o"].astype(float).values, m[f"{c}_n"].astype(float).values
    both_nan = np.isnan(x) & np.isnan(y)
    rel = np.abs(x - y) / np.maximum(np.abs(x), 1e-9)
    rows.append((c, np.mean(both_nan | (rel < 1e-6)), np.mean(both_nan | (rel < 0.01)), np.nanmedian(rel)))
print(pd.DataFrame(rows, columns=["column", "exact (1e-6)", "within 1%", "median rel. diff"]).round(4).to_string(index=False))
same_order = list(a.object_id) == list(b.object_id)
print("row order identical (needed for identical splits):", same_order)
exact = all(r[1] > 0.999 for r in rows)
print("VERDICT:", "reproduces the original table" if exact and same_order and not only_a and not only_b
      else "does NOT exactly reproduce the original; report the differences above in the README")
