"""Write the exact object identifiers of every split and target sample to splits/ (no labels beyond
what the public catalogues already contain). Usage: python data/export_splits.py"""
import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "phase1"))
import common as C  # noqa: E402

out = os.path.join(REPO, "splits"); os.makedirs(out, exist_ok=True)
plas, _ = C.load_plasticc()
rows = []
for seed in range(5):
    tr, te = train_test_split(plas, test_size=1570 / len(plas), stratify=plas["target"], random_state=seed)
    rows += [{"object_id": i, "split_seed": seed, "split": "train"} for i in tr["object_id"]]
    rows += [{"object_id": i, "split_seed": seed, "split": "test"} for i in te["object_id"]]
pd.DataFrame(rows).to_csv(os.path.join(out, "plasticc_splits.csv"), index=False)
tr, te = train_test_split(plas, test_size=1570 / len(plas), stratify=plas["target"], random_state=42)
pd.concat([pd.DataFrame({"object_id": tr["object_id"], "split": "train"}), pd.DataFrame({"object_id": te["object_id"], "split": "test"})]
          ).to_csv(os.path.join(out, "plasticc_split_legacy_seed42.csv"), index=False)
des = C.load_des(); lab = C.des_labels(des, "corrected"); keep = lab.isin(C.SHARED).values
id_col = next(c for c in ("snid", "SNID", "object_id", "cid", "CID") if c in des.columns)
pd.DataFrame({"snid": des[id_col][keep].values, "class_code": lab[keep].values,
              "confident_label": C.des_labels(des, "confirmed_only")[keep].isin(C.SHARED).values}).to_csv(os.path.join(out, "des_objects.csv"), index=False)
for name, col in [("ztf", "ztfid"), ("yse", "ysename")]:
    try:
        df, l = C.load_target(name); k = l.isin(C.SHARED).values
        extra = {"in_bts": df["in_bts"][k].values} if name == "yse" else {}
        pd.DataFrame({col: df[col][k].values, "class_code": l[k].values, **extra}).to_csv(os.path.join(out, f"{name}_objects.csv"), index=False)
    except FileNotFoundError as e:
        print("skipped", name, e)
print("wrote", sorted(os.listdir(out)))
