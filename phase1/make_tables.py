"""
Build every table from results/*.json. Never type a number into the paper by hand:
re-run this script and copy the generated .tex/.md instead.

Usage:  python make_tables.py      -> writes tables/*.md and tables/*.tex
"""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, "results", "*.json")))}
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)


def ms(r, key, nd=3):
    v = [x[key] for x in r["runs"] if key in x]
    if not v:
        return "--"
    return f"{np.mean(v):.{nd}f}" if len(v) == 1 else f"{np.mean(v):.{nd}f} ± {np.std(v):.{nd}f}"


def write(name, header, rows, caption):
    md = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    open(os.path.join(HERE, "tables", name + ".md"), "w").write(f"**{caption}**\n\n" + "\n".join(md) + "\n")
    esc = lambda s: s.replace("±", r"$\pm$").replace("_", r"\_").replace("→", r"$\rightarrow$")
    tex = [r"\begin{table}", r"\centering", r"\caption{" + esc(caption) + "}", r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\hline",
           " & ".join(map(esc, header)) + r" \\", r"\hline"] + [" & ".join(map(esc, r)) + r" \\" for r in rows] + [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(HERE, "tables", name + ".tex"), "w").write("\n".join(tex) + "\n")


# Table 1: every configuration, one row each
rows = []
for k, r in RES.items():
    c = r["config"]
    rows.append([k, c["preprocessing"], "65 (wrong)" if c.get("legacy_class_codes") else "95 (correct)",
                 ", ".join(c.get("drop_features", [])) or "none", str(len(r["runs"])),
                 ms(r, "plasticc_macro_f1"), ms(r, "transfer_macro_f1")])
write("all_configs", ["Config", "Preprocessing", "SLSN-I code", "Dropped", "Seeds", "PLAsTiCC F1", "Transfer F1"], rows,
      "Every Phase 1 configuration (mean ± SD over split seeds)")

# Table 2: the three-way benchmark (new baseline)
b = RES.get("10_corrected_consistent")
if b:
    rows = [["PLAsTiCC in-domain", str(b["runs"][0]["n_test"]), ms(b, "plasticc_macro_f1")],
            ["DES-only 5-fold CV", str(b["runs"][0]["n_des"]), ms(b, "des_cv_macro_f1_mean_of_folds")],
            ["PLAsTiCC → DES transfer", str(b["runs"][0]["n_des"]), ms(b, "transfer_macro_f1")]]
    write("benchmark", ["Benchmark", "N test", "Macro-F1"], rows,
          "Three-way benchmark, corrected classes, consistent preprocessing (5 split seeds)")

    # Table 3: per-class F1
    names = b["class_order"]
    rows = []
    for n in names:
        pv = [x["plasticc_per_class_f1"][n] for x in b["runs"]]
        tv = [x["transfer_per_class_f1"][n] for x in b["runs"]]
        dv = [x["des_cv_per_class_f1"][n] for x in b["runs"]]
        rows.append([n, f"{np.mean(pv):.3f} ± {np.std(pv):.3f}", f"{np.mean(dv):.3f} ± {np.std(dv):.3f}", f"{np.mean(tv):.3f} ± {np.std(tv):.3f}"])
    write("per_class", ["Class", "PLAsTiCC in-domain", "DES-only CV", "Transfer"], rows,
          "Per-class F1, corrected classes, consistent preprocessing (mean ± SD over 5 split seeds)")

    # Table 4: summed transfer confusion matrix
    cm = np.sum([x["transfer_confusion"] for x in b["runs"]], axis=0) / len(b["runs"])
    rows = [[f"true {n}"] + [f"{v:.1f}" for v in row] for n, row in zip(names, cm)]
    write("transfer_confusion", ["", *[f"pred {n}" for n in names]], rows,
          "Transfer confusion matrix (DES objects, mean over 5 split seeds)")

# Table 5: feature-removal effects under both preprocessing modes
pairs = [("consistent", "10_corrected_consistent", "11_corrected_consistent_drop_absmag", "12_corrected_consistent_drop_redshift"),
         ("per-domain", "20_corrected_per_domain", "21_corrected_per_domain_drop_absmag", "22_corrected_per_domain_drop_redshift")]
rows = []
for lab, full, nomag, noz in pairs:
    if all(k in RES for k in (full, nomag, noz)):
        rows.append([lab, ms(RES[full], "transfer_macro_f1"), ms(RES[nomag], "transfer_macro_f1"), ms(RES[noz], "transfer_macro_f1")])
write("feature_removal", ["Preprocessing", "All features", "− abs_peak_mag", "− redshift"], rows,
      "Transfer macro-F1 after removing one feature (corrected classes, 5 split seeds)")
print("Wrote", len(os.listdir(os.path.join(HERE, "tables"))), "files to tables/")
