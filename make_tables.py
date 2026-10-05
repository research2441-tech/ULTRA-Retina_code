"""Generate table markup (for build.py) from runs/results.json and the bootstrap outputs."""
import json, os, numpy as np

R = json.load(open("runs/results.json"))
B = json.load(open("runs/boot_test.json")) if os.path.exists("runs/boot_test.json") else {}
BE = json.load(open("runs/boot_extE.json")) if os.path.exists("runs/boot_extE.json") else {}
BA = json.load(open("runs/boot_abl.json")) if os.path.exists("runs/boot_abl.json") else {}
NICE = {"ULTRA": "ULTRA-Retina", "CTFilter": "CT-Filter", "LTMD": "LT-MD", "GRUD": "GRU-D", "CSFusion": "CS-Fusion",
        "LandmarkLR": "Landmark-LR", "LTMD-Ens": "LT-MD-Ens", "LTMD-MCD": "LT-MD-MCD"}
ORDER = ["LandmarkLR", "CSFusion", "GRUD", "LTMD", "LTMD-MCD", "LTMD-Ens", "CTFilter", "ULTRA"]
out = []


def f(v, d=3, sd=True):
    return f"{v[0]:.{d}f} $\\pm$ {v[1]:.{d}f}" if sd else f"{v[0]:.{d}f}"


def best(vals, higher):
    arr = np.array(vals)
    return int(np.argmax(arr) if higher else np.argmin(arr))


def table(label, width, widths, caption, header, rows, notes=()):
    s = [f"!table {label} | {width} | {','.join(map(str, widths))} | {caption}", "| " + " | ".join(header) + " |"]
    s += ["| " + " | ".join(r) + " |" for r in rows]
    s += [f"!note {n}" for n in notes]
    return "\n".join(s)


def bold_best(rows_vals, cols_higher, fmt):
    """rows_vals: list of lists of (mean, sd); returns formatted strings with best in bold."""
    out_rows = [[None] * len(cols_higher) for _ in rows_vals]
    for j, hi in enumerate(cols_higher):
        b = best([r[j][0] for r in rows_vals], hi)
        for i, r in enumerate(rows_vals):
            txt = fmt(r[j])
            out_rows[i][j] = ("!b" + txt) if i == b else txt
    return out_rows


# ---------------------------------------------------------------- Table: main progression (internal test)
M = R["main"]
keys = [("auc12", 1), ("auc24", 1), ("cidx", 1), ("ibs", 0), ("ece12", 0), ("nll12", 0)]
rv = [[M[f"{m}:main"]["test"][k] for k, _ in keys] for m in ORDER]
fm = bold_best(rv, [h for _, h in keys], lambda v: f(v))
rows = [[NICE[m]] + fm[i] for i, m in enumerate(ORDER)]
out.append(table("tab:main", "wide", [1.3, 1, 1, 1, 1, 1, 1],
                 "Progression forecasting on the internal test set (mean $\\pm$ s.d. over five seeds)",
                 ["Method", "AUC$_{12}$ $\\uparrow$", "AUC$_{24}$ $\\uparrow$", "C-index $\\uparrow$", "IBS $\\downarrow$",
                  "ECE$_{12}$ $\\downarrow$", "NLL$_{12}$ $\\downarrow$"], rows,
                 ["Best mean in bold. All learned methods share the frozen encoders, observation pooling, heads and modality dropout; 270 test eyes, 1047 landmarks with known 12-month status."]))

# ---------------------------------------------------------------- Table: secondary tasks
keys2 = [("cls_auc", 1), ("cls_bacc", 1), ("cls_mcc", 1), ("sev_qwk", 1), ("sev_acc", 1), ("les_ap", 1), ("les_dice", 1)]
rv = [[M[f"{m}:main"]["test"][k] for k, _ in keys2] for m in ORDER if m not in ("LTMD-MCD",)]
names = [m for m in ORDER if m not in ("LTMD-MCD",)]
fm = bold_best(rv, [h for _, h in keys2], lambda v: f"{v[0]:.3f}")
rows = [[NICE[m]] + fm[i] for i, m in enumerate(names)]
out.append(table("tab:secondary", "col", [1.5, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                 "Diagnosis, severity and lesion localisation on the internal test set",
                 ["Method", "Dx AUC", "Dx BAcc", "Dx MCC", "QWK", "Sev Acc", "Les AP", "Dice"], rows,
                 ["Means over five seeds; s.d. $\\le$ 0.020 for all entries. Dx: diagnosis; QWK: severity quadratic-weighted kappa; Les: lesion grid."]))

# ---------------------------------------------------------------- Table: ablation
A = R["abl"]
ab = [("main", "Full model"), ("no_align", "w/o anatomy alignment ($\\lambda_a=\\lambda_o=0$)"),
      ("no_factor", "w/o block factorisation"), ("no_aging", "w/o evidence ageing ($\\mathbf{Q}=\\mathbf{0}$)"),
      ("no_velocity", "w/o velocity state"), ("homosc", "homoscedastic noise"),
      ("no_integrate", "w/o belief integration ($L=1$)"), ("no_innov", "w/o innovation loss ($\\lambda_p=0$)"),
      ("no_axis", "w/o ordinal axis and $\\psi_j$")]
keys3 = [("test", "auc12", 1), ("test", "ibs", 0), ("test", "ece12", 0), ("test", "sev_qwk", 1), ("extE", "auc12", 1), ("stale", "ece12", 0)]
rows = []
for t, lab in ab:
    v = A[f"ULTRA:{t}"]
    row = [lab]
    for s, k, hi in keys3:
        txt = f"{v[s][k][0]:.3f}"
        if t != "main":
            d = v[s][k][0] - A["ULTRA:main"][s][k][0]
            txt += f" ({d:+.3f})"
        row.append(txt)
    rows.append(row)
out.append(table("tab:abl", "wide", [2.3, 1, 1, 1, 1, 1, 1],
                 "Component-removal ablation (mean over five seeds; change from the full model in parentheses)",
                 ["Variant", "Test AUC$_{12}$", "Test IBS", "Test ECE$_{12}$", "Test QWK", "Centre E AUC$_{12}$",
                  "Stale ECE$_{12}$"], rows,
                 ["Stale: landmarks of the test and external sets whose last OCT is older than 12 months or absent. Paired bootstrap intervals for the changes are given in the text."]))

# ---------------------------------------------------------------- Table: external generalisation
keys4 = [("extD", "auc12", 1), ("extD", "ibs", 0), ("extD", "ece12", 0), ("extD", "sev_qwk", 1),
         ("extE", "auc12", 1), ("extE", "ibs", 0), ("extE", "ece12", 0), ("extE", "sev_qwk", 1)]
names = [m for m in ORDER if m != "LTMD-MCD"]
rv = [[M[f"{m}:main"][s][k] for s, k, _ in keys4] for m in names]
fm = bold_best(rv, [h for _, _, h in keys4], lambda v: f"{v[0]:.3f}")
rows = [[NICE[m]] + fm[i] for i, m in enumerate(names)]
out.append(table("tab:ext", "wide", [1.3, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                 "Generalisation to external centres with unseen devices (means over five seeds)",
                 ["Method", "D: AUC$_{12}$", "D: IBS", "D: ECE$_{12}$", "D: QWK", "E: AUC$_{12}$", "E: IBS",
                  "E: ECE$_{12}$", "E: QWK"], rows,
                 ["Centre D: unseen device, OCT at 68% of visits. Centre E: unseen device, OCT at 23% of visits, median interval 11.2 months."]))

# ---------------------------------------------------------------- Table: statistics
if B:
    rows = []
    for m in ["LandmarkLR", "CSFusion", "GRUD", "LTMD", "LTMD-Ens", "CTFilter"]:
        k = f"{m}:main"
        if k not in B:
            continue
        r = [NICE[m]]
        for met in ["auc12", "ibs", "ece12"]:
            d = B[k][met]
            r.append(f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
            r.append(f"{d['p_holm']:.3f}" if d["p_holm"] >= 0.001 else "<0.001")
        if BE and k in BE:
            d = BE[k]["auc12"]
            r.append(f"{d['diff']:+.3f} ({d['p_holm']:.3f})" if d["p_holm"] >= 0.001 else f"{d['diff']:+.3f} (<0.001)")
        rows.append(r)
    out.append(table("tab:stats", "wide", [1.1, 1.45, 0.55, 1.45, 0.55, 1.45, 0.55, 1.15],
                     "Paired eye-level bootstrap comparison of ULTRA-Retina against each comparator (difference = ULTRA-Retina minus comparator)",
                     ["Comparator", "$\\Delta$AUC$_{12}$ [95% CI]", "$p_{\\mathrm{Holm}}$", "$\\Delta$IBS [95% CI]", "$p_{\\mathrm{Holm}}$",
                      "$\\Delta$ECE$_{12}$ [95% CI]", "$p_{\\mathrm{Holm}}$", "Centre E $\\Delta$AUC$_{12}$ ($p_{\\mathrm{Holm}}$)"], rows,
                     ["Internal test set unless stated; 1000 resamples of eyes, seed-averaged metrics, Holm correction across the six comparators. Positive $\\Delta$AUC and negative $\\Delta$IBS or $\\Delta$ECE favour ULTRA-Retina."]))

# ---------------------------------------------------------------- Table: robustness (corruption + uncertainty)
C = R["corrupt"]; AU = R["aurc"]
rows = []
for m in ["CSFusion", "GRUD", "LTMD", "LTMD-MCD", "CTFilter", "ULTRA"]:
    if m not in C:
        continue
    c = C[m]
    rows.append([NICE[m], f"{c['clean_auc12'][0]:.3f}", f"{c['corrupt_auc12'][0]:.3f}", f"{c['corrupt_ece12'][0]:.3f}",
                 f"{c['corrupt_sev_qwk'][0]:.3f}", f"{c['unc_auroc'][0]:.3f} $\\pm$ {c['unc_auroc'][1]:.3f}",
                 f"{AU[m][0]:.4f}" if m in AU else "--"])
if "LTMD-Ens" in AU:
    rows.append([NICE["LTMD-Ens"], "--", "--", "--", "--", "--", f"{AU['LTMD-Ens'][0]:.4f}"])
out.append(table("tab:robust", "wide", [1.3, 0.9, 0.9, 0.9, 0.9, 1.25, 1.0],
                 "Robustness to image corruption and quality of the uncertainty score",
                 ["Method", "Clean AUC$_{12}$", "Corrupted AUC$_{12}$", "Corrupted ECE$_{12}$", "Corrupted QWK",
                  "Flagging AUROC", "AURC $\\downarrow$"], rows,
                 ["Half of the internal test visits are corrupted (fundus blur, under-illumination and noise; OCT low signal and extra speckle) and re-encoded with the frozen encoder. Flagging AUROC: separation of corrupted from clean visits by the uncertainty score (Monte Carlo spread of $\\rho_{12}$ for ULTRA-Retina and LT-MD-MCD, binary entropy of $\\rho_{12}$ otherwise). AURC: area under the risk-coverage curve of the 12-month Brier loss over test and external landmarks."]))

# ---------------------------------------------------------------- Table: efficiency
if "eff" in R:
    E = R["eff"]
    rows = []
    for m in ["CSFusion", "GRUD", "LTMD", "CTFilter", "ULTRA"]:
        e = E[m]
        state = {"CSFusion": "0", "GRUD": "386", "LTMD": "$512K$", "CTFilter": "96", "ULTRA": "240"}[m]
        rows.append([NICE[m], f"{e['params']/1e3:.0f}k", f"{e['ms_per_visit']:.3f}", state])
    rows.append(["Encoders $E_F+E_O$", f"{E['encoders']['params']/1e3:.0f}k", f"{E['encoders']['ms_per_visit']:.2f}", "--"])
    out.append(table("tab:eff", "col", [1.4, 0.8, 0.9, 1.0], "Computational cost (CPU, one thread, batch of 64 eyes)",
                     ["Model", "Params", "ms / visit", "State / eye"], rows,
                     ["Stage-2 parameters exclude the shared frozen encoders (listed separately, without decoders). ULTRA-Retina latency includes 32-sample belief integration. State / eye: numbers that must be kept between visits; $K$ is the number of past visits (two-layer key-value cache for LT-MD)."]))

open("runs/tables.md", "w").write("\n\n".join(out) + "\n")
print("tables written", len(out))
