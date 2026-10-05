"""Aggregate every experiment into results.json (tables) and draw Figs. 3-8 (grayscale)."""
import json, glob, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data as D, metrics as Mt, evaluate as EV, robustness as RB

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                     "font.size": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
FIG = "figs"
SEEDS = [0, 1, 2, 3, 4]
MAIN = [("LandmarkLR", "main"), ("CSFusion", "main"), ("GRUD", "main"), ("LTMD", "main"), ("LTMD-MCD", "main"),
        ("LTMD-Ens", "main"), ("CTFilter", "main"), ("ULTRA", "main")]
ABL = [("ULTRA", t) for t in ["main", "no_align", "no_factor", "no_aging", "no_velocity", "homosc", "no_integrate",
                             "no_innov", "no_axis"]]
STY = {"ULTRA": dict(c="0.0", ls="-", m="o"), "CTFilter": dict(c="0.35", ls="--", m="s"),
       "LTMD": dict(c="0.55", ls="-.", m="^"), "GRUD": dict(c="0.7", ls=":", m="D"), "CSFusion": dict(c="0.45", ls=(0, (5, 1, 1, 1)), m="v"),
       "LTMD-Ens": dict(c="0.25", ls=":", m="P"), "LTMD-MCD": dict(c="0.6", ls="--", m="X")}
NICE = {"ULTRA": "ULTRA-Retina", "CTFilter": "CT-Filter", "LTMD": "LT-MD", "GRUD": "GRU-D", "CSFusion": "CS-Fusion",
        "LandmarkLR": "Landmark-LR", "LTMD-Ens": "LT-MD-Ens", "LTMD-MCD": "LT-MD-MCD"}


def available(name, tag):
    return [s for s in SEEDS if os.path.exists(f"runs/pred_{name}_{tag}_s{s}.npz")]


def stale_rows(tab, cols, rows):
    st = RB.staleness(tab, cols)
    return rows[~np.isfinite(st[rows]) | (st[rows] > 12)]


def summarise(models):
    tab, cols, sp, rows, G, eid = EV.setup()
    res = {}
    for name, tag in models:
        seeds = available(name, tag)
        if not seeds:
            continue
        per = {s: [] for s in ["test", "extD", "extE", "stale"]}
        for sd in seeds:
            P = EV.load_pred(name, tag, sd)
            for s in ["test", "extD", "extE"]:
                per[s].append(Mt.all_metrics(P, tab, cols, rows[s], G))
            r = stale_rows(tab, cols, np.concatenate([rows["test"], rows["extD"], rows["extE"]]))
            per["stale"].append(Mt.progression_metrics(P["S"][r], tab[r, cols.index("event")], tab[r, cols.index("tte")], G))
        res[f"{name}:{tag}"] = {s: {k: [float(np.mean([m[k] for m in v])), float(np.std([m[k] for m in v]))] for k in v[0]}
                                for s, v in per.items()}
        res[f"{name}:{tag}"]["n_seeds"] = len(seeds)
    return res


def uncertainty(models):
    """AURC of the 12-month Brier loss using each model's uncertainty score (test + external)."""
    tab, cols, sp, rows, G, eid = EV.setup()
    r = np.concatenate([rows["test"], rows["extD"], rows["extE"]])
    y = Mt.landmark_labels(tab[r, cols.index("event")], tab[r, cols.index("tte")], 12); ok = ~np.isnan(y)
    out, curves = {}, {}
    for name, tag in models:
        vals, cs = [], []
        for sd in available(name, tag):
            P = EV.load_pred(name, tag, sd)
            p = Mt.risk_at(P["S"][r], 12)[ok]
            unc = P["risk_sd"][r][ok] if name in ("ULTRA", "LTMD-MCD", "LTMD-Ens") else RB.binary_entropy(p)
            a, cum = Mt.aurc(unc, (p - y[ok]) ** 2)
            vals.append(a); cs.append(cum[::-1][np.linspace(0, len(cum) - 1, 50).astype(int)])
        if vals:
            out[name] = [float(np.mean(vals)), float(np.std(vals))]
            curves[name] = np.mean(cs, 0).tolist()
    return out, curves


def fig_sweep():
    """Fig: test-time OCT availability sweep."""
    data = []
    for s in SEEDS:
        f = f"runs/octsweep_s{s}.json"
        if os.path.exists(f):
            data += [dict(d, seed=s) for d in json.load(open(f))]
    if not data:
        return None
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.15), gridspec_kw=dict(wspace=0.38))
    qs = sorted(set(d["q"] for d in data))
    summary = {}
    for name in ["CSFusion", "GRUD", "LTMD", "CTFilter", "ULTRA"]:
        for k, ax, lab in [("auc12", axs[0], "AUC$_{12}$"), ("ece12", axs[1], "ECE$_{12}$"), ("sev_qwk", axs[2], "Severity QWK")]:
            mu = [np.mean([d[k] for d in data if d["model"] == name and d["q"] == q and d["split"] == "test"]) for q in qs]
            sd = [np.std([d[k] for d in data if d["model"] == name and d["q"] == q and d["split"] == "test"]) for q in qs]
            st = STY[name]
            ax.errorbar(qs, mu, yerr=sd, color=st["c"], ls=st["ls"], marker=st["m"], ms=3.5, lw=1.0, capsize=1.5,
                        label=NICE[name])
            ax.set_xlabel("Test-time OCT availability $q$"); ax.set_ylabel(lab)
            summary.setdefault(name, {})[k] = dict(zip([str(q) for q in qs], [float(x) for x in mu]))
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.75, 1.32), ncol=5, fontsize=6.8, handlelength=2.6)
    for ax in axs:
        ax.set_xticks(qs)
    fig.subplots_adjust(top=0.80, bottom=0.2, left=0.085, right=0.99)
    fig.savefig(f"{FIG}/fig_sweep.png", dpi=500)
    return summary


def fig_staleness():
    data = []
    for s in SEEDS:
        f = f"runs/staleness_s{s}.json"
        if os.path.exists(f):
            data += json.load(open(f))
    if not data:
        return None
    bins = ["current", "<=12m", ">12m", "never"]
    labels = ["OCT now", "$\\leq$12 mo", ">12 mo", "never"]
    models = ["CSFusion", "GRUD", "LTMD", "LTMD-Ens", "CTFilter", "ULTRA"]
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.3), gridspec_kw=dict(wspace=0.25, width_ratios=[1.25, 1]))
    w = 0.13
    hatches = ["", "////", "....", "xxxx", "\\\\\\\\", ""]
    fills = ["0.85", "white", "white", "white", "white", "0.1"]
    summ = {}
    for i, name in enumerate(models):
        mu = [np.mean([d["ece12"] for d in data if d["model"] == name and d["bin"] == b]) for b in bins]
        sd = [np.std([d["ece12"] for d in data if d["model"] == name and d["bin"] == b]) for b in bins]
        x = np.arange(4) + (i - 2.5) * w
        axs[0].bar(x, mu, w * 0.92, yerr=sd, color=fills[i], edgecolor="black", hatch=hatches[i], lw=0.5,
                   error_kw=dict(lw=0.5, capsize=1), label=NICE[name])
        summ[name] = dict(zip(bins, [float(v) for v in mu]))
        n = [np.mean([d["n"] for d in data if d["model"] == name and d["bin"] == b]) for b in bins]
        summ[name + "_n"] = dict(zip(bins, [float(v) for v in n]))
        sdr = [np.mean([d["risk_sd"] for d in data if d["model"] == name and d["bin"] == b]) for b in bins]
        summ[name + "_risk_sd"] = dict(zip(bins, [float(v) for v in sdr]))
    axs[0].set_xticks(np.arange(4)); axs[0].set_xticklabels(labels, fontsize=6.8); axs[0].set_ylabel("ECE$_{12}$")
    axs[0].set_xlabel("Time since last OCT"); axs[1].set_xlabel("Time since last OCT")
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=3, fontsize=6.4)
    # right: ULTRA posterior risk spread by staleness
    for name, st in [("ULTRA", STY["ULTRA"]), ("LTMD-Ens", STY["LTMD-Ens"])]:
        v = [summ[name + "_risk_sd"][b] for b in bins]
        axs[1].plot(range(4), v, color=st["c"], ls=st["ls"], marker=st["m"], ms=4, lw=1.1, label=NICE[name])
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels(labels, fontsize=6.8)
    axs[1].set_ylabel("Mean s.d. of $\\rho_{12}$")
    axs[1].legend(loc="upper left", fontsize=6.6)
    fig.subplots_adjust(top=0.78, bottom=0.19, left=0.07, right=0.99)
    fig.savefig(f"{FIG}/fig_staleness.png", dpi=500)
    return summ


def fig_reliability():
    tab, cols, sp, rows, G, eid = EV.setup()
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.3), gridspec_kw=dict(wspace=0.32))
    out = {}
    for ax, split, title in [(axs[0], "test", "Internal test"), (axs[1], "extE", "External centre E")]:
        r = rows[split]
        y = Mt.landmark_labels(tab[r, cols.index("event")], tab[r, cols.index("tte")], 12); ok = ~np.isnan(y)
        ax.plot([0, 1], [0, 1], color="0.75", lw=0.7, ls="-")
        for name in ["LTMD", "LTMD-Ens", "CTFilter", "ULTRA"]:
            ps, ys = [], []
            for sd in available(name, "main"):
                P = EV.load_pred(name, "main", sd)
                ps.append(Mt.risk_at(P["S"][r], 12)[ok]); ys.append(y[ok])
            p = np.concatenate(ps); yy = np.concatenate(ys)
            edges = np.quantile(p, np.linspace(0, 1, 11)); idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
            mx = [p[idx == b].mean() for b in range(10)]; my = [yy[idx == b].mean() for b in range(10)]
            st = STY[name]
            ax.plot(mx, my, color=st["c"], ls=st["ls"], marker=st["m"], ms=3.2, lw=1.0, label=NICE[name])
            out[f"{name}:{split}"] = list(zip(map(float, mx), map(float, my)))
        ax.set_xlabel("Predicted 12-month risk"); ax.set_ylabel("Observed frequency"); ax.set_title(title, fontsize=7.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axs[0].legend(loc="upper left", fontsize=6.4)
    au, curves = uncertainty([(m, "main") for m in ["CSFusion", "GRUD", "LTMD", "LTMD-MCD", "LTMD-Ens", "CTFilter", "ULTRA"]])
    cov = np.linspace(1, 0.02, 50)
    for name in ["LTMD", "LTMD-Ens", "CTFilter", "ULTRA"]:
        st = STY[name]
        axs[2].plot(cov, curves[name], color=st["c"], ls=st["ls"], lw=1.1, label=NICE[name])
    axs[2].set_xlabel("Coverage"); axs[2].set_ylabel("Brier$_{12}$ of retained"); axs[2].set_title("Risk-coverage", fontsize=7.5)
    axs[2].invert_xaxis()
    fig.subplots_adjust(top=0.9, bottom=0.17, left=0.075, right=0.99)
    fig.savefig(f"{FIG}/fig_reliability.png", dpi=500)
    return out, au


def fig_policy():
    data = []
    for s in SEEDS:
        f = f"runs/policy_test_s{s}.json"
        if os.path.exists(f):
            data += [dict(d, seed=s) for d in json.load(open(f))]
    if not data:
        return None
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.25), gridspec_kw=dict(wspace=0.38))
    pol = {"value_mc": ("Acquisition value $V_k$", dict(c="0.0", ls="-", m="o")),
           "value": ("Approximation $\\tilde{V}_k$", dict(c="0.0", ls=":", m="o", mfc="white")),
           "uncert": ("Uncertainty trigger", dict(c="0.45", ls="--", m="s")),
           "risk": ("Risk trigger", dict(c="0.6", ls="-.", m="^")),
           "random": ("Random", dict(c="0.75", ls=":", m="D")),
           "interval": ("Fixed interval", dict(c="0.3", ls=(0, (5, 1, 1, 1)), m="v"))}
    grid = np.round(np.arange(0.2, 0.96, 0.1), 2)
    summ = {}
    for p, (lab, st) in pol.items():
        curves = []
        for sd in SEEDS:
            rows = sorted([(d["budget"], d["auc12"], d["ibs"], d["sens80"]) for d in data if d["policy"] == p and d["seed"] == sd])
            if not rows:
                continue
            rows = np.array(rows)
            ys = []
            for j in (1, 2, 3):
                y = np.interp(grid, rows[:, 0], rows[:, j], left=np.nan, right=np.nan)
                ys.append(y)
            curves.append(np.stack(ys, 1))
        if not curves:
            continue
        C = np.array(curves)                                    # seeds, grid, 3
        mu = np.nanmean(C, 0); sd_ = np.nanstd(C, 0)
        ok = (~np.isnan(C[:, :, 0])).sum(0) >= 3                 # at least three seeds cover this budget
        summ[p] = {"budget": grid[ok].tolist(), "auc12": mu[ok, 0].tolist(), "ibs": mu[ok, 1].tolist(),
                   "sens80": mu[ok, 2].tolist(), "auc12_sd": sd_[ok, 0].tolist()}
        for ax, j in zip(axs, [0, 1, 2]):
            ax.plot(grid[ok], mu[ok, j], color=st["c"], ls=st["ls"], marker=st["m"], ms=3.2, lw=1.0, label=lab,
                    mfc=st.get("mfc", st["c"]), mec=st["c"])
    for nm in ("none", "all"):
        g = [d for d in data if d["policy"] == nm]
        summ[nm] = [float(np.mean([d["budget"] for d in g])), float(np.mean([d["auc12"] for d in g])),
                    float(np.mean([d["ibs"] for d in g])), float(np.mean([d["sens80"] for d in g]))]
        for ax, j in zip(axs, [1, 2, 3]):
            ax.axhline(summ[nm][j], color="0.8", lw=0.8, ls="--" if nm == "none" else "-", zorder=0)
    axs[0].text(0.21, summ["all"][1] + 0.0008, "OCT at all visits", ha="left", va="bottom", fontsize=6.0, color="0.4")
    axs[0].text(0.55, summ["none"][1] + 0.0008, "no OCT", ha="left", va="bottom", fontsize=6.0, color="0.4")
    axs[0].set_ylabel("AUC$_{12}$"); axs[1].set_ylabel("IBS"); axs[2].set_ylabel("Sensitivity at 80% specificity")
    for ax in axs:
        ax.set_xlabel("OCT budget (fraction of visits)"); ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.75, 1.36), ncol=3, fontsize=6.6, handlelength=2.8)
    fig.subplots_adjust(top=0.76, bottom=0.19, left=0.07, right=0.99)
    fig.savefig(f"{FIG}/fig_policy.png", dpi=500)
    return summ


def failure(models=("LTMD", "GRUD", "CTFilter", "ULTRA")):
    """Subgroup analysis on internal test + external landmarks: disease, change points, regression, degraded visits."""
    from sklearn.metrics import roc_auc_score
    tab, cols, sp, rows, G, eid = EV.setup()
    c = lambda n: tab[:, cols.index(n)]
    r = np.concatenate([rows["test"], rows["extD"], rows["extE"]])
    groups = {"DR": c("etype")[r] == 1, "AMD": c("etype")[r] == 2, "Glaucoma": c("etype")[r] == 3,
              "Healthy at baseline": c("etype")[r] == 0, "Rate change point": c("cp")[r] == 1,
              "No change point": (c("cp")[r] == 0) & (c("etype")[r] > 0), "Treatment regression": c("reg")[r] == 1,
              "Degraded acquisition": c("bad")[r] == 1}
    y = Mt.landmark_labels(c("event")[r], c("tte")[r], 12)
    out = {}
    for name in models:
        for g, msk in groups.items():
            vals, eces, ns = [], [], []
            for sd in available(name, "main"):
                P = EV.load_pred(name, "main", sd)
                ok = msk & ~np.isnan(y)
                p = Mt.risk_at(P["S"][r], 12)[ok]
                if len(np.unique(y[ok])) > 1:
                    vals.append(roc_auc_score(y[ok], p)); eces.append(Mt.ece(p, y[ok]))
                ns.append(int(ok.sum()))
            out.setdefault(g, {})[name] = [float(np.mean(vals)), float(np.mean(eces)), ns[0], float(np.nanmean(y[msk]))]
    return out


def recalibration(models=("LandmarkLR", "CSFusion", "GRUD", "LTMD", "LTMD-Ens", "CTFilter", "ULTRA")):
    """Platt recalibration of the 12-month risk fitted on the calibration split; ECE before/after per split."""
    from sklearn.linear_model import LogisticRegression
    tab, cols, sp, rows, G, eid = EV.setup()
    ev, tte = tab[:, cols.index("event")], tab[:, cols.index("tte")]
    y = Mt.landmark_labels(ev, tte, 12)
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / np.clip(1 - p, 1e-4, 1))
    out = {}
    for name in models:
        acc = {}
        for sd in available(name, "main"):
            P = EV.load_pred(name, "main", sd)
            p = Mt.risk_at(P["S"], 12)
            rc = rows["cal"][~np.isnan(y[rows["cal"]])]
            lr = LogisticRegression(C=1e4).fit(lg(p[rc])[:, None], y[rc])
            for s in ["test", "extD", "extE"]:
                r = rows[s][~np.isnan(y[rows[s]])]
                q = lr.predict_proba(lg(p[r])[:, None])[:, 1]
                acc.setdefault(s, []).append((Mt.ece(p[r], y[r]), Mt.ece(q, y[r])))
        out[name] = {s: [float(np.mean([a for a, b in v])), float(np.mean([b for a, b in v]))] for s, v in acc.items()}
    return out


def corruption():
    out = {}
    for s in SEEDS:
        f = f"runs/corrupt_s{s}.json"
        if os.path.exists(f):
            for d in json.load(open(f)):
                for k, v in d.items():
                    if k != "model":
                        out.setdefault(d["model"], {}).setdefault(k, []).append(v)
    return {m: {k: [float(np.mean(v)), float(np.std(v))] for k, v in d.items()} for m, d in out.items()}


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    R = {"main": summarise(MAIN), "abl": summarise(ABL),
         "sens": summarise([("ULTRA", t) for t in ["nS12", "nS48", "lam005", "lam1", "poct0", "poct6"]] +
                           [("LTMD", t) for t in ["poct0", "poct6"]])}
    R["sweep"] = fig_sweep()
    R["stale"] = fig_staleness()
    R["reliab"], R["aurc"] = fig_reliability()
    R["policy"] = fig_policy()
    R["corrupt"] = corruption()
    R["failure"] = failure()
    R["recal"] = recalibration()
    if os.path.exists("runs/efficiency.json"):
        R["eff"] = json.load(open("runs/efficiency.json"))
    json.dump(R, open("runs/results.json", "w"), indent=1)
    print("written")
