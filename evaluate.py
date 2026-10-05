"""Aggregate metrics over seeds and run eye-level paired bootstrap tests (Holm-corrected)."""
import sys, json, glob, numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score
import data as D, metrics as Mt

SEEDS = [0, 1, 2, 3, 4]


def setup():
    _, _, tab, cols = D.load_table()
    sp = D.split_eyes(tab, cols)
    eid = D.col(tab, cols, "eid").astype(int)
    rows = {s: np.where(np.isin(eid, sp[s]))[0] for s in sp}
    # censoring distribution from the training split (IPCW)
    tr = rows["train"]
    G = Mt.km_censor(tab[tr, cols.index("tte")], tab[tr, cols.index("event")])
    return tab, cols, sp, rows, G, eid


def load_pred(name, tag, seed):
    f = f"runs/pred_{name}_{tag}_s{seed}.npz"
    z = np.load(f)
    return {k: z[k] for k in z.files}


def summarise(models, splits=("test", "extD", "extE"), seeds=SEEDS, out=None):
    tab, cols, sp, rows, G, eid = setup()
    res = {}
    for (name, tag) in models:
        key = f"{name}:{tag}"
        res[key] = {}
        for s in splits:
            ms = []
            for sd in seeds:
                try:
                    P = load_pred(name, tag, sd)
                except FileNotFoundError:
                    continue
                ms.append(Mt.all_metrics(P, tab, cols, rows[s], G))
            if not ms:
                continue
            res[key][s] = {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms])), len(ms))
                           for k in ms[0]}
    if out:
        json.dump(res, open(out, "w"), indent=1)
    return res


def fast_metrics(S, cls, sev, tab, cols, r, G):
    c = lambda n: tab[r, cols.index(n)]
    ev, tte = c("event"), c("tte")
    o = {}
    for h in (12, 24):
        y = Mt.landmark_labels(ev, tte, h); m = ~np.isnan(y)
        p = Mt.risk_at(S[r], h)[m]
        o[f"auc{h}"] = roc_auc_score(y[m], p)
        if h == 12:
            o["ece12"] = Mt.ece(p, y[m])
    o["ibs"] = Mt.ipcw_brier(S[r], ev, tte, G)[1]
    o["cls_auc"] = roc_auc_score(c("cls").astype(int), cls[r], multi_class="ovr", average="macro")
    o["sev_qwk"] = cohen_kappa_score(c("stage").astype(int), sev[r].argmax(1), weights="quadratic")
    return o


def bootstrap(ref, others, split="test", B=1000, seeds=SEEDS, out=None):
    """Paired eye-level bootstrap of seed-averaged metric differences (ref - other); two-sided p, Holm."""
    tab, cols, sp, rows, G, eid = setup()
    eyes = sp[split]
    rows_of = {e: np.where(eid == e)[0] for e in eyes}
    preds = {m: [load_pred(*m, s) for s in seeds] for m in [ref] + others}
    rng = np.random.default_rng(0)
    keys = ["auc12", "auc24", "ibs", "ece12", "cls_auc", "sev_qwk"]
    diffs = {m: {k: [] for k in keys} for m in others}
    base = {}
    for bi in range(B + 1):
        sel = eyes if bi == 0 else rng.choice(eyes, len(eyes), replace=True)
        r = np.concatenate([rows_of[e] for e in sel])
        vals = {}
        for m in [ref] + others:
            ms = [fast_metrics(P["S"], P["cls"], P["sev"], tab, cols, r, G) for P in preds[m]]
            vals[m] = {k: np.mean([x[k] for x in ms]) for k in keys}
        for m in others:
            for k in keys:
                d = vals[ref][k] - vals[m][k]
                if bi == 0:
                    base[(m, k)] = d
                else:
                    diffs[m][k].append(d)
    res = {}
    for m in others:
        name = f"{m[0]}:{m[1]}"
        res[name] = {}
        for k in keys:
            d = np.array(diffs[m][k])
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            res[name][k] = {"diff": float(base[(m, k)]), "lo": float(np.percentile(d, 2.5)),
                            "hi": float(np.percentile(d, 97.5)), "p": float(max(p, 1.0 / B))}
    # Holm across comparisons within each metric
    for k in keys:
        ps = sorted([(res[n][k]["p"], n) for n in res])
        mlt = len(ps); run = 0
        for i, (p, n) in enumerate(ps):
            run = max(run, min(1.0, (mlt - i) * p))
            res[n][k]["p_holm"] = run
    if out:
        json.dump(res, open(out, "w"), indent=1)
    return res
