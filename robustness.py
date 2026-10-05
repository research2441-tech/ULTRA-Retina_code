"""E4/E8 robustness and uncertainty experiments.

oct_sweep  : test-time OCT availability q in {1, .75, .5, .25, 0} (fundus always available).
staleness  : metrics stratified by time since the last OCT under the recorded acquisition pattern.
corrupt    : 50% of test visits re-rendered-equivalent corruption (fundus blur + under-illumination + noise;
             OCT low signal + extra speckle), re-encoded with the same frozen encoder; reports accuracy on the
             corrupted subset and AUROC of the uncertainty score for flagging corrupted examinations.
ensemble   : builds the 5-member LTMD deep ensemble predictions from independently trained members.
"""
import sys, json, numpy as np, torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
import data as D, models as M, train as T, metrics as Mt, evaluate as EV, pretrain as PT, encoders as E

torch.set_num_threads(2)
MODELS = ["ULTRA", "CTFilter", "LTMD", "GRUD", "CSFusion"]


def load_model(name, tag, seed, flags=None):
    if flags is None:
        flags = json.load(open(f"runs/hist_{name}_{tag}_s{seed}.json"))["flags"]
    m = T.make_model(name, flags)
    m.load_state_dict(torch.load(f"runs/{name}_{tag}_s{seed}.pt")); m.eval()
    return m


def binary_entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def oct_sweep(seed, models=MODELS, qs=(1.0, 0.75, 0.5, 0.25, 0.0), splits=("test", "extD")):
    tab, cols, sp, rows, G, eid = EV.setup()
    coh = T.Cohort("full", seed)
    rng = np.random.default_rng(100 + seed)
    out = []
    masks = {q: (rng.random(len(tab)) < q).astype(float) for q in qs}
    for name in models:
        m = load_model(name, "main", seed)
        for s in splits:
            for q in qs:
                P = T.predict(m, coh, coh.sp[s], mF_over=np.ones(len(tab)), mO_over=masks[q])
                r = Mt.all_metrics(P, tab, cols, rows[s], G)
                out.append(dict(model=name, split=s, q=q, **r))
                print(name, s, q, round(r["auc12"], 3), round(r["ece12"], 3), round(r["sev_qwk"], 3), flush=True)
    json.dump(out, open(f"runs/octsweep_s{seed}.json", "w"), indent=1)


def staleness(tab, cols):
    c = lambda n: D.col(tab, cols, n)
    eid, k, t, ho = c("eid").astype(int), c("k").astype(int), c("t"), c("has_oct")
    st = np.full(len(tab), np.inf)
    order = np.lexsort((k, eid)); cur, last = -1, None
    for i in order:
        if eid[i] != cur:
            cur, last = eid[i], None
        if ho[i]:
            last = t[i]
        st[i] = (t[i] - last) if last is not None else np.inf
    return st


def staleness_eval(seed, models=MODELS + ["LandmarkLR", "LTMD-MCD", "LTMD-Ens"]):
    tab, cols, sp, rows, G, eid = EV.setup()
    st = staleness(tab, cols)
    allr = np.concatenate([rows["test"], rows["extD"], rows["extE"]])
    bins = {"current": st == 0, "<=12m": (st > 0) & (st <= 12), ">12m": (st > 12) & np.isfinite(st), "never": ~np.isfinite(st)}
    c = lambda n: D.col(tab, cols, n)
    out = []
    for name in models:
        try:
            P = EV.load_pred(name, "main", seed)
        except FileNotFoundError:
            continue
        for b, m in bins.items():
            r = allr[m[allr]]
            y = Mt.landmark_labels(c("event")[r], c("tte")[r], 12); ok = ~np.isnan(y)
            p = Mt.risk_at(P["S"][r], 12)[ok]
            out.append(dict(model=name, bin=b, n=int(ok.sum()), auc12=roc_auc_score(y[ok], p),
                            ece12=Mt.ece(p, y[ok]), brier12=float(np.mean((p - y[ok]) ** 2)),
                            mean_risk=float(p.mean()), obs_rate=float(y[ok].mean()),
                            risk_sd=float(np.nanmean(P["risk_sd"][r][ok]))))
    json.dump(out, open(f"runs/staleness_s{seed}.json", "w"), indent=1)
    return out


def corrupt_images(F, O, rng):
    f = F.astype(np.float32) / 255; o = O.astype(np.float32) / 255
    f = np.stack([np.stack([gaussian_filter(ch, 1.3) for ch in im]) for im in f]) * 0.62
    f = np.clip(f + rng.normal(0, 0.05, f.shape), 0, 1)
    o = np.stack([np.stack([gaussian_filter(b, 0.8) for b in vol]) for vol in o]) * 0.55
    o = np.clip(o * rng.gamma(2.0, 0.5, o.shape), 0, 1)
    return (f * 255).astype(np.uint8), (o * 255).astype(np.uint8)


def corrupt_eval(seed, models=MODELS + ["LTMD-MCD"]):
    F, O, tab, cols = D.load()
    sp = D.split_eyes(tab, cols)
    eid = D.col(tab, cols, "eid").astype(int)
    test_rows = np.where(np.isin(eid, sp["test"]))[0]
    rng = np.random.default_rng(500 + seed)
    flag = np.zeros(len(tab), bool); flag[test_rows[rng.random(len(test_rows)) < 0.5]] = True
    ck = torch.load(f"enc_full_s{seed}.pt", weights_only=False)
    ef, eo = E.FundusEncoder(), E.OCTEncoder(); ef.load_state_dict(ck["ef"]); eo.load_state_dict(ck["eo"])
    fr = np.where(flag)[0]
    Fc, Oc = corrupt_images(F[fr], O[fr], rng)
    tf, to = PT.encode_all(ef, eo, Fc, Oc, None, None, None, None)
    z = np.load(f"tok_full_s{seed}.npz"); TF, TO = z["F"].copy(), z["O"].copy()
    TF[fr], TO[fr] = tf, to
    np.savez(f"tok_corrupt_s{seed}.npz", F=TF, O=TO)
    tabx, colsx, spx, rows, G, _ = EV.setup()
    coh = T.Cohort("corrupt", seed)
    c = lambda n: D.col(tab, cols, n)
    out = []
    for name in models:
        base = name.split("-")[0]
        m = load_model(base, "main", seed)
        P = T.predict(m, coh, sp["test"], mc_dropout=30 if name.endswith("MCD") else 0)
        r = test_rows
        risk = Mt.risk_at(P["S"][r], 12)
        unc = P["risk_sd"][r] if name in ("ULTRA", "LTMD-MCD") else binary_entropy(risk)
        res = dict(model=name, unc_auroc=roc_auc_score(flag[r], unc))
        for lab, msk in [("corrupt", flag[r]), ("clean", ~flag[r])]:
            rr = r[msk]
            mm = Mt.all_metrics(P, tab, cols, rr, G)
            res.update({f"{lab}_{k}": v for k, v in mm.items() if k in ("auc12", "ece12", "ibs", "sev_qwk", "cls_auc")})
        out.append(res); print(res, flush=True)
    json.dump(out, open(f"runs/corrupt_s{seed}.json", "w"), indent=1)


def ensemble(seed, members=5):
    preds = [EV.load_pred("LTMD", "main", seed)] + [EV.load_pred("LTMD", "ens", seed + 10 * j) for j in range(1, members)]
    out = {k: np.mean([p[k] for p in preds], 0) for k in ["S", "cls", "sev", "grid"]}
    out["risk_sd"] = np.std([1 - p["S"][:, 1] for p in preds], 0)
    np.savez_compressed(f"runs/pred_LTMD-Ens_main_s{seed}.npz", **out)


def mcd(seed):
    tab, cols, sp, rows, G, eid = EV.setup()
    coh = T.Cohort("full", seed)
    m = load_model("LTMD", "main", seed)
    eyes = np.concatenate([sp[s] for s in ("val", "cal", "test", "extD", "extE")])
    P = T.predict(m, coh, eyes, mc_dropout=30)
    np.savez_compressed(f"runs/pred_LTMD-MCD_main_s{seed}.npz", **P)


if __name__ == "__main__":
    what, seed = sys.argv[1], int(sys.argv[2])
    {"sweep": oct_sweep, "stale": staleness_eval, "corrupt": corrupt_eval, "ens": ensemble, "mcd": mcd}[what](seed)
