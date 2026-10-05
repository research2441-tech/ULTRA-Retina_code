"""Experiment E10: belief-driven OCT acquisition under a budget.

At each visit the fundus photograph is acquired first. The belief is time-updated and fundus-updated; the
policy then decides whether to acquire OCT. The value of OCT is the closed-form expected reduction of the
variance of 12-month progression risk: the Kalman posterior covariance does not depend on the yet-unseen
OCT values, only on the expected OCT noise r_bar (mean heteroscedastic noise on the calibration split).
Policies: value (proposed), uncertainty-only, risk-threshold, random, fixed-interval.
"""
import sys, json, numpy as np, torch
import data as D, models as M, train as T, metrics as Mt, evaluate as EV

torch.set_num_threads(2)


@torch.no_grad()
def risk_var(model, p, v, Ppp, Ppv, Pvv, eps):
    S, sd = model.hazard(p[:, None], v[:, None], Ppp[:, None], Ppv[:, None], Pvv[:, None], eps=eps[:, :, None])
    return (1 - S[:, 0, 1]), sd[:, 0] ** 2


@torch.no_grad()
def preposterior_mc(model, p, v, Ppp, Ppv, Pvv, rbar, eps, n_y=16):
    """Exact Monte Carlo preposterior value: variance, over simulated OCT observations, of the posterior mean
    12-month risk (law of total variance: equals the expected reduction of risk variance)."""
    B, n = p.shape
    g = torch.Generator().manual_seed(11)
    S0 = Ppp + rbar
    means = []
    for _ in range(n_y):
        y = p + torch.sqrt(S0) * torch.randn(B, n, generator=g)
        st = model.update(p, v, Ppp, Ppv, Pvv, y, rbar.expand(B, n), model.HO.expand(B, n))[:5]
        r, _ = risk_var(model, *st, eps)
        means.append(r)
    return torch.stack(means).var(0)


@torch.no_grad()
def rollout(model, coh, eyes, policy, thr, rbarO, rng, n_mc=32):
    """Return OCT acquisition mask (per table row) chosen sequentially by the policy."""
    mO_all = np.zeros(len(coh.tab))
    mF_all = np.ones(len(coh.tab))
    for i in range(0, len(eyes), 64):
        el = list(eyes[i:i + 64])
        b = coh.batch(el, mF_over=mF_all, mO_over=np.ones(len(coh.tab)))
        yF, rF = model.observe(b["tF"], model.embF, model.yF, model.rF, model.rF_c)
        yO, rO = model.observe(b["tO"], model.embO, model.yO, model.rO, model.rO_c)
        B, K = b["t"].shape; n = model.n
        p = model.p0.expand(B, n).clone(); v = torch.zeros(B, n)
        Ppp = torch.exp(model.lPp0).expand(B, n).clone(); Ppv = torch.zeros(B, n)
        Pvv = torch.exp(model.lPv0).expand(B, n).clone() if model.velocity else torch.zeros(B, n)
        prev = b["t"][:, 0]
        g = torch.Generator().manual_seed(7)
        eps = torch.randn(n_mc, B, n, generator=g)
        since = np.zeros(B)
        for k in range(K):
            dt = (b["t"][:, k] - prev) / 12.0
            p, v, Ppp, Ppv, Pvv = model.predict(p, v, Ppp, Ppv, Pvv, dt)
            mf = b["mF"][:, k, None] * model.HF
            p, v, Ppp, Ppv, Pvv, _ = model.update(p, v, Ppp, Ppv, Pvv, yF[:, k], rF[:, k], mf)
            valid = b["valid"][:, k].numpy() > 0
            risk, var0 = risk_var(model, p, v, Ppp, Ppv, Pvv, eps)
            # preposterior: covariance after a hypothetical OCT with expected noise
            mo = model.HO.expand(B, n)
            _, _, Pq, Pqv, Pqvv, _ = model.update(p, v, Ppp, Ppv, Pvv, p, rbarO.expand(B, n), mo)
            _, var1 = risk_var(model, p, v, Pq, Pqv, Pqvv, eps)
            value = (var0 - var1).numpy()
            if policy == "value_mc":
                value = preposterior_mc(model, p, v, Ppp, Ppv, Pvv, rbarO, eps).numpy()
            since += dt.numpy()
            if policy in ("value", "value_mc"):
                dec = value > thr
            elif policy == "uncert":
                dec = var0.numpy() > thr
            elif policy == "risk":
                dec = risk.numpy() > thr
            elif policy == "random":
                dec = rng.random(B) < thr
            elif policy == "interval":                       # OCT when the last OCT is older than thr months
                dec = (since * 12 >= thr) | (k == 0)
            elif policy == "all":
                dec = np.ones(B, bool)
            else:
                dec = np.zeros(B, bool)
            dec &= valid
            since[dec] = 0
            mo = torch.tensor(dec, dtype=torch.float32)[:, None] * model.HO
            p, v, Ppp, Ppv, Pvv, _ = model.update(p, v, Ppp, Ppv, Pvv, yO[:, k], rO[:, k], mo)
            prev = b["t"][:, k]
            idx = b["idx"][:, k]
            mO_all[idx[idx >= 0]] = dec[idx >= 0]
    return mO_all, mF_all


def evaluate_policy(model, coh, eyes, mO, mF, G):
    P = T.predict(model, coh, eyes, mF_over=mF, mO_over=mO)
    rows = np.where(np.isin(coh.eid, eyes))[0]
    c = lambda n: D.col(coh.tab, coh.cols, n)
    r = Mt.progression_metrics(P["S"][rows], c("event")[rows], c("tte")[rows], G)
    r.update(Mt.severity_metrics(P["sev"][rows], c("stage")[rows].astype(int)))
    y = Mt.landmark_labels(c("event")[rows], c("tte")[rows], 12); m = ~np.isnan(y)
    risk = Mt.risk_at(P["S"][rows], 12)[m]; yy = y[m]
    thr = np.quantile(risk[yy == 0], 0.8)                      # sensitivity at 80% specificity
    r["sens80"] = float((risk[yy == 1] > thr).mean())
    r["budget"] = float(mO[rows].mean())
    return r


def main(seed, split="test"):
    coh = T.Cohort("full", seed)
    import robustness as R
    model = R.load_model("ULTRA", "main", seed)
    tab, cols, sp, rows, G, eid = EV.setup()
    with torch.no_grad():                                      # expected OCT noise from the calibration split
        cb = coh.batch(list(coh.sp["cal"]))
        _, rO = model.observe(cb["tO"], model.embO, model.yO, model.rO, model.rO_c)
        m = (cb["mO"] > 0)
        rbar = rO[m].mean(0)
    eyes = coh.sp[split]
    rng = np.random.default_rng(seed)
    out = []
    grids = {"value": None, "value_mc": None, "uncert": None, "risk": None, "random": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
             "interval": [0, 6, 12, 18, 24, 36, 48]}
    # thresholds for value / uncert / risk are chosen as quantiles on the calibration split so that
    # each traces the same range of budgets
    cal_stats = {}
    for pol in [p for p in grids if grids[p] is None]:
        vals = probe(model, coh, list(coh.sp["cal"]), pol, rbar)
        cal_stats[pol] = vals
        grids[pol] = list(np.quantile(vals, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]))
    for pol in ["none", "all"]:
        mO, mF = rollout(model, coh, eyes, pol, 0, rbar, rng)
        out.append(dict(policy=pol, thr=0, **evaluate_policy(model, coh, eyes, mO, mF, G)))
    for pol, ths in grids.items():
        for th in ths:
            mO, mF = rollout(model, coh, eyes, pol, th, rbar, rng)
            out.append(dict(policy=pol, thr=float(th), **evaluate_policy(model, coh, eyes, mO, mF, G)))
            print(out[-1], flush=True)
    json.dump(out, open(f"runs/policy_{split}_s{seed}.json", "w"), indent=1)


@torch.no_grad()
def probe(model, coh, eyes, pol, rbar):
    """Distribution of the decision statistic at visits after the first (all-OCT history)."""
    vals = []
    b = coh.batch(eyes, mF_over=np.ones(len(coh.tab)), mO_over=np.ones(len(coh.tab)))
    out = model(b)
    p, v, Ppp, Ppv, Pvv = out["state"]
    B, K, n = p.shape
    # one step ahead: predict to the next visit, fundus update, then the statistic
    yF, rF = model.observe(b["tF"], model.embF, model.yF, model.rF, model.rF_c)
    g = torch.Generator().manual_seed(7); eps = torch.randn(32, B, n, generator=g)
    for k in range(1, K):
        dt = (b["t"][:, k] - b["t"][:, k - 1]) / 12.0
        s = model.predict(p[:, k - 1], v[:, k - 1], Ppp[:, k - 1], Ppv[:, k - 1], Pvv[:, k - 1], dt)
        s = model.update(*s, yF[:, k], rF[:, k], b["mF"][:, k, None] * model.HF)[:5]
        risk, var0 = risk_var(model, *s, eps)
        _, _, Pq, Pqv, Pqvv, _ = model.update(*s, s[0], rbar.expand(B, n), model.HO.expand(B, n))
        _, var1 = risk_var(model, s[0], s[1], Pq, Pqv, Pqvv, eps)
        if pol == "value_mc":
            val = preposterior_mc(model, *s, rbar, eps).numpy()
        else:
            val = {"value": var0 - var1, "uncert": var0, "risk": risk}[pol].numpy()
        vals.append(val[b["valid"][:, k].numpy() > 0])
    return np.concatenate(vals)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "test")
