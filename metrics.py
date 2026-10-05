"""Evaluation metrics: landmark progression discrimination/calibration, IPCW Brier, C-index,
diagnosis, severity, lesion localisation, uncertainty."""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, cohen_kappa_score)


def landmark_labels(event, tte, h):
    y = np.where((event > 0) & (tte <= h), 1.0, np.where(tte > h, 0.0, np.nan))
    return y


def risk_at(S, h):
    j = {6: 0, 12: 1, 18: 2, 24: 3}[h]
    return 1 - S[:, j]


def ece(p, y, nb=10):
    bins = np.clip((p * nb).astype(int), 0, nb - 1)
    e = 0.0
    for b in range(nb):
        m = bins == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def km_censor(time, event):
    """Kaplan-Meier estimate of the censoring survival G(t)."""
    order = np.argsort(time)
    t, c = time[order], 1 - event[order]
    uniq = np.unique(t)
    G, g = [], 1.0
    n = len(t)
    for u in uniq:
        at_risk = (t >= u).sum(); d = c[t == u].sum()
        g *= 1 - d / max(at_risk, 1)
        G.append(g)
    G = np.array(G)
    return lambda s: np.interp(s, uniq, G, left=1.0, right=G[-1]) if len(uniq) else np.ones_like(s)


def ipcw_brier(S, event, tte, G):
    """Graf IPCW Brier at 6, 12, 18, 24 months and its integrated mean."""
    out = []
    for j, h in enumerate([6, 12, 18, 24]):
        surv = S[:, j]
        ev = (event > 0) & (tte <= h)
        alive = tte > h
        w1 = np.where(ev, 1.0 / np.maximum(G(tte - 1e-6), 1e-3), 0.0)
        w2 = np.where(alive, 1.0 / max(G(np.array([h]))[0], 1e-3), 0.0)
        out.append(np.mean(w1 * surv ** 2 + w2 * (1 - surv) ** 2))
    return np.array(out), float(np.mean(out))


def cindex(risk, event, tte, horizon=24):
    t = np.minimum(tte, horizon); e = (event > 0) & (tte <= horizon)
    order = np.argsort(t)
    t, e, r = t[order], e[order], risk[order]
    num = den = 0.0
    for i in np.where(e)[0]:
        m = t > t[i]
        den += m.sum()
        num += (r[i] > r[m]).sum() + 0.5 * (r[i] == r[m]).sum()
    return num / max(den, 1)


def progression_metrics(S, event, tte, G):
    r = {}
    for h in (12, 24):
        y = landmark_labels(event, tte, h); m = ~np.isnan(y)
        p = risk_at(S, h)[m]; yy = y[m]
        r[f"auc{h}"] = roc_auc_score(yy, p)
        r[f"brier{h}"] = float(np.mean((p - yy) ** 2))
        r[f"ece{h}"] = ece(p, yy)
        r[f"nll{h}"] = float(-np.mean(yy * np.log(np.clip(p, 1e-6, 1)) + (1 - yy) * np.log(np.clip(1 - p, 1e-6, 1))))
    _, r["ibs"] = ipcw_brier(S, event, tte, G)
    r["cidx"] = cindex(risk_at(S, 24), event, tte)
    return r


def diagnosis_metrics(P, y):
    pred = P.argmax(1)
    return {"cls_auc": roc_auc_score(y, P, multi_class="ovr", average="macro"),
            "cls_bacc": balanced_accuracy_score(y, pred), "cls_f1": f1_score(y, pred, average="macro"),
            "cls_mcc": matthews_corrcoef(y, pred)}


def severity_metrics(P, y):
    pred = P.argmax(1)
    return {"sev_qwk": cohen_kappa_score(y, pred, weights="quadratic"), "sev_acc": float((pred == y).mean())}


def lesion_metrics(P, Y):
    p, y = P.ravel(), Y.ravel()
    b = p > 0.5
    dice = 2 * (b & (y > 0)).sum() / max(b.sum() + (y > 0).sum(), 1)
    return {"les_ap": average_precision_score(y, p), "les_dice": float(dice)}


def all_metrics(pred, tab, cols, rows, G):
    c = lambda n: tab[rows, cols.index(n)]
    r = progression_metrics(pred["S"][rows], c("event"), c("tte"), G)
    r.update(diagnosis_metrics(pred["cls"][rows], c("cls").astype(int)))
    r.update(severity_metrics(pred["sev"][rows], c("stage").astype(int)))
    r.update(lesion_metrics(pred["grid"][rows], np.stack([c(f"g{j}") for j in range(16)], -1)))
    return r


def aurc(unc, loss):
    """Area under the risk-coverage curve (lower is better): reject most uncertain first."""
    order = np.argsort(unc)
    l = loss[order]
    cum = np.cumsum(l) / np.arange(1, len(l) + 1)
    return float(cum.mean()), cum
