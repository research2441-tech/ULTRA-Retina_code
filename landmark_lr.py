"""Classical baseline: landmark discrete-time hazard logistic regression with last-observation-carried-forward
OCT features and time since last OCT; multinomial logistic regression for diagnosis and severity."""
import sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import data as D


def features(tab, cols, TF, TO):
    c = lambda n: D.col(tab, cols, n)
    eid, k, t = c("eid").astype(int), c("k").astype(int), c("t")
    hasF, hasO = c("has_fun"), c("has_oct")
    fF = np.concatenate([TF[:, :16].mean(1), TF[:, 16]], 1).astype(np.float32)
    fO = np.concatenate([TO[:, :16].mean(1), TO[:, 16]], 1).astype(np.float32)
    X = np.zeros((len(tab), 128 + 128 + 3), np.float32)
    order = np.lexsort((k, eid))
    lastO, lastt, cur = None, None, -1
    for i in order:
        if eid[i] != cur:
            cur, lastO, lastt = eid[i], None, None
        if hasO[i]:
            lastO, lastt = fO[i], t[i]
        X[i, :128] = fF[i] * hasF[i]
        if lastO is not None:
            X[i, 128:256] = lastO
        X[i, 256] = hasF[i]; X[i, 257] = hasO[i]
        X[i, 258] = (t[i] - lastt) / 12.0 if lastt is not None else 5.0
    return X


def run(seed, tok="full"):
    _, _, tab, cols = D.load_table()
    z = np.load(f"tok_{tok}_s{seed}.npz"); TF, TO = z["F"].astype(np.float32), z["O"].astype(np.float32)
    sp = D.split_eyes(tab, cols)
    c = lambda n: D.col(tab, cols, n)
    eid = c("eid").astype(int)
    tr = np.where(np.isin(eid, sp["train"]))[0]
    X = features(tab, cols, TF, TO)
    sc = StandardScaler().fit(X[tr]); pca = PCA(32, random_state=seed).fit(sc.transform(X[tr]))
    Z = np.concatenate([pca.transform(sc.transform(X)), X[:, 256:]], 1)
    # person-period expansion
    ev, tte = c("event"), c("tte")
    rowsX, rowsY = [], []
    for i in tr:
        for j in range(4):
            if tte[i] <= 6 * j:
                break
            y = 1 if (ev[i] > 0 and 6 * j < tte[i] <= 6 * (j + 1)) else 0
            if ev[i] == 0 and tte[i] < 6 * (j + 1):
                break                                   # censored inside the bin: not at risk for the full bin
            oh = np.zeros(4); oh[j] = 1
            rowsX.append(np.concatenate([Z[i], oh])); rowsY.append(y)
            if y:
                break
    haz = LogisticRegression(max_iter=2000, C=0.5).fit(np.array(rowsX), np.array(rowsY))
    h = np.stack([haz.predict_proba(np.concatenate([Z, np.tile(np.eye(4)[j], (len(Z), 1))], 1))[:, 1]
                  for j in range(4)], 1)
    S = np.cumprod(1 - h, 1)
    clf = LogisticRegression(max_iter=3000, C=0.5).fit(Z[tr], c("cls")[tr].astype(int))
    sev = LogisticRegression(max_iter=3000, C=0.5).fit(Z[tr], c("stage")[tr].astype(int))
    # lesion grid: per-cell logistic on the shared halves of the current cell tokens
    G = np.zeros((len(tab), 16), np.float32)
    for j in range(16):
        Xc = np.concatenate([TF[:, j, :32] * c("has_fun")[:, None], TO[:, j, :32] * c("has_oct")[:, None],
                             c("has_fun")[:, None], c("has_oct")[:, None]], 1)
        lr = LogisticRegression(max_iter=1000, C=0.5).fit(Xc[tr], c(f"g{j}")[tr].astype(int))
        G[:, j] = lr.predict_proba(Xc)[:, 1]
    np.savez_compressed(f"runs/pred_LandmarkLR_main_s{seed}.npz", S=S.astype(np.float32),
                        cls=clf.predict_proba(Z).astype(np.float32), sev=sev.predict_proba(Z).astype(np.float32),
                        grid=G, risk_sd=np.zeros(len(tab), np.float32))


if __name__ == "__main__":
    run(int(sys.argv[1]))
