"""Reference ceilings used in Section V-A (simulation-only quantities, never available to any model):
gradient boosting on the true current stage + diagnosis, and on the exact latent severity; RMSE of a
ridge probe from single-visit tokens to the latent severity; test landmark count; centre-level mean risk."""
import json, numpy as np, data as D, metrics as Mt, evaluate as EV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
_, _, tab, cols = D.load_table(); c = lambda n: D.col(tab, cols, n)
sp = D.split_eyes(tab, cols); eid = c("eid").astype(int)
u, st, et = c("u"), c("stage"), c("etype")
gap = np.array([0.5, 1.5, 2.5, 3.5, 99])[st.astype(int)] - u
y = Mt.landmark_labels(c("event"), c("tte"), 12)
tr = np.isin(eid, sp["train"]) & ~np.isnan(y); te = np.isin(eid, sp["test"]) & ~np.isnan(y)
out = {"test_landmarks_12m": int(te.sum())}
for name, F in [("oracle_stage_type", [st, et == 1, et == 2, et == 3]), ("oracle_latent", [u, gap, et == 1, et == 2, et == 3])]:
    X = np.stack(F, 1).astype(float)
    m = GradientBoostingClassifier(random_state=0).fit(X[tr], y[tr])
    out[name] = float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
z = np.load("tok_full_s0.npz"); X = np.concatenate([z["F"].reshape(len(u), -1), z["O"].reshape(len(u), -1)], 1).astype(np.float32)
a = np.isin(eid, sp["train"]); b = np.isin(eid, sp["test"])
p = Ridge(alpha=10).fit(X[a], u[a]).predict(X[b]); out["ridge_rmse_latent"] = float(np.sqrt(np.mean((p - u[b]) ** 2)))
tab2, cols2, sp2, rows, G, _ = EV.setup()
for s in ["test", "extD", "extE"]:
    r = rows[s][~np.isnan(y[rows[s]])]
    out[f"mean_risk_ULTRA_{s}"] = float(np.mean([Mt.risk_at(EV.load_pred("ULTRA", "main", sd)["S"], 12)[r].mean() for sd in range(5)]))
    out[f"observed_rate_{s}"] = float(y[r].mean())
json.dump(out, open("runs/oracle.json", "w"), indent=1); print(out)
