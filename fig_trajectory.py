"""Fig. 5: belief trajectory of one test eye with sparse OCT (centre C): block-wise posterior s.d. and
12-month risk with its Monte Carlo spread on a monthly grid between visits."""
import os; os.makedirs("figs", exist_ok=True)
import numpy as np, torch, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data as D, train as T, robustness as R

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                     "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
torch.set_num_threads(2)
seed = 0
coh = T.Cohort("full", seed)
m = R.load_model("ULTRA", "main", seed)
tab, cols = coh.tab, coh.cols
c = lambda n: D.col(tab, cols, n)
eid = c("eid").astype(int)
cands = []
for e in coh.sp["test"]:
    r = coh.rows_by_eye[e]
    if c("centre")[r[0]] == 2 and len(r) >= 5 and c("etype")[r[0]] > 0 and 0 < c("has_oct")[r].sum() < len(r) - 1 \
            and c("event")[r[0]] == 1 and c("tte")[r[0]] > 12:
        cands.append((e, len(r)))
e = sorted(cands, key=lambda x: -x[1])[0][0]
rows = coh.rows_by_eye[e]
b = coh.batch([e])
with torch.no_grad():
    yF, rF = m.observe(b["tF"], m.embF, m.yF, m.rF, m.rF_c)
    yO, rO = m.observe(b["tO"], m.embO, m.yO, m.rO, m.rO_c)
    n = m.n; K = b["t"].shape[1]
    p = m.p0.expand(1, n).clone(); v = torch.zeros(1, n)
    Ppp = torch.exp(m.lPp0).expand(1, n).clone(); Ppv = torch.zeros(1, n); Pvv = torch.exp(m.lPv0).expand(1, n).clone()
    t = b["t"][0].numpy(); grid, sdS, sdO, risk, rsd = [], [], [], [], []
    g = torch.Generator().manual_seed(3); eps = torch.randn(64, 1, 1, n, generator=g)
    nS = 24; nF = 12
    prev = t[0]
    def record(tt, st):
        S, sd = m.hazard(*[x[:, None] for x in st], eps=eps)
        grid.append(tt); risk.append(float(1 - S[0, 0, 1])); rsd.append(float(sd[0, 0]))
        sdS.append(float(torch.sqrt(st[2][0, :nS]).mean())); sdO.append(float(torch.sqrt(st[2][0, nS + nF:]).mean()))
    for k in range(K):
        # monthly propagation between visits (no observation)
        for tt in np.arange(prev + 1, t[k], 1.0):
            st = m.predict(p, v, Ppp, Ppv, Pvv, torch.tensor([(tt - prev) / 12.0], dtype=torch.float32))
            record(tt, st)
        st = m.predict(p, v, Ppp, Ppv, Pvv, torch.tensor([(t[k] - prev) / 12.0], dtype=torch.float32))
        st = m.update(*st, yF[:, k], rF[:, k], b["mF"][:, k, None] * m.HF)[:5]
        st = m.update(*st, yO[:, k], rO[:, k], b["mO"][:, k, None] * m.HO)[:5]
        p, v, Ppp, Ppv, Pvv = st
        record(t[k], st)
        prev = t[k]
grid = np.array(grid); risk = np.array(risk); rsd = np.array(rsd)
hasO = c("has_oct")[rows] > 0
stage = c("stage")[rows].astype(int)
fig, axs = plt.subplots(3, 1, figsize=(3.45, 3.9), sharex=True,
                        gridspec_kw=dict(hspace=0.12, height_ratios=[0.32, 1, 1.1]))
ax0 = axs[0]
for tk, ho, sg in zip(t, hasO, stage):
    ax0.plot([tk], [0.0], marker="o", ms=4.6, mfc="black" if ho else "white", mec="black", mew=0.7, ls="none")
    ax0.text(tk, 0.55, str(sg), ha="center", va="bottom", fontsize=6.2)
ax0.set_ylim(-0.6, 1.5); ax0.set_yticks([]); ax0.spines["left"].set_visible(False)
ax0.set_ylabel("visits", fontsize=6.6, rotation=0, ha="right", va="center")
ax0.text(1.0, 1.25, "filled: fundus + OCT; open: fundus only; number: stage", transform=ax0.transAxes,
         ha="right", va="bottom", fontsize=6.0)
axs[1].plot(grid, sdS, color="0.0", lw=1.1, label="shared block $\\mathcal{S}$")
axs[1].plot(grid, sdO, color="0.45", lw=1.1, ls="--", label="OCT-private block $\\mathcal{O}$")
axs[1].set_ylabel("Mean posterior s.d.")
ymax = max(max(sdO), max(sdS))
axs[1].set_ylim(0, ymax * 1.45)
axs[1].legend(loc="upper left", fontsize=6.4, ncol=2, handlelength=2.2, columnspacing=1.0)
axs[2].fill_between(grid, np.clip(risk - rsd, 0, 1), np.clip(risk + rsd, 0, 1), color="0.85", lw=0, label="$\\pm$1 s.d. over belief")
axs[2].plot(grid, risk, color="0.0", lw=1.1, label="$\\rho_{12}$")
ev_t = t[0] + c("tte")[rows[0]]
axs[2].axvline(ev_t, color="0.4", lw=0.8, ls=":")
axs[2].text(ev_t - 0.8, 0.97, "confirmed\nprogression", fontsize=6.2, va="top", ha="right")
axs[2].set_ylim(0, 1); axs[2].set_ylabel("12-month risk"); axs[2].set_xlabel("Time since first visit (months)")
axs[2].legend(loc="upper left", fontsize=6.4, ncol=1)
fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.10)
fig.savefig("figs/fig_trajectory.png", dpi=500)
json.dump({"eye": int(e), "t": t.tolist(), "hasO": hasO.tolist(), "stage": stage.tolist(), "event_t": float(ev_t),
           "sdO_start": float(sdO[0]), "sdO_max": float(max(sdO)), "sdO_min": float(min(sdO)), "sdS_max": float(max(sdS))}, open("runs/traj.json", "w"))
print("eye", e, t, hasO, stage)
