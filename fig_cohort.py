"""Fig. 2: example longitudinal eyes from the simulated cohort (grayscale: fundus luminance, central B-scan)."""
import os; os.makedirs("figs", exist_ok=True)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data as D
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif", "font.size": 7})
F, O, tab, cols = D.load()
c = lambda n: D.col(tab, cols, n)
sp = D.split_eyes(tab, cols); eid = c("eid").astype(int)
names = {1: "Diabetic retinopathy", 2: "Age-related macular degeneration", 3: "Glaucoma"}
chosen = {}
for et in [1, 2, 3]:
    for e in sp["test"]:
        r = np.where(eid == e)[0]
        r = r[np.argsort(c("k")[r])]
        if c("etype")[r[0]] == et and len(r) >= 5 and c("stage")[r[-1]] - c("stage")[r[0]] >= 2 and c("u")[r[0]] > 0.6:
            chosen[et] = r; break
fig, axes = plt.subplots(3, 6, figsize=(7.16, 3.75), gridspec_kw=dict(wspace=0.06, hspace=0.42, width_ratios=[1, 1, 1, 1, 1, 1]))
for i, et in enumerate([1, 2, 3]):
    r = chosen[et]
    picks = [r[0], r[len(r) // 2], r[-1]]
    for j, row in enumerate(picks):
        im = F[row].astype(float) / 255
        g = 0.30 * im[0] + 0.59 * im[1] + 0.11 * im[2]
        lo, hi = np.percentile(g, [1, 99.5])
        ax = axes[i, j]; ax.imshow(g, cmap="gray", vmin=lo, vmax=hi); ax.set_xticks([]); ax.set_yticks([])
        ax.add_patch(plt.Rectangle((16 - 0.5, 20 - 0.5), 24, 24, fill=False, ec="white", lw=0.6, ls="--"))
        ax.set_title(f"$t$={c('t')[row]:.0f} mo, stage {int(c('stage')[row])}", fontsize=6.5, pad=2)
        ax2 = axes[i, j + 3]; ax2.imshow(O[row][4], cmap="gray", vmin=0, vmax=255, aspect="auto"); ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_title(f"OCT B-scan, $t$={c('t')[row]:.0f} mo", fontsize=6.5, pad=2)
    axes[i, 0].set_ylabel(names[et].replace(" macular", "\nmacular").replace("Diabetic ", "Diabetic\n"), fontsize=6.8)
fig.text(0.29, 0.005, "Fundus (luminance); dashed square = OCT macular field", ha="center", fontsize=6.6)
fig.text(0.74, 0.005, "Central OCT B-scan (depth $\\times$ width)", ha="center", fontsize=6.6)
fig.subplots_adjust(left=0.07, right=0.995, top=0.93, bottom=0.06)
fig.savefig("figs/fig2_cohort.png", dpi=500)
print({k: len(v) for k, v in chosen.items()})
