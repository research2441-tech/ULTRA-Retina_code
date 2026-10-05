"""Fig. 1: ULTRA-Retina architecture (grayscale).

Every connector is an explicit orthogonal path between named anchor points, so directions and connections are
exact. An automated audit checks (i) box-box overlap, (ii) text escaping its box or touching another box,
(iii) text-text overlap, and (iv) crossings between connectors of different paths.
"""
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif"})
FS = 6.4
fig = plt.figure(figsize=(7.16, 4.0))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
BOX, PATHS, OWN = {}, [], {}
LS = {"-": "-", "--": (0, (4, 2.2)), ":": (0, (1, 1.5))}


def box(key, x, y, w, h, text, fc="white", fs=FS):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7", fc=fc, ec="black", lw=0.8))
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.3)
    BOX[key] = (x, y, w, h); OWN[t] = key


def frame(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc="none", ec="0.45",
                                lw=0.6, ls=(0, (3, 2))))
    t = ax.text(x + w / 2, y + h - 0.6, title, ha="center", va="top", fontsize=FS, style="italic", color="0.15")
    OWN[t] = None


def A(key, side, f=0.5):
    x, y, w, h = BOX[key]
    return {"l": (x, y + h * f), "r": (x + w, y + h * f), "t": (x + w * f, y + h), "b": (x + w * f, y)}[side]


def path(points, style="-", color="black", lw=0.85, name=None):
    """Orthogonal polyline; arrow head on the final segment, pointing at the last point."""
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=LS[style], solid_capstyle="butt")
    head(points[-2], points[-1], color, lw)
    PATHS.append((name or str(len(PATHS)), points))


def head(p, q, color, lw):
    """Solid arrow head at q, aligned with the segment p->q (keeps heads crisp on dashed or dotted lines)."""
    import math
    L = math.hypot(q[0] - p[0], q[1] - p[1]); u = ((q[0] - p[0]) / L, (q[1] - p[1]) / L)
    s = (q[0] - 1.2 * u[0], q[1] - 1.2 * u[1])
    ax.add_patch(FancyArrowPatch(s, q, arrowstyle="-|>,head_length=0.45,head_width=0.22", lw=lw, ls="-",
                                 color=color, shrinkA=0, shrinkB=0, mutation_scale=10))


def label(x, y, s, **kw):
    t = ax.text(x, y, s, fontsize=kw.pop("fs", FS - 0.4), **kw); OWN[t] = None


# ------------------------------------------------------------------ geometry
RF, RO = 41.0, 29.0                     # centre lines of the fundus row and the OCT row
H = 5.0
box("xF", 2.0, RF - H / 2, 14.5, H, "Fundus photograph\n$x_k^{F}$ (if $m_k^{F}=1$)")
box("xO", 2.0, RO - H / 2, 14.5, H, "Macular OCT volume\n$x_k^{O}$ (if $m_k^{O}=1$)")
frame(19.0, 23.0, 21.0, 26.5, "Stage 1 (frozen)")
box("eF", 20.5, RF - H / 2, 18.0, H, "Fundus encoder $E_F$\n16 cell + 1 global token")
box("eO", 20.5, RO - H / 2, 18.0, H, "OCT encoder $E_O$\n16 cell + 1 global token")
box("al", 20.5, 33.2, 18.0, 3.6, "shared halves aligned by $\\mathcal{L}_{\\mathrm{nce}}$", fc="0.92", fs=FS - 0.6)
box("oF", 42.5, RF - H / 2, 12.0, H, "Observation\n$y_k^{F},\\ r_k^{F}$")
box("oO", 42.5, RO - H / 2, 12.0, H, "Observation\n$y_k^{O},\\ r_k^{O}$")
frame(57.0, 18.5, 22.0, 37.0, "Evidence-ageing belief filter")
box("tu", 59.0, 47.0, 16.0, 5.0, "Time update over $\\Delta_k$:\n$\\mathbf{F}(\\Delta_k)$, $\\mathbf{Q}(\\Delta_k)$")
box("uf", 59.0, RF - H / 2, 16.0, H, "Fundus update on\nblocks $\\mathcal{S}\\cup\\mathcal{F}$")
box("uo", 59.0, RO - H / 2, 16.0, H, "OCT update on\nblocks $\\mathcal{S}\\cup\\mathcal{O}$")
box("bel", 59.0, 20.5, 16.0, 3.5, "Belief $\\mathcal{B}_k$", fc="0.88")
box("val", 82.0, 47.0, 15.8, 6.0, "OCT acquisition value\n$V_k$ and order $a_k$")
box("haz", 82.0, 36.5, 15.8, 7.0, "Belief-integrated\nhazard (prior $\\psi_j$ + MC)\n$S_j,\\ \\rho_{12},\\ \\rho_{24}$")
box("hd", 82.0, 26.0, 15.8, 6.5, "Diagnosis, severity\nand lesion-grid heads")
box("loss", 42.5, 4.0, 38.0, 6.5,
    "$\\mathcal{L}=\\mathcal{L}_{\\mathrm{surv}}+\\frac{1}{2}(\\mathcal{L}_{\\mathrm{cls}}+\\mathcal{L}_{\\mathrm{sev}}"
    "+\\mathcal{L}_{\\mathrm{grid}})+\\lambda_{p}\\mathcal{L}_{\\mathrm{innov}}$", fc="0.95", fs=FS + 0.4)

# ------------------------------------------------------------------ forward pass (solid)
path([A("xF", "r"), A("eF", "l")], name="f1"); path([A("xO", "r"), A("eO", "l")], name="f2")
path([A("eF", "r"), A("oF", "l")], name="f3"); path([A("eO", "r"), A("oO", "l")], name="f4")
path([A("eF", "b"), A("al", "t")], name="f5"); path([A("eO", "t"), A("al", "b")], name="f6")
path([A("oF", "r"), A("uf", "l")], name="f7"); path([A("oO", "r"), A("uo", "l")], name="f8")
path([A("tu", "b"), A("uf", "t")], name="f9"); path([A("uf", "b"), A("uo", "t")], name="f10")
path([A("uo", "b"), A("bel", "t")], name="f11")
# recurrence k -> k+1 on the right edge of the filter column
yb = A("bel", "r", 0.75)[1]
path([A("bel", "r", 0.75), (77.2, yb), (77.2, A("tu", "r")[1]), A("tu", "r")], name="rec")
label(77.6, 37.0, "$k\\!\\rightarrow\\!k\\!+\\!1$", rotation=90, ha="left", va="center")
# belief bus to the heads
bus_x, yb0 = 80.6, A("bel", "r", 0.25)[1]
ax.plot([A("bel", "r", 0.25)[0], bus_x], [yb0, yb0], color="black", lw=0.85)
ax.plot([bus_x, bus_x], [yb0, A("val", "l")[1]], color="black", lw=0.85)
PATHS.append(("bus", [A("bel", "r", 0.25), (bus_x, yb0), (bus_x, A("val", "l")[1])]))
path([(bus_x, A("hd", "l")[1]), A("hd", "l")], name="b1")
path([(bus_x, A("haz", "l")[1]), A("haz", "l")], name="b2")
path([(bus_x, A("val", "l")[1]), A("val", "l")], name="b3")
path([A("haz", "t"), A("val", "b")], name="f12")
# acquisition loop (dotted) over the top margin and down the left margin into the OCT input
path([A("val", "t"), (A("val", "t")[0], 56.0), (0.8, 56.0), (0.8, RO), A("xO", "l")], style=":", lw=1.0, name="acq")
label(40.0, 56.6, "acquisition decision $a_k$ sets $m_{k}^{O}$ for the current visit", ha="center", va="bottom")

# ------------------------------------------------------------------ reverse-mode gradient (dashed)
G = "0.3"
path([A("loss", "t", 0.12), A("oO", "b", (A("loss", "t", 0.12)[0] - 42.5) / 12.0)], style="--", color=G, name="g1")
gx = A("bel", "b")[0]
path([(gx, A("loss", "t")[1]), A("bel", "b")], style="--", color=G, name="g2")
path([A("loss", "r", 0.7), (A("hd", "b")[0], A("loss", "r", 0.7)[1]), A("hd", "b")], style="--", color=G, name="g3")
path([A("loss", "r", 0.3), (99.3, A("loss", "r", 0.3)[1]), (99.3, A("haz", "r")[1]), A("haz", "r")], style="--", color=G, name="g4")
label(21.0, 13.0, "no gradient enters $E_F$ or $E_O$\nduring stage 2", ha="left", va="center", color="0.2")

# ------------------------------------------------------------------ legend
for i, (st, lab, col) in enumerate([("-", "forward pass", "black"), ("--", "reverse-mode gradient", G),
                                    (":", "acquisition loop", "black")]):
    yy = 9.0 - 3.0 * i
    ax.plot([2.0, 7.0], [yy, yy], color=col, lw=0.85 if st != ":" else 1.0, ls=LS[st])
    head((2.0, yy), (7.0, yy), col, 0.85)
    label(8.0, yy, lab, ha="left", va="center")

import os; os.makedirs("figs", exist_ok=True); fig.savefig("figs/fig1_architecture.png", dpi=600)


# ------------------------------------------------------------------ audit
def audit():
    bad = []
    for a, b in itertools.combinations(BOX, 2):
        x1, y1, w1, h1 = BOX[a]; x2, y2, w2, h2 = BOX[b]
        if x1 < x2 + w2 and x2 < x1 + w1 and y1 < y2 + h2 and y2 < y1 + h1:
            bad.append(("box-box", a, b))
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = ax.transData.inverted()
    ext = {}
    for t, owner in OWN.items():
        bb = t.get_window_extent(r); (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        ext[t] = (x0, y0, x1, y1)
        for k, (bx, by, bw, bh) in BOX.items():
            hit = x0 < bx + bw and bx < x1 and y0 < by + bh and by < y1
            inside = x0 >= bx + 0.3 and x1 <= bx + bw - 0.3 and y0 >= by + 0.2 and y1 <= by + bh - 0.2
            if k == owner and not inside:
                bad.append(("text-escapes-box", k))
            if k != owner and hit:
                bad.append(("text-touches-box", t.get_text()[:25], k))
        if x0 < 0 or x1 > 100 or y0 < 0 or y1 > 60:
            bad.append(("text-outside-canvas", t.get_text()[:25]))
    for t1, t2 in itertools.combinations(ext, 2):
        a, b = ext[t1], ext[t2]
        if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
            bad.append(("text-text", t1.get_text()[:20], t2.get_text()[:20]))
    segs = [(n, p, q) for n, pts in PATHS for p, q in zip(pts[:-1], pts[1:])]
    for (n1, a, b), (n2, c, d) in itertools.combinations(segs, 2):
        if n1 == n2 or {n1, n2} <= {"bus", "b1", "b2", "b3"}:
            continue
        if seg_cross(a, b, c, d):
            bad.append(("connector-crossing", n1, n2))
    # connectors must not pass through boxes other than at their own endpoints
    for n, p, q in segs:
        for k, (bx, by, bw, bh) in BOX.items():
            if seg_box(p, q, (bx + 0.05, by + 0.05, bw - 0.1, bh - 0.1)):
                bad.append(("connector-through-box", n, k))
    # connectors must not cross text
    for n, p, q in segs:
        for t, (x0, y0, x1, y1) in ext.items():
            if OWN[t] is None and seg_box(p, q, (x0, y0, x1 - x0, y1 - y0)):
                bad.append(("connector-through-text", n, t.get_text()[:20]))
    return bad


def seg_cross(a, b, c, d):
    def o(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return 0 if abs(v) < 1e-9 else (1 if v > 0 else -1)
    if max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0]) or \
       max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1]):
        return False
    shared = any(abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6 for p in (a, b) for q in (c, d))
    o1, o2, o3, o4 = o(a, b, c), o(a, b, d), o(c, d, a), o(c, d, b)
    return (o1 * o2 < 0 and o3 * o4 < 0) and not shared


def seg_box(p, q, rect):
    x, y, w, h = rect
    n = 60
    for i in range(1, n):
        s = i / n
        px, py = p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1])
        if x < px < x + w and y < py < y + h:
            return True
    return False


if __name__ == "__main__":
    res = audit()
    print("FIG1 AUDIT:", "PASS" if not res else res)
