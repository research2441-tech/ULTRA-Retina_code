"""Mechanistic longitudinal fundus + OCT cohort simulator (RetinaSim-L).

Every eye carries a latent continuous severity u(t) whose speed varies between eyes and can change
abruptly (change points, treatment-like regression). Biomarkers are generated from u(t) and rendered
into a 3x64x64 colour fundus photograph and an 8x32x32 macular OCT volume. Some biomarkers are visible
in only one modality (peripheral haemorrhages and cup-to-disc ratio in the fundus; small drusen,
subretinal fluid and RNFL thinning in OCT); macular exudates, drusen and haemorrhages are visible in
both at the same anatomical cell, which is what the cross-modal alignment exploits.

Eyes are right-eye oriented after preprocessing (left eyes are mirrored at acquisition and flipped back).
"""
import numpy as np
from scipy.ndimage import gaussian_filter

H = W = 64
MAC_C = np.array([28.0, 32.0])        # macula (x, y) in the right-eye frame
DISC_C = np.array([48.0, 31.0])       # optic disc centre
MAC_X0, MAC_Y0, MAC_S = 16.0, 20.0, 24.0   # macular square covered by OCT and by the 4x4 lesion grid
N_SCAN, OCT_D, OCT_W = 8, 32, 32
STAGE_THR = np.array([0.5, 1.5, 2.5, 3.5])
CLASSES = ["none", "DR", "AMD", "glaucoma"]

# acquisition device per centre: fundus gain, gamma, colour cast, blur, noise, vignette;
# OCT speckle looks (lower = noisier), signal, blur; mean visit interval (months); OCT availability
CENTRES = {
    "A": dict(gain=1.00, gamma=1.00, cast=(1.00, 1.00, 1.00), blur=0.45, noise=0.025, vig=0.30,
              looks=6.0, signal=1.00, oblur=0.40, interval=6.0, p_oct=0.90, p_fun=0.97),
    "B": dict(gain=0.92, gamma=1.10, cast=(1.05, 0.95, 0.90), blur=0.60, noise=0.035, vig=0.38,
              looks=5.0, signal=0.92, oblur=0.50, interval=9.0, p_oct=0.60, p_fun=0.96),
    "C": dict(gain=1.08, gamma=0.92, cast=(0.95, 1.04, 1.08), blur=0.50, noise=0.030, vig=0.25,
              looks=4.0, signal=0.88, oblur=0.45, interval=12.0, p_oct=0.35, p_fun=0.95),
    # external centres: unseen devices
    "D": dict(gain=0.85, gamma=1.22, cast=(1.10, 0.90, 0.85), blur=0.80, noise=0.045, vig=0.45,
              looks=3.0, signal=0.80, oblur=0.65, interval=8.0, p_oct=0.70, p_fun=0.95),
    "E": dict(gain=1.15, gamma=0.85, cast=(0.90, 1.08, 1.12), blur=0.70, noise=0.040, vig=0.20,
              looks=3.5, signal=0.85, oblur=0.55, interval=14.0, p_oct=0.25, p_fun=0.94),
}
TYPE_P = {"A": [0.30, 0.30, 0.20, 0.20], "B": [0.30, 0.34, 0.18, 0.18], "C": [0.32, 0.30, 0.18, 0.20],
          "D": [0.30, 0.30, 0.20, 0.20], "E": [0.28, 0.36, 0.16, 0.20]}

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def stage_of(u):
    return np.searchsorted(STAGE_THR, u, side="right")


# ----------------------------------------------------------------------------- latent course
def simulate_course(rng, etype, months):
    """Monthly latent severity on grid 0..months. Returns u (months+1,), change-point flag, regression flag."""
    t = np.arange(months + 1, dtype=np.float64)
    if etype == 0:
        u = np.full_like(t, rng.uniform(0.0, 0.25))
        onset = rng.random() < 0.10                      # incident disease in an initially healthy eye
        if onset:
            t0 = rng.uniform(6, months)
            r = rng.lognormal(np.log(0.45), 0.5)
            u = u + np.clip(t - t0, 0, None) / 12.0 * r
        return u + np.cumsum(rng.normal(0, 0.012, t.size)), int(onset), 0
    u0 = rng.uniform(0.3, 2.9)
    rate = rng.lognormal(np.log(0.38), 0.75)            # stage units per year, eye-specific
    rate_t = np.full(t.size, rate)
    cp = 0
    if rng.random() < 0.40:                               # change point: acceleration or deceleration
        tc = rng.uniform(0, months)
        rate_t[t >= tc] = rate * np.exp(rng.normal(0.2, 0.7))
        cp = 1
    du = rate_t / 12.0 + rng.normal(0, 0.03, t.size)
    u = u0 + np.concatenate([[0.0], np.cumsum(du[1:])])
    reg = 0
    if etype == 1 and rng.random() < 0.15:               # treatment-like regression in DR
        tr = rng.uniform(6, months)
        u = np.where(t >= tr, u - 0.9, u)
        reg = 1
    return np.clip(u, 0.0, 4.6), cp, reg


# ----------------------------------------------------------------------------- lesion sites
def make_sites(rng, etype):
    """Persistent lesion sites (x, y, threshold, size, kind). Kinds: 0 haem, 1 exudate, 2 drusen, 3 pigment."""
    sites = []
    def add(n, kind, lo, hi, mac_frac, size_lo, size_hi):
        for _ in range(n):
            if rng.random() < mac_frac:
                x = rng.uniform(MAC_X0 + 1, MAC_X0 + MAC_S - 1); y = rng.uniform(MAC_Y0 + 1, MAC_Y0 + MAC_S - 1)
            else:
                ang = rng.uniform(0, 2 * np.pi); rad = rng.uniform(8, 27)
                x = 32 + rad * np.cos(ang); y = 32 + rad * np.sin(ang)
            sites.append((x, y, rng.uniform(lo, hi), rng.uniform(size_lo, size_hi), kind))
    if etype == 1:
        add(55, 0, 0.35, 4.2, 0.30, 0.6, 1.2)
        add(22, 1, 1.30, 4.0, 0.75, 0.5, 0.9)
    elif etype == 2:
        add(36, 2, 0.20, 3.2, 1.00, 0.5, 1.1)
        add(9, 3, 1.80, 3.6, 1.00, 0.6, 1.0)
    elif etype == 0:
        add(3, 2, 0.00, 0.30, 1.00, 0.5, 0.8)            # age-related hard drusen: a confounder
    return np.array(sites, dtype=np.float32).reshape(-1, 5)


def biomarkers(rng, etype, u, sites, eye):
    """State of every biomarker at one visit."""
    bm = {}
    vis = np.zeros(len(sites), dtype=bool)
    if len(sites):
        vis = u > sites[:, 2]
        flick = rng.random(len(sites)) < 0.85          # haemorrhages come and go
        vis &= np.where(sites[:, 4] == 0, flick, True)
    bm["vis"] = vis
    bm["dme"] = float(etype == 1 and u > 2.0 and rng.random() < 0.65) * min(1.0, (u - 2.0) / 1.5 + 0.3)
    bm["nv_dr"] = float(etype == 1 and u > 3.5)
    bm["nvamd"] = float(etype == 2 and u > 3.0) * min(1.0, (u - 3.0) / 1.0 + 0.4)
    bm["amd_haem"] = float(etype == 2 and u > 3.4)
    g = u if etype == 3 else 0.0
    bm["cdr"] = float(np.clip(eye["cdr0"] + 0.13 * g, 0.15, 0.95))
    bm["rnfl"] = float(np.clip(1.0 - 0.16 * g, 0.2, 1.1))
    bm["disc_haem"] = float(etype == 3 and u > 1.0 and rng.random() < 0.10)
    return bm


def lesion_grid(sites, bm):
    """4x4 macular cell labels: any visible macular lesion, fluid or pigment touching the cell."""
    gl = np.zeros((4, 4), dtype=np.uint8)
    for (x, y, th, s, k), v in zip(sites, bm["vis"]):
        if not v:
            continue
        cx = (x - MAC_X0) / 6.0; cy = (y - MAC_Y0) / 6.0
        if 0 <= cx < 4 and 0 <= cy < 4:
            gl[int(cy), int(cx)] = 1
    if bm["dme"] > 0 or bm["nvamd"] > 0:
        gl[1:3, 1:3] = 1                                   # foveal cells: macular fluid
    return gl


# ----------------------------------------------------------------------------- rendering
def _blob(cx, cy, sig):
    return np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * sig * sig))


def render_fundus(rng, eye, sites, bm, dev, quality):
    img = np.empty((3, H, W), np.float32)
    base = np.array([0.80, 0.38, 0.15], np.float32)
    r2 = ((XX - 32) ** 2 + (YY - 32) ** 2) / (30.0 ** 2)
    illum = 1.0 - dev["vig"] * r2 + eye["illum"] * (XX - 32) / 64.0
    for c in range(3):
        img[c] = base[c] * illum
    mac = _blob(*MAC_C, 5.0)
    img *= (1 - 0.28 * mac)[None]

    def paint(mask, col, alpha=1.0):
        m = np.clip(mask * alpha, 0, 1)[None]
        img[:] = img * (1 - m) + np.asarray(col, np.float32)[:, None, None] * m

    # vessels: persistent arcs from the disc
    for (a0, curv, length) in eye["vessels"]:
        s = np.linspace(0, 1, 60)
        ang = a0 + curv * s
        xs = DISC_C[0] + np.cumsum(np.cos(ang)) * length / 60
        ys = DISC_C[1] + np.cumsum(np.sin(ang)) * length / 60
        vm = np.zeros((H, W), np.float32)
        for x, y in zip(xs[::2], ys[::2]):
            vm = np.maximum(vm, _blob(x, y, 0.65))
        paint(vm, (0.50, 0.10, 0.07), 0.85)
    disc = np.clip(1.6 - np.sqrt((XX - DISC_C[0]) ** 2 + (YY - DISC_C[1]) ** 2) / 4.2, 0, 1)
    paint(disc, (0.96, 0.82, 0.55))
    cup = np.clip(1.6 - np.sqrt((XX - DISC_C[0]) ** 2 + (YY - DISC_C[1]) ** 2) / (4.6 * bm["cdr"]), 0, 1)
    paint(cup, (1.00, 0.96, 0.82), 0.9)
    if bm["disc_haem"]:
        paint(_blob(DISC_C[0] - 1, DISC_C[1] + 5, 0.9), (0.40, 0.04, 0.03))
    cols = {0: (0.42, 0.05, 0.03), 1: (1.00, 0.92, 0.42), 2: (0.95, 0.82, 0.48), 3: (0.22, 0.10, 0.05)}
    for (x, y, th, s, k), v in zip(sites, bm["vis"]):
        if not v:
            continue
        grow = 1.0 if k != 2 else min(1.5, 0.7 + 0.4 * (bm["u"] - th))
        paint(_blob(x, y, s * grow), cols[int(k)], 0.95 if k != 2 else 0.75)
    if bm["nv_dr"]:
        for j in range(6):
            paint(_blob(DISC_C[0] + rng.normal(0, 3), DISC_C[1] + rng.normal(0, 3), 0.5), (0.55, 0.05, 0.05))
    if bm["amd_haem"]:
        paint(_blob(MAC_C[0] + 1, MAC_C[1] + 1, 2.8), (0.35, 0.05, 0.02), 0.9)
    # acquisition: misregistration, colour cast, gamma, blur, noise
    sh = rng.normal(0, 0.7, 2)
    img = np.roll(img, (int(round(sh[1])), int(round(sh[0]))), axis=(1, 2))
    img *= np.asarray(dev["cast"], np.float32)[:, None, None] * dev["gain"] * quality["gain"]
    img = np.clip(img, 0, 1) ** dev["gamma"]
    bl = dev["blur"] + quality["blur"]
    img = np.stack([gaussian_filter(ch, bl) for ch in img])
    img += rng.normal(0, dev["noise"] + quality["noise"], img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def render_oct(rng, eye, sites, bm, dev, quality):
    vol = np.zeros((N_SCAN, OCT_D, OCT_W), np.float32)
    xs = MAC_X0 + (np.arange(OCT_W) + 0.5) * MAC_S / OCT_W
    zz = np.arange(OCT_D, dtype=np.float32)[:, None]
    tilt = rng.normal(0, 0.03)
    for j in range(N_SCAN):
        yb = MAC_Y0 + (j + 0.5) * MAC_S / N_SCAN
        d2 = (xs - MAC_C[0]) ** 2 + (yb - MAC_C[1]) ** 2
        z_ilm = 8.0 + 3.0 * np.exp(-d2 / (2 * 3.0 ** 2))
        z_ilm -= 4.5 * bm["dme"] * np.exp(-d2 / (2 * 5.0 ** 2))
        z_rpe = np.full(OCT_W, 24.0)
        for (x, y, th, s, k), v in zip(sites, bm["vis"]):
            if v and k == 2 and abs(y - yb) < 2.2:
                h = min(2.6, 0.9 + 0.9 * (bm["u"] - th)) * s
                z_rpe -= h * np.exp(-(xs - x) ** 2 / (2 * (1.1 * s) ** 2))
        if bm["nvamd"] > 0:
            z_rpe -= 2.5 * bm["nvamd"] * np.exp(-d2 / (2 * 3.5 ** 2))       # PED
        side = 1.0 + 0.35 * (xs - MAC_C[0]) / 12.0                          # thicker toward the disc
        t_rnfl = (1.2 + 2.4 * (1 - np.exp(-d2 / (2 * 6.0 ** 2)))) * bm["rnfl"] * side
        z_nfl = z_ilm + t_rnfl
        z_onl = z_nfl + 0.45 * (z_rpe - 2 - z_nfl)
        off = tilt * (np.arange(OCT_W) - OCT_W / 2) + quality["zshift"]
        z_ilm, z_nfl, z_onl, z_rpe = z_ilm + off, z_nfl + off, z_onl + off, z_rpe + off
        b = np.full((OCT_D, OCT_W), 0.04, np.float32)
        b = np.where((zz >= z_ilm) & (zz < z_nfl), 0.85, b)
        inner = (zz >= z_nfl) & (zz < z_onl)
        b = np.where(inner, 0.42 + 0.12 * np.sin((zz - z_nfl) * 1.9), b)
        b = np.where((zz >= z_onl) & (zz < z_rpe - 2), 0.18, b)
        b = np.where((zz >= z_rpe - 2) & (zz < z_rpe + 1.5), 0.95, b)
        b = np.where(zz >= z_rpe + 1.5, 0.48 + 0.08 * np.sin(zz * 2.7 + xs[None] * 0.8), b)
        if bm["dme"] > 0:                                                    # intraretinal cysts
            cyst = np.exp(-((xs - MAC_C[0]) ** 2) / (2 * (2.5 + 2 * bm["dme"]) ** 2) - (yb - MAC_C[1]) ** 2 / 30)
            zc = (z_nfl + z_onl) / 2 + 1.0
            b = np.where((np.abs(zz - zc) < 1.8 * bm["dme"] + 0.5) & (cyst > 0.35), 0.03, b)
        if bm["nvamd"] > 0:                                                  # subretinal fluid
            srf = np.exp(-d2 / (2 * 3.0 ** 2))
            b = np.where((zz >= z_rpe - 2 - 2.2 * bm["nvamd"] * srf) & (zz < z_rpe - 2) & (srf > 0.25), 0.03, b)
        for (x, y, th, s, k), v in zip(sites, bm["vis"]):
            if v and k == 1 and abs(y - yb) < 1.6:                            # hyper-reflective foci
                col = int(np.clip(round((x - MAC_X0) / MAC_S * OCT_W), 0, OCT_W - 1))
                zf = int(np.clip(round(z_onl[col] - 0.5), 0, OCT_D - 1))
                b[max(zf - 1, 0):zf + 1, max(col - 1, 0):col + 1] = 0.92
            if v and k == 0 and abs(y - yb) < 1.2:                            # haemorrhage shadow
                col = int(np.clip(round((x - MAC_X0) / MAC_S * OCT_W), 0, OCT_W - 1))
                b[:, max(col - 1, 0):col + 1][zz[:, 0] > z_rpe[col]] *= 0.55
        vol[j] = b
    vol *= dev["signal"] * quality["signal"]
    vol = np.stack([gaussian_filter(v, dev["oblur"] + quality["oblur"]) for v in vol])
    looks = dev["looks"]
    vol = vol * rng.gamma(looks, 1.0 / looks, vol.shape).astype(np.float32)
    vol += rng.normal(0, 0.02, vol.shape).astype(np.float32)
    return np.clip(vol, 0, 1)


# ----------------------------------------------------------------------------- cohort
def visit_quality(rng):
    bad = rng.random() < 0.12                              # occasional poor-quality acquisition
    return dict(gain=rng.normal(1.0, 0.05), blur=(rng.uniform(0.6, 1.6) if bad else 0.0),
                noise=(0.03 if bad else 0.0), signal=(rng.uniform(0.6, 0.8) if bad else 1.0),
                oblur=(rng.uniform(0.4, 0.9) if bad else 0.0), zshift=rng.normal(0, 0.8), bad=bad)


def simulate_eye(rng, centre, eid):
    dev = CENTRES[centre]
    etype = rng.choice(4, p=TYPE_P[centre])
    fu = rng.uniform(24, 66)                               # follow-up length (months)
    months = int(np.ceil(fu)) + 30
    u, cp, reg = simulate_course(rng, etype, months)
    eye = dict(cdr0=rng.normal(0.36, 0.07) if rng.random() > 0.08 else rng.uniform(0.5, 0.6),
               illum=rng.normal(0, 0.15),
               vessels=[(a, rng.normal(0, 0.9), rng.uniform(26, 44)) for a in
                        [np.pi - 0.6 + rng.normal(0, .15), np.pi - 0.25 + rng.normal(0, .1),
                         np.pi + 0.25 + rng.normal(0, .1), np.pi + 0.6 + rng.normal(0, .15),
                         -0.5 + rng.normal(0, .2), 0.5 + rng.normal(0, .2)]])
    sites = make_sites(rng, etype)
    # irregular visit schedule
    times = [0.0]
    while True:
        k = 3.0
        gap = np.clip(rng.gamma(k, dev["interval"] / k), 2.0, 30.0)
        if times[-1] + gap > fu:
            break
        times.append(times[-1] + gap)
    times = np.array(times)
    recs = []
    for t in times:
        ui = float(np.interp(t, np.arange(u.size), u))
        bm = biomarkers(rng, etype, ui, sites, eye); bm["u"] = ui
        q = visit_quality(rng)
        f = render_fundus(rng, eye, sites, bm, dev, q)
        o = render_oct(rng, eye, sites, bm, dev, q)
        st = int(stage_of(ui))
        dom = int(etype) if (ui >= 0.5 and etype > 0) else 0
        recs.append(dict(t=t, u=ui, stage=st, cls=dom, grid=lesion_grid(sites, bm), fundus=f, oct=o,
                         bad=q["bad"], has_oct=rng.random() < dev["p_oct"], has_fun=rng.random() < dev["p_fun"]))
    # an initially healthy eye that develops disease takes a fixed incident class
    if etype == 0:
        inc = 1 if rng.random() < 0.6 else 2
        for r in recs:
            r["cls"] = inc if r["u"] >= 0.5 else 0
    # at least one modality per visit
    for r in recs:
        if not r["has_oct"] and not r["has_fun"]:
            r["has_fun"] = True
    # time to confirmed progression from the latent course: first month after the visit at which the stage
    # exceeds the visit stage and stays above it for three consecutive months; right-censored at the last visit
    stage_m = stage_of(u)
    for i, r in enumerate(recs):
        ev, tt = 0, times[-1] - times[i]
        m0 = int(np.ceil(times[i] + 1e-9))
        for m in range(m0, int(np.floor(times[-1])) + 1):
            if m + 2 < len(stage_m) and (stage_m[m:m + 3] > r["stage"]).all():
                ev, tt = 1, m - times[i]
                break
        r["event"], r["tte"] = ev, tt
    meta = dict(eid=eid, centre=centre, etype=int(etype), cp=cp, reg=reg)
    return meta, recs


def build_cohort(seed, n_per_centre, out_path):
    rng = np.random.default_rng(seed)
    F, O, rows = [], [], []
    eid = 0
    for centre, n in n_per_centre.items():
        for _ in range(n):
            meta, recs = simulate_eye(rng, centre, eid)
            for k, r in enumerate(recs):
                F.append((r["fundus"] * 255).astype(np.uint8)); O.append((r["oct"] * 255).astype(np.uint8))
                rows.append((eid, "ABCDE".index(centre), meta["etype"], meta["cp"], meta["reg"], k, r["t"], r["u"],
                             r["stage"], r["cls"], int(r["has_fun"]), int(r["has_oct"]), int(r["bad"]),
                             r["event"], r["tte"]) + tuple(r["grid"].ravel().tolist()))
            eid += 1
    rows = np.array(rows, dtype=np.float64)
    cols = ["eid", "centre", "etype", "cp", "reg", "k", "t", "u", "stage", "cls", "has_fun", "has_oct", "bad",
            "event", "tte"] + [f"g{i}" for i in range(16)]
    np.savez_compressed(out_path, fundus=np.stack(F), oct=np.stack(O), table=rows, cols=np.array(cols))
    return rows, cols
