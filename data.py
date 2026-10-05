"""Cohort loading, patient-level splits and normalisation."""
import numpy as np

SPLIT_SEED = 7


def load(path="cohort.npz"):
    z = np.load(path)
    cols = list(z["cols"])
    tab = z["table"]
    return z["fundus"], z["oct"], tab, cols


def col(tab, cols, name):
    return tab[:, cols.index(name)]


def split_eyes(tab, cols):
    """Eye-level (= patient-level, one study eye per patient) partition of development centres A-C.
    External centres D (index 3) and E (index 4) are held out entirely."""
    eid = col(tab, cols, "eid").astype(int); cen = col(tab, cols, "centre").astype(int)
    rng = np.random.default_rng(SPLIT_SEED)
    sp = {}
    dev = np.unique(eid[cen < 3])
    perm = rng.permutation(dev)
    n = len(perm)
    a, b, c = int(0.60 * n), int(0.75 * n), int(0.85 * n)
    sp["train"], sp["val"], sp["cal"], sp["test"] = perm[:a], perm[a:b], perm[b:c], perm[c:]
    sp["extD"] = np.unique(eid[cen == 3]); sp["extE"] = np.unique(eid[cen == 4])
    return sp


def visit_mask(tab, cols, eyes):
    return np.isin(col(tab, cols, "eid").astype(int), eyes)


def norm_stats(F, O, idx):
    f = F[idx[::7]].astype(np.float32) / 255.0
    o = O[idx[::7]].astype(np.float32) / 255.0
    return f.mean((0, 2, 3)), f.std((0, 2, 3)), float(o.mean()), float(o.std())


def load_table(path="cohort.npz"):
    z = np.load(path)
    return None, None, z["table"], list(z["cols"])
