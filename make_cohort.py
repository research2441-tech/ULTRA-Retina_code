"""Generate the development (A, B, C) and external (D, E) cohorts."""
import sys, numpy as np, simulator as S
if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    rows, cols = S.build_cohort(seed, {"A": 600, "B": 600, "C": 600, "D": 400, "E": 400}, "cohort.npz")
    t = rows; c = list(cols)
    print("visits", len(t), "eyes", len(np.unique(t[:, 0])))
    for ce in range(5):
        m = t[:, 1] == ce
        print("ABCDE"[ce], "visits", m.sum(), "oct", t[m, c.index("has_oct")].mean().round(3),
              "event", t[m, c.index("event")].mean().round(3), "stage", np.bincount(t[m, c.index("stage")].astype(int), minlength=5))
