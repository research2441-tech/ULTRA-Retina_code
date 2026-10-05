import sys, numpy as np, evaluate as EV
tab, cols, sp, rows, G, eid = EV.setup()
import metrics as Mt
for arg in [a for a in sys.argv[1:] if ":" in a]:
    name, tag, seed = arg.split(":")
    P = EV.load_pred(name, tag, int(seed))
    for s in (sys.argv[1].split(",") if "," in sys.argv[1] or sys.argv[1] in ("val","test","extD","extE") else ["test", "extD", "extE"]):
        r = Mt.all_metrics(P, tab, cols, rows[s], G)
        print(f"{name:9s} {tag:10s} {s:5s} " + " ".join(f"{k}={v:.3f}" for k, v in r.items()))
