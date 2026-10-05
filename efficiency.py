"""E6: parameters, CPU latency per visit and peak memory for every stage-2 model (seed 0), plus encoders."""
import time, json, resource, tracemalloc, numpy as np, torch
import train as T, robustness as R, encoders as E

torch.set_num_threads(1)


def count(m):
    return sum(p.numel() for p in m.parameters())


def main():
    coh = T.Cohort("full", 0)
    eyes = list(coh.sp["test"][:64])
    b = coh.batch(eyes)
    nvis = int(b["valid"].sum())
    out = {}
    for name in ["CSFusion", "GRUD", "LTMD", "CTFilter", "ULTRA"]:
        m = R.load_model(name, "main", 0); m.eval()
        with torch.no_grad():
            m(b)
            ts = []
            for _ in range(5):
                t0 = time.perf_counter(); m(b); ts.append(time.perf_counter() - t0)
            tracemalloc.start(); m(b); cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        out[name] = {"params": count(m), "ms_per_visit": 1000 * np.median(ts) / nvis, "peak_mb": peak / 2 ** 20}
        print(name, out[name], flush=True)
    ef, eo = E.FundusEncoder(), E.OCTEncoder()
    x = torch.randn(64, 3, 64, 64); v = torch.randn(64, 8, 32, 32)
    with torch.no_grad():
        ef(x); eo(v)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); ef(x); eo(v); ts.append(time.perf_counter() - t0)
    enc_params = count(ef) - count(ef.dec) + count(eo) - count(eo.dec)
    out["encoders"] = {"params": enc_params, "ms_per_visit": 1000 * np.median(ts) / 64}
    print(out["encoders"])
    json.dump(out, open("runs/efficiency.json", "w"), indent=1)


if __name__ == "__main__":
    main()
