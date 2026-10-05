"""Stage 2 training / inference for ULTRA and all baselines on cached tokens.

usage: python train.py MODEL SEED [TAG] [key=value ...]
   e.g. python train.py ULTRA 0 full
        python train.py ULTRA 0 no_aging aging=0
        python train.py ULTRA 0 no_align tok=noalign
"""
import sys, time, json, copy, numpy as np, torch, torch.nn.functional as Fn
import data as D, models as M

torch.set_num_threads(2)
EDGES = np.array([0, 6, 12, 18, 24.0])


class Cohort:
    def __init__(self, tok="full", seed=0):
        _, _, tab, cols = D.load_table()
        self.tab, self.cols = tab, cols
        z = np.load(f"tok_{tok}_s{seed}.npz")
        self.TF, self.TO = z["F"], z["O"]
        self.sp = D.split_eyes(tab, cols)
        c = lambda n: D.col(tab, cols, n)
        self.eid = c("eid").astype(int); self.k = c("k").astype(int)
        order = np.lexsort((self.k, self.eid))
        self.rows_by_eye = {}
        for i in order:
            self.rows_by_eye.setdefault(self.eid[i], []).append(i)
        self.Kmax = max(len(v) for v in self.rows_by_eye.values())

    def batch(self, eyes, mF_over=None, mO_over=None):
        """Padded tensors for a list of eyes. Optional availability overrides (dict row->0/1)."""
        c = lambda n: D.col(self.tab, self.cols, n)
        K = max(len(self.rows_by_eye[e]) for e in eyes)
        B = len(eyes)
        idx = np.full((B, K), -1)
        for i, e in enumerate(eyes):
            r = self.rows_by_eye[e]; idx[i, :len(r)] = r
        valid = idx >= 0
        ii = np.where(valid, idx, idx[:, :1])
        g = lambda name: np.where(valid, c(name)[ii], 0)
        b = {"tF": torch.tensor(self.TF[ii]).float(), "tO": torch.tensor(self.TO[ii]).float(),
             "t": torch.tensor(np.where(valid, c("t")[ii], np.nan)).float(),
             "valid": torch.tensor(valid).float(), "idx": idx}
        # padded times: carry last time forward so dt = 0
        t = b["t"].numpy()
        for i in range(B):
            n = valid[i].sum(); t[i, n:] = t[i, n - 1]
        b["t"] = torch.tensor(t)
        mF = g("has_fun"); mO = g("has_oct")
        if mF_over is not None:
            mF = np.where(valid, mF_over[ii], 0)
        if mO_over is not None:
            mO = np.where(valid, mO_over[ii], 0)
        b["mF"] = torch.tensor(mF * valid).float(); b["mO"] = torch.tensor(mO * valid).float()
        b["cls"] = torch.tensor(g("cls")).long(); b["sev"] = torch.tensor(g("stage")).long()
        b["grid"] = torch.tensor(np.stack([g(f"g{j}") for j in range(16)], -1)).float()
        b["event"] = torch.tensor(g("event")).float(); b["tte"] = torch.tensor(g("tte")).float()
        return b


def surv_nll(S, event, tte, valid):
    """Discrete-time survival NLL over 4 six-month bins with right censoring (S: B,K,4 survival)."""
    Sp = torch.cat([torch.ones_like(S[..., :1]), S], -1)                 # S_{-1}=1
    j = torch.clamp((tte / 6).floor().long(), 0, 3)
    ev = (event > 0) & (tte < 24)
    f = (Sp.gather(-1, j[..., None]) - Sp.gather(-1, (j + 1)[..., None])).squeeze(-1)
    nb = torch.clamp((torch.minimum(tte, torch.full_like(tte, 24)) / 6).floor().long(), 0, 4)
    sc = Sp.gather(-1, nb[..., None]).squeeze(-1)
    nll = torch.where(ev, -torch.log(f.clamp(min=1e-6)), -torch.log(sc.clamp(min=1e-6)))
    return (nll * valid).sum() / valid.sum()


def losses(out, b, lam_pred=0.2):
    v = b["valid"]
    L = {"surv": surv_nll(out["S"], b["event"], b["tte"], v)}
    vm = v > 0
    L["cls"] = Fn.cross_entropy(out["cls"][vm], b["cls"][vm])
    L["sev"] = Fn.cross_entropy(out["sev"][vm], b["sev"][vm])
    L["grid"] = Fn.binary_cross_entropy_with_logits(out["grid"][vm], b["grid"][vm])
    tot = L["surv"] + 0.5 * (L["cls"] + L["sev"] + L["grid"])
    if "innov" in out and lam_pred > 0:
        n = out["state"].shape[-1]
        L["innov"] = (out["innov"] * v).sum() / (v.sum() * n)
        tot = tot + lam_pred * L["innov"]
    return tot, {k: round(float(x.detach()), 4) for k, x in L.items()}


def augment(b, rng, p_oct=0.35, p_fun=0.10):
    """Modality dropout applied identically to every model during training."""
    mF, mO = b["mF"].numpy().copy(), b["mO"].numpy().copy()
    dO = (rng.random(mO.shape) < p_oct) & (mF > 0)
    mO[dO] = 0
    dF = (rng.random(mF.shape) < p_fun) & (mO > 0)
    mF[dF] = 0
    b = dict(b); b["mF"] = torch.tensor(mF); b["mO"] = torch.tensor(mO)
    return b


def make_model(name, flags):
    kw = {}
    if name == "ULTRA":
        for k in ["factor", "aging", "velocity", "hetero", "integrate", "axis"]:
            if k in flags:
                kw[k] = bool(int(flags[k]))
        if "nS" in flags:
            s = int(flags["nS"]); kw.update(nS=s, nF=s // 2, nO=s // 2)
        if "hz" in flags:
            kw["hz"] = flags["hz"]
        if "n_mc" in flags:
            kw["n_mc"] = int(flags["n_mc"])
    if name in ("LTMD", "CSFusion", "GRUD") and "drop" in flags:
        kw["p"] = float(flags["drop"])
    return M.build(name, **kw)


def train(name, seed, coh, flags, epochs=40, bs=32, patience=8, verbose=True):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = make_model(name, flags)
    lam_pred = float(flags.get("lam_pred", 0.2))
    opt = torch.optim.AdamW(model.parameters(), lr=float(flags.get("lr", 2e-3)), weight_decay=1e-4)
    tr, va = list(coh.sp["train"]), list(coh.sp["val"])
    vb = coh.batch(va)
    best, best_state, bad = 1e9, None, 0
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), bs):
            b = augment(coh.batch(list(perm[i:i + bs])), rng, p_oct=float(flags.get("p_oct", 0.35)))
            out = model(b)
            loss, _ = losses(out, b, lam_pred)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            torch.manual_seed(1234)
            _, parts = losses(model(vb), vb, lam_pred)
            vl = parts["surv"] + 0.5 * (parts["cls"] + parts["sev"] + parts["grid"])   # selection: task loss only
        hist.append(parts)
        if vl < best - 1e-4:
            best, best_state, bad = float(vl), copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose:
            print(f"{name} s{seed} ep{ep} val {float(vl):.4f} {parts} {time.time()-t0:.0f}s", flush=True)
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def predict(model, coh, eyes, mF_over=None, mO_over=None, n_mc=32, mc_dropout=0):
    """Per-visit predictions keyed by table row."""
    model.eval()
    if mc_dropout:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
    res = {}
    for i in range(0, len(eyes), 64):
        b = coh.batch(list(eyes[i:i + 64]), mF_over, mO_over)
        torch.manual_seed(99)
        if isinstance(model, M.ULTRA):
            model_nmc = model.n_mc; model.n_mc = n_mc if model.integrate else 1
            out = model(b); model.n_mc = model_nmc
        else:
            if mc_dropout:
                outs = [model(b) for _ in range(mc_dropout)]
                Ss = torch.stack([o["S"] for o in outs])
                out = {k: torch.stack([o[k] for o in outs]).mean(0) for k in ["cls", "sev", "grid"]}
                out["cls"] = torch.log(torch.stack([torch.softmax(o["cls"], -1) for o in outs]).mean(0))
                out["sev"] = torch.log(torch.stack([torch.softmax(o["sev"], -1) for o in outs]).mean(0))
                out["S"] = Ss.mean(0); out["risk_sd"] = (1 - Ss[..., 1]).std(0)
            else:
                out = model(b)
        idx = b["idx"]
        for key, val in [("S", out["S"]), ("cls", torch.softmax(out["cls"], -1)), ("sev", torch.softmax(out["sev"], -1)),
                         ("grid", torch.sigmoid(out["grid"])),
                         ("risk_sd", out.get("risk_sd", torch.zeros(idx.shape)))]:
            a = val.numpy()
            res.setdefault(key, []).append((idx, a))
    final = {}
    n = len(coh.tab)
    for key, lst in res.items():
        shape = lst[0][1].shape[2:]
        arr = np.full((n,) + shape, np.nan, np.float32)
        for idx, a in lst:
            m = idx >= 0
            arr[idx[m]] = a[m]
        final[key] = arr
    return final


def run(name, seed, tag, flags, splits=("val", "cal", "test", "extD", "extE")):
    coh = Cohort(flags.get("tok", "full"), int(flags.get("tokseed", seed)))
    model, hist = train(name, seed, coh, flags)
    torch.save(model.state_dict(), f"runs/{name}_{tag}_s{seed}.pt")
    eyes = np.concatenate([coh.sp[s] for s in splits])
    P = predict(model, coh, eyes)
    np.savez_compressed(f"runs/pred_{name}_{tag}_s{seed}.npz", **P)
    json.dump({"hist": hist, "flags": flags}, open(f"runs/hist_{name}_{tag}_s{seed}.json", "w"))
    return model, coh


if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    flags = dict(a.split("=") for a in sys.argv[4:])
    import os; os.makedirs("runs", exist_ok=True)
    run(name, seed, tag, flags)
