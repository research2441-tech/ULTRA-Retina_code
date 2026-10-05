"""Stage 1: self-supervised pretraining of the modality encoders on unlabelled development-train visits.

Loss = masked reconstruction (fundus) + masked reconstruction (OCT)
     + lambda_a * anatomy-aware cross-modal InfoNCE (paired visits only) + lambda_o * shared/private decorrelation.
Variant 'noalign' sets lambda_a = lambda_o = 0 (ablation A1).
After training, every visit of every centre is encoded once and cached (frozen-encoder protocol).
"""
import sys, time, json, numpy as np, torch
import data as D, encoders as E

torch.set_num_threads(2)


def mask_fundus(x, rng, p=0.4):
    B = x.shape[0]
    m = (torch.rand(B, 1, 8, 8, generator=rng) < p).float()
    m = m.repeat_interleave(8, 2).repeat_interleave(8, 3)
    return x * (1 - m), m


def mask_oct(v, rng, p=0.4):
    B = v.shape[0]
    m = (torch.rand(B, 8, 1, 4, generator=rng) < p).float().repeat_interleave(8, 3).expand(B, 8, 32, 32)
    return v * (1 - m), m


def inst_norm_f(x):
    """Per-image, per-channel standardisation (removes device gain, colour cast and most of the gamma shift)."""
    return (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-3)


def inst_norm_o(v):
    return (v - v.mean((1, 2, 3), keepdim=True)) / (v.std((1, 2, 3), keepdim=True) + 1e-3)


def photometric(x, v, g):
    """Acquisition augmentation used only during pretraining: gamma, channel gain, blur proxy, noise, OCT signal."""
    B = x.shape[0]
    r = lambda *s: torch.rand(*s, generator=g)
    gam = 0.75 + 0.55 * r(B, 1, 1, 1)
    x = x.clamp(1e-4, 1) ** gam * (0.85 + 0.3 * r(B, 3, 1, 1))
    if r(1).item() < 0.5:
        x = torch.nn.functional.avg_pool2d(x, 3, 1, 1)
    x = x + 0.04 * r(B, 1, 1, 1) * torch.randn(x.shape, generator=g)
    v = v * (0.7 + 0.5 * r(B, 1, 1, 1))
    v = v * (1 + 0.25 * r(B, 1, 1, 1) * torch.randn(v.shape, generator=g))
    return x, v


def main(seed=0, variant="full", epochs=12, bs=64):
    torch.manual_seed(seed); np.random.seed(seed)
    F, O, tab, cols = D.load()
    sp = D.split_eyes(tab, cols)
    tr = np.where(D.visit_mask(tab, cols, sp["train"]))[0]
    fm, fs, om, osd = D.norm_stats(F, O, tr)
    has_f = D.col(tab, cols, "has_fun")[tr] > 0; has_o = D.col(tab, cols, "has_oct")[tr] > 0
    lam_a, lam_o = (0.5, 0.1) if variant == "full" else (0.0, 0.0)
    ef, eo = E.FundusEncoder(), E.OCTEncoder()
    opt = torch.optim.AdamW(list(ef.parameters()) + list(eo.parameters()), lr=2e-3, weight_decay=0.05)
    steps = epochs * (len(tr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(seed)
    fm_t = torch.tensor(fm)[None, :, None, None]; fs_t = torch.tensor(fs)[None, :, None, None]
    log = []
    t0 = time.time()
    for ep in range(epochs):
        perm = np.random.permutation(len(tr))
        acc = np.zeros(4); n = 0
        for b in range(len(tr) // bs):
            ii = perm[b * bs:(b + 1) * bs]; idx = tr[ii]
            x, v = photometric(torch.tensor(F[idx]).float() / 255, torch.tensor(O[idx]).float() / 255, g)
            x, v = inst_norm_f(x), inst_norm_o(v)
            hf = torch.tensor(has_f[ii]); ho = torch.tensor(has_o[ii])
            xm, mf = mask_fundus(x, g); vm, mo = mask_oct(v, g)
            tf, rf = ef(xm, decode=True); to, ro = eo(vm, decode=True)
            lf = (((rf - x) ** 2) * mf).sum((1, 2, 3)) / (mf.sum((1, 2, 3)) * 3 + 1)
            lo = (((ro - v) ** 2) * mo).sum((1, 2, 3)) / (mo.sum((1, 2, 3)) + 1)
            lf = (lf * hf).sum() / hf.sum().clamp(min=1); lo = (lo * ho).sum() / ho.sum().clamp(min=1)
            pair = hf & ho
            la = E.info_nce(tf[pair, :16, :E.D_S], to[pair, :16, :E.D_S]) if (lam_a > 0 and pair.sum() > 4) else torch.zeros(())
            lorth = (E.orth_loss(tf[hf]) + E.orth_loss(to[ho])) if lam_o > 0 else torch.zeros(())
            loss = lf + lo + lam_a * la + lam_o * lorth
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            acc += [lf.item(), lo.item(), la.item(), lorth.item()]; n += 1
        log.append((acc / n).tolist())
        print(f"seed {seed} {variant} ep {ep} recF {acc[0]/n:.4f} recO {acc[1]/n:.4f} nce {acc[2]/n:.3f} "
              f"orth {acc[3]/n:.4f} {time.time()-t0:.0f}s", flush=True)
    torch.save({"ef": ef.state_dict(), "eo": eo.state_dict(), "fm": fm, "fs": fs, "om": om, "os": osd, "log": log},
               f"enc_{variant}_s{seed}.pt")
    tokF, tokO = encode_all(ef, eo, F, O, fm, fs, om, osd)
    np.savez(f"tok_{variant}_s{seed}.npz", F=tokF, O=tokO)


@torch.no_grad()
def encode_all(ef, eo, F, O, fm, fs, om, osd, bs=256):
    ef.eval(); eo.eval()
    tf, to = [], []
    for b in range(0, len(F), bs):
        x = inst_norm_f(torch.tensor(F[b:b + bs]).float() / 255)
        v = inst_norm_o(torch.tensor(O[b:b + bs]).float() / 255)
        tf.append(ef(x).numpy().astype(np.float16)); to.append(eo(v).numpy().astype(np.float16))
    return np.concatenate(tf), np.concatenate(to)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "full",
         int(sys.argv[3]) if len(sys.argv) > 3 else 12)
