"""Stage 2 longitudinal models on cached encoder tokens.

ULTRA      : factorised evidence-ageing belief filter (shared / fundus-private / OCT-private blocks,
             per-dimension position-velocity Gaussian belief, white-acceleration process noise,
             heteroscedastic modality noise) + belief-integrated discrete-time hazard.
CSFusion   : cross-sectional late fusion of the current visit.
GRUD       : GRU-D with per-modality masks and time-since-observation decay.
LTMD       : longitudinal transformer with continuous time embedding, modality dropout and gated tokens.
CTFilter   : continuous-time latent Kalman filter (joint state, homoscedastic, random walk, point hazard).
All share the observation embedding and the task heads so that only the temporal/fusion mechanism differs.
"""
import math, torch, torch.nn as nn, torch.nn.functional as Fn

D_TOK, D_S = 64, 32
BINS_M = torch.tensor([3.0, 9.0, 15.0, 21.0])       # bin midpoints (months) of [0,6),[6,12),[12,18),[18,24)
NB = 4


def mlp(i, h, o, p=0.0):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, o))


class ObsEmbed(nn.Module):
    """Tokens (B,K,17,64) -> visit embedding (B,K,128) with attention pooling over macular cells."""
    def __init__(self, d=128):
        super().__init__()
        self.q = nn.Parameter(torch.randn(D_TOK) * 0.1)
        self.net = mlp(3 * D_TOK, d, d)

    def forward(self, tok):
        cell, glob = tok[..., :16, :], tok[..., 16, :]
        a = torch.softmax((cell * self.q).sum(-1) / math.sqrt(D_TOK), -1)
        att = (a[..., None] * cell).sum(-2)
        return self.net(torch.cat([cell.mean(-2), att, glob], -1))


class Heads(nn.Module):
    """Diagnosis (4), severity (5), 4x4 lesion grid and (optionally) a point hazard head."""
    def __init__(self, c, point_hazard=True, p=0.0):
        super().__init__()
        self.cls = mlp(c, 128, 4, p); self.sev = mlp(c, 128, 5, p)
        self.ctx = nn.Linear(c, 32)
        self.grid = mlp(2 * D_S + 2 + 32 + 16, 64, 1, p)
        self.cellpos = nn.Parameter(torch.eye(16), requires_grad=False)
        self.haz = mlp(c + NB, 128, 1, p) if point_hazard else None

    def lesion(self, ctx, tF, tO, mF, mO):
        B, K = ctx.shape[:2]
        sF = tF[..., :16, :D_S] * mF[..., None, None]; sO = tO[..., :16, :D_S] * mO[..., None, None]
        z = torch.cat([sF, sO, mF[..., None, None].expand(B, K, 16, 1), mO[..., None, None].expand(B, K, 16, 1),
                       self.ctx(ctx)[..., None, :].expand(B, K, 16, 32), self.cellpos.expand(B, K, 16, 16)], -1)
        return self.grid(z).squeeze(-1)

    def point_hazard(self, ctx):
        B, K, _ = ctx.shape
        oh = torch.eye(NB).expand(B, K, NB, NB)
        h = self.haz(torch.cat([ctx[..., None, :].expand(B, K, NB, ctx.shape[-1]), oh], -1)).squeeze(-1)
        return torch.sigmoid(h)                               # B,K,NB


def surv_from_hazard(h):
    """h: (...,NB) hazards -> survival after each bin."""
    return torch.cumprod(1 - h, -1)


# ============================================================================= ULTRA
class ULTRA(nn.Module):
    def __init__(self, nS=24, nF=12, nO=12, factor=True, aging=True, velocity=True, hetero=True,
                 integrate=True, n_mc=16, hz="extrap", axis=True):
        super().__init__()
        self.hz, self.axis = hz, axis
        self.factor, self.aging, self.velocity, self.hetero, self.integrate = factor, aging, velocity, hetero, integrate
        self.n = n = nS + nF + nO
        self.n_mc = n_mc if integrate else 1
        idxF = list(range(nS)) + list(range(nS, nS + nF)) if factor else list(range(n))
        idxO = list(range(nS)) + list(range(nS + nF, n)) if factor else list(range(n))
        self.register_buffer("HF", torch.zeros(n)); self.HF[idxF] = 1
        self.register_buffer("HO", torch.zeros(n)); self.HO[idxO] = 1
        self.embF, self.embO = ObsEmbed(), ObsEmbed()
        self.yF, self.yO = nn.Linear(128, n), nn.Linear(128, n)
        self.rF, self.rO = nn.Linear(128, n), nn.Linear(128, n)
        self.rF_c = nn.Parameter(torch.zeros(n)); self.rO_c = nn.Parameter(torch.zeros(n))
        self.p0 = nn.Parameter(torch.zeros(n))
        self.lPp0 = nn.Parameter(torch.zeros(n)); self.lPv0 = nn.Parameter(torch.full((n,), -2.0))
        self.lqp = nn.Parameter(torch.full((n,), -3.0)); self.lqv = nn.Parameter(torch.full((n,), -3.0))
        self.heads = Heads(4 * n, point_hazard=False)
        ax_in = 2 if axis else 0
        self.haz = mlp(n + n + NB + ax_in, 128, 1) if hz == "extrap" else mlp(4 * n + NB + ax_in, 128, 1)
        # ordinal severity axis eta = w^T p with cumulative-probit thresholds theta_1 < ... < theta_4
        self.w = nn.Parameter(torch.randn(n) / math.sqrt(n))
        self.th0 = nn.Parameter(torch.tensor(-1.5)); self.dth = nn.Parameter(torch.full((3,), 0.5))

    # --- process noise rates (per year)
    def q(self):
        if not self.aging:
            return torch.zeros_like(self.lqp), torch.zeros_like(self.lqv)
        qv = Fn.softplus(self.lqv) if self.velocity else torch.zeros_like(self.lqv)
        return Fn.softplus(self.lqp), qv

    def predict(self, p, v, Ppp, Ppv, Pvv, dt):
        """Time update over dt (years) for a white-acceleration (constant-velocity) model per dimension."""
        qp, qv = self.q()
        dt = dt[..., None]
        p2 = p + dt * v
        Ppp2 = Ppp + 2 * dt * Ppv + dt ** 2 * Pvv + qv * dt ** 3 / 3 + qp * dt
        Ppv2 = Ppv + dt * Pvv + qv * dt ** 2 / 2
        Pvv2 = Pvv + qv * dt
        return p2, v, Ppp2, Ppv2, Pvv2

    @staticmethod
    def update(p, v, Ppp, Ppv, Pvv, y, r, mask):
        """Diagonal Kalman update on observed position dimensions (mask: B x n of 0/1)."""
        S = Ppp + r
        kp, kv = Ppp / S, Ppv / S
        e = y - p
        p2 = p + mask * kp * e; v2 = v + mask * kv * e
        Ppp2 = Ppp - mask * kp * Ppp; Ppv2 = Ppv - mask * kp * Ppv; Pvv2 = Pvv - mask * kv * Ppv
        nll = 0.5 * (torch.log(S) + e ** 2 / S + math.log(2 * math.pi))
        return p2, v2, Ppp2, Ppv2, Pvv2, (nll * mask).sum(-1)

    def observe(self, tok, emb, ylin, rlin, rconst):
        h = emb(tok)
        y = 3.0 * torch.tanh(ylin(h) / 3.0)                       # bounded observation embedding
        r = (Fn.softplus(rlin(h)) if self.hetero else Fn.softplus(rconst).expand_as(y)) + 1e-3
        return y, r

    def forward(self, b, eps=None):
        tF, tO, mF, mO, t = b["tF"], b["tO"], b["mF"], b["mO"], b["t"]
        B, K = t.shape
        yF, rF = self.observe(tF, self.embF, self.yF, self.rF, self.rF_c)
        yO, rO = self.observe(tO, self.embO, self.yO, self.rO, self.rO_c)
        n = self.n
        p = self.p0.expand(B, n); v = torch.zeros(B, n)
        Ppp = torch.exp(self.lPp0).expand(B, n); Ppv = torch.zeros(B, n)
        Pvv = torch.exp(self.lPv0).expand(B, n) if self.velocity else torch.zeros(B, n)
        prev = t[:, 0]
        outs, innov = [], []
        for k in range(K):
            dt = (t[:, k] - prev).clamp(min=0) / 12.0
            p, v, Ppp, Ppv, Pvv = self.predict(p, v, Ppp, Ppv, Pvv, dt)
            # innovation likelihood with stop-gradient on the observation value (trains prior, dynamics, ageing, noise)
            mf = mF[:, k, None] * self.HF; mo = mO[:, k, None] * self.HO
            p, v, Ppp, Ppv, Pvv, nf = self.update(p, v, Ppp, Ppv, Pvv, yF[:, k].detach(), rF[:, k], mf)
            p, v, Ppp, Ppv, Pvv, no = self.update(p, v, Ppp, Ppv, Pvv, yO[:, k].detach(), rO[:, k], mo)
            innov.append(nf + no)
            prev = t[:, k]
            outs.append((p, v, Ppp, Ppv, Pvv))
        # the filtered state used by the heads must carry gradients into y and r as well: re-run with grads
        p = self.p0.expand(B, n); v = torch.zeros(B, n)
        Ppp = torch.exp(self.lPp0).expand(B, n); Ppv = torch.zeros(B, n)
        Pvv = torch.exp(self.lPv0).expand(B, n) if self.velocity else torch.zeros(B, n)
        prev = t[:, 0]; states = []
        for k in range(K):
            dt = (t[:, k] - prev).clamp(min=0) / 12.0
            p, v, Ppp, Ppv, Pvv = self.predict(p, v, Ppp, Ppv, Pvv, dt)
            mf = mF[:, k, None] * self.HF; mo = mO[:, k, None] * self.HO
            p, v, Ppp, Ppv, Pvv, _ = self.update(p, v, Ppp, Ppv, Pvv, yF[:, k], rF[:, k], mf)
            p, v, Ppp, Ppv, Pvv, _ = self.update(p, v, Ppp, Ppv, Pvv, yO[:, k], rO[:, k], mo)
            prev = t[:, k]
            states.append(torch.stack([p, v, Ppp, Ppv, Pvv], 0))
        st = torch.stack(states, 2)                                    # 5,B,K,n
        p, v, Ppp, Ppv, Pvv = st
        ctx = torch.cat([p, v, torch.log(Ppp + 1e-6), torch.log(Pvv + 1e-6)], -1)
        out = {"cls": self.heads.cls(ctx), "sev": self.heads.sev(ctx),
               "grid": self.heads.lesion(ctx, tF, tO, mF, mO), "innov": torch.stack(innov, 1)}
        self._pi = None
        if self.axis:
            mu, var = self.axis_moments(p, v, Ppp, Ppv, Pvv)
            pi = self.stage_probs(mu, var)
            out["sev"] = torch.log(pi)
            self._pi = pi
        out["S"], out["risk_sd"] = self.hazard(p, v, Ppp, Ppv, Pvv, eps)
        out["state"] = st
        return out

    def thresholds(self):
        return self.th0 + torch.cat([torch.zeros(1), torch.cumsum(Fn.softplus(self.dth), 0)])

    def axis_moments(self, p, v, Ppp, Ppv, Pvv, months=0.0):
        """Mean and variance of the severity coordinate eta at a horizon (months) under the belief."""
        if months > 0:
            p, v, Ppp, Ppv, Pvv = self.horizon_belief(p, v, Ppp, Ppv, Pvv, months)
        w2 = self.w ** 2
        return (p * self.w).sum(-1), (Ppp * w2).sum(-1)

    def stage_probs(self, mu, var):
        """Belief-integrated cumulative probit: P(stage >= j) = Phi((mu - theta_j) / sqrt(1 + var))."""
        th = self.thresholds()
        z = (mu[..., None] - th) / torch.sqrt(1 + var[..., None])
        ge = torch.distributions.Normal(0., 1.).cdf(z)                                  # ...,4
        ge = torch.cat([torch.ones_like(ge[..., :1]), ge, torch.zeros_like(ge[..., :1])], -1)
        return (ge[..., :-1] - ge[..., 1:]).clamp(min=1e-6)                            # ...,5

    def crossing(self, p, v, Ppp, Ppv, Pvv, months, pi):
        """Threshold-crossing prior: sum_c pi_c * P(eta(t+tau) > theta_{c+1})."""
        mu, var = self.axis_moments(p, v, Ppp, Ppv, Pvv, months)
        th = self.thresholds()
        z = (mu[..., None] - th) / torch.sqrt(1 + var[..., None])
        up = torch.distributions.Normal(0., 1.).cdf(z)                                  # P(eta > theta_{c+1}), c=0..3
        psi = (pi[..., :4] * up).sum(-1).clamp(1e-5, 1 - 1e-5)
        return torch.stack([psi, torch.log(psi) - torch.log1p(-psi)], -1)

    def horizon_belief(self, p, v, Ppp, Ppv, Pvv, months):
        dt = torch.full(p.shape[:-1], months / 12.0)
        return self.predict(p, v, Ppp, Ppv, Pvv, dt)

    def hazard(self, p, v, Ppp, Ppv, Pvv, eps=None, n_mc=None):
        M = n_mc or self.n_mc
        if not self.integrate:
            M = 1
        if eps is None:
            eps = torch.randn(M, *p.shape) if self.integrate else torch.zeros(1, *p.shape)
        hs = []
        pi = self.stage_probs(*self.axis_moments(p, v, Ppp, Ppv, Pvv)) if self.axis else None
        if self.hz == "current":
            # sample the current (position, velocity) belief per dimension and let the head map it to each bin
            e2 = torch.randn(eps.shape) if self.integrate else torch.zeros_like(eps)
            sp_ = p[None] + torch.sqrt(Ppp + 1e-8)[None] * eps
            rho = (Ppv / torch.sqrt(Ppp * Pvv + 1e-8)).clamp(-0.999, 0.999)
            sv_ = v[None] + torch.sqrt(Pvv + 1e-8)[None] * (rho[None] * eps + torch.sqrt(1 - rho[None] ** 2) * e2)
            lv = torch.cat([torch.log(Ppp + 1e-6), torch.log(Pvv + 1e-6)], -1)[None].expand(eps.shape[0], *p.shape[:-1], 2 * self.n)
            for j in range(NB):
                oh = torch.zeros(*sp_.shape[:-1], NB); oh[..., j] = 1
                feats = [sp_, sv_, lv, oh]
                if self.axis:
                    feats.append(self.crossing(p, v, Ppp, Ppv, Pvv, float(BINS_M[j]) + 3, pi)[None].expand(*sp_.shape[:-1], 2))
                hs.append(torch.sigmoid(self.haz(torch.cat(feats, -1)).squeeze(-1)))
            h = torch.stack(hs, -1)
            Sm = surv_from_hazard(h)
            S = Sm.mean(0)
            risk_sd = (1 - Sm[..., 1]).std(0) if Sm.shape[0] > 1 else torch.zeros_like(S[..., 1])
            return S, risk_sd
        for j in range(NB):
            pj, vj, Pj, _, _ = self.horizon_belief(p, v, Ppp, Ppv, Pvv, float(BINS_M[j]))
            s = pj[None] + torch.sqrt(Pj + 1e-8)[None] * eps                         # M,B,K,n
            oh = torch.zeros(*s.shape[:-1], NB); oh[..., j] = 1
            feats = [s, vj[None].expand_as(s), oh]
            if self.axis:
                feats.append(self.crossing(p, v, Ppp, Ppv, Pvv, float(BINS_M[j]) + 3, pi)[None].expand(*s.shape[:-1], 2))
            hs.append(torch.sigmoid(self.haz(torch.cat(feats, -1)).squeeze(-1)))
        h = torch.stack(hs, -1)                                                       # M,B,K,NB
        Sm = surv_from_hazard(h)
        S = Sm.mean(0)
        risk_sd = (1 - Sm[..., 1]).std(0) if Sm.shape[0] > 1 else torch.zeros_like(S[..., 1])
        return S, risk_sd


# ============================================================================= baselines
class CSFusion(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.embF, self.embO = ObsEmbed(), ObsEmbed()
        self.fuse = mlp(256 + 2, 128, 128, p)
        self.heads = Heads(128, p=p)

    def context(self, b):
        hF = self.embF(b["tF"]) * b["mF"][..., None]; hO = self.embO(b["tO"]) * b["mO"][..., None]
        return self.fuse(torch.cat([hF, hO, b["mF"][..., None], b["mO"][..., None]], -1))

    def forward(self, b, eps=None):
        c = self.context(b)
        h = self.heads.point_hazard(c)
        return {"cls": self.heads.cls(c), "sev": self.heads.sev(c),
                "grid": self.heads.lesion(c, b["tF"], b["tO"], b["mF"], b["mO"]), "S": surv_from_hazard(h)}


class GRUD(CSFusion):
    def __init__(self, p=0.1):
        super().__init__(p)
        self.cell = nn.GRUCell(256 + 2 + 3, 128)
        self.gx = nn.Linear(2, 256); self.gh = nn.Linear(3, 128)

    def context(self, b):
        hF = self.embF(b["tF"]); hO = self.embO(b["tO"])
        mF, mO, t = b["mF"], b["mO"], b["t"]
        B, K = t.shape
        h = torch.zeros(B, 128); lastF = torch.zeros(B, 128); lastO = torch.zeros(B, 128)
        tF = t[:, 0].clone(); tO = t[:, 0].clone(); prev = t[:, 0]
        cs = []
        for k in range(K):
            dF = (t[:, k] - tF) / 12.0; dO = (t[:, k] - tO) / 12.0; dt = (t[:, k] - prev) / 12.0
            delta = torch.stack([dF, dO], -1)
            gx = torch.exp(-Fn.relu(self.gx(delta)))
            xF = mF[:, k, None] * hF[:, k] + (1 - mF[:, k, None]) * gx[:, :128] * lastF
            xO = mO[:, k, None] * hO[:, k] + (1 - mO[:, k, None]) * gx[:, 128:] * lastO
            h = h * torch.exp(-Fn.relu(self.gh(torch.stack([dF, dO, dt], -1))))
            h = self.cell(torch.cat([xF, xO, mF[:, k, None], mO[:, k, None], delta, dt[:, None]], -1), h)
            lastF = torch.where(mF[:, k, None] > 0, hF[:, k], lastF); lastO = torch.where(mO[:, k, None] > 0, hO[:, k], lastO)
            tF = torch.where(mF[:, k] > 0, t[:, k], tF); tO = torch.where(mO[:, k] > 0, t[:, k], tO); prev = t[:, k]
            cs.append(h)
        return torch.stack(cs, 1)


class TimeEmb(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.register_buffer("w", torch.exp(torch.linspace(math.log(1 / 120), math.log(2.0), d // 2)))

    def forward(self, t):
        a = t[..., None] * self.w
        return torch.cat([torch.sin(a), torch.cos(a)], -1)


class LTMD(CSFusion):
    """Longitudinal transformer: gated modality tokens -> visit token -> causal self-attention over visits."""
    def __init__(self, p=0.1, layers=2):
        super().__init__(p)
        self.pF, self.pO = nn.Linear(128, 128), nn.Linear(128, 128)
        self.gate = nn.Linear(128, 1)
        self.miss = nn.Embedding(4, 128)
        self.temb = TimeEmb(); self.tproj = nn.Linear(128, 128)
        el = nn.TransformerEncoderLayer(128, 4, 256, p, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(el, layers)

    def context(self, b):
        mF, mO, t = b["mF"], b["mO"], b["t"]
        eF = self.pF(self.embF(b["tF"])); eO = self.pO(self.embO(b["tO"]))
        gF = torch.sigmoid(self.gate(eF)) * mF[..., None]; gO = torch.sigmoid(self.gate(eO)) * mO[..., None]
        vis = (gF * eF + gO * eO) / (gF + gO + 1e-6)
        code = (mF + 2 * mO).long()
        x = vis + self.miss(code) + self.tproj(self.temb(t))
        K = t.shape[1]
        causal = torch.triu(torch.ones(K, K, dtype=torch.bool), 1)
        pad = b["valid"] < 0.5
        return self.tr(x, mask=causal, src_key_padding_mask=pad)


class CTFilter(ULTRA):
    """Closest continuous-time competitor: joint latent state observed by both modalities,
    homoscedastic noise, random-walk dynamics, hazard on the posterior mean."""
    def __init__(self):
        super().__init__(factor=False, aging=True, velocity=False, hetero=False, integrate=False)


def build(name, **kw):
    return {"ULTRA": ULTRA, "CSFusion": CSFusion, "GRUD": GRUD, "LTMD": LTMD, "CTFilter": CTFilter}[name](**kw)
