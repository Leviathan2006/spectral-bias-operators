import torch


def kgrid(n, dev):
    ky = torch.fft.fftfreq(n, 1 / n).to(dev)
    kx = torch.fft.rfftfreq(n, 1 / n).to(dev)
    return kx[None] ** 2 + ky[:, None] ** 2


def hnorm(x, s, k2):
    return (((1 + k2) ** s) * torch.fft.rfft2(x).abs() ** 2).sum((-2, -1))


def relh(p, g, s, k2):
    return (hnorm(p - g, s, k2).sum() / (hnorm(g, s, k2).sum() + 1e-12)).sqrt().item()


def band(p, g, kc, k2):
    ep = torch.fft.rfft2(p - g).abs() ** 2
    gg = torch.fft.rfft2(g).abs() ** 2
    hi = (k2 > kc ** 2).float()
    f = lambda mm: ((ep * mm).sum() / ((gg * mm).sum() + 1e-12)).sqrt().item()
    return f(hi), f(1 - hi)


def _radial(field2d):
    h, wc = field2d.shape
    w = (wc - 1) * 2
    ky = torch.fft.fftfreq(h, 1 / h)
    kx = torch.fft.rfftfreq(w, 1 / w)
    kk = (kx[None] ** 2 + ky[:, None] ** 2).sqrt().round().long()
    return kk


def spec_err(p, g):
    ep = (torch.fft.rfft2(p - g).abs() ** 2).mean(0)
    gg = (torch.fft.rfft2(g).abs() ** 2).mean(0)
    kk = _radial(ep)
    nb = int(kk.max()) + 1
    a = torch.zeros(nb)
    b = torch.zeros(nb)
    for j in range(nb):
        mm = kk == j
        if mm.any():
            a[j] = ep[mm].sum()
            b[j] = gg[mm].sum()
    return (a / (b + 1e-12)).numpy()


def espec(x):
    p = torch.fft.rfft2(x).abs() ** 2
    kk = _radial(p)
    o = torch.zeros(int(kk.max()) + 1)
    for j in range(o.shape[0]):
        mm = kk == j
        if mm.any():
            o[j] = p[mm].mean()
    return o.numpy()


@torch.no_grad()
def rollout(md, ic, mu, sd, steps, dev):
    steps = min(steps, ic.shape[1] - 1)
    x = ic[:, :1].to(dev)
    seq = [x.clone()]
    for _ in range(steps):
        x = md.predict(x)
        seq.append(x.clone())
    pr = torch.cat(seq, 1).cpu() * sd + mu
    gt = ic[:, :steps + 1] * sd + mu
    return pr, gt


def rmse_corr(pr, gt):
    d = ((pr - gt) ** 2).mean((0, 2, 3)).sqrt().numpy()
    b, t = pr.shape[:2]
    a = pr.reshape(b, t, -1)
    c = gt.reshape(b, t, -1)
    a = a - a.mean(-1, keepdim=True)
    c = c - c.mean(-1, keepdim=True)
    co = ((a * c).sum(-1) / (a.norm(dim=-1) * c.norm(dim=-1) + 1e-12)).mean(0).numpy()
    return d, co


def boch(pr, gt, s, k2):
    if not torch.isfinite(pr).all():
        return float('nan')
    num = hnorm(pr - gt, s, k2).sum(1)
    den = hnorm(gt, s, k2).sum(1)
    return 100 * (num.sqrt() / (den.sqrt() + 1e-12)).mean().item()
