import math
import torch


def grf(n, dev, alpha=2.5, tau=7.0):
    km = n // 2
    wn = torch.cat([torch.arange(0, km), torch.arange(-km, 0)]).repeat(n, 1)
    kx, ky = wn.t(), wn
    sig = tau ** (0.5 * (2 * alpha - 2))
    r = (n ** 2) * math.sqrt(2) * sig * ((4 * math.pi ** 2 * (kx ** 2 + ky ** 2) + tau ** 2) ** (-alpha / 2))
    r[0, 0] = 0
    return r.to(dev)


def sample_ic(r, m, dev):
    c = torch.randn(m, *r.shape, dtype=torch.cfloat, device=dev)
    return torch.fft.ifftn(r * c, dim=(-2, -1)).real


def step_ns(w0, f, nu, drag, T, dt, nrec):
    n = w0.size(-1)
    km = n // 2
    ns = math.ceil(T / dt)
    wh = torch.fft.rfft2(w0)
    fh = torch.fft.rfft2(f)
    if fh.dim() < wh.dim():
        fh = fh[None]
    every = ns // nrec
    ky = torch.cat([torch.arange(0, km), torch.arange(-km, 0)]).to(w0.device).repeat(n, 1)
    kx = ky.t()
    kx, ky = kx[..., :km + 1], ky[..., :km + 1]
    lap = 4 * math.pi ** 2 * (kx ** 2 + ky ** 2)
    lap[0, 0] = 1
    lin = nu * lap + drag
    de = ((ky.abs() <= 2 / 3 * km) & (kx.abs() <= 2 / 3 * km)).float()[None]
    out = torch.zeros(*w0.shape, nrec, device=w0.device)
    c = 0
    for i in range(ns):
        psi = wh / lap
        u = torch.fft.irfft2(2j * math.pi * ky * psi, s=(n, n))
        v = torch.fft.irfft2(-2j * math.pi * kx * psi, s=(n, n))
        wx = torch.fft.irfft2(2j * math.pi * kx * wh, s=(n, n))
        wy = torch.fft.irfft2(2j * math.pi * ky * wh, s=(n, n))
        nl = de * torch.fft.rfft2(u * wx + v * wy)
        wh = (-dt * nl + dt * fh + (1 - 0.5 * dt * lin) * wh) / (1 + 0.5 * dt * lin)
        if (i + 1) % every == 0 and c < nrec:
            out[..., c] = torch.fft.irfft2(wh, s=(n, n))
            c += 1
    return out


def coarsen(x, n):
    if x.shape[-1] == n:
        return x
    N = x.shape[-1]
    xf = torch.fft.rfft2(x, norm='forward')
    k = n // 2
    lo = torch.cat([xf[..., :k, :k + 1], xf[..., N - k:, :k + 1]], -2)
    return torch.fft.irfft2(lo, s=(n, n), norm='forward')


def make(cfg, ntraj, nframe, dev, tag):
    gr = cfg['gres']
    r = grf(gr, dev)
    xs = torch.linspace(0, 1, gr + 1, device=dev)[:-1]
    _, gy = torch.meshgrid(xs, xs, indexing='ij')
    kf = cfg['kf']
    f = cfg['famp'] * kf * 2 * math.pi * torch.sin(2 * math.pi * kf * gy)
    T = nframe * cfg['dtr']
    out = torch.zeros(ntraj, nframe, cfg['res'], cfg['res'])
    ch = 200 if gr <= 64 else 48
    for s in range(0, ntraj, ch):
        e = min(s + ch, ntraj)
        w = step_ns(sample_ic(r, e - s, dev), f, cfg['nu'], cfg['drag'], T, cfg['dt'], nframe)
        w = coarsen(w.permute(0, 3, 1, 2), cfg['res'])
        if not torch.isfinite(w).all():
            raise RuntimeError('blew up, drop dt or gres')
        out[s:e] = w.cpu()
        print(f'  {tag} {e}/{ntraj}')
    return out


class Pairs(torch.utils.data.Dataset):
    def __init__(self, d, mu, sd):
        self.d = ((d - mu) / sd).float()
        self.T = d.shape[1]

    def __len__(self):
        return self.d.shape[0] * (self.T - 1)

    def __getitem__(self, i):
        a, b = divmod(i, self.T - 1)
        return self.d[a, b][None], self.d[a, b + 1][None]
