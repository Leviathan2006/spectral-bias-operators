import torch
import torch.nn as nn
import torch.nn.functional as F


def coords(b, h, w, dev):
    x = torch.linspace(0, 1, h, device=dev).view(1, 1, h, 1).expand(b, 1, h, w)
    y = torch.linspace(0, 1, w, device=dev).view(1, 1, 1, w).expand(b, 1, h, w)
    return x, y


class Spec(nn.Module):
    def __init__(self, ci, co, m):
        super().__init__()
        self.m = m
        sc = 1 / (ci * co)
        self.a = nn.Parameter(sc * torch.rand(ci, co, m, m, dtype=torch.cfloat))
        self.b = nn.Parameter(sc * torch.rand(ci, co, m, m, dtype=torch.cfloat))

    def forward(self, x):
        n, _, h, w = x.shape
        xf = torch.fft.rfft2(x)
        m = self.m
        o = torch.zeros(n, self.a.shape[1], h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        o[:, :, :m, :m] = torch.einsum('bixy,ioxy->boxy', xf[:, :, :m, :m], self.a)
        o[:, :, -m:, :m] = torch.einsum('bixy,ioxy->boxy', xf[:, :, -m:, :m], self.b)
        return torch.fft.irfft2(o, s=(h, w))


class Mix(nn.Module):
    # 1x1 channel mix as a matmul; cuDNN's 1x1 conv has bad launch latency on blackwell
    def __init__(self, ci, co):
        super().__init__()
        self.w = nn.Parameter(torch.randn(co, ci) * ci ** -0.5)
        self.b = nn.Parameter(torch.zeros(co))

    def forward(self, x):
        return torch.einsum('bihw,oi->bohw', x, self.w) + self.b.view(1, -1, 1, 1)


class Trunk(nn.Module):
    def __init__(self, w, m, L, cin, nstep=0):
        super().__init__()
        self.inp = Mix(cin, w)
        self.emb = nn.Embedding(nstep + 1, w) if nstep else None
        self.sp = nn.ModuleList(Spec(w, w, m) for _ in range(L))
        self.lin = nn.ModuleList(Mix(w, w) for _ in range(L))
        self.o1 = Mix(w, 128)
        self.o2 = Mix(128, 1)

    def forward(self, x, k=None):
        z = self.inp(x)
        if self.emb is not None and k is not None:
            z = z + self.emb(k)[:, :, None, None]
        for sp, li in zip(self.sp, self.lin):
            z = F.gelu(sp(z) + li(z))
        return self.o2(F.gelu(self.o1(z)))


class FNO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = Trunk(cfg['w'], cfg['modes'], cfg['L'], 3)

    def predict(self, a):
        b, _, h, w = a.shape
        gx, gy = coords(b, h, w, a.device)
        return self.net(torch.cat([a, gx, gy], 1))

    def loss(self, a, y):
        return ((self.predict(a) - y) ** 2).mean()


class Whitened(FNO):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.lam = cfg['lam']
        self.beta = cfg.get('beta', 1.0)
        self.mom = 0.99
        self.register_buffer('ps', torch.zeros(cfg['res'], cfg['res'] // 2 + 1))
        self.seen = 0

    def loss(self, a, y):
        with torch.no_grad():
            e = (torch.fft.rfft2(y.squeeze(1)).abs() ** 2).mean(0)
            self.ps.mul_(self.mom).add_((1 - self.mom) * e)
            self.seen += 1
        sp = self.ps / (1 - self.mom ** self.seen)
        wt = sp.clamp_min(self.lam * sp.max()) ** (-self.beta)
        err = torch.fft.rfft2((self.predict(a) - y).squeeze(1)).abs() ** 2
        return (err * wt).mean() / wt.mean()


class Refiner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.K = cfg['K']
        self.smin = cfg['smin']
        self.net = Trunk(cfg['w'], cfg['modes'], cfg['L'], 4, nstep=self.K)

    def sig(self, k):
        return (self.smin ** (k.float() / self.K)).view(-1, 1, 1, 1)

    def run(self, xin, a, k):
        b, _, h, w = a.shape
        gx, gy = coords(b, h, w, a.device)
        return self.net(torch.cat([xin, a, gx, gy], 1), k)

    def loss(self, a, y):
        b = a.shape[0]
        k = torch.randint(0, self.K + 1, (b,), device=a.device)
        mask = (k >= 1).view(-1, 1, 1, 1)
        z = torch.randn_like(y)
        sg = self.sig(k)
        xin = torch.where(mask, y + sg * z, torch.zeros_like(y))
        tgt = torch.where(mask, z, y)
        return ((self.run(xin, a, k) - tgt) ** 2).mean()

    @torch.no_grad()
    def _refine(self, a, keep=False):
        b = a.shape[0]
        z0 = torch.zeros_like(a)
        u = self.run(z0, a, torch.zeros(b, dtype=torch.long, device=a.device))
        seq = [u.clone()]
        for k in range(1, self.K + 1):
            sg = self.smin ** (k / self.K)
            un = u + sg * torch.randn_like(u)
            kk = torch.full((b,), k, dtype=torch.long, device=a.device)
            u = un - sg * self.run(un, a, kk)
            if keep:
                seq.append(u.clone())
        return seq if keep else u

    def predict(self, a):
        return self._refine(a, keep=False)

    def trace(self, a):
        return self._refine(a, keep=True)


def build(name, cfg):
    return {'FNO': FNO, 'PDE-Refiner': Refiner, 'Whitened': Whitened}[name](cfg)
