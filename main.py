import os
import time
import numpy as np
import torch

import data
import nets
import metrics as M
import plots

SMOKE = os.environ.get('SMOKE', '0') == '1'

C = dict(
    models=['FNO', 'PDE-Refiner', 'Whitened'],
    res=64, gres=128, nu=1e-3, drag=0.05, dt=5e-4, dtr=0.25, kf=4, famp=1.0,
    ntrain=300, ntest=40, ftrain=30, ftest=40,
    w=32, modes=32, L=4, K=3, smin=(1e-3) ** 0.5, lam=1e-3, beta=0.5,
    epochs=80, bs=32, lr=1e-3, wd=1e-4, workers=2,
    kfrac=0.25, nprobe=8, broll=8, roll=60, nshow=16,
    out='out', seed=0,
)
if SMOKE:
    C.update(res=32, gres=32, dt=1e-3, dtr=0.5, ntrain=60, ntest=8, ftrain=14, ftest=20,
             w=12, modes=16, L=2, K=3, epochs=6, workers=0, nprobe=4, broll=5, roll=15, nshow=6)

os.makedirs(C['out'], exist_ok=True)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(C['seed']); np.random.seed(C['seed'])
torch.backends.cudnn.benchmark = True
print('device', dev, '| smoke', SMOKE)


def probe_metrics(md, px, py, ic, mu, sd, kc, k2):
    md.eval()
    pr = []
    for i in range(0, px.shape[0], 128):
        pr.append(md.predict(px[i:i + 128].to(dev)).squeeze(1).cpu() * sd + mu)
    pr = torch.cat(pr, 0)
    l2 = 100 * M.relh(pr, py, 0, k2)
    h1 = 100 * M.relh(pr, py, 1, k2)
    hi, lo = M.band(pr, py, kc, k2)
    rp, rg = M.rollout(md, ic, mu, sd, C['broll'], dev)
    b0 = M.boch(rp, rg, 0, k2)
    b1 = M.boch(rp, rg, 1, k2)
    return dict(l2=l2, h1=h1, hik=100 * hi, lok=100 * lo, b0=b0, b1=b1), pr


def train(name, loader, px, py, ic, mu, sd, kc, k2):
    md = nets.build(name, C).to(dev)
    opt = torch.optim.AdamW(md.parameters(), lr=C['lr'], weight_decay=C['wd'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C['epochs'])
    np_ = sum(p.numel() for p in md.parameters())
    print(f'\n[{name}] {np_:,} params')
    ck = sorted({1, C['epochs'] // 4, C['epochs'] // 2, C['epochs']})
    h = {k: [] for k in ['ep', 'l2', 'h1', 'hik', 'lok', 'b0', 'b1', 'tl']}
    h['curves'] = {}
    for ep in range(1, C['epochs'] + 1):
        md.train(); t0 = time.time(); acc = 0; nb = 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            ls = md.loss(xb, yb)
            ls.backward(); opt.step()
            acc += ls.item(); nb += 1
        sch.step()
        with torch.no_grad():
            mv, pr = probe_metrics(md, px, py, ic, mu, sd, kc, k2)
        h['ep'].append(ep); h['tl'].append(acc / max(nb, 1))
        for k in ['l2', 'h1', 'hik', 'lok', 'b0', 'b1']:
            h[k].append(mv[k])
        if ep in ck:
            h['curves'][ep] = M.spec_err(pr, py)
        print(f'  ep {ep:3d} loss {acc/max(nb,1):.2e}  L2 {mv["l2"]:5.2f}  hi-k {mv["hik"]:6.2f}  '
              f'H1 {mv["h1"]:5.2f}  B1 {mv["b1"]:6.2f}  ({time.time()-t0:.0f}s)')
    return md, h


def main():
    tr = data.make(C, C['ntrain'], C['ftrain'], dev, 'train')
    te = data.make(C, C['ntest'], C['ftest'], dev, 'test')
    mu, sd = tr.mean(), tr.std()
    print('mu %.3f sd %.3f' % (mu, sd))

    loader = torch.utils.data.DataLoader(data.Pairs(tr, mu, sd), batch_size=C['bs'], shuffle=True,
                                         num_workers=C['workers'], drop_last=True,
                                         pin_memory=(dev.type == 'cuda'))
    P = te[:C['nprobe']]
    px = ((P[:, :-1] - mu) / sd).reshape(-1, 1, C['res'], C['res'])
    py = P[:, 1:].reshape(-1, C['res'], C['res'])
    ic = (te[:C['nprobe']] - mu) / sd
    k2 = M.kgrid(C['res'], 'cpu')
    kc = C['kfrac'] * C['res']

    hist, mods = {}, {}
    for nm in C['models']:
        mods[nm], hist[nm] = train(nm, loader, px, py, ic, mu, sd, kc, k2)

    o = C['out']
    plots.hik_epoch(hist, o)
    plots.fprinciple(hist['FNO'], o)
    plots.bochner(hist, o)
    plots.gap(hist, o)
    plots.losses(hist, o)
    plots.curve_grid(hist, o)

    ics = (te[:C['nshow']] - mu) / sd
    roll, pred1, spec1, curves, finals = {}, {}, {}, {}, {}
    nb = min(16, py.shape[0])
    gt_s = np.mean([M.espec(py[b]) for b in range(nb)], 0)
    for nm in C['models']:
        pr, gt = M.rollout(mods[nm], ics, mu, sd, C['roll'], dev)
        roll[nm] = M.rmse_corr(pr, gt)
        p1 = torch.cat([mods[nm].predict(px[i:i + 128].to(dev)).detach().squeeze(1).cpu()
                        for i in range(0, px.shape[0], 128)], 0) * sd + mu
        pred1[nm] = p1
        spec1[nm] = np.mean([M.espec(p1[b]) for b in range(nb)], 0)
        curves[nm] = M.spec_err(p1, py)
        finals[nm] = dict(l2=hist[nm]['l2'][-1], h1=hist[nm]['h1'][-1],
                          hik=hist[nm]['hik'][-1], b1=hist[nm]['b1'][-1])

    sample = {nm: M.rollout(mods[nm], ics[:1], mu, sd, C['roll'], dev)[0][0] for nm in C['models']}
    gt_sample = M.rollout(mods[C['models'][0]], ics[:1], mu, sd, C['roll'], dev)[1][0]
    steps = [s for s in (0, C['roll'] // 4, C['roll'] // 2, C['roll']) if s <= C['roll']]

    plots.snapshots(gt_sample, sample, steps, o)
    plots.err_maps(gt_sample, sample, steps[len(steps) // 2], o)
    plots.rmse(roll, o)
    plots.corr(roll, o)
    plots.spectrum(gt_s, spec1, o)
    plots.spec_error(curves, o)
    plots.fourier_heat(pred1, py, o)
    plots.bars(finals, o)

    tr_seq = mods['PDE-Refiner'].trace(px[:nb].to(dev))
    step_specs = [np.mean([M.espec((s.squeeze(1).cpu() * sd + mu)[b]) for b in range(nb)], 0) for s in tr_seq]
    plots.refine_spectra(step_specs, gt_s, o)

    print('\nfinal')
    for nm in C['models']:
        f = finals[nm]
        print(f'  {nm:<12} L2 {f["l2"]:.2f}  hi-k {f["hik"]:.2f}  H1 {f["h1"]:.2f}  '
              f'Boch-H1 {f["b1"]:.2f}  gap {f["h1"]/max(f["l2"],1e-6):.2f}')


if __name__ == '__main__':
    main()
