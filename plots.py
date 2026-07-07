import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from metrics import espec, spec_err

COL = {'FNO': '#d62728', 'PDE-Refiner': '#1f77b4', 'Whitened': '#2ca02c'}


def _save(fig, out, name):
    fig.tight_layout()
    p = os.path.join(out, name)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print('  wrote', name)


def snapshots(gt, preds, steps, out):
    rows = ['truth'] + list(preds)
    fig, ax = plt.subplots(len(rows), len(steps), figsize=(2.2 * len(steps), 2.2 * len(rows)))
    vm = float(gt[steps[0]].abs().max())
    for j, s in enumerate(steps):
        ax[0, j].imshow(gt[s], cmap='RdBu_r', vmin=-vm, vmax=vm)
        ax[0, j].set_title(f't={s}', fontsize=9)
        for i, nm in enumerate(preds, 1):
            ax[i, j].imshow(preds[nm][s], cmap='RdBu_r', vmin=-vm, vmax=vm)
    for i, r in enumerate(rows):
        ax[i, 0].set_ylabel(r, fontsize=9)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    _save(fig, out, 'fields.png')


def err_maps(gt, preds, step, out):
    fig, ax = plt.subplots(1, len(preds), figsize=(3.2 * len(preds), 3))
    if len(preds) == 1:
        ax = [ax]
    for a, nm in zip(ax, preds):
        e = (preds[nm][step] - gt[step]).abs()
        im = a.imshow(e, cmap='magma')
        a.set_title(f'{nm}  |err| @t={step}', fontsize=9)
        a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046)
    _save(fig, out, 'error_maps.png')


def fourier_heat(preds, gt, out):
    fig, ax = plt.subplots(1, len(preds), figsize=(3.4 * len(preds), 3))
    if len(preds) == 1:
        ax = [ax]
    for a, nm in zip(ax, preds):
        e = (torch.fft.rfft2(preds[nm] - gt).abs() ** 2).mean(0).clamp_min(1e-12).log10()
        e = torch.fft.fftshift(e, dim=0)
        im = a.imshow(e.numpy(), cmap='inferno', aspect='auto')
        a.set_title(f'{nm}  log|e(k)|^2', fontsize=9)
        a.set_xlabel('kx'); a.set_ylabel('ky')
        fig.colorbar(im, ax=a, fraction=0.046)
    _save(fig, out, 'fourier_error.png')


def rmse(roll, out):
    fig, a = plt.subplots(figsize=(6, 4.2))
    for nm, (d, _) in roll.items():
        a.plot(d, lw=2, color=COL.get(nm), label=nm)
    a.set_xlabel('rollout step'); a.set_ylabel('RMSE'); a.set_ylim(bottom=0)
    a.legend(); a.grid(alpha=.3); a.set_title('autoregressive rollout error')
    _save(fig, out, 'rollout_rmse.png')


def corr(roll, out):
    fig, a = plt.subplots(figsize=(6, 4.2))
    for nm, (_, c) in roll.items():
        a.plot(c, lw=2, color=COL.get(nm), label=nm)
    a.axhline(0.95, ls='--', c='gray')
    a.set_xlabel('rollout step'); a.set_ylabel('vorticity correlation')
    a.legend(); a.grid(alpha=.3); a.set_title('correlation with ground truth')
    _save(fig, out, 'rollout_corr.png')


def spectrum(gt_s, model_s, out):
    fig, a = plt.subplots(figsize=(6, 4.2))
    a.plot(gt_s, 'k', lw=2.5, label='truth')
    for nm, s in model_s.items():
        a.plot(s, lw=2, color=COL.get(nm), label=nm)
    a.set_yscale('log'); a.set_xlabel('|k|'); a.set_ylabel('energy')
    a.legend(); a.grid(alpha=.3); a.set_title('energy spectrum (final)')
    _save(fig, out, 'spectrum.png')


def spec_error(curves, out):
    fig, a = plt.subplots(figsize=(6, 4.2))
    for nm, c in curves.items():
        a.plot(c, lw=2, color=COL.get(nm), label=nm)
    a.set_yscale('log'); a.set_xlabel('|k|'); a.set_ylabel(r'$\langle|\Delta u|^2\rangle/\langle|u|^2\rangle$')
    a.legend(); a.grid(alpha=.3); a.set_title('relative error per wavenumber')
    _save(fig, out, 'spectral_error.png')


def hik_epoch(hist, out):
    fig, a = plt.subplots(figsize=(6.4, 4.4))
    for nm, h in hist.items():
        a.plot(h['ep'], h['hik'], lw=2.3, color=COL.get(nm), label=f'{nm} high-k')
        a.plot(h['ep'], h['l2'], lw=1.3, ls='--', color=COL.get(nm), label=f'{nm} L2')
    a.set_yscale('log'); a.set_xlabel('epoch'); a.set_ylabel('relative error (%)')
    a.legend(fontsize=8); a.grid(alpha=.3); a.set_title('high-k error over training')
    _save(fig, out, 'highk_epoch.png')


def fprinciple(h, out):
    fig, a = plt.subplots(figsize=(6, 4.2))
    a.plot(h['ep'], h['lok'], lw=2.2, color='#2ca02c', label='low-k')
    a.plot(h['ep'], h['hik'], lw=2.2, color='#d62728', label='high-k')
    a.plot(h['ep'], h['l2'], lw=1.5, ls='--', color='k', label='total L2')
    a.set_yscale('log'); a.set_xlabel('epoch'); a.set_ylabel('relative error (%)')
    a.legend(); a.grid(alpha=.3); a.set_title('FNO: low-k learned first, high-k stalls')
    _save(fig, out, 'f_principle.png')


def bochner(hist, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for nm, h in hist.items():
        ax[0].plot(h['ep'], h['b0'], lw=2.2, color=COL.get(nm), label=nm)
        ax[1].plot(h['ep'], h['b1'], lw=2.2, color=COL.get(nm), label=nm)
    ax[0].set_title('Bochner-H0'); ax[1].set_title('Bochner-H1 (high-freq weighted)')
    for a in ax:
        a.set_xlabel('epoch'); a.set_ylabel('rel. error (%)'); a.legend(fontsize=8); a.grid(alpha=.3)
    _save(fig, out, 'bochner.png')


def gap(hist, out):
    fig, a = plt.subplots(figsize=(6, 4.2))
    for nm, h in hist.items():
        g = np.array(h['h1']) / np.maximum(np.array(h['l2']), 1e-6)
        a.plot(h['ep'], g, lw=2.3, color=COL.get(nm), label=nm)
    a.axhline(1.0, ls=':', c='gray')
    a.set_xlabel('epoch'); a.set_ylabel('H1 / L2'); a.legend(); a.grid(alpha=.3)
    a.set_title('spectral-bias gap (1 = unbiased)')
    _save(fig, out, 'gap.png')


def losses(hist, out):
    fig, a = plt.subplots(figsize=(6, 4.2))
    for nm, h in hist.items():
        a.plot(h['ep'], h['tl'], lw=2, color=COL.get(nm), label=nm)
    a.set_yscale('log'); a.set_xlabel('epoch'); a.set_ylabel('train loss')
    a.legend(); a.grid(alpha=.3); a.set_title('training loss')
    _save(fig, out, 'train_loss.png')


def bars(finals, out):
    keys = ['l2', 'h1', 'hik', 'b1']
    lab = ['L2', 'H1', 'high-k', 'Boch-H1']
    names = list(finals)
    x = np.arange(len(keys))
    wbar = 0.8 / len(names)
    fig, a = plt.subplots(figsize=(7.5, 4.4))
    for i, nm in enumerate(names):
        a.bar(x + i * wbar, [finals[nm][k] for k in keys], wbar, label=nm, color=COL.get(nm))
    a.set_xticks(x + wbar * (len(names) - 1) / 2)
    a.set_xticklabels(lab)
    a.set_ylabel('final rel. error (%)'); a.legend(); a.grid(alpha=.3, axis='y')
    a.set_title('final metrics')
    _save(fig, out, 'final_bars.png')


def refine_spectra(step_specs, gt_s, out):
    fig, a = plt.subplots(figsize=(6, 4.2))
    a.plot(gt_s, 'k', lw=2.5, label='truth')
    for i, s in enumerate(step_specs):
        a.plot(s, lw=1.8, label=('init (k=0)' if i == 0 else f'refine {i}'))
    a.set_yscale('log'); a.set_xlabel('|k|'); a.set_ylabel('energy')
    a.legend(fontsize=8); a.grid(alpha=.3); a.set_title('PDE-Refiner: spectrum per refinement step')
    _save(fig, out, 'refine_steps.png')


def curve_grid(hist, out):
    n = len(hist)
    fig, ax = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    for a, (nm, h) in zip(ax[0], hist.items()):
        for ep in sorted(h['curves']):
            a.plot(h['curves'][ep], lw=1.8, label=f'ep {ep}')
        a.set_yscale('log'); a.set_title(nm, fontsize=10)
        a.set_xlabel('|k|'); a.legend(fontsize=7); a.grid(alpha=.3)
    ax[0][0].set_ylabel('rel. error per k')
    _save(fig, out, 'error_front.png')
