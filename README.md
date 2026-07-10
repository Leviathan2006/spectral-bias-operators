# Spectral bias in neural operators

A small study of the spectral bias of Fourier Neural Operators on 2D Kolmogorov turbulence, and two
solutions.

An MSE-trained FNO looks great on aggregate error (a few percent relative L2) while being almost
completely wrong on the high wavenumbers: the loss is dominated by the high-amplitude low-k modes, so
low-amplitude high-k structure never gets a gradient. Over an autoregressive rollout the nonlinear term
folds those high-k errors back into the large scales and the forecast drifts. This repo measures that
directly with frequency-weighted metrics and compares three models:

- **FNO** — standard Fourier neural operator, MSE loss.
- **PDE-Refiner** — the multistep denoising-refinement scheme of Lippe et al. (2023). k=0 predicts the
  signal; k=1..K add Gaussian noise at an exponentially shrinking scale and predict/subtract it, which
  gives every frequency an O(1) target. Same FNO backbone as the baseline.
- **Whitened** — same FNO, but the loss is reweighted by the inverse of the running target spectrum,
  `Σ_k |e_k|² / (s_k² + λ)`. This is the deterministic analogue of what the denoising does implicitly:
  it equalizes the per-mode gradient without any noise at inference. `λ` plays the role of `σ_min`.

## Metrics

- relative L2 (H⁰) and Sobolev H¹ error (H¹ upweights high wavenumbers)
- low-k / high-k relative error split at `|k| = kfrac·N`
- Bochner norm over a short rollout, `L²(0,T; Hˢ)` for s = 0, 1
- per-wavenumber relative error curve and the energy spectrum

## Data

Vorticity form of the 2D incompressible Navier–Stokes on a periodic box with Kolmogorov forcing
`f = χ·sin(k y)` plus linear drag, integrated pseudospectrally (Crank–Nicolson diffusion, 2/3
dealiasing). The solver runs at `gres` and is spectrally coarsened to `res`. Reynolds number ≈ 1000.

## Run

```
pip install -r requirements.txt
python main.py            # ~15-30 min on a single GPU
SMOKE=1 python main.py    # tiny/fast sanity run
```

All hyperparameters are in the `C` dict at the top of `main.py`. Figures land in `out/`.

`python bench.py` times the core fft / complex-matmul / conv kernels — handy to check the GPU stack
is healthy before a long run (each op should be well under a millisecond).

## Figures produced

`highk_epoch` (high-k error vs epoch for each model), `f_principle` (FNO learns low-k first, high-k
stalls), `bochner`, `gap` (H¹/L2), `train_loss`, `error_front` (per-wavenumber error at several epochs),
`fields` (rollout snapshots), `error_maps`, `rollout_rmse`, `rollout_corr`, `spectrum`,
`spectral_error`, `fourier_error` (2D Fourier error), `final_bars`, and `refine_steps` (spectrum after
each PDE-Refiner refinement step).

## Layout

```
data.py      solver, Gaussian-random-field ICs, dataset
nets.py      FNO, PDE-Refiner, whitened-loss FNO
metrics.py   frequency-weighted metrics + rollout
plots.py     all figures
main.py      config, training loop, orchestration
```
