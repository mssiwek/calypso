# calypso

[![Try it on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://calypso.streamlit.app/)
[![PyPI](https://img.shields.io/pypi/v/calypso-emulator.svg)](https://pypi.org/project/calypso-emulator/)

**calypso** (Circumbinary Accretion Lightcurves Yielded via Predictive Sequence Outputs) is a parameter-conditioned stochastic surrogate model for circumbinary accretion time-series. Given a binary eccentricity `eb` and mass ratio `qb`, calypso returns synthetic accretion-rate light curves for the total binary (`Mb`) and each component (`M1`, `M2`), drawn from a multivariate Gaussian over a global PCA basis trained on hydrodynamic simulations.

**▶ Try it interactively:** [calypso.streamlit.app](https://calypso.streamlit.app/) — sliders for $e_b$, $q_b$, mass, redshift, LSST band; renders synthetic light curves in real time.

## Method (Short)

calypso is built from a suite of 100 2D hydrodynamic simulations of circumbinary accretion disks (`Arepo`, Navier-Stokes), spanning `eb ∈ [0.0, 0.8]` and `qb ∈ [0.1, 1.0]`. From each simulation, 500 detrended 10-orbit windows of the concatenated `(Mb, M1, M2)` time series form the training matrix. The model then consists of three layers:

1. **Global PCA basis.** A single SVD over the entire training matrix yields a basis in which each window is represented by `k_final = 142` coefficients (capturing ≳90% of the variance).
2. **Per-binary multivariate Gaussian.** For each training `(eb, qb)`, the empirical mean and covariance of the coefficient vectors across the 500 windows define a `k`-dimensional Gaussian. This captures the *aleatoric* uncertainty of the accretion process — including precession-driven long-term modulation — directly in the latent space.
3. **Cholesky-space interpolation.** To predict at unseen `(eb, qb)`, mean vectors and Cholesky factors of the per-binary covariances are linearly interpolated across the parameter grid, then recombined into a positive semi-definite covariance. Sampling from the resulting Gaussian and projecting back through the PCA basis produces synthetic time series.

Epistemic uncertainty is implemented but disabled by default (it inflates predictive variance beyond what the held-out test set supports). See the paper for the full derivation, validation against 13 held-out simulations, and parameter-space evaluation.

## Installation

Requires Python ≥ 3.12.

```bash
pip install calypso-emulator
```

For the demo notebooks and apps:

```bash
pip install "calypso-emulator[demo]"
```

From source (editable):

```bash
git clone https://github.com/mssiwek/calypso.git
cd calypso
pip install -e .
```

## Quickstart

```python
import calypso

emu = calypso.load_emulator()

# Stochastic samples: dict with 'Mb', 'M1', 'M2' arrays of shape (n_samples, T)
samples = emu.predict(eb=0.35, qb=0.75, n_samples=16)

# Mean prediction (no sampling): dict with arrays of shape (T,)
mean = emu.predict_mean(eb=0.35, qb=0.75)

print(emu.component_names)  # ['Mb', 'M1', 'M2']
```

`predict` returns a dictionary keyed by component name (`Mb`, `M1`, `M2`) plus an `'parameters'` record. Each component array has shape `(n_samples, T)` where `T` is the per-component window length (1000 points = 10 binary orbits at 100 points per orbit). Pass `return_coefficients=True` to also retrieve the sampled PCA coefficients.

For reproducible sampling:

```python
import numpy as np
rng = np.random.default_rng(42)
samples = emu.predict(eb=0.35, qb=0.75, n_samples=16, rng=rng)
```

## Public API

- `calypso.load_emulator(artifact_name=None, force_download=False) → PCAEmulator` — load the default trained emulator (downloads the runtime artifact on first call).
- `PCAEmulator.predict(eb, qb, n_samples=1, ...)` — sample synthetic time series.
- `PCAEmulator.predict_mean(eb, qb, ...)` — mean time series, no sampling.
- `PCAEmulator.component_names`, `PCAEmulator.n_components`.

Lower-level building blocks (`PCAModel`, `CholeskyInterpolator`, `fit_pca_svd_multicomponent`, `train_cholesky_interpolator`, `reconstruct_components`) are also exposed for users who want to retrain or extend the model.

## Runtime artifacts

The trained emulator (PCA basis + per-binary Cholesky factors) is shipped as a single binary artifact rather than packaged into the wheel — keeping the install lightweight and decoupling code releases from retraining. On the first call to `load_emulator()`, calypso resolves the artifact name from a packaged manifest and downloads it from Zenodo into a local cache. Subsequent calls reuse the cached file.

To override the cache location:

```bash
export CALYPSO_ARTIFACTS_DIR=/path/to/cache
# or, equivalently:
export CALYPSO_WEIGHTS_DIR=/path/to/cache
```

## Demos

Notebooks and example scripts live in [`demo/`](demo/) and are intended to be read alongside the source rather than shipped via PyPI. They cover the basic prediction workflow, an alpha-disk luminosity model, an LSST photometry pipeline, and a Streamlit app.

## Citation

If you use calypso in your work, please cite:

> Siwek et al. (2026), *calypso: a Parameter-Conditioned Stochastic Surrogate Model for Circumbinary Accretion Time-Series*. (in prep.)

## License

MIT. See [LICENSE](LICENSE).
