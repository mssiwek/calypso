# calypso

[![Try it on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://calypso.streamlit.app/)
[![PyPI](https://img.shields.io/pypi/v/calypso-emulator.svg)](https://pypi.org/project/calypso-emulator/)
[![DOI (code)](https://zenodo.org/badge/DOI/10.5281/zenodo.20028473.svg)](https://doi.org/10.5281/zenodo.20028473)
[![DOI (artifact)](https://zenodo.org/badge/DOI/10.5281/zenodo.20027761.svg)](https://doi.org/10.5281/zenodo.20027761)

**calypso** (Circumbinary Accretion Lightcurves Yielded via Probabilistic Spectral Operators) is a parameter-conditioned stochastic surrogate model for circumbinary accretion time-series. Given a binary eccentricity `eb` and mass ratio `qb`, calypso returns synthetic accretion-rate light curves for the total binary (`Mb`) and each component (`M1`, `M2`), drawn from a multivariate Gaussian over a global PCA basis trained on hydrodynamic simulations.

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

## Likelihood for parameter inference

Because the emulator is a multivariate Gaussian over PCA coefficients, the likelihood of $(e_b, q_b)$ given an observed time series $y_\mathrm{obs}$ has a closed form. Project the observation onto the orthonormal PCA basis $V$,

$$c_\mathrm{obs} = V^\top (y_\mathrm{obs} - \bar y),$$

where $\bar y$ is the training-set column mean. Then for any candidate $(e_b, q_b)$,

$$\log L(e_b, q_b \mid y_\mathrm{obs}) = -\tfrac{1}{2} (c_\mathrm{obs} - \mu)^\top \Sigma^{-1} (c_\mathrm{obs} - \mu) - \tfrac{1}{2} \log |\Sigma| + \mathrm{const},$$

with $(\mu, \Sigma) = (\mu(e_b, q_b), \Sigma(e_b, q_b))$ the interpolated coefficient mean and covariance. The first term is the squared Mahalanobis distance between the observed coefficients and the predicted mean; the second penalises parameter points where $\Sigma$ has a large determinant. Because $V$ is orthonormal the change-of-variables Jacobian is unity, so this is exactly the likelihood in time-series space (see Appendix A of the paper for the derivation).

Computing it from the public API:

```python
import numpy as np
import calypso

emu = calypso.load_emulator()

# Pre-process y_obs identically to the training data: concatenated
# (Mb, M1, M2) in that order, same de-trending and time grid.
y_obs = ...  # shape (3T,)

V_T   = emu.pca_model.components_   # (K, 3T) -- rows are basis vectors
bar_y = emu.pca_model.mean_         # (3T,)
c_obs = V_T @ (y_obs - bar_y)       # (K,)

def log_likelihood(eb, qb):
    mu, Sigma = emu.interpolator.predict_distribution(eb, qb)
    _, logdet = np.linalg.slogdet(Sigma)
    delta = c_obs - mu
    mahal = delta @ np.linalg.solve(Sigma, delta)
    return -0.5 * mahal - 0.5 * logdet
```

Combine with any prior on the training-grid support and feed into MCMC or nested sampling to obtain a posterior on $(e_b, q_b)$.

**Caveats.** The likelihood assumes `y_obs` has been pre-processed identically to the training data. For real photometric observations (e.g. LSST), this requires first inverting the radiative-transfer pipeline to recover an estimate of the accretion-rate time series before projecting onto the PCA basis. The likelihood also inherits the modelling assumptions of the training simulations (locally isothermal EOS, fixed $h/r$, fixed $\alpha$).

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

If you use calypso in your work, please cite **the software** and (when available) the paper.

**Software:**

> Siwek, M. (2026). *calypso: a parameter-conditioned stochastic surrogate model for circumbinary accretion time-series*. Zenodo. https://doi.org/10.5281/zenodo.20028473

**Trained model + dataset:**

> Siwek, M. (2026). *calypso: trained PCA runtime artifact and training/evaluation dataset*. Zenodo. https://doi.org/10.5281/zenodo.20027761

**Paper:**

> Siwek et al. (2026), *calypso: a Parameter-Conditioned Stochastic Surrogate Model for Circumbinary Accretion Time-Series*. (in prep.)

The DOIs above are *concept DOIs* — they always resolve to the latest version. For paper reproducibility, cite the version DOIs instead: code v1.0.0 = `10.5281/zenodo.20028474`, artifact v1 = `10.5281/zenodo.20027762`.

## License

MIT. See [LICENSE](LICENSE).
