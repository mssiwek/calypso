"""
calypso interactive lightcurve explorer — Streamlit edition.

Generates synthetic LSST-band lightcurves from calypso-emulated accretion
variability onto circumbinary disk systems.

Directions:
cd /Users/magda/Work/projects/ongoing/CALYPSO/calypso-dev/calypso
conda activate py312_calypso
streamlit run demo/streamlit/streamlit_app.py
"""

import sys
from pathlib import Path

# Make the parent demo/ directory importable so we can reuse
# alpha_disk, demo_constants, lsst, and lum_calc.
_DEMO_DIR = Path(__file__).resolve().parent.parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import alpha_disk as ad
from demo_constants import G, MSUN, YR, PC, ETA
from lsst import LSST_FILTERS
from photometry import lum_to_mags

import calypso

# Paper colour scheme (Okabe–Ito)
COLORS = {"Mb": "#0072B2", "M1": "#009E73", "M2": "#D55E00"}

# ---------------------------------------------------------------------------
# Initial UI settings — change these to set the default app state
# ---------------------------------------------------------------------------

EB0 = 0.0
QB0 = 1.0
LOG_MB0 = 7.0
LOG_AB0 = -3.0
Z0 = 0.5
F_EDD0 = 0.1
BAND0 = "g"  # must be a key of LSST_FILTERS

CAL_EPISTEMIC0 = True
CAL_SHOW_DRAWS0 = True
CAL_N_DRAWS0 = 10
CAL_SHOW_STATS0 = True
CAL_N_STATS0 = 64

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="calypso timeseries emulator",
    layout="wide",
)

st.title("calypso: a parameter-conditioned stochastic timeseries emulator")
st.markdown(
    "calypso generates synthetic timeseries conditioned on two real-valued "
    "parameters. The data is decomposed into a PCA basis, and the resulting "
    "coefficients — both their central values and uncertainties — are modeled "
    "as a parameter-conditioned multivariate Gaussian. Drawing samples from "
    "this distribution and recombining them with the basis yields new "
    "realizations of the timeseries. "
    "This demo applies it to **circumbinary accretion**: explore how lightcurve "
    "morphology changes with binary eccentricity $e_b$ and mass ratio $q_b$. "
    "Accretion rates are emulated by calypso; luminosities are computed via an "
    "$\\alpha$-disk model with blackbody emission integrated over LSST "
    "photometric bands."
)

with st.expander("About calypso — method details", expanded=True):
    st.markdown(
        r"""
**calypso** (Circumbinary Accretion Lightcurves Yielded via Predictive Sequence Outputs)
is a parameter-conditioned stochastic surrogate model for circumbinary accretion
time-series. Given a binary eccentricity $e_b$ and mass ratio $q_b$, calypso returns
synthetic accretion-rate light curves for the total binary ($\dot{M}_b$) and each
component ($\dot{M}_1$, $\dot{M}_2$).

**Training data.** 100 2D hydrodynamic simulations of circumbinary accretion disks
(`Arepo`, Navier-Stokes), spanning $e_b \in [0.0, 0.8]$ and $q_b \in [0.1, 1.0]$.
From each simulation, 500 detrended 10-orbit windows of the concatenated
$(\dot{M}_b, \dot{M}_1, \dot{M}_2)$ time series form the training matrix.

**Method.** The model is built in three layers:

1. **Global PCA basis.** A single SVD over the entire training matrix yields a
   basis in which each window is represented by $k = 142$ coefficients
   (capturing $\gtrsim 90\%$ of the variance).
2. **Per-binary multivariate Gaussian.** For each training $(e_b, q_b)$, the
   empirical mean and covariance of the coefficient vectors across the 500
   windows define a $k$-dimensional Gaussian. This captures the *aleatoric*
   uncertainty of the accretion process — including precession-driven long-term
   modulation — directly in the latent space.
3. **Cholesky-space interpolation.** To predict at unseen $(e_b, q_b)$, mean
   vectors and Cholesky factors of the per-binary covariances are linearly
   interpolated across the parameter grid, then recombined into a positive
   semi-definite covariance. Sampling from the resulting Gaussian and projecting
   back through the PCA basis produces the synthetic time series shown below.

**Epistemic uncertainty** is implemented but disabled by default (it inflates
predictive variance beyond what the held-out test set supports). Toggle it in
the sidebar to inspect its effect.

See Siwek et al. (2026) for the full derivation, validation against 13 held-out
simulations, and parameter-space evaluation.
        """
    )

# ---------------------------------------------------------------------------
# Cached resources (loaded once per server session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading calypso emulator...")
def _load_emulator():
    return calypso.load_emulator()


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

NBINS = 100
NORB = 10
LOG_MODEL = True
N_ANNULI = 10

emulator = _load_emulator()
lsst_filters = {band: dict(vals) for band, vals in LSST_FILTERS.items()}

# ---------------------------------------------------------------------------
# Physics helpers (cached where possible)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_prediction_samples(eb, qb, n_samples, seed, epistemic_enabled=True):
    pred = emulator.predict(eb, qb, n_samples=n_samples, rng=np.random.default_rng(seed),
                            epistemic_enabled=epistemic_enabled)
    pred_M1 = 10 ** pred["M1"] if LOG_MODEL else pred["M1"]
    pred_M2 = 10 ** pred["M2"] if LOG_MODEL else pred["M2"]
    return pred_M1, pred_M2


@st.cache_data(show_spinner=False)
def get_t(ab, mb):
    Pb = 2 * np.pi * np.sqrt(ab ** 3 / (G * mb))
    t = np.linspace(0, NORB, NBINS * NORB) * Pb
    t0 = Pb / 2
    return t, t0, Pb


@st.cache_data(show_spinner=False)
def precompute_disk_radii(t, t0, mb, ab, eb, qb):
    radius_primary, radius_secondary = [], []
    qb_re = 1.0 / qb
    for t_i in t:
        radius_primary.append(ad.radius_minidisk(t_i, t0, mb, ab, eb, qb_re, n_annuli=N_ANNULI, r_ER=True))
        radius_secondary.append(ad.radius_minidisk(t_i, t0, mb, ab, eb, qb, n_annuli=N_ANNULI, r_ER=True))
    return np.asarray(radius_primary), np.asarray(radius_secondary)


def band_luminosity_samples(mdot_samples, radius_grid, lam_cm, mb, qb, f_edd):
    mdot_phys = np.clip(np.asarray(mdot_samples, dtype=float), 0.0, None) * ad.m_edd(mb, f_edd, eta=ETA)
    mi = mb * (qb / (1 + qb))
    temperature = ad.T_R(radius_grid[None, :, :], mi, mdot_phys[:, :, None])
    radiance = ad.planck_lambda(lam_cm[:, None, None, None], temperature[None, :, :, :])
    surface_integrand = 4 * np.pi ** 2 * radius_grid[None, None, :, :] * radiance
    spectral_luminosity = np.trapezoid(surface_integrand, radius_grid[None, None, :, :], axis=-1)
    return np.trapezoid(spectral_luminosity, lam_cm, axis=0)


def get_band_wavelength_grid(z, band_name):
    band_obs = lsst_filters[band_name]["lambda_min"], lsst_filters[band_name]["lambda_max"]
    band_rest = [b / (1 + z) for b in band_obs]
    return np.logspace(np.log10(band_rest[0]), np.log10(band_rest[1]), 10)


@st.cache_data(show_spinner=False)
def compute_curves(eb, qb, mb, ab, z, band, n_samples, seed, f_edd, epistemic_enabled=True):
    t, t0, _ = get_t(ab, mb)
    m1_samples, m2_samples = get_prediction_samples(eb, qb, n_samples=n_samples, seed=seed,
                                                     epistemic_enabled=epistemic_enabled)
    lam_cm = get_band_wavelength_grid(z, band)
    radius_primary, radius_secondary = precompute_disk_radii(t, t0, mb, ab, eb, qb)
    qb_re = 1.0 / qb
    lum1 = band_luminosity_samples(m1_samples, radius_primary, lam_cm, mb, qb_re, f_edd)
    lum2 = band_luminosity_samples(m2_samples, radius_secondary, lam_cm, mb, qb, f_edd)
    mag1 = np.asarray(lum_to_mags(lum1, band, z))
    mag2 = np.asarray(lum_to_mags(lum2, band, z))
    magb = np.asarray(lum_to_mags(lum1 + lum2, band, z))
    return t, mag1, mag2, magb


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.header("Binary parameters")
eb = st.sidebar.slider("Eccentricity $e_b$", 0.0, 0.8, EB0, 0.01)
qb = st.sidebar.slider("Mass ratio $q_b$", 0.1, 1.0, QB0, 0.01)
log_mb = st.sidebar.slider(r"$\log_{10}(M_b / M_\odot)$", 2.0, 10.0, LOG_MB0, 0.1)
log_ab = st.sidebar.slider(r"$\log_{10}(a_b / \mathrm{pc})$", -4.0, 0.0, LOG_AB0, 0.1)
z = st.sidebar.slider("Redshift $z$", 0.0, 3.0, Z0, 0.01)
f_edd = st.sidebar.slider(r"Eddington fraction $f_{\mathrm{Edd}}$", 0.001, 1.0, F_EDD0, 0.001)

st.sidebar.header("Display")
_band_keys = list(LSST_FILTERS.keys())
band = st.sidebar.selectbox("LSST band", _band_keys, index=_band_keys.index(BAND0))

st.sidebar.subheader("calypso (emulated)")
cal_epistemic = st.sidebar.checkbox("Epistemic uncertainty", value=CAL_EPISTEMIC0, key="cal_epi")
cal_show_draws = st.sidebar.checkbox("Show realisations", value=CAL_SHOW_DRAWS0, key="cal_draws")
cal_n_draws = st.sidebar.slider("Realisations", 1, 64, CAL_N_DRAWS0, 1, key="cal_n") if cal_show_draws else 0
cal_show_stats = st.sidebar.checkbox("Show mean +/- std", value=CAL_SHOW_STATS0, key="cal_stats")
cal_n_stats = st.sidebar.slider("Samples for stats", 8, 256, CAL_N_STATS0, 8, key="cal_nstats") if cal_show_stats else 0

if "seed" not in st.session_state:
    st.session_state.seed = 0
if st.sidebar.button("Resample"):
    st.session_state.seed += 1

# ---------------------------------------------------------------------------
# Derived physical values
# ---------------------------------------------------------------------------

mb = 10 ** log_mb * MSUN
ab = 10 ** log_ab * PC

# ---------------------------------------------------------------------------
# Compute curves
# ---------------------------------------------------------------------------

# Total emulator samples needed: draws shown individually + stats pool
cal_total = max(cal_n_draws, cal_n_stats)

with st.spinner("Computing lightcurves..."):
    t, mag1, mag2, magb = compute_curves(
        eb, qb, mb, ab, z, band, cal_total, st.session_state.seed, f_edd, cal_epistemic,
    )

t_years = t / YR
mag_limit = lsst_filters[band]["mag_limit_single"]


def _draw_alpha(n):
    """Line alpha that decreases with the number of overlaid draws."""
    return np.clip(0.9 / np.sqrt(n), 0.08, 0.9)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
fig.subplots_adjust(hspace=0.12)
buffer = 0.02

# Collect all plotted mag values for y-limit computation
all_magb_vals = []
all_mag12_vals = []

# -- calypso draws --
if cal_show_draws and cal_n_draws >= 1:
    alpha_c = _draw_alpha(cal_n_draws)
    for i in range(cal_n_draws):
        label_b = r"$\dot{M}_b$" if i == 0 else None
        label_1 = r"$\dot{M}_1$" if i == 0 else None
        label_2 = r"$\dot{M}_2$" if i == 0 else None
        ax1.plot(t_years, magb[i], lw=1.6, color=COLORS["Mb"], alpha=alpha_c, label=label_b, zorder=2)
        ax2.plot(t_years, mag1[i], lw=1.6, color=COLORS["M1"], alpha=alpha_c, label=label_1, zorder=2)
        ax2.plot(t_years, mag2[i], lw=1.6, color=COLORS["M2"], alpha=alpha_c, label=label_2, zorder=2)
        all_magb_vals.append(magb[i])
        all_mag12_vals.extend([mag1[i], mag2[i]])

# -- calypso mean +/- std --
if cal_show_stats and cal_n_stats >= 2:
    magb_mean, magb_std = magb[:cal_n_stats].mean(0), magb[:cal_n_stats].std(0)
    mag1_mean, mag1_std = mag1[:cal_n_stats].mean(0), mag1[:cal_n_stats].std(0)
    mag2_mean, mag2_std = mag2[:cal_n_stats].mean(0), mag2[:cal_n_stats].std(0)
    lbl_b_stats = rf"$\dot{{M}}_b$ mean $\pm 1\sigma$ ({cal_n_stats})" if not cal_show_draws else None
    lbl_1_stats = rf"primary $\dot{{M}}_1$ mean $\pm 1\sigma$ ({cal_n_stats})" if not cal_show_draws else None
    lbl_2_stats = rf"secondary $\dot{{M}}_2$ mean $\pm 1\sigma$ ({cal_n_stats})" if not cal_show_draws else None
    ax1.plot(t_years, magb_mean, lw=2.2, color=COLORS["Mb"], label=lbl_b_stats, zorder=4)
    ax1.fill_between(t_years, magb_mean - magb_std, magb_mean + magb_std, color=COLORS["Mb"], alpha=0.25, zorder=1)
    ax2.plot(t_years, mag1_mean, lw=2.2, color=COLORS["M1"], label=lbl_1_stats, zorder=4)
    ax2.fill_between(t_years, mag1_mean - mag1_std, mag1_mean + mag1_std, color=COLORS["M1"], alpha=0.25, zorder=1)
    ax2.plot(t_years, mag2_mean, lw=2.2, color=COLORS["M2"], label=lbl_2_stats, zorder=4)
    ax2.fill_between(t_years, mag2_mean - mag2_std, mag2_mean + mag2_std, color=COLORS["M2"], alpha=0.25, zorder=1)
    all_magb_vals.extend([magb_mean - magb_std, magb_mean + magb_std])
    all_mag12_vals.extend([mag1_mean - mag1_std, mag1_mean + mag1_std,
                           mag2_mean - mag2_std, mag2_mean + mag2_std])

# -- Y-limits and decorations --
if all_magb_vals:
    stacked = np.column_stack(all_magb_vals)
    ylo1 = np.nanmin(stacked) - buffer
    yhi1 = np.nanmax(stacked) + buffer
else:
    ylo1, yhi1 = 0.0, 1.0
ax1.set_ylim(ylo1, yhi1)
ax1.axhspan(mag_limit, yhi1, color="pink", alpha=0.3, label="LSST lim (sv)", zorder=0)
ax1.set_ylim(ylo1, yhi1)
ax1.invert_yaxis()
ax1.set_ylabel(r"$m_{\rm AB}$", fontsize=14)
ax1.set_title(
    rf"$\dot{{M}}_b$ magnitude in {band}-band, $e_{{\rm b}}$={eb:.2f}, $q_{{\rm b}}$={qb:.2f}",
    fontsize=14,
)
ax1.legend(fontsize=10, loc="upper right")
ax1.grid(alpha=0.3, linestyle="--")
plt.setp(ax1.get_xticklabels(), visible=False)

if all_mag12_vals:
    stacked = np.column_stack(all_mag12_vals)
    all_lo = np.nanmin(stacked) - buffer
    all_hi = np.nanmax(stacked) + buffer
else:
    all_lo, all_hi = 0.0, 1.0
ax2.set_ylim(all_lo, all_hi)
ax2.axhspan(mag_limit, all_hi, color="pink", alpha=0.3, zorder=0)
ax2.set_ylim(all_lo, all_hi)
ax2.invert_yaxis()
ax2.set_ylabel(r"$m_{\rm AB}$", fontsize=14)
ax2.set_title(r"$\dot{M}_1$ and $\dot{M}_2$", fontsize=14)
ax2.legend(fontsize=10, loc="upper right")
ax2.grid(alpha=0.3, linestyle="--")

# x-axis labels in years
n_labels = 5
dt = (t[-1] - t[0]) / n_labels
xlabels = np.array([(t[0] + i * dt) for i in range(n_labels + 1)]) / YR
xlabels_str = [rf"$t_0 + ${i * dt / YR:.1f} yr" for i in range(n_labels + 1)]
ax2.set_xticks(xlabels)
ax2.set_xticklabels(xlabels_str)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

