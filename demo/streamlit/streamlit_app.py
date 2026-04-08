"""
CALYPSO interactive lightcurve explorer — Streamlit edition.

Generates synthetic LSST-band lightcurves from calypso-emulated accretion
variability onto circumbinary disk systems.
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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CALYPSO Lightcurve Explorer",
    layout="wide",
)

st.title("CALYPSO: Circumbinary Accretion Lightcurve Explorer")
st.markdown(
    "Explore how lightcurve morphology changes with binary parameters.  "
    "Accretion rates are emulated by **CALYPSO**; luminosities are computed "
    "via an $\\alpha$-disk model with blackbody emission integrated over "
    "LSST photometric bands."
)

# ---------------------------------------------------------------------------
# Cached resources (loaded once per server session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading CALYPSO emulator...")
def _load_emulator():
    return calypso.load_emulator()

emulator = _load_emulator()
lsst_filters = {band: dict(vals) for band, vals in LSST_FILTERS.items()}

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

NBINS = 100
NORB = 10
LOG_MODEL = True
N_ANNULI = 10
F_EDD = 0.1

# ---------------------------------------------------------------------------
# Physics helpers (cached where possible)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_prediction_samples(eb, qb, n_samples, seed):
    pred = emulator.predict(eb, qb, n_samples=n_samples, rng=np.random.default_rng(seed))
    pred_M1 = 10 ** pred["M1"] if LOG_MODEL else pred["M1"]
    pred_M2 = 10 ** pred["M2"] if LOG_MODEL else pred["M2"]
    return pred_M1, pred_M2


def get_t(ab, mb):
    Pb = 2 * np.pi * np.sqrt(ab ** 3 / (G * mb))
    t = np.linspace(0, NORB, NBINS * NORB) * Pb
    t0 = Pb / 2
    return t, t0, Pb


def precompute_disk_radii(t, t0, mb, ab, eb, qb):
    radius_primary, radius_secondary = [], []
    qb_re = 1.0 / qb
    for t_i in t:
        radius_primary.append(ad.radius_minidisk(t_i, t0, mb, ab, eb, qb_re, n_annuli=N_ANNULI))
        radius_secondary.append(ad.radius_minidisk(t_i, t0, mb, ab, eb, qb, n_annuli=N_ANNULI))
    return np.asarray(radius_primary), np.asarray(radius_secondary)


def band_luminosity_samples(mdot_samples, radius_grid, lam_cm, mb, qb, f_edd):
    mdot_phys = np.clip(np.asarray(mdot_samples, dtype=float), 0.0, None) * ad.m_edd(mb, f_edd, eta=ETA)
    mi = mb * (qb / (1 + qb))
    temperature = ad.T_R(radius_grid[None, :, :], mi, mdot_phys[:, :, None])
    radiance = ad.planck_lambda(lam_cm[:, None, None, None], temperature[None, :, :, :])
    surface_integrand = 4 * np.pi ** 2 * radius_grid[None, None, :, :] * radiance
    spectral_luminosity = np.trapz(surface_integrand, radius_grid[None, None, :, :], axis=-1)
    return np.trapz(spectral_luminosity, lam_cm, axis=0)


def get_band_wavelength_grid(z, band_name):
    band_obs = lsst_filters[band_name]["lambda_min"], lsst_filters[band_name]["lambda_max"]
    band_rest = [b / (1 + z) for b in band_obs]
    return np.logspace(np.log10(band_rest[0]), np.log10(band_rest[1]), 10)


def compute_curves(eb, qb, mb, ab, z, band, n_samples, seed):
    t, t0, _ = get_t(ab, mb)
    m1_samples, m2_samples = get_prediction_samples(eb, qb, n_samples=n_samples, seed=seed)
    lam_cm = get_band_wavelength_grid(z, band)
    radius_primary, radius_secondary = precompute_disk_radii(t, t0, mb, ab, eb, qb)
    lum1 = band_luminosity_samples(m1_samples, radius_primary, lam_cm, mb, qb, F_EDD)
    lum2 = band_luminosity_samples(m2_samples, radius_secondary, lam_cm, mb, qb, F_EDD)
    mag1 = np.asarray(lum_to_mags(lum1, band, z))
    mag2 = np.asarray(lum_to_mags(lum2, band, z))
    magb = np.asarray(lum_to_mags(lum1 + lum2, band, z))
    return t, mag1, mag2, magb


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.header("Binary parameters")
eb = st.sidebar.slider("Eccentricity $e_b$", 0.0, 0.8, 0.66, 0.01)
qb = st.sidebar.slider("Mass ratio $q_b$", 0.1, 1.0, 0.96, 0.01)
log_mb = st.sidebar.slider(r"$\log_{10}(M_b / M_\odot)$", 2.0, 10.0, 7.0, 0.1)
log_ab = st.sidebar.slider(r"$\log_{10}(a_b / \mathrm{pc})$", -4.0, 0.0, -3.0, 0.1)
z = st.sidebar.slider("Redshift $z$", 0.001, 10.0, 0.5, 0.01)

st.sidebar.header("Display")
band = st.sidebar.selectbox("LSST band", list(LSST_FILTERS.keys()), index=1)
show_stats = st.sidebar.checkbox("Show mean +/- std", value=False)
n_samples = st.sidebar.slider("Number of samples", 1, 64, 32, 1) if show_stats else 1

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

with st.spinner("Computing lightcurves..."):
    t, mag1, mag2, magb = compute_curves(eb, qb, mb, ab, z, band, n_samples, st.session_state.seed)

t_years = t / YR
mag_limit = lsst_filters[band]["mag_limit_single"]

# ---------------------------------------------------------------------------
# Aggregate samples
# ---------------------------------------------------------------------------

if show_stats and n_samples > 1:
    magb_mean, magb_std = magb.mean(axis=0), magb.std(axis=0)
    mag1_mean, mag1_std = mag1.mean(axis=0), mag1.std(axis=0)
    mag2_mean, mag2_std = mag2.mean(axis=0), mag2.std(axis=0)
    mode_label = rf"mean $\pm$ 1$\sigma$ over {n_samples} samples"
else:
    magb_mean, magb_std = magb[0], np.zeros_like(magb[0])
    mag1_mean, mag1_std = mag1[0], np.zeros_like(mag1[0])
    mag2_mean, mag2_std = mag2[0], np.zeros_like(mag2[0])
    mode_label = "single stochastic sample"

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
fig.subplots_adjust(hspace=0.12)
buffer = 0.02

# -- Top panel: combined Mb --
ax1.plot(t_years, magb_mean, lw=2.0, color="blue", label=r"$\dot{M}_b$", zorder=2)
if show_stats and n_samples > 1:
    ax1.fill_between(t_years, magb_mean - magb_std, magb_mean + magb_std, color="blue", alpha=0.3, zorder=1)
ylo1 = np.nanmin(magb_mean - magb_std) - buffer
yhi1 = np.nanmax(magb_mean + magb_std) + buffer
ax1.set_ylim(ylo1, yhi1)
ax1.axhspan(mag_limit, yhi1, color="pink", alpha=0.3, label="LSST lim (sv)", zorder=0)
ax1.set_ylim(ylo1, yhi1)
ax1.invert_yaxis()
ax1.set_ylabel(r"$m_{\rm AB}$", fontsize=14)
ax1.set_title(
    rf"$\dot{{M}}_b$ magnitude in {band}-band, $e_{{\rm b}}$={eb:.2f}, $q_{{\rm b}}$={qb:.2f} | {mode_label}",
    fontsize=14,
)
ax1.legend(fontsize=11, loc="upper right")
ax1.grid(alpha=0.3, linestyle="--")
plt.setp(ax1.get_xticklabels(), visible=False)

# -- Bottom panel: M1 + M2 --
ax2.plot(t_years, mag1_mean, lw=2.0, color="teal", label=r"$\dot{M}_1$", zorder=2)
ax2.plot(t_years, mag2_mean, lw=2.0, color="purple", label=r"$\dot{M}_2$", zorder=2)
if show_stats and n_samples > 1:
    ax2.fill_between(t_years, mag1_mean - mag1_std, mag1_mean + mag1_std, color="teal", alpha=0.3, zorder=1)
    ax2.fill_between(t_years, mag2_mean - mag2_std, mag2_mean + mag2_std, color="purple", alpha=0.3, zorder=1)
all_lo = min(np.nanmin(mag1_mean - mag1_std), np.nanmin(mag2_mean - mag2_std)) - buffer
all_hi = max(np.nanmax(mag1_mean + mag1_std), np.nanmax(mag2_mean + mag2_std)) + buffer
ax2.set_ylim(all_lo, all_hi)
ax2.axhspan(mag_limit, all_hi, color="pink", alpha=0.3, zorder=0)
ax2.set_ylim(all_lo, all_hi)
ax2.invert_yaxis()
ax2.set_ylabel(r"$m_{\rm AB}$", fontsize=14)
ax2.set_title(rf"$\dot{{M}}_1$ and $\dot{{M}}_2$ | {mode_label}", fontsize=14)
ax2.legend(fontsize=11, loc="upper right")
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
