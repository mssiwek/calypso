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
from calypso.data.loader import load_single_component

# Paper colour scheme (Okabe–Ito)
COLORS = {"Mb": "#0072B2", "M1": "#009E73", "M2": "#D55E00"}
# Darker variants for true (simulation) curves
COLORS_TRUE = {"Mb": "#005A8C", "M1": "#007A59", "M2": "#A94600"}

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

TRUE_SHOW_DRAWS0 = False
TRUE_N_DRAWS0 = 5
TRUE_SHOW_STATS0 = False

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


NWINDOWS_TRUE = 500  # window count used in the simulation datasets


@st.cache_resource(show_spinner="Loading simulation time series...")
def _load_true_ts():
    """Load true time series, index by (split, comp, eb, qb)."""
    windows = {}
    for split in ("test", "train"):
        for comp in ("Mb", "M1", "M2"):
            X, y = load_single_component(split, comp, NWINDOWS_TRUE, NORB, NBINS, LOG_MODEL)
            for i in range(len(X)):
                eb_r = round(float(X[i, 0]), 2)
                qb_r = round(float(X[i, 1]), 2)
                windows.setdefault((split, comp, eb_r, qb_r), []).append(y[i])
    return windows


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


def _find_true_windows(windows, eb, qb, comp, n, seed, tol=0.02):
    """Return up to *n* true time-series windows (linear scale) or None."""
    rng = np.random.default_rng(seed)
    eb_r, qb_r = round(eb, 2), round(qb, 2)
    for split in ("test", "train"):
        for (s, c, e, q), arrs in windows.items():
            if s == split and c == comp and abs(e - eb_r) < tol and abs(q - qb_r) < tol:
                idxs = rng.choice(len(arrs), size=min(n, len(arrs)), replace=False)
                out = np.array([arrs[i] for i in idxs])
                return 10 ** out if LOG_MODEL else out
    return None



@st.cache_data(show_spinner=False)
def compute_true_draws(eb, qb, mb, ab, z, band, n_draws, seed, f_edd):
    """Convert *n_draws* true simulation windows to magnitudes."""
    windows = _load_true_ts()
    true_seed = seed + 999
    m1_true = _find_true_windows(windows, eb, qb, "M1", n_draws, true_seed)
    m2_true = _find_true_windows(windows, eb, qb, "M2", n_draws, true_seed)
    mb_true = _find_true_windows(windows, eb, qb, "Mb", n_draws, true_seed)
    if m1_true is None or m2_true is None or mb_true is None:
        return None

    t, t0, _ = get_t(ab, mb)
    lam_cm = get_band_wavelength_grid(z, band)
    radius_primary, radius_secondary = precompute_disk_radii(t, t0, mb, ab, eb, qb)

    qb_re = 1.0 / qb
    lum1 = band_luminosity_samples(m1_true, radius_primary, lam_cm, mb, qb_re, f_edd)
    lum2 = band_luminosity_samples(m2_true, radius_secondary, lam_cm, mb, qb, f_edd)
    lumb = band_luminosity_samples(mb_true, radius_primary, lam_cm, mb, qb, f_edd)

    mag1 = np.asarray(lum_to_mags(lum1, band, z))
    mag2 = np.asarray(lum_to_mags(lum2, band, z))
    magb = np.asarray(lum_to_mags(lumb, band, z))
    return mag1, mag2, magb


@st.cache_data(show_spinner="Computing true statistics...")
def compute_true_stats(eb, qb, mb, ab, z, band, f_edd):
    """Push ALL true windows through the pipeline, return mean/std in mag space."""
    windows = _load_true_ts()
    # Gather all windows (linear scale) per component
    comps = {}
    for comp in ("M1", "M2", "Mb"):
        w = _find_true_windows(windows, eb, qb, comp, NWINDOWS_TRUE, seed=0)
        if w is None:
            return None
        comps[comp] = w

    t, t0, _ = get_t(ab, mb)
    lam_cm = get_band_wavelength_grid(z, band)
    radius_primary, radius_secondary = precompute_disk_radii(t, t0, mb, ab, eb, qb)

    result = {}
    for comp, radius in [("M1", radius_primary), ("M2", radius_secondary), ("Mb", radius_primary)]:
        lum = band_luminosity_samples(comps[comp], radius, lam_cm, mb, qb, f_edd)
        mag = np.asarray(lum_to_mags(lum, band, z))  # (N_windows, T)
        result[comp] = {"mean": mag.mean(axis=0), "std": mag.std(axis=0)}

    return result


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

st.sidebar.subheader("CALYPSO (emulated)")
cal_epistemic = st.sidebar.checkbox("Epistemic uncertainty", value=CAL_EPISTEMIC0, key="cal_epi")
cal_show_draws = st.sidebar.checkbox("Show realisations", value=CAL_SHOW_DRAWS0, key="cal_draws")
cal_n_draws = st.sidebar.slider("Realisations", 1, 64, CAL_N_DRAWS0, 1, key="cal_n") if cal_show_draws else 0
cal_show_stats = st.sidebar.checkbox("Show mean +/- std", value=CAL_SHOW_STATS0, key="cal_stats")
cal_n_stats = st.sidebar.slider("Samples for stats", 8, 256, CAL_N_STATS0, 8, key="cal_nstats") if cal_show_stats else 0

st.sidebar.subheader("True (simulation)")
true_show_draws = st.sidebar.checkbox("Show realisations", value=TRUE_SHOW_DRAWS0, key="true_draws")
true_n_draws = st.sidebar.slider("Realisations", 1, 64, TRUE_N_DRAWS0, 1, key="true_n") if true_show_draws else 0
true_show_stats = st.sidebar.checkbox("Show mean +/- std", value=TRUE_SHOW_STATS0, key="true_stats")

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
    true_draw_curves = None
    if true_show_draws and true_n_draws >= 1:
        true_draw_curves = compute_true_draws(
            eb, qb, mb, ab, z, band, true_n_draws, st.session_state.seed, f_edd,
        )
        if true_draw_curves is None:
            st.warning(
                f"No simulation windows available at eb={eb:.2f}, qb={qb:.2f}; "
                "true realisations not shown. Try an (eb, qb) on the simulation grid."
            )
    true_stat_curves = None
    if true_show_stats:
        true_stat_curves = compute_true_stats(eb, qb, mb, ab, z, band, f_edd)
        if true_stat_curves is None:
            st.warning(
                f"No simulation windows available at eb={eb:.2f}, qb={qb:.2f}; "
                "true mean/std not shown."
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

# -- CALYPSO draws --
if cal_show_draws and cal_n_draws >= 1:
    alpha_c = _draw_alpha(cal_n_draws)
    for i in range(cal_n_draws):
        label_b = r"synth $\dot{M}_b$" if i == 0 else None
        label_1 = r"synth $\dot{M}_1$" if i == 0 else None
        label_2 = r"synth $\dot{M}_2$" if i == 0 else None
        ax1.plot(t_years, magb[i], lw=1.6, color=COLORS["Mb"], alpha=alpha_c, label=label_b, zorder=2)
        ax2.plot(t_years, mag1[i], lw=1.6, color=COLORS["M1"], alpha=alpha_c, label=label_1, zorder=2)
        ax2.plot(t_years, mag2[i], lw=1.6, color=COLORS["M2"], alpha=alpha_c, label=label_2, zorder=2)
        all_magb_vals.append(magb[i])
        all_mag12_vals.extend([mag1[i], mag2[i]])

# -- CALYPSO mean +/- std --
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

# -- True draws --
if true_draw_curves is not None:
    alpha_t = _draw_alpha(len(true_draw_curves[0]))
    for i in range(len(true_draw_curves[0])):
        label_b = r"true $\dot{M}_b$" if i == 0 else None
        label_1 = r"true $\dot{M}_1$" if i == 0 else None
        label_2 = r"true $\dot{M}_2$" if i == 0 else None
        ax1.plot(t_years, true_draw_curves[2][i], lw=1.4, ls="--", color=COLORS_TRUE["Mb"], alpha=alpha_t, label=label_b, zorder=3)
        ax2.plot(t_years, true_draw_curves[0][i], lw=1.4, ls="--", color=COLORS_TRUE["M1"], alpha=alpha_t, label=label_1, zorder=3)
        ax2.plot(t_years, true_draw_curves[1][i], lw=1.4, ls="--", color=COLORS_TRUE["M2"], alpha=alpha_t, label=label_2, zorder=3)
        all_magb_vals.append(true_draw_curves[2][i])
        all_mag12_vals.extend([true_draw_curves[0][i], true_draw_curves[1][i]])

# -- True mean +/- std (from precomputed stats — fast) --
if true_stat_curves is not None:
    ts = true_stat_curves
    _show_true_stats_label = true_draw_curves is None
    lbl_tb = r"true $\dot{M}_b$ mean $\pm 1\sigma$" if _show_true_stats_label else None
    lbl_t1 = r"true primary $\dot{M}_1$ mean $\pm 1\sigma$" if _show_true_stats_label else None
    lbl_t2 = r"true secondary $\dot{M}_2$ mean $\pm 1\sigma$" if _show_true_stats_label else None
    ax1.plot(t_years, ts["Mb"]["mean"], lw=2.2, ls="--", color=COLORS_TRUE["Mb"], label=lbl_tb, zorder=5)
    ax1.fill_between(t_years, ts["Mb"]["mean"] - ts["Mb"]["std"], ts["Mb"]["mean"] + ts["Mb"]["std"], color=COLORS_TRUE["Mb"], alpha=0.18, zorder=1)
    ax2.plot(t_years, ts["M1"]["mean"], lw=2.2, ls="--", color=COLORS_TRUE["M1"], label=lbl_t1, zorder=5)
    ax2.fill_between(t_years, ts["M1"]["mean"] - ts["M1"]["std"], ts["M1"]["mean"] + ts["M1"]["std"], color=COLORS_TRUE["M1"], alpha=0.18, zorder=1)
    ax2.plot(t_years, ts["M2"]["mean"], lw=2.2, ls="--", color=COLORS_TRUE["M2"], label=lbl_t2, zorder=5)
    ax2.fill_between(t_years, ts["M2"]["mean"] - ts["M2"]["std"], ts["M2"]["mean"] + ts["M2"]["std"], color=COLORS_TRUE["M2"], alpha=0.18, zorder=1)
    all_magb_vals.extend([ts["Mb"]["mean"] - ts["Mb"]["std"], ts["Mb"]["mean"] + ts["Mb"]["std"]])
    all_mag12_vals.extend([ts["M1"]["mean"] - ts["M1"]["std"], ts["M1"]["mean"] + ts["M1"]["std"],
                           ts["M2"]["mean"] - ts["M2"]["std"], ts["M2"]["mean"] + ts["M2"]["std"]])

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

