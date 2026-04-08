"""Photometry helpers for converting luminosities to AB magnitudes."""

import numpy as np
from functools import lru_cache

from alpha_disk import c
from lsst import get_lsst_filter


@lru_cache(maxsize=1)
def _cosmology():
    from astropy.cosmology import FlatLambdaCDM

    return FlatLambdaCDM(H0=70, Om0=0.3)


def _luminosity_distance_cm(z):
    import astropy.units as u

    return _cosmology().luminosity_distance(z).to(u.cm).value


def lum_to_mags(lum, band, z):
    """
    Convert band-integrated luminosity to AB magnitude.
    """
    band_meta = get_lsst_filter(band)
    lum = np.asarray(lum, dtype=float)

    luminosity_distance = _luminosity_distance_cm(z)
    flux_band = lum / (4 * np.pi * luminosity_distance**2 * (1 + z))

    lambda_min = band_meta["lambda_min"] / (1 + z)
    lambda_max = band_meta["lambda_max"] / (1 + z)
    delta_lambda = lambda_max - lambda_min
    if delta_lambda <= 0:
        raise ValueError(f"Invalid rest-frame bandpass width for band {band!r} at z={z}")

    flux_lambda = flux_band / delta_lambda
    f_nu = (band_meta["lambda_pivot"] ** 2 * flux_lambda) / c

    magnitudes = np.full_like(f_nu, np.inf, dtype=float)
    valid = np.isfinite(f_nu) & (f_nu > 0)
    magnitudes[valid] = -2.5 * np.log10(f_nu[valid]) - 48.6
    return magnitudes.item() if magnitudes.ndim == 0 else magnitudes
