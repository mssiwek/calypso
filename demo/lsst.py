"""LSST filter metadata used by the demo photometry utilities."""

from types import MappingProxyType


LSST_FILTERS = MappingProxyType(
    {
        "u": {
            "lambda_min": 3200e-8,
            "lambda_max": 4000e-8,
            "lambda_pivot": 3740e-8,
            "mag_limit_single": 23.9,
            "mag_limit_coadd": 26.1,
        },
        "g": {
            "lambda_min": 4000e-8,
            "lambda_max": 5500e-8,
            "lambda_pivot": 4870e-8,
            "mag_limit_single": 25.0,
            "mag_limit_coadd": 27.4,
        },
        "r": {
            "lambda_min": 5500e-8,
            "lambda_max": 7000e-8,
            "lambda_pivot": 6220e-8,
            "mag_limit_single": 24.7,
            "mag_limit_coadd": 27.5,
        },
        "i": {
            "lambda_min": 7000e-8,
            "lambda_max": 8200e-8,
            "lambda_pivot": 7540e-8,
            "mag_limit_single": 24.0,
            "mag_limit_coadd": 26.8,
        },
        "z": {
            "lambda_min": 8200e-8,
            "lambda_max": 9200e-8,
            "lambda_pivot": 8690e-8,
            "mag_limit_single": 23.3,
            "mag_limit_coadd": 26.1,
        },
        "y": {
            "lambda_min": 9200e-8,
            "lambda_max": 10500e-8,
            "lambda_pivot": 9710e-8,
            "mag_limit_single": 22.1,
            "mag_limit_coadd": 24.9,
        },
    }
)


def get_lsst_filter(band: str) -> dict:
    try:
        return dict(LSST_FILTERS[band])
    except KeyError as exc:
        raise KeyError(f"Unknown LSST band: {band!r}") from exc


# Backward-compatible alias used by existing notebooks and scripts.
lsst_filters = LSST_FILTERS
