import numpy as np
from scipy.optimize import newton

try:
    from .lsst import LSST_FILTERS
except ImportError:
    from lsst import LSST_FILTERS

G = 6.67430e-8  # Gravitational constant [cm^3/g/s^2]
sigma = 5.670374419e-5  # Stefan-Boltzmann constant [erg/s/cm^2/K^4]
h = 6.62607015e-27  # Planck's constant [erg s]
k = 1.380649e-16  # Boltzmann's constant [erg/K]
c = 2.99792458e10  # Speed of light [cm/s]
msun = 1.9885e33  # Solar mass [grams]

def kepler_eq(E, M, e):
    return E - e * np.sin(E) - M

def rb_of_t(t, mb, ab, eb, t0=np.pi):
    """
    Compute the binary separation r(t) at time t.

    Parameters
    ----------
    t : float or array-like
        Time(s) at which to evaluate the separation [seconds]
    mb : float
        Total mass of binary [grams]
    ab : float
        Semi-major axis [cm]
    eb : float
        Eccentricity (0 <= e < 1)
    t0 : float
        Time of periastron passage [seconds]
        Default is pi, which corresponds to periastron at t=0.

    Returns
    -------
    r : float or array-like
        Separation between the two components [cm]
    """
    n = np.sqrt(G * mb / ab**3)  # mean motion [rad/s]
    M = n * (np.asarray(t) - t0)  # mean anomaly

    def solve_E(Mi):
        # Initial guess: E ~ M
        return newton(kepler_eq, Mi, args=(Mi, eb), tol=1e-10, maxiter=100)

    # Solve for eccentric anomaly
    E = np.vectorize(solve_E)(M)

    # Compute separation
    r = ab * (1 - eb * np.cos(E))
    return r


def Eggleton_Roche_Radius(t, t0, mb, ab, eb, qb):
	"""
	Calculate the Roche radius for a binary system using Eggleton's formula.

	Parameters
	----------
    t : float or array-like
        Time(s) at which to evaluate the Roche radius [seconds]
    t0 : float
        Time of periastron passage [seconds]
	ab : float
		Semi-major axis of the binary orbit [cm]
    eb : float 
        Eccentricity of the binary orbit
    qb : float
        Mass ratio of the binary. 
        IMPORTANT: This is either m1/m2 or m2/m1, depending on which star's Roche radius you want to calculate.
        E.g., If you want the Roche radius of the primary star, qb = m1/m2.
              However, if you want the Roche radius of the secondary star, qb = m2/m1.

	Returns
	-------
	R : float
		Roche radius of the primary star [cm]
	"""
	return 0.49 * (qb**(2/3)) * rb_of_t(t, mb, ab, eb, t0) / (0.6 * qb**(2/3) + np.log(1 + qb**(1/3)))

def steady_csd_radius(ab):
    """
    Calculate the radius of a steady circumbinary disk (CSD) around a black hole.

    Returns
    -------
    r_csd : float
        Radius of the steady CSD [cm]
    """
    # For a steady CSD, we assume it extends to a radius of 0.27 ab, according to Roedig+2015
    # This is an arbitrary choice for demonstration purposes.
    r_csd = 0.27 * ab  # 1000 times the Schwarzschild radius
    return r_csd


def radius_minidisk(t, t0, mb, ab, eb, qb, n_annuli=100, r_ER=True):
    """
    Calculate the disk annuli around a black hole.

    Parameters
	----------
    t : float or array-like
        Time(s) at which to evaluate the disk radius [seconds]
    t0 : float
        Time of periastron passage [seconds]
    mb : float
        Mass of the accreting star [grams]
	ab : float
		Semi-major axis of the binary orbit [cm]
    eb : float 
        Eccentricity of the binary orbit
    qb : float
        Mass ratio of the binary. 
        IMPORTANT: This is either m1/m2 or m2/m1, depending on which star's Roche radius you want to calculate.
        E.g., If you want the Roche radius of the primary star, qb = m1/m2.
              However, if you want the Roche radius of the secondary star, qb = m2/m1.


    Returns
    -------
    r_minidisk : float
        Radii of the disk annuli [cm]
    """
    
    m_i = (qb/ (1 + qb)) * mb  # mass of the accreting star
    
    # Within 20rg, the disk actually mostly emits in X-rays, 
    # so any thermal/optical emission calcuated from this region must be discarded.
    r_min = 20 * G * m_i / (c**2)
    
    if r_ER:
        r_max = Eggleton_Roche_Radius(t, t0, mb, ab, eb, qb)  # Roche radius
    else:
        r_max = steady_csd_radius(ab)
    if r_max < r_min:
        print("r_max:", r_max)
        print("r_min:", r_min)
        raise ValueError("Roche radius is smaller than minimum disk radius. Check parameters.")
    
    r_minidisk = np.linspace(r_min, r_max, n_annuli)

    return r_minidisk


def T_R(R, mi, midot):
    """
    Disk temperature profile (simplified thin-disk model).
    """
    return (3 * G * mi * midot / (8 * np.pi * sigma * R**3))**0.25


def planck_lambda(wavelength_cm, T):
    """
    Planck function B_lambda for spectral radiance.

    Parameters
    ----------
    wavelength_cm : float or ndarray
        Wavelength in cm
    T : float
        Temperature in Kelvin

    Returns
    -------
    B_lambda : float or ndarray
        Spectral radiance in erg/s/cm^2/sr/cm
    """

    lam = np.asarray(wavelength_cm, dtype=float)
    temp = np.asarray(T, dtype=float)

    lam, temp = np.broadcast_arrays(lam, temp)
    out = np.zeros_like(lam, dtype=float)

    valid = (lam > 0) & (temp > 0)
    if not np.any(valid):
        return out

    exponent = h * c / (lam[valid] * k * temp[valid])
    exponent = np.clip(exponent, None, 700.0)
    denom = np.expm1(exponent)
    out[valid] = (2 * h * c**2) / (lam[valid] ** 5 * denom)
    return out

def L_lambda_disk(lam_cm, R_grid, T_grid):
    """
    Compute L_lambda by summing over annuli.

    Parameters
    ----------
    lam_cm : float
        Wavelength [cm]
    R_grid : ndarray
        Radii of annuli [cm]
    T_grid : ndarray
        Temperature at each annulus [K]

    Returns
    -------
    L_lambda : float
        Spectral luminosity [erg/s/cm]
    """
    radius = np.asarray(R_grid, dtype=float)
    temperature = np.asarray(T_grid, dtype=float)
    surface_integrand = 4 * np.pi**2 * radius * planck_lambda(lam_cm, temperature)
    return np.trapz(surface_integrand, radius)


def spectral_luminosity_grid(lam_cm, R_grid, T_grid):
    """
    Compute the spectral luminosity across a wavelength grid for a disk.

    Parameters
    ----------
    lam_cm : array-like
        Wavelength grid in cm.
    R_grid : array-like
        Annulus radii in cm.
    T_grid : array-like
        Temperature profile in Kelvin.

    Returns
    -------
    ndarray
        Spectral luminosity evaluated on the wavelength grid.
    """
    lam = np.asarray(lam_cm, dtype=float)
    radius = np.asarray(R_grid, dtype=float)
    temperature = np.asarray(T_grid, dtype=float)

    if lam.ndim != 1:
        raise ValueError("lam_cm must be one-dimensional")
    if radius.ndim != 1 or temperature.ndim != 1 or radius.shape != temperature.shape:
        raise ValueError("R_grid and T_grid must be one-dimensional arrays with matching shape")

    radiance = planck_lambda(lam[:, None], temperature[None, :])
    surface_integrand = 4 * np.pi**2 * radius[None, :] * radiance
    return np.trapz(surface_integrand, radius, axis=1)


def band_luminosity(lam_cm, R_grid, T_grid):
    """
    Integrate spectral luminosity over a wavelength band.
    """
    spectral_luminosity = spectral_luminosity_grid(lam_cm, R_grid, T_grid)
    return np.trapz(spectral_luminosity, np.asarray(lam_cm, dtype=float))

def lambda_wien(T):
    """
        Compute the wavelength at which blackbody emits maximum spectral radiance (Wien's displacement law)
        
        Parameters
        ----------
        T: float
            Temperature [K]
            
        Returns
        -------
        lambda_max: float
            wavelength of max spectral radiance [cm]
    """
    b = 0.289777 # in cm * Kelvin
    lambda_max = b/T
    return(lambda_max)

def m_edd(mi, f_edd, eta=0.1):
    """
    Compute the physical mass accretion rate [g/s] for a given black hole mass in grams.

    Parameters
    ----------
    mi : float
        Mass of the black hole [grams]
    f_edd : float
        Eddington fraction (dimensionless)
    eta : float
        Radiative efficiency (default 0.1)

    Returns
    -------
    mdot : float
        Mass accretion rate [g/s]
    """
    G = 6.67430e-8       # cm^3 g^-1 s^-2
    c = 2.99792458e10    # cm/s
    m_p = 1.6726e-24     # g
    sigma_T = 6.6524e-25 # cm^2

    L_edd = (4 * np.pi * G * mi * m_p * c) / sigma_T  # erg/s
    mdot = f_edd * L_edd / (eta * c**2)               # g/s
    return mdot


def get_lsst_filters():
    return {band: dict(values) for band, values in LSST_FILTERS.items()}
