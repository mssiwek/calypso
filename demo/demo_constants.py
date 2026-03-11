""" CONSTANTS USED IN ACCRETION VARIABILITY PROJECT """
import numpy as np

""" Binary parameters """
Pb = 2.*np.pi
ab = 1
omega = 1

""" Disk parameters """ 
Sigma0 = 1.e-4  # Surface density normalization [g/cm^2]

""" Standard figure sizes and formatting """
colors = [
    (0.0039, 0.4510, 0.6980),
    (0.8706, 0.5608, 0.0196),
    (0.0078, 0.6196, 0.4510),
    (0.8353, 0.3686, 0.0000),
    (0.8000, 0.4706, 0.7373),
    (0.7922, 0.5686, 0.3804),
    (0.9843, 0.6863, 0.8941),
    (0.5804, 0.5804, 0.5804),
]

figwidth = 13
figheight = 6
fontsize = 20
ticksize = 10
tickwidth = 2.5
linewidth = 2
cPrim = colors[3]
cSec = colors[2]
cComb = colors[0]

""" Physical constants in cgs units """
G = 6.67430e-8  # Gravitational constant [cm^3/g/s^2]
SIGMA = 5.670374419e-5  # Stefan-Boltzmann constant [erg/s/cm^2/K^4]
SIGMA_T = 6.6524587321e-25  # Thomson cross-section [cm^2]
HP = 6.62607015e-27  # Planck's constant [erg s]
KB = 1.380649e-16  # Boltzmann's constant [erg/K]
C = 2.99792458e10  # Speed of light [cm/s]
MSUN = 1.9885e33  # Solar mass [grams]

""" Derived constants """
MYR = 3.15576e13  # Megayear in seconds
YR = 3.15576e7  # Year in seconds
PC = 3.085677581491367e18  # Parsec in cm
ETA = 0.1  # Efficiency factor for accretion


