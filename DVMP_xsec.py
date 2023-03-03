### Module for calculating meson production cross-sections and TFFs

import numpy as np
import Evolution as ev
import Observables as obs

from numpy import cos as Cos
from numpy import sin as Sin
from numpy import real as Real
from numpy import imag as Imag
from numpy import conjugate as Conjugate
from scipy.integrate import quad

from numba import njit, vectorize

"""

********************************Masses, decay constants, etc.***********************

"""

M_p = 0.938
M_n = 0.940
M_rho = 0.775
M_phi = 1.019
M_jpsi = 3.097
gevtonb = 389.9 * 1000
alphaEM = 1 / 137.036




"""

******************************Cross-sections for proton target (currently for virtual photon scattering sub-process)*********************************

"""


@njit
def prefac_rho(y: float, xB: float, t: float, Q: float, phi: int): # Multiply with gamma factor (photon flux) for full cross section. Here for rho and phi we use the R(Q^2) fit from Muller and Lautenschlager.
    
    return gevtonb * ev.AlphaS(2, 2, Q)**2 * ((1 - y) / (1 - y - y**2 / 2) + (1 + 2.2 * Q**2 / M_rho**2)**0.451 * M_rho**2 / Q**2)  * alphaEM * (xB ** 2) * np.pi * ev.CF ** 2 / (1 - xB) / (Q ** 6) / np.sqrt(1 - eps(xB, Q) ** 2) / ev.NC ** 2

@njit
def prefac_phi(y: float, xB: float, t: float, Q: float, phi: int):
    
    return gevtonb * ev.AlphaS(2, 2, Q)**2 * ((1 - y) / (1 - y - y**2 / 2) + (1 + 25.4 * Q**2 / M_phi**2)**0.180 * M_phi**2 / Q**2)  * alphaEM * (xB ** 2) * np.pi * ev.CF ** 2 / (1 - xB) / (Q ** 6) / np.sqrt(1 - eps(xB, Q) ** 2) / ev.NC ** 2

@njit
def prefac_jpsi(y: float, xB: float, t: float, Q: float, phi: int):
    
    return gevtonb * ev.AlphaS(2, 2, Q)**2 * alphaEM * (xB ** 2) * np.pi * ev.CF ** 2 / (1 - xB) / (Q ** 6) / np.sqrt(1 - eps(xB, Q) ** 2) / ev.NC ** 2

# Jpsi xsec is calculated as longitudinal piece only, compared to total xsec from data converted to longitudinal piece with experimental values for R and error propagation


@njit
def eps(xB: float, Q: float):
    
    return 2 * xB * M_p / Q

@njit
def m2(xB: float, Q: float, t: float):
    
    return -1 * (Q ** 2 - M_p ** 2 + (t / 2)) ** 3 * (1 / ((2 * Q ** 2) / xB - Q ** 2 - M_p ** 2 + t) ) ** 2 / ( 2 * Q ** 4)

@njit
def K(xB: float, Q: float, t: float):
    
    return np.sqrt(-1 * M_p ** 2 * xB ** 2 * ((1 - ((m2(xB, Q, t) - t) / Q ** 2)) ** 2 + 4 * m2(xB, Q, t) * (1 - (t / 4 / (M_p ** 2))) / (Q ** 2)) - (1 - xB) * t * (1 - (xB * (m2(xB, Q, t) - t) / (Q ** 2))))

@vectorize(["float64(float64,float64,float64,float64,float64,complex128,complex128)"])
def dsigma_rho_dt(y: float, xB: float, t: float, Q: float, phi: float, HTFF_rho: complex, ETFF_rho: complex):
    
    """
    
    differential cross section in t
    
    """
    
    """
    
    unpolarized nucleon to unpolarized nucleon
    
    """
    return prefac_rho(y, xB, t, Q, phi) * ((4 * (1 - xB) * (1 - xB * (m2(xB, Q, t) - t) / (Q ** 2)) - (m2(xB, Q, t) * eps(xB, Q) ** 2 / (M_p ** 2))) / ((2 - xB - (xB * (m2(xB, Q, t) - t) / Q ** 2)) ** 2) * (HTFF_rho - ((xB ** 2 * ((1 + (m2(xB, Q, t) - t) / Q ** 2) ** 2) + 4 * xB * t / Q ** 2) * ETFF_rho / (4 * (1 - xB) * (1 - xB * (m2(xB, Q, t) - t) / Q ** 2) - (m2(xB, Q, t) * eps(xB,Q) ** 2 / M_p ** 2)))) * Conjugate((HTFF_rho - ((xB ** 2 * ((1 + (m2(xB, Q, t) - t) / Q ** 2) ** 2) + 4 * xB * t / Q ** 2) * ETFF_rho / (4 * (1 - xB) * (1 - xB * (m2(xB, Q, t) - t) / Q ** 2) - (m2(xB, Q, t) * eps(xB,Q) ** 2 / M_p ** 2))))) + (K(xB, Q, t) ** 2 * ETFF_rho * Conjugate(ETFF_rho) / 4 / M_p ** 2 / ((1 - xB) * (1 - xB * (m2(xB, Q, t) - t) / Q ** 2) - (m2(xB, Q, t) * eps(xB, Q) ** 2 / 4 / M_p ** 2))))
    
   

@vectorize(["float64(float64,float64,float64,float64,float64,complex128,complex128)"])    
def dsigma_phi_dt(y: float, xB: float, t: float, Q: float, phi: float, HTFF_phi: complex, ETFF_phi: complex):
    
    """
    
    unpolarized nucleon to unpolarized nucleon
    
    """
    return prefac_phi(y, xB, t, Q, phi) * ((4 * (1 - xB) * (1 - xB * (m2(xB, Q, t) - t) / (Q ** 2)) - (m2(xB, Q, t) * eps(xB, Q) ** 2 / (M_p ** 2))) / ((2 - xB - (xB * (m2(xB, Q, t) - t) / Q ** 2)) ** 2) * (HTFF_phi - ((xB ** 2 * ((1 + (m2(xB, Q, t) - t) / Q ** 2) ** 2) + 4 * xB * t / Q ** 2) * ETFF_phi / (4 * (1 - xB) * (1 - xB * (m2(xB, Q, t) - t) / Q ** 2) - (m2(xB, Q, t) * eps(xB,Q) ** 2 / M_p ** 2)))) * Conjugate((HTFF_phi - ((xB ** 2 * ((1 + (m2(xB, Q, t) - t) / Q ** 2) ** 2) + 4 * xB * t / Q ** 2) * ETFF_phi / (4 * (1 - xB) * (1 - xB * (m2(xB, Q, t) - t) / Q ** 2) - (m2(xB, Q, t) * eps(xB,Q) ** 2 / M_p ** 2))))) + (K(xB, Q, t) ** 2 * ETFF_phi * Conjugate(ETFF_phi) / 4 / M_p ** 2 / ((1 - xB) * (1 - xB * (m2(xB, Q, t) - t) / Q ** 2) - (m2(xB, Q, t) * eps(xB, Q) ** 2 / 4 / M_p ** 2))))
    
    
    
    
@vectorize(["float64(float64,float64,float64,float64,float64,complex128,complex128)"])    
def dsigma_Jpsi_dt(y: float, xB: float, t: float, Q: float, phi: float, HTFF_jpsi: complex, ETFF_jpsi: complex):
    
    """
    
    unpolarized nucleon to unpolarized nucleon
    
    """
    
    return prefac_jpsi(y, xB, t, Q, phi) * ((4 * (1 - xB) * (1 - xB * (m2(xB, Q, t) - t) / (Q ** 2)) - (m2(xB, Q, t) * eps(xB, Q) ** 2 / (M_p ** 2))) / ((2 - xB - (xB * (m2(xB, Q, t) - t) / Q ** 2)) ** 2) * (HTFF_jpsi - ((xB ** 2 * ((1 + (m2(xB, Q, t) - t) / Q ** 2) ** 2) + 4 * xB * t / Q ** 2) * ETFF_jpsi / (4 * (1 - xB) * (1 - xB * (m2(xB, Q, t) - t) / Q ** 2) - (m2(xB, Q, t) * eps(xB,Q) ** 2 / M_p ** 2)))) * Conjugate((HTFF_jpsi - ((xB ** 2 * ((1 + (m2(xB, Q, t) - t) / Q ** 2) ** 2) + 4 * xB * t / Q ** 2) * ETFF_jpsi / (4 * (1 - xB) * (1 - xB * (m2(xB, Q, t) - t) / Q ** 2) - (m2(xB, Q, t) * eps(xB,Q) ** 2 / M_p ** 2))))) + (K(xB, Q, t) ** 2 * ETFF_jpsi * Conjugate(ETFF_jpsi) / 4 / M_p ** 2 / ((1 - xB) * (1 - xB * (m2(xB, Q, t) - t) / Q ** 2) - (m2(xB, Q, t) * eps(xB, Q) ** 2 / 4 / M_p ** 2))))

    
