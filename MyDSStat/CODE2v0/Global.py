import numpy as np
import healpy as hp
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from astropy.cosmology import FlatLambdaCDM
from numba import njit

# --------------------- Global Constants ----------------------------------
href = 67  # Hubble constant reference value
Om0GLOB = 0.319  # Matter density
Xi0Glob = 1.0  # Cosmological coupling constant
clight = 2.99792458 * 10**5  # Speed of light in km/s
cosmoflag = FlatLambdaCDM(H0=href, Om0=Om0GLOB)

# --------------------- Cosmology Functions ----------------------------------

@njit
def E_z(z, H0, Om=Om0GLOB):
    """
    Helper function for Hubble parameter as a function of redshift.
    """
    return np.sqrt(Om * (1 + z)**3 + (1 - Om))

def r_z(z, H0, Om=Om0GLOB):
    """
    Scalar comoving distance r(z).
    """
    c = clight
    integrand = lambda x: 1 / E_z(x, H0, Om)
    integral, error = quad(integrand, 0, z)
    return integral * c / H0

def Dl_z(z, H0, Om=Om0GLOB):
    """
    Scalar luminosity distance D_L(z).
    """
    return r_z(z, H0, Om) * (1 + z)

def r_z_vectorized(z, H0, Om=Om0GLOB):
    """
    Vectorized comoving distance r(z) for array inputs.
    """
    c = clight
    integrand = lambda x: 1 / E_z(x, H0, Om)
    integral = quad_vec(integrand, 0, z)[0]  # Vectorized integration
    return integral * c / H0

def Dl_z_vectorized(z, H0, Om=Om0GLOB):
    """
    Vectorized luminosity distance D_L(z) for array inputs.
    """
    return r_z_vectorized(z, H0, Om) * (1 + z)

# --------------------- Precomputed Distance ------------------------------

def precompute_r_z(H0, Om=Om0GLOB, z_max=10, num_points=10000):
    """
    Precomputes comoving distance r(z) and creates an interpolation function.
    """
    z_grid = np.linspace(0, z_max, num_points)
    integrand = lambda x: 1 / E_z(x, H0, Om)
    r_values = [quad(integrand, 0, z)[0] for z in z_grid]  # Compute r(z) for grid
    r_interp = interp1d(z_grid, np.array(r_values) * clight / H0, kind='cubic', fill_value="extrapolate")
    return r_interp

# Precompute r(z) for Dl_z_approx
r_interp = precompute_r_z(H0=href, Om=Om0GLOB)

def r_z_approx(z):
    """
    Comoving distance r(z) using precomputed interpolation.
    """
    return r_interp(z)

def Dl_z_approx(z, H0, Om=Om0GLOB):
    """
    Luminosity distance D_L(z) using precomputed interpolation.
    """
    return r_z_approx(z) * (1 + z)

# --------------------- Redshift and Hubble Functions ---------------------

def z_from_dcom(dc_val):
    """
    Returns redshift for a given comoving distance dc (in Mpc).
    """
    func = lambda z: cosmoflag.comoving_distance(z).value - dc_val
    z = fsolve(func, 0.02)
    return z[0]

def h_of_z_dl(z, dl):
    """
    Solves for H0 given a redshift z and luminosity distance dl.
    """
    func = lambda h: Dl_z(z, h, Om0GLOB) - dl
    heq = fsolve(func, 30)[0]
    return heq

def z_from_dL(dL_val):
    """
    Returns redshift for a given luminosity distance dL (in Mpc).
    """
    func = lambda z: cosmoflag.luminosity_distance(z).value - dL_val
    z = fsolve(func, 0.02)
    return z[0]

# --------------------- HEALPix Utilities ---------------------------------

def compute_area(nside, all_pixels, p_posterior, level=0.99):
    """
    Computes the area of the level% credible region in square degrees.
    """
    pixarea = hp.nside2pixarea(nside)
    return get_credible_region_pixels(all_pixels, p_posterior, level=level).size * pixarea * (180 / np.pi)**2

def _get_credible_region_pth(p_posterior, level=0.99):
    """
    Finds the probability threshold for the x% credible region (default 99%).
    """
    prob_sorted = np.sort(p_posterior)[::-1]
    prob_sorted_cum = np.cumsum(prob_sorted)
    idx = np.searchsorted(prob_sorted_cum, level)
    return prob_sorted[idx]

def get_credible_region_pixels(all_pixels, p_posterior, level=0.99):
    """
    Returns the pixels within the level% credible region.
    """
    return all_pixels[p_posterior > _get_credible_region_pth(p_posterior, level=level)]

# --------------------- File and Run Settings -----------------------------

# List of GW data files to process
fname = ['GWtest07.fits', 'GWtest08.fits']

# Name of the runpath folder for saving results
runpath = 'FirstBatch'
