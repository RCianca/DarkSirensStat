import numpy as np
import healpy as hp
from scipy.integrate import quad, quad_vec,simpson
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
# def r_z_vectorized(z, H0, Om=Om0GLOB):
#     """
#     Vectorized comoving distance r(z) for array inputs.
#     Handles both scalar and array inputs for z.
#     """
#     c = clight
#     integrand = lambda x: 1 / E_z(x, H0, Om)
#     if np.isscalar(z):
#         integral = quad_vec(integrand, 0, z)[0]
#     else:
#         integral = np.array([quad_vec(integrand, 0, zi)[0] for zi in z])
#     return integral * c / H0
@njit
def r_z_vectorized(z, H0, Om=Om0GLOB, num_points=500):
    """
    Vectorized comoving distance r(z) for array inputs using Simpson's rule.
    Handles both scalar and array inputs for z.
    """
    c = clight

    def integrand(x):
        return 1 / E_z(x, H0, Om)

    if np.isscalar(z):
        x_grid = np.linspace(0, z, num_points)
        y_values = integrand(x_grid)
        if len(x_grid) == 0 or len(y_values) == 0:
            raise ValueError("Empty integration grid or invalid values")
        integral = simpson(y_values, x_grid)
    else:
        integral = np.array([
            simpson(
                integrand(np.linspace(0, zi, num_points)),
                np.linspace(0, zi, num_points)
            ) for zi in z
        ])
    if integral is None or np.isnan(integral).any():
        raise ValueError("Integration failed, returned None or NaN")
    return integral * c / H0


def Dl_z_vectorized(z, H0, Om=Om0GLOB):
    """
    Vectorized luminosity distance D_L(z) for array inputs.
    """
    return r_z_vectorized(z, H0, Om) * (1 + z)

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
# Not to test: GWtest03
fname = [
'GWtest52.fits','GWtest01.fits','GWtest02.fits','GWtest03.fits','GWtest04.fits','GWtest05.fits',
'GWtest06.fits','GWtest07.fits','GWtest08.fits','GWtest09.fits','GWtest10.fits','GWtest11.fits',
'GWtest12.fits','GWtest13.fits','GWtest14.fits','GWtest15.fits','GWtest16.fits','GWtest17.fits',
'GWtest18.fits','GWtest19.fits','GWtest20.fits','GWtest21.fits','GWtest22.fits','GWtest23.fits',
'GWtest24.fits','GWtest25.fits','GWtest26.fits','GWtest27.fits','GWtest28.fits','GWtest29.fits',
'GWtest30.fits','GWtest31.fits','GWtest32.fits','GWtest33.fits','GWtest34.fits','GWtest35.fits',
'GWtest36.fits','GWtest37.fits','GWtest38.fits','GWtest39.fits','GWtest40.fits','GWtest41.fits',
'GWtest42.fits','GWtest43.fits','GWtest44.fits','GWtest45.fits','GWtest46.fits','GWtest47.fits',
'GWtest49.fits','GWtest49.fits','GWtest50.fits','GWtest51.fits','GWtest53.fits','GWtest54.fits',
]

# Name of the runpath folder for saving results
runpath = 'test-frac005_54DS'
#Host Catalogue to read
to_read = 'Uniform_paper_sampled_frac_005.txt'
