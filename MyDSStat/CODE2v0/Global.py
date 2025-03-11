import numpy as np
import healpy as hp
from scipy.integrate import simpson #quad, quad_vec
#from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from astropy.cosmology import FlatLambdaCDM
from numba import njit
import os 

# --------------------- Global Constants ----------------------------------
href = 67  # Hubble constant reference value
Om0GLOB = 0.319  # Matter density
Xi0Glob = 1.0  # Cosmological coupling constant
clight = 2.99792458 * 10**5  # Speed of light in km/s
cosmoflag = FlatLambdaCDM(H0=href, Om0=Om0GLOB)


# Global variable for log directory
log_folder = "."

def set_log_folder(folder_path):
    """Set the global log folder for saving debug logs."""
    global log_folder
    log_folder = folder_path


# --------------------- Cosmology Functions ----------------------------------
# def r_z(z, H0, Om=Om0GLOB):
#     """
#     Scalar comoving distance r(z).
#     """
#     c = clight
#     integrand = lambda x: 1 / E_z(x, H0, Om)
#     integral, error = quad(integrand, 0, z)
#     return integral * c / H0

# def Dl_z(z, H0, Om=Om0GLOB):
#     """
#     Scalar luminosity distance D_L(z).
#     """
#     return r_z(z, H0, Om) * (1 + z)
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
def E_z(z, H0, Om=Om0GLOB):
    """
    Helper function for Hubble parameter as a function of redshift.
    """
    return np.sqrt(Om * (1 + z)**3 + (1 - Om))
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
########################################DEBUG#########################################
# def r_z_vectorized(z, H0, Om=Om0GLOB, num_points=500):
#     c = clight
#     def integrand(x):
#         return 1 / E_z(x, H0, Om)

#     try:
#         if np.isscalar(z):
#             x_grid = np.linspace(0, z, num_points)
#             y_values = integrand(x_grid)
#             integral = simpson(y_values, x_grid)
#         else:
#             integral = np.array([
#                 simpson(
#                     integrand(np.linspace(0, zi, num_points)),
#                     np.linspace(0, zi, num_points)
#                 ) for zi in z
#             ])
#     except Exception as e:
#         log_file = os.path.join(log_folder, "debug_global.log")
#         with open(log_file, "a") as f:
#             f.write(f"Error in r_z_vectorized: {e}\n")
#         return np.nan

#     if np.isnan(integral).any():
#         log_file = os.path.join(log_folder, "debug_global.log")
#         with open(log_file, "a") as f:
#             f.write(f"NaN in r_z_vectorized output for z={z}, H0={H0}\n")

#     return integral * c / H0
##############################################################################################

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

def InputEvents(start, end):
    """
    Generates a list of GW test file names in the format 'GWtestXX.fits'
    where XX ranges from start to end (inclusive).

    Parameters:
    start (int): Starting number for file names (inclusive).
    end (int): Ending number for file names (inclusive).

    Returns:
    list: List of file names in the specified range.
    """
    if start == end:
        return [f"GWtest{start:02d}.fits"]
    return [f"GWtest{num:02d}.fits" for num in range(start, end + 1)]

def ThresholdInput(th_value, start, stop):
    """
    Selects elements from the numpy file in the folder /DS_th/th_{th_value}/ 
    between the indices `start` and `stop`.

    Parameters:
        th_value (int): Threshold value used to determine the folder.
        start (int): Start index.
        stop (int): Stop index.

    Returns:
        list or str: A list of filenames if start != stop, else a single filename.
    """
    folder_path = f"DS_th/th_{th_value}"
    file_path = os.path.join(folder_path, "saved_events.npy")

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return []

    # Load the saved events from the numpy file
    events = np.load(file_path)

    # Handle stop being too large
    stop = min(stop, len(events))

    # If start and stop are the same, return a single filename
    if start == stop:
        return events[start] if start < len(events) else None

    # Otherwise, return the list of events within range
    return events[start:stop]



#PARAMETES FOR THE CORE SCRIPT#######################
#print('Loading GW data')
working_dir = os.getcwd()
MapPath = os.path.join(working_dir, 'Events/Uniform/TestRun01/')
start=1100
stop=1199
pix_threshold=200
H0min, H0max = 40, 100
which_beta='Beta2v0'#'Beta_fast'#'Beta2v0'
# List of GW data files to process
#fname = InputEvents(start,stop)
th_start=0
th_stop=100
fname = ThresholdInput(pix_threshold, th_start, th_stop)
#fname=['GWtest1100.fits']

# Name of the runpath folder for saving results
runpath = 'Paper-Uniform_old_dens_testrun01_few_1100_1199'
#Host Catalogue to read
to_read = 'Uniform_paper_sampled_density_of_version_one_testrun01.txt'
#Uniform_paper_sampled_frac_005-host
#Uniform_paper_sampled_density_of_version_one
#Uniform_paper_sampled_density_of_version_one_testrun01.txt
#Uniform_paper_sampled_density_of_version_one_testrun01_few.txt

