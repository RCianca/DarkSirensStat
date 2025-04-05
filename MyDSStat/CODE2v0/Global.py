import numpy as np
import healpy as hp
from scipy.integrate import simpson
from scipy.optimize import fsolve
from astropy.cosmology import FlatLambdaCDM
from numba import njit
import os 
import warnings

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

@njit
def E_z(z, H0, Om=Om0GLOB):
    """
    Helper function for Hubble parameter as a function of redshift.
    """
    return np.sqrt(Om * (1 + z)**3 + (1 - Om))

# Cache per i calcoli di r_z
_r_z_cache = {}

def r_z_vectorized(z, H0, Om=Om0GLOB, num_points=100):
    """
    Vectorized comoving distance r(z) for array inputs using Simpson's rule.
    Optimized with caching and reduced integration points.
    """
    c = clight
    
    # Per input scalare, usa cache
    if np.isscalar(z):
        cache_key = (z, H0, Om)
        if cache_key in _r_z_cache:
            return _r_z_cache[cache_key]
        
        x_grid = np.linspace(0, z, num_points)
        y_values = 1.0 / E_z(x_grid, H0, Om)
        integral = simpson(y_values, x_grid)
        result = integral * c / H0
        
        # Limita dimensione cache
        if len(_r_z_cache) < 10000:
            _r_z_cache[cache_key] = result
            
        return result
    else:
        # Per array, calcola per ogni z
        results = []
        for zi in z:
            cache_key = (zi, H0, Om)
            if cache_key in _r_z_cache:
                results.append(_r_z_cache[cache_key])
            else:
                x_grid = np.linspace(0, zi, num_points)
                y_values = 1.0 / E_z(x_grid, H0, Om)
                try:
                    integral = simpson(y_values, x_grid)
                    result = integral * c / H0
                    if len(_r_z_cache) < 10000:
                        _r_z_cache[cache_key] = result
                    results.append(result)
                except Exception as e:
                    warnings.warn(f"Integration error: {e}")
                    results.append(0.0)
        return np.array(results)


def Dl_z_vectorized(z, H0, Om=Om0GLOB):
    """
    Vectorized luminosity distance D_L(z) for array inputs.
    """
    return r_z_vectorized(z, H0, Om) * (1 + z)

# Funzione approssimata per stimare z da dL (per pre-filtraggio)
def z_from_dL_approx(dL_val, H0):
    """
    Quick approximation of z from luminosity distance.
    Using simple relation z ≈ H0 * dL / c for small z.
    """
    # Approssimazione semplice: z ≈ H0 * dL / c
    z_approx = H0 * dL_val / clight
    
    # Applica correzione basata su cosmologia
    if z_approx < 0.1:
        return z_approx
    elif z_approx < 0.5:
        return z_approx * 0.9  # Correzione per z medi
    else:
        return z_approx * 0.8  # Correzione per z alti

# --------------------- Redshift and Hubble Functions ---------------------
def z_from_dcom(dc_val):
    """
    Returns redshift for a given comoving distance dc (in Mpc).
    """
    func = lambda z: cosmoflag.comoving_distance(z).value - dc_val
    z = fsolve(func, 0.02)
    return z[0]

# Cache per z_from_dL
_z_from_dL_cache = {}

def z_from_dL(dL_val):
    """
    Returns redshift for a given luminosity distance dL (in Mpc).
    """
    # Usa cache se possibile
    cache_key = dL_val
    if cache_key in _z_from_dL_cache:
        return _z_from_dL_cache[cache_key]
    
    # Calcola redshift
    func = lambda z: cosmoflag.luminosity_distance(z).value - dL_val
    z = fsolve(func, 0.02)[0]
    
    # Salva in cache
    if len(_z_from_dL_cache) < 10000:
        _z_from_dL_cache[cache_key] = z
    
    return z

def h_of_z_dl(z, dl):
    """
    Solves for H0 given a redshift z and luminosity distance dl.
    """
    # Per redshift piccoli
    if z < 0.1:
        return clight * z / (dl / (1 + z))
    
    # Per redshift maggiori
    func = lambda h: Dl_z_vectorized(z, h, Om0GLOB) - dl
    heq = fsolve(func, 30)[0]
    return heq

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

def ImprovedInputEvets(folder_path, start, end):
    """
    Lists .fits files in the specified folder and selects a range.
    """
    try:
        all_files = os.listdir(folder_path)
    except OSError as e:
        print(f"Error accessing folder {folder_path}: {e}")
        return []
    
    fits_files = [f for f in all_files if f.endswith('.fits')]
    
    if not fits_files:
        return []
        
    if start >= len(fits_files):
        print(f"Start index {start} exceeds the number of files {len(fits_files)}")
        return []
    
    end = min(end, len(fits_files))
    
    return fits_files[start:end]

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



# --------------------- Script Parameters ---------------------------------
working_dir = os.getcwd()
MapPath = os.path.join(working_dir, 'Events/Uniform/TestRun02/')
start = 1100
stop = 1199
pix_threshold = 100
H0min, H0max = 40, 100
H0Grid = np.linspace(H0min, H0max, 1000)
which_beta = 'Beta2v0'  # 'Beta_fast'
debug = 0
th_start = 0
th_stop = 100
how_many_sigma = 5
fname = ImprovedInputEvets(MapPath, th_start, th_stop)
runpath = 'TestRun02-speedup-0_100'
to_read = 'Uniform_paper_sampled_density_of_version_one_testrun02.txt'
# Altre impostazioni globali



#Uniform_paper_sampled_frac_005-host
#Uniform_paper_sampled_density_of_version_one
#Uniform_paper_sampled_density_of_version_one_testrun01.txt
#Uniform_paper_sampled_density_of_version_one_testrun01_few.txt
#Uniform_paper_sampled_density_of_version_one_testrun02.txt

