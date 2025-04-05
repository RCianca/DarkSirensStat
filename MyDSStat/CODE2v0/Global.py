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

@njit
def E_z(z, H0, Om=Om0GLOB):
    """
    Helper function for Hubble parameter as a function of redshift.
    """
    return np.sqrt(Om * (1 + z)**3 + (1 - Om))

# Cache per i calcoli di r_z
_r_z_cache = {}

def r_z_vectorized(z, H0, Om=Om0GLOB, num_points=500):
    """
    Vectorized comoving distance r(z) for array inputs using Simpson's rule.
    Optimized with caching for repeated calculations.
    
    Args:
        z: Redshift value or array
        H0: Hubble constant
        Om: Matter density parameter (default: Om0GLOB)
        num_points: Number of integration points
        
    Returns:
        Comoving distance in Mpc
    """
    c = clight

    def integrand(x):
        return 1 / E_z(x, H0, Om)

    # Per valori scalari, usa cache
    if np.isscalar(z):
        cache_key = (z, H0, Om)
        if cache_key in _r_z_cache:
            return _r_z_cache[cache_key]
        
        x_grid = np.linspace(0, z, num_points)
        y_values = integrand(x_grid)
        
        if len(x_grid) == 0 or len(y_values) == 0:
            raise ValueError("Empty integration grid or invalid values")
            
        integral = simpson(y=y_values, x=x_grid)
        
        if integral is None or np.isnan(integral):
            raise ValueError("Integration failed, returned None or NaN")
            
        result = integral * c / H0
        
        # Memorizza il risultato nella cache se non è troppo grande
        if len(_r_z_cache) < 10000:
            _r_z_cache[cache_key] = result
        
        # Gestisci la dimensione della cache per evitare problemi di memoria
        elif len(_r_z_cache) >= 12000:  # Con un po' di margine rispetto al limite
            # Rimuovi casualmente alcune voci dalla cache
            keys_to_remove = list(_r_z_cache.keys())[:2000]  # Rimuovi 2000 voci
            for key in keys_to_remove:
                _r_z_cache.pop(key)
                
        return result
    else:
        # Per array, processa ogni valore singolarmente per usare la cache
        results = []
        for zi in z:
            cache_key = (zi, H0, Om)
            if cache_key in _r_z_cache:
                results.append(_r_z_cache[cache_key])
            else:
                x_grid = np.linspace(0, zi, num_points)
                y_values = integrand(x_grid)
                
                if len(x_grid) == 0 or len(y_values) == 0:
                    raise ValueError(f"Empty integration grid or invalid values for z={zi}")
                    
                integral = simpson(y=y_values, x=x_grid)
                
                if integral is None or np.isnan(integral):
                    raise ValueError(f"Integration failed, returned None or NaN for z={zi}")
                    
                result = integral * c / H0
                
                # Memorizza il risultato nella cache
                if len(_r_z_cache) < 10000:
                    _r_z_cache[cache_key] = result
                    
                results.append(result)
        
        return np.array(results)

def Dl_z_vectorized(z, H0, Om=Om0GLOB):
    """
    Vectorized luminosity distance D_L(z) for array inputs.
    Uses caching for improved performance on repeated calculations.
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

def ImprovedInputEvets(folder_path, start, end):
    """
    Lists .fits files in the specified folder and selects a range based on start and end indices. returns up to end -1

    Parameters:
    folder_path (str): The path to the folder containing the .fits files.
    start (int): The start index for file selection.
    end (int): The end index for file selection.

    Returns:
    list: A list of selected .fits files.
    """
    # List all files in the folder
    all_files = os.listdir(folder_path)

    # Filter the .fits files
    fits_files = [f for f in all_files if f.endswith('.fits')]

    # Select the range of files
    selected_files = fits_files[start:end]

    return selected_files

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
MapPath = os.path.join(working_dir, 'Events/Uniform/TestRun02/')
start=1100
stop=1199
pix_threshold=100
H0min, H0max = 40, 100
H0Grid = np.linspace(H0min, H0max, 1000)
which_beta='Beta2v0'#'Beta_fast'#'Beta2v0'
debug=0
# List of GW data files to process
#fname = InputEvents(start,stop)
th_start=20
th_stop=40
how_many_sigma=5
fname = ImprovedInputEvets(MapPath, th_start, th_stop)
#fname=['GWtest225903.fits']

# Name of the runpath folder for saving results
runpath = 'TestRun02-new-20_40'
#Host Catalogue to read
to_read = 'Uniform_paper_sampled_density_of_version_one_testrun02.txt'
#Uniform_paper_sampled_frac_005-host
#Uniform_paper_sampled_density_of_version_one
#Uniform_paper_sampled_density_of_version_one_testrun01.txt
#Uniform_paper_sampled_density_of_version_one_testrun01_few.txt
#Uniform_paper_sampled_density_of_version_one_testrun02.txt

