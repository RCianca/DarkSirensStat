import os
import sys
import copy
import numpy as np
import pandas as pd 
from astropy.cosmology import FlatLambdaCDM 
import matplotlib.pyplot as plt
import healpy as hp
from astropy.table import Table
from ligo.skymap.io import fits
import h5py
from multiprocessing import Pool
import multiprocessing
import pickle
from numba import jit
import warnings

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd())))
sys.path.append(SCRIPT_DIR)
import gwfast.gwfastGlobals as glob
import gwfast 
from gwfast.waveforms import IMRPhenomD_NRTidalv2
from gwfast.waveforms import IMRPhenomD
from gwfast.waveforms import IMRPhenomHM

from gwfast.signal import GWSignal
from gwfast.network import DetNet
from gwfast import fisherTools
from fisherTools import CovMatr, compute_localization_region, check_covariance, fixParams
from gwfastUtils import GPSt_to_LMST
from gwfast.gwfastUtils import load_population

###########################################################################################################################
def sample_multivariate_gaussian(mean, cov, num_samples):
    return np.random.multivariate_normal(mean, cov, num_samples)

def list_perm(lista, permutazione):
    tmp = []
    for e in permutazione:
        tmp.append(lista[e])
    return tmp

def cat2parameter(args):
    catalogue, keys = args
    
    missing_columns = [key for key in keys if key not in catalogue.columns]
    if missing_columns:
        raise ValueError(f"Some keys are missing in the DataFrame: {missing_columns}")
    
    # Reorder the catalogue according to the order in keys
    catalogue_permuted = catalogue[keys]
    
    return catalogue_permuted

def permutation(args):
    mean, cov, keys = args
    
    dL_pos = keys.index('dL')
    theta_pos = keys.index('theta')
    phi_pos = keys.index('phi')
    iota_pos = keys.index('iota')
    eta_pos = keys.index('eta')
    phicoal_pos = keys.index('Phicoal')
    tcoal_pos = keys.index('tcoal')
    psi_pos = keys.index('psi')
    remaining_indices = list(set(range(len(keys))) - {dL_pos, theta_pos, phi_pos,
                                                     tcoal_pos, psi_pos, iota_pos, eta_pos, phicoal_pos})
    perm = [dL_pos, tcoal_pos, psi_pos, iota_pos, eta_pos, phicoal_pos] + remaining_indices + [theta_pos, phi_pos]
    
    mean_permuted = np.array(mean)[perm]
    cov_permuted = cov[np.ix_(perm, perm)]
    keys_permuted = list_perm(keys, perm)
    
    return mean_permuted, cov_permuted, keys_permuted

# Funzioni per gestire i problemi ai poli mediante rotazione della sfera
def rotate_coordinates(theta, phi, rot_theta=np.pi/4):
    """
    Applica una rotazione rigida alle coordinate sferiche.
    Trasla l'evento lontano dai poli di rot_theta radianti.
    
    Args:
        theta: coordinate theta (colatitude, 0 = polo nord, π = polo sud)
        phi: coordinate phi (longitude)
        rot_theta: angolo di rotazione in radianti (default: 45 gradi)
        
    Returns:
        theta_rotated, phi_rotated: coordinate ruotate
    """
    # Converti coordinate sferiche in coordinate cartesiane
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # Crea matrice di rotazione attorno all'asse y
    # Questa rotazione sposta i punti lontano dai poli
    cos_rot = np.cos(rot_theta)
    sin_rot = np.sin(rot_theta)
    
    # Applica rotazione
    x_rot = x * cos_rot + z * sin_rot
    y_rot = y
    z_rot = -x * sin_rot + z * cos_rot
    
    # Riconverti in coordinate sferiche
    theta_rot = np.arccos(z_rot)
    phi_rot = np.arctan2(y_rot, x_rot)
    
    # Assicurati che phi sia nell'intervallo [0, 2π)
    phi_rot = np.mod(phi_rot, 2*np.pi)
    
    return theta_rot, phi_rot

def inverse_rotate_coordinates(theta, phi, rot_theta=np.pi/4):
    """
    Applica la rotazione inversa per tornare alle coordinate originali.
    
    Args:
        theta: coordinate theta ruotate
        phi: coordinate phi ruotate
        rot_theta: angolo di rotazione originale (da invertire)
        
    Returns:
        theta_original, phi_original: coordinate originali
    """
    # La rotazione inversa è lo stesso tipo di rotazione ma con angolo negativo
    return rotate_coordinates(theta, phi, -rot_theta)

def rotate_skymap(skymap, rot_theta=np.pi/4, inverse=False):
    """
    Ruota una mappa HEALPix completa.
    
    Args:
        skymap: mappa HEALPix
        rot_theta: angolo di rotazione in radianti
        inverse: se True, applica rotazione inversa
        
    Returns:
        rotated_skymap: mappa ruotata
    """
    nside = hp.npix2nside(len(skymap))
    
    # Ottieni le coordinate di tutti i pixel
    ipix = np.arange(len(skymap))
    theta, phi = hp.pix2ang(nside, ipix)
    
    # Applica rotazione (diretta o inversa)
    rot_angle = -rot_theta if inverse else rot_theta
    theta_rot, phi_rot = rotate_coordinates(theta, phi, rot_angle)
    
    # Converti le coordinate ruotate in nuovi indici di pixel
    ipix_rot = hp.ang2pix(nside, theta_rot, phi_rot)
    
    # Crea nuova mappa ruotata
    rotated_skymap = np.zeros_like(skymap)
    rotated_skymap[ipix] = skymap[ipix_rot]
    
    return rotated_skymap

def process_near_pole_event(samples, theta_mean, phi_mean, nside=128, pole_threshold=0.1):
    """
    Processa un evento vicino al polo applicando una rotazione.
    
    Args:
        samples: array di campioni con theta, phi nelle ultime due colonne
        theta_mean, phi_mean: coordinate medie dell'evento
        nside: risoluzione HEALPix
        pole_threshold: soglia per identificare eventi vicino ai poli
        
    Returns:
        sky_map: mappa di probabilità
        pixels: array di pixel corrispondenti ai campioni
    """
    # Determina se l'evento è vicino a un polo
    is_near_pole = (theta_mean < pole_threshold) or (theta_mean > (np.pi - pole_threshold))
    
    if not is_near_pole:
        # Elaborazione standard se non siamo vicino a un polo
        theta = samples[:, -2]
        phi = samples[:, -1]
        theta_hp = np.mod(theta, np.pi)
        phi_hp = np.mod(phi, 2*np.pi)
        pixels = hp.ang2pix(nside, theta_hp, phi_hp)
        
        # Genera la mappa
        sky_map = np.zeros(hp.nside2npix(nside))
        np.add.at(sky_map, pixels, 1)
        sky_map = sky_map / np.sum(sky_map)
        
        return sky_map, pixels, is_near_pole
    
    # Se siamo vicino a un polo, applichiamo rotazione
    print(f"Event near pole detected (theta={theta_mean:.4f}). Applying rotation.")
    
    # Angolo di rotazione: 45 gradi (π/4)
    rot_theta = np.pi/4
    
    # Estrai theta e phi dai campioni
    theta = samples[:, -2]
    phi = samples[:, -1]
    
    # Applica rotazione ai campioni
    theta_rot, phi_rot = rotate_coordinates(theta, phi, rot_theta)
    
    # Converti in pixel HEALPix
    theta_hp = np.mod(theta_rot, np.pi)
    phi_hp = np.mod(phi_rot, 2*np.pi)
    pixels_rot = hp.ang2pix(nside, theta_hp, phi_hp)
    
    # Genera mappa ruotata
    sky_map_rot = np.zeros(hp.nside2npix(nside))
    np.add.at(sky_map_rot, pixels_rot, 1)
    sky_map_rot = sky_map_rot / np.sum(sky_map_rot)
    
    # Ruota la mappa indietro alla posizione originale
    sky_map = rotate_skymap(sky_map_rot, rot_theta, inverse=True)
    
    # Calcola i pixel originali (non ruotati) per il processing successivo
    theta_orig = np.mod(theta, np.pi)
    phi_orig = np.mod(phi, 2*np.pi)
    pixels = hp.ang2pix(nside, theta_orig, phi_orig)
    
    return sky_map, pixels, is_near_pole

def cond_inpix(pix, samples_in_pixel):
    # Create the alpha vector with the fixed values and mean of other parameters
    dL_pos = columns.get_loc('dL')
    theta_pos = columns.get_loc('theta')
    phi_pos = columns.get_loc('phi')

    theta_fixed, phi_fixed = hp.pix2ang(nside, pix)
    alpha = np.zeros(2)
    alpha[0] = theta_fixed
    alpha[1] = phi_fixed
    
    mean_new = perm_mean[-2:]
    
    # Partition the permuted covariance matrix
    Sigma_xx = perm_cov[-2:, -2:]
    Sigma_xy = perm_cov[-2:, 0:-2]
    Sigma_yx = perm_cov[0:-2, -2:]
    Sigma_yy = perm_cov[0:-2, 0:-2]
    
    try:
        mu_cond = perm_mean[0:-2] + Sigma_yx @ np.linalg.inv(Sigma_xx) @ (alpha - DS_angs)
        Sigma_cond = Sigma_yy - Sigma_yx @ np.linalg.inv(Sigma_xx) @ Sigma_xy
        
        mu = mu_cond[0]
        std = np.sqrt(max(Sigma_cond[0, 0], 1e-10))  # Evita valori negativi
    except np.linalg.LinAlgError:
        # Se l'inversione fallisce, usa una stima più semplice
        mu = np.mean(samples_in_pixel[:, 0]) if len(samples_in_pixel) > 0 else perm_mean[0]
        std = np.std(samples_in_pixel[:, 0]) if len(samples_in_pixel) > 0 else np.sqrt(perm_cov[0, 0])
        
    return mu, std

def process_pixel(args):
    pix = args
    pix = int(pix)
    if not isinstance(pix, int):
        raise TypeError(f"Expected integer for pixel, but got {type(pix)}")
        pix = int(pix)  # Explicitly cast to Python int
    pixel_indices = np.where(pixels == pix)[0]
    samples_in_pixel = samples[pixel_indices]

    mu, std = cond_inpix(pix, samples_in_pixel)
    distance_sampled = samples_in_pixel[:, 0]
    
    return pix, mu, std, distance_sampled

def parallel_process_pixels(unique_pixels):
    with Pool(multiprocessing.cpu_count()) as pool:
        # Use map to distribute the unique pixels to each worker
        results = pool.map(process_pixel, unique_pixels)
    return results

# --------------------- HEALPix Utilities ---------------------------------

def compute_area(nside, all_pixels, p_posterior, level=0.9):
    """
    Computes the area of the level% credible region in square degrees.
    """
    pixarea = hp.nside2pixarea(nside)
    return get_credible_region_pixels(all_pixels, p_posterior, level=level).size * pixarea * (180 / np.pi)**2

def _get_credible_region_pth(p_posterior, level=0.9):
    """
    Finds the probability threshold for the x% credible region (default 90%).
    """
    prob_sorted = np.sort(p_posterior)[::-1]
    prob_sorted_cum = np.cumsum(prob_sorted)
    idx = np.searchsorted(prob_sorted_cum, level)
    return prob_sorted[idx]

def get_credible_region_pixels(all_pixels, p_posterior, level=0.9):
    """
    Returns the pixels within the level% credible region.
    """
    return all_pixels[p_posterior > _get_credible_region_pth(p_posterior, level=level)]


############################################################################################################################

# Configure ET and the PSD
ETdet = {'ET': copy.deepcopy(glob.detectors).pop('ETS')}
print(ETdet)
ETdet['ET']['psd_path'] = os.path.join(glob.detPath, 'ET-0000A-18.txt')
mySignalsET = {}
for d in ETdet.keys():
    mySignalsET[d] = GWSignal((IMRPhenomHM()),
                psd_path= ETdet[d]['psd_path'],
                detector_shape = ETdet[d]['shape'],
                det_lat= ETdet[d]['lat'],
                det_long=ETdet[d]['long'],
                det_xax=ETdet[d]['xax'],
                verbose=True,
                useEarthMotion = False,
                fmin=2.,
                IntTablePath=None)

myET = DetNet(mySignalsET)
folder='Uniform/TestRun04/'
CAT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'
SCRIPT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/'
COV_SAVE_PATH='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Events/'+folder


os.chdir(CAT_FOLDER)
DS_Cat= pd.read_csv('DS_From_Parent_Uniform_Complete_SNR.txt')
os.chdir(SCRIPT_FOLDER)

H0GLOB= 67
Om0GLOB=0.319
Xi0Glob =1.
cosmoeuclid = FlatLambdaCDM(H0=H0GLOB, Om0=Om0GLOB)

ParNums = IMRPhenomHM().ParNums
print(ParNums)
totalds=DS_Cat.shape[0]
DS_Cat=DS_Cat[DS_Cat['SNR']>100]
print('Number of DSs with SNR more than 100 {}. {}%'.format(DS_Cat.shape[0],100*DS_Cat.shape[0]/totalds))
print(DS_Cat.head(5))
start_index=49181
iteration_count = 0
max_iterations=700

# Log file per tenere traccia degli eventi con problemi ai poli
pole_log_file = os.path.join(COV_SAVE_PATH, "pole_issues.log")
with open(pole_log_file, "w") as f:
    f.write("Event Index, Is Near Pole, Original Area, Rotated Area, GWfast Area\n")

for event_index, row in DS_Cat.iloc[start_index:].iterrows():
    if iteration_count % 10 == 0:
        print('Computed {} maps'.format(iteration_count))
    if iteration_count >= max_iterations:
        print("Reached maximum iteration limit. Stopping.")
        break
    print(f"Processing event {event_index}")
    
    Allevents_DS = {
        'Mc': np.array([row['MC'] * (1 + row['z'])]),
        'eta': np.array([row['q'] / (1 + row['q']) ** 2]),
        'dL': np.array([row['Luminosity Distance'] / 1000.0]),
        'theta': np.array([row['theta']]),
        'phi': np.array([row['phi']]),
        'iota': np.array([np.arccos(row['cos_iota'])]),
        'psi': np.array([row['psi'] / 2]),
        'tcoal': np.array([row['tcoal']]),
        'Phicoal': np.array([row['Phicoal']]),
        'chi1z': np.array([row['chi1z']]),
        'chi2z': np.array([row['chi2z']])
    }
    
    # Verifica se siamo vicino ai poli
    theta_val = Allevents_DS['theta'][0]
    pole_threshold = 0.1  # ~6 gradi dal polo
    is_near_pole = (theta_val < pole_threshold) or (theta_val > (np.pi - pole_threshold))
    
    if is_near_pole:
        print(f"Note: Event {event_index} is near a pole (theta = {theta_val:.4f})")
    
    my_DS_theta = Allevents_DS['theta']
    
    # Compute Fisher and Covariance matrix
    totFET = myET.FisherMatr(Allevents_DS)
    totCov_ET, inversion_err_ET = CovMatr(totFET)
    
    # Compute localization area
    area_deg2 = compute_localization_region(totCov_ET, ParNums, Allevents_DS['theta'])

    if area_deg2 <= 25 or is_near_pole:  # Anche se l'area stimata GWfast è grande, se è vicino ai poli proviamo comunque
        iteration_count += 1
        np.save(COV_SAVE_PATH + f'Cov_SNR_more_than_100_{event_index}', totCov_ET)
        print(f"Saved covariance matrix for event {event_index}")
        gwfast.gwfastUtils.save_data(COV_SAVE_PATH+f'SNR_more_than_100_{event_index}.h5', Allevents_DS)

        #######Start the map making. I have to save and load beacuse I don't know if indices are mixed and for now it works if I load 
        #Reading Files
        Cov_file=f'Cov_SNR_more_than_100_{event_index}.npy'
        Population=f'SNR_more_than_100_{event_index}.h5'
        tosave=load_population(COV_SAVE_PATH+Population)
        allcov = np.load(COV_SAVE_PATH+Cov_file, allow_pickle=True)
        ###################Permutations###################################
        Allevents_DS_fromfile = pd.DataFrame.from_dict(tosave, orient='columns')
        keys=list(Allevents_DS_fromfile.columns)
        parameters=IMRPhenomHM().ParNums
        parameters_list=list(IMRPhenomHM().ParNums.keys())
        args=Allevents_DS_fromfile,parameters_list
        Allevents_DS=cat2parameter(args)
        keys = list(Allevents_DS.columns)
        print(f"Generating map {event_index}")

        ##############Generation----fix iteration logic is no more a loop
        columns=Allevents_DS.columns
        # Construct mean vector and covariance matrix for the selected event
        mean = np.array(Allevents_DS.iloc[0])
        cov = np.float64(allcov[:, :, 0])
        
        condition_number = np.linalg.cond(cov)
        if condition_number > 10**12:
            epsilon = 1e-10 * np.trace(cov)
            cov += np.eye(cov.shape[0]) * epsilon
            print('condition number was too high, used eigenvalues regularisation')
        
        try:
            np.linalg.cholesky(cov)
            print('Cov Matrix is Cholesky approved')
        except:
            print('Cov not positive semi-defined')
            print('Increasing Epsilon')
            cov += np.eye(cov.shape[0]) / epsilon
            epsilon = 1e-8 * np.trace(cov)
            cov += np.eye(cov.shape[0]) * epsilon
            np.linalg.cholesky(cov)           

        # Permutation and Cholesky decomposition
        args = mean, cov, parameters_list
        perm_mean, perm_cov, perm_keys = permutation(args)
        
        L = np.linalg.cholesky(perm_cov)
        z = np.random.randn(10**8, len(perm_mean))
        samples = perm_mean + z @ L.T
        
        # Ottieni theta e phi medi
        theta_mean = perm_mean[-2]
        phi_mean = perm_mean[-1]
        
        # Genera la mappa di skyprob applicando rotazione se necessario
        nside = 128
        sky_map, pixels, was_rotated = process_near_pole_event(samples, theta_mean, phi_mean, 
                                                    nside=nside, pole_threshold=pole_threshold)

        # Compute the area of the 90% credible region
        all_pixels = np.arange(hp.nside2npix(nside))
        gw_area = compute_area(nside, all_pixels, sky_map, level=0.9)
        
        # Registra informazioni sugli eventi con problemi ai poli
        with open(pole_log_file, "a") as f:
            f.write(f"{event_index}, {is_near_pole}, {gw_area:.2f}, {was_rotated}, {area_deg2:.2f}\n")

        print('Number of unique pixels {}'.format(len(np.unique(pixels))))
        allsky = hp.nside2npix(nside) * hp.nside2pixarea(nside, degrees=True)
        print('Area GW 90%={} deg^2'.format(gw_area))
        print('Percentage of sky={}%'.format(100*gw_area/allsky))
        print('GWfast Area ={}'.format(area_deg2))
        os.chdir(COV_SAVE_PATH)

        if (iteration_count % 100 == 0) or is_near_pole:
            plt.figure(figsize=(12, 8))
            hp.mollview(sky_map, title=f'GWtest{event_index}-skyprob', nest=False, hold=True)
            plt.savefig(f'GWtest{event_index}.pdf')
            plt.close() 
    
        mean_pix = hp.ang2pix(nside, theta_mean, phi_mean)
        theta_DS, phi_DS = hp.pix2ang(nside, mean_pix)
        DS_angs = np.zeros(2)
        DS_angs[0] = theta_DS
        DS_angs[1] = phi_DS   
        all_mu = np.zeros(hp.nside2npix(nside))
        all_std = np.zeros(hp.nside2npix(nside))
        unique_pixels = np.unique(pixels)
        luminosity_distance_samples = {}
        
        # Definiamo le variabili globali necessarie per il processing dei pixel
        global samples, pixels, perm_mean, perm_cov, DS_angs, nside
        # Qui non serve riassegnare queste variabili, ma dichiarare che sono globali
        results = parallel_process_pixels(unique_pixels)

        for pix, mu, std, distance_sampled in results:
            all_mu[pix] = mu
            all_std[pix] = std
            luminosity_distance_samples[pix] = distance_sampled

        mod_postnorm = np.ones(hp.nside2npix(nside))

        # Save the map with an incremental name
        fname = f'GWtest{event_index}.fits'
        dat = Table([sky_map, all_mu, all_std, mod_postnorm],
                    names=('PROB', 'DISTMU', 'DISTSIGMA', 'DISTNORM'))
        os.chdir(COV_SAVE_PATH)
        fits.write_sky_map(fname, dat, nest=False)
        print(f'Map {fname} saved')

    else:
        print(f"Skipping event {event_index}, area too large: {area_deg2:.2f} deg^2\n")