import os
import sys

import copy
import numpy as np
import pandas as pd 
from astropy.cosmology import FlatLambdaCDM 
import matplotlib.pyplot as plt

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


import healpy as hp



from astropy.table import Table

from ligo.skymap.io import fits
import os
import sys

from gwfast.gwfastUtils import load_population


import h5py
from multiprocessing import Pool
import multiprocessing
import pickle
from numba import jit, njit


###########################################################################################################################
def ensure_scalar(value):
    """Convert NumPy arrays to scalar values safely."""
    if isinstance(value, np.ndarray):
        return float(value.item()) if value.size == 1 else float(value[0])
    return float(value)

def list_perm(lista,permutazione):
    tmp=[]
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
    mean,cov,keys=args
    
    dL_pos = keys.index('dL')
    theta_pos = keys.index('theta')
    phi_pos = keys.index('phi')
    iota_pos=keys.index('iota')
    eta_pos=keys.index('eta')
    phicoal_pos=keys.index('Phicoal')
    tcoal_pos=keys.index('tcoal')
    psi_pos=keys.index('psi')
    remaining_indices = list(set(range(len(keys))) - {dL_pos, theta_pos,phi_pos
                                                    ,tcoal_pos,psi_pos,iota_pos,eta_pos,phicoal_pos})
    perm = [dL_pos,tcoal_pos,psi_pos,iota_pos,eta_pos,phicoal_pos] + remaining_indices +[theta_pos,phi_pos]
    #mean_permuted = np.array(mean)[perm]
    mean_permuted = np.array(mean)[perm]
    cov_permuted = cov[np.ix_(perm, perm)]
    keys_permuted=list_perm(keys,perm)
    return mean_permuted,cov_permuted,keys_permuted


def cond_inpix(pix,samples_in_pixel):
# Create the alpha vector with the fixed values and mean of other parameters
    #columns = Allevents_DS.columns # global variable
    dL_pos = columns.get_loc('dL')
    theta_pos = columns.get_loc('theta')
    phi_pos = columns.get_loc('phi')

    # Create the permutation order with 'dL' first, 'theta' second, and 'phi' third
    #remaining_indices = list(set(range(len(columns))) - {dL_pos, theta_pos, phi_pos})

    theta_fixed, phi_fixed = hp.pix2ang(nside,pix)
    alpha = np.zeros(2)
    alpha[0] = theta_fixed
    alpha[1] = phi_fixed
    #alpha[2:] = samples_in_pixel[:, remaining_indices].mean(axis=0)  # Use the mean of the other parameters in this pixel
    mean_new = perm_mean[-2:]
    #theta_mean=mean_new[0]
    #phi_mean=mean_new[1]
    #mean_pix=hp.ang2pix(nside,theta_mean,phi_mean)
    #theta_DS, phi_DS = hp.pix2ang(nside,mean_pix)
    #DS_angs = np.zeros(2)
    #DS_angs[0] = theta_fixed
    #DS_angs[1] = phi_fixed    
    # Partition the permuted covariance matrix
    Sigma_xx = perm_cov[-2:, -2:]
    Sigma_xy = perm_cov[-2:, 0:-2]
    Sigma_yx = perm_cov[0:-2, -2:]
    Sigma_yy = perm_cov[0:-2, 0:-2]
    mu_cond = perm_mean[0:-2] + Sigma_yx @ np.linalg.inv(Sigma_xx) @ (alpha - DS_angs)#DS_angs#mean_new
    Sigma_cond = Sigma_yy - Sigma_yx @ np.linalg.inv(Sigma_xx) @ Sigma_xy
    
    mu = mu_cond[0]#mu_cond[0]#np.mean(new_samples)
    std = np.sqrt(Sigma_cond[0,0])
    return mu,std#, new_samples

def process_pixel(args):
    pix = args
    pix=int(pix)
    if not isinstance(pix, int):
        raise TypeError(f"Expected integer for pixel, but got {type(pix)}")
        pix = int(pix)  # Explicitly cast to Python int
    pixel_indices = np.where(pixels == pix)[0]
    samples_in_pixel = samples[pixel_indices]

    mu,std = cond_inpix(pix,samples_in_pixel)
    distance_sampled = samples_in_pixel[:,0]
    
    return pix, mu, std ,distance_sampled

def parallel_process_pixels(unique_pixels):
    with Pool(multiprocessing.cpu_count()) as pool:
        # Use map to distribute the unique pixels to each worker
        results = pool.map(process_pixel, unique_pixels)
    return results

# Aggiungi queste funzioni ottimizzate
@njit
def generate_samples_batch(perm_mean, L, batch_size):
    """
    Genera un batch di campioni da una distribuzione normale multivariata.
    Questa funzione è ottimizzata con njit.
    
    Args:
        perm_mean: vettore media
        L: fattore di Cholesky della matrice di covarianza
        batch_size: dimensione del batch
        
    Returns:
        batch_samples: array di campioni
    """
    z_batch = np.random.randn(batch_size, len(perm_mean))
    return perm_mean + z_batch @ L.T

@njit
def compute_healpix_angles(theta, phi):
    """
    Calcola gli angoli HEALPix corretti (modulo).
    Questa funzione è ottimizzata con njit.
    
    Args:
        theta: angoli theta
        phi: angoli phi
        
    Returns:
        theta_hp, phi_hp: angoli HEALPix
    """
    theta_hp = np.mod(theta, np.pi)
    phi_hp = np.mod(phi, 2 * np.pi)
    return theta_hp, phi_hp
#-----------------------------------------------------------------#

# Poi modifica la funzione adaptive_sampling come segue:

#-----------------------------------------------------------------#
def adaptive_sampling(perm_mean, perm_cov, nside=128):
    """
    Implementa il campionamento incrementale per generare mappe del cielo
    con un numero ottimizzato di campioni.
    
    Args:
        perm_mean: vettore media dopo permutazione
        perm_cov: matrice covarianza dopo permutazione
        nside: risoluzione HEALPix
        
    Returns:
        samples: tutti i campioni generati
        pixels: array di pixel corrispondenti ai campioni
        sky_map: mappa di probabilità
    """
    # Calcola la decomposizione di Cholesky
    L = np.linalg.cholesky(perm_cov)
    
    # Parametri per il campionamento incrementale
    max_samples = 10**8  # Limite massimo di campioni
    samples_per_batch = 10**6  # Dimensione del batch
    samples_list = []
    pixels_list = []
    unique_pixels_set = set()
    
    # Calcolo del numero atteso di pixel
    total_sky_pixels = hp.nside2npix(nside)
    total_sky_area = 4 * np.pi * (180/np.pi)**2  # circa 41.253 deg²
    pixels_per_deg2 = total_sky_pixels / total_sky_area
    expected_pixels = int(25 * pixels_per_deg2)  # Numero atteso di pixel in 25 deg²
    
    # Target e parametri
    pixel_coverage_target = min(expected_pixels * 3, total_sky_pixels)
    min_sample_batches = 5
    
    print(f"Area prevista: circa 25 deg² (~{expected_pixels} pixel)")
    print(f"Target di copertura: {pixel_coverage_target} pixel")
    print(f"Avviando campionamento incrementale...")
    
    # Variabili per la convergenza
    prev_unique_count = 0
    stable_iterations = 0
    required_stable_iterations = 3
    convergence_tolerance = 0.02  # 2% di cambiamento
    
    for batch_idx in range(0, max_samples // samples_per_batch):
        # Usa la funzione ottimizzata per generare campioni
        batch_samples = generate_samples_batch(perm_mean, L, samples_per_batch)
        
        # Estrai theta e phi dai campioni
        theta_batch = batch_samples[:, -2]
        phi_batch = batch_samples[:, -1]
        
        # Calcola gli angoli HEALPix con la funzione ottimizzata
        theta_hp, phi_hp = compute_healpix_angles(theta_batch, phi_batch)
        
        # Calcola i pixel
        batch_pixels = hp.ang2pix(nside, theta_hp, phi_hp)
        
        # Aggiungi alle liste
        samples_list.append(batch_samples)
        pixels_list.append(batch_pixels)
        
        # Aggiorna i pixel unici
        prev_unique_count = len(unique_pixels_set)
        unique_pixels_set.update(batch_pixels)
        current_unique_count = len(unique_pixels_set)
        
        # Calcola la percentuale di cambiamento
        if prev_unique_count > 0:
            percent_change = (current_unique_count - prev_unique_count) / prev_unique_count
        else:
            percent_change = 1.0
        
        # Stampa progresso
        total_samples = (batch_idx + 1) * samples_per_batch
        print(f"Batch {batch_idx+1}: {current_unique_count} pixel unici (Δ: {percent_change:.2%}), {total_samples:,} campioni")
        
        # Verifica convergenza
        if percent_change < convergence_tolerance:
            stable_iterations += 1
        else:
            stable_iterations = 0
        
        # Criteri di terminazione
        min_samples_reached = batch_idx >= min_sample_batches
        convergence_reached = stable_iterations >= required_stable_iterations
        
        if min_samples_reached and convergence_reached:
            print(f"Convergenza raggiunta dopo {total_samples:,} campioni con {current_unique_count} pixel unici")
            break
            
        if (batch_idx + 1) * samples_per_batch >= max_samples:
            print(f"Raggiunto limite massimo di {max_samples:,} campioni con {current_unique_count} pixel unici")
            break
    
    # Combina tutti i batch
    samples = np.vstack(samples_list)
    pixels = np.concatenate(pixels_list)
    
    print(f"Campionamento completato: generati {samples.shape[0]:,} campioni con {len(unique_pixels_set)} pixel unici")
    
    # Genera la mappa
    sky_map = np.zeros(hp.nside2npix(nside))
    np.add.at(sky_map, pixels, 1)
    sky_map = sky_map / np.sum(sky_map)
    
    return samples, pixels, sky_map
#-----------------------------------------------------------------#

def is_near_pole(theta, pole_threshold=0.1):
    """
    Verifica se un evento è troppo vicino ai poli celesti.
    
    Args:
        theta: angolo di declinazione in radianti (0 = polo nord, pi = polo sud)
        pole_threshold: soglia in radianti per considerare l'evento vicino a un polo
        
    Returns:
        bool: True se l'evento è vicino a un polo, False altrimenti
    """
    # Verifico vicinanza al polo nord (theta vicino a 0)
    # o al polo sud (theta vicino a pi)
    return theta < pole_threshold or (np.pi - theta) < pole_threshold

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


############################################################################################################################

# Configure ET and the PSD
ETdet = {'ET': copy.deepcopy(glob.detectors).pop('ETS') }
print(ETdet)
ETdet['ET']['psd_path'] = os.path.join(glob.detPath, 'ET-0000A-18.txt')
mySignalsET = {}
for d in ETdet.keys():
    #print(d)
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
folder='Uniform/TestRun06/'
CAT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'
SCRIPT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/'
COV_SAVE_PATH='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Events/'+folder


os.chdir(CAT_FOLDER)
DS_Cat= pd.read_csv('DS_From_Parent_Uniform_Complete_SNR.txt')
os.chdir(SCRIPT_FOLDER)

H0GLOB= 67#67.9 #69
Om0GLOB=0.319
Xi0Glob =1.
cosmoeuclid = FlatLambdaCDM(H0=H0GLOB, Om0=Om0GLOB)

ParNums = IMRPhenomHM().ParNums
print(ParNums)
totalds=DS_Cat.shape[0]
DS_Cat=DS_Cat[DS_Cat['SNR']>100]
print('Number of DSs with SNR more than 100 {}. {}%'.format(DS_Cat.shape[0],100*DS_Cat.shape[0]/totalds))
print(DS_Cat.head(5))
start_index=6168
iteration_count = 0
max_iterations=600
steps=0
for event_index, row in DS_Cat.iloc[start_index:].iterrows():
    steps += 1
    if iteration_count%10==0:
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
    my_DS_theta=Allevents_DS['theta']
    # Compute Fisher and Covariance matrix
    totFET = myET.FisherMatr(Allevents_DS)
    totCov_ET, inversion_err_ET = CovMatr(totFET)
    
    # Compute localization area
    area_deg2=compute_localization_region(totCov_ET,ParNums,Allevents_DS['theta'])

    if area_deg2 <= 25:
        # Controllo evento vicino ai poli maybe to cut out -------------------------------------------------------------------------
        pole_threshold_rad = 0.3  # circa 8.6 gradi dal polo range :#0.1 --- 0.3
        Val=is_near_pole(Allevents_DS['theta'][0], pole_threshold=pole_threshold_rad)
        print(f"(Is near pole is giving {Val})")
        if is_near_pole(Allevents_DS['theta'][0], pole_threshold=pole_threshold_rad):
            print(f"Skipping event {event_index}, too close to celestial pole (theta = {Allevents_DS['theta'][0]:.4f} rad)")
            continue
            #----------------------------------------------------------------------------------------------------------------------
        iteration_count += 1
        np.save(COV_SAVE_PATH + f'Cov_SNR_more_than_100_{event_index}', totCov_ET)
        print(f"Saved covariance matrix for event {event_index}")
        gwfast.gwfastUtils.save_data(COV_SAVE_PATH+f'SNR_more_than_100_{event_index}.h5', Allevents_DS)
        print('GWfast Area ={}'.format(area_deg2))

        #######Start the map making. I have to save and load beacuse I don't know if indices are mixed and for now it works if I load 
        #Reading Files
        Cov_file=f'Cov_SNR_more_than_100_{event_index}.npy'
        Population=f'SNR_more_than_100_{event_index}.h5'
        tosave=load_population(COV_SAVE_PATH+Population)
        allcov = np.load(COV_SAVE_PATH+Cov_file, allow_pickle=True)
        ###################Permutations###################################
        Allevents_DS_fromfile = pd.DataFrame.from_dict(tosave, orient='columns')
        #print('Catalogue has shape {}'.format(Allevents_DS_fromfile.shape))
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
        #print("Eigenvalues before permutation:", np.linalg.eigvalsh(cov))
        condition_number = np.linalg.cond(cov)
        #print("Condition number:", condition_number)
        if condition_number>10**12:
            epsilon = 1e-10 * np.trace(cov)
            cov += np.eye(cov.shape[0]) * epsilon
            print('condition number was too hight, used eigenvalues regularisation')
        #condition_number = np.linalg.cond(cov)
        #print("Condition number:", condition_number)       
        
        max_attempts = 5
        attempt = 0
        cholesky_success = False

        while attempt < max_attempts and not cholesky_success:
            try:
                np.linalg.cholesky(cov)
                print(f'Cov Matrix è positiva definita dopo {attempt+1} tentativi')
                cholesky_success = True
            except np.linalg.LinAlgError:
                attempt += 1
                if attempt == max_attempts:
                    print(f'Fallito dopo {max_attempts} tentativi, skippo evento {event_index}')
                    break
                    
                # Incrementa epsilon in modo esponenziale ad ogni tentativo
                epsilon = 1e-8 * np.trace(cov) * (10**attempt)
                print(f'Tentativo {attempt+1}/{max_attempts}: aumento epsilon a {epsilon:.2e}')
                
                # Applica la regolarizzazione
                cov_orig = cov.copy()  # Salva la matrice originale
                cov = cov_orig + np.eye(cov_orig.shape[0]) * epsilon

        # Verifica se la decomposizione è riuscita, altrimenti salta questo evento
        if not cholesky_success:
            print(f'Impossibile rendere la matrice positiva definita, skippo evento {event_index}')
            continue


        # Permutation and Cholesky decomposition
#-----------------------------------------------------------------#
        args = mean, cov, parameters_list
        perm_mean, perm_cov, perm_keys = permutation(args)

        # Imposta la risoluzione HEALPix
        nside = 128
        
        # Utilizza il campionamento adattivo per generare la mappa
        samples, pixels, sky_map = adaptive_sampling(perm_mean, perm_cov, nside)
#-----------------------------------------------------------------#
        # Compute the area of the 90% credible region
        all_pixels = np.arange(hp.nside2npix(nside))
        gw_area = compute_area(nside, all_pixels, sky_map, level=0.9)


        print('Number of unique pixels {}'.format(len(np.unique(pixels))))
        allsky=hp.nside2npix(nside)*hp.nside2pixarea(nside,degrees=True)
        print('Area GW 90%={} deg^2'.format(gw_area))
        print('Percentage of sky={}%'.format(100*gw_area/allsky))
        print('GWfast Area ={}'.format(area_deg2))
        os.chdir(COV_SAVE_PATH)
        ##### INsert here a chck on true area an save only if less than 25. iteration_count-=1
        if ensure_scalar(gw_area) > 25:
            iteration_count -= 1
            print('GW area too large after Monte Carlo')
            
            # Elimina i file salvati in precedenza
            cov_file_path = COV_SAVE_PATH + f'Cov_SNR_more_than_100_{event_index}'
            h5_file_path = COV_SAVE_PATH + f'SNR_more_than_100_{event_index}.h5'
            
            try:
                # Rimuovi il file della matrice di covarianza
                if os.path.exists(cov_file_path):
                    os.remove(cov_file_path)
                    print(f"File rimosso: {cov_file_path}")
                
                # Rimuovi anche il file h5 salvato
                if os.path.exists(h5_file_path):
                    os.remove(h5_file_path)
                    print(f"File rimosso: {h5_file_path}")
            except Exception as e:
                print(f"Errore durante la rimozione dei file: {e}")
            
            continue
        else:
            if (iteration_count % 100==0):
                plt.figure(figsize=(12, 8))
                hp.mollview(sky_map, title=f'GWtest{event_index}-skyprob', nest=False, hold=True)
                plt.savefig(f'GWtest{event_index}.pdf')
                plt.close() 
        
            theta_mean=perm_mean[-2]
            phi_mean=perm_mean[-1]
            mean_pix=hp.ang2pix(nside,theta_mean,phi_mean)
            theta_DS, phi_DS = hp.pix2ang(nside,mean_pix) #to fix DS in the pix. If galaxies are in the same piz, ang dist must be 0, you are in the same pix
            DS_angs = np.zeros(2)
            DS_angs[0] = theta_DS
            DS_angs[1] = phi_DS   
            all_mu = np.zeros(hp.nside2npix(nside))
            all_std = np.zeros(hp.nside2npix(nside))
            unique_pixels = np.unique(pixels)
            luminosity_distance_samples = {}
            results = parallel_process_pixels(unique_pixels)

            for pix, mu, std,distance_sampled in results:
                all_mu[pix] = mu
                all_std[pix] = std
                luminosity_distance_samples[pix] = distance_sampled #to check dist in each pix. Not used after test good

            mod_postnorm = np.ones(hp.nside2npix(nside))

            # Save the map with an incremental name
            fname = f'GWtest{event_index}.fits'
            dat = Table([sky_map, all_mu, all_std, mod_postnorm],
                        names=('PROB', 'DISTMU', 'DISTSIGMA', 'DISTNORM'))
            os.chdir(COV_SAVE_PATH)
            fits.write_sky_map(fname, dat, nest=False)
            print(f'Map {fname} saved')
            print(f" Elaborated {steps} lines, add to start_index\n")

    else:
        print(f"Skipping event {event_index}, area too large")
        print(f" Elaborated {steps} lines, add to start_index\n")

