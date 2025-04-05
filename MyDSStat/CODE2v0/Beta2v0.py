import numpy as np
from Global import *
import os
import sys

import multiprocessing
from functools import partial
from numba import njit, prange
#from multiprocessing import Pool

from GalaxyCat import GalCat
from SkyMap import GWskymap


#for now is a function to use in CODE2v0-Fast-MultipleDS, so all args will be passed by the script. Used as main, you need to give the arguments
def beta_caller(name,args):
    if name == 'Beta_fast':
        beta= Beta_fast(args)
    else:
        beta= Beta2v0_pix(args)
    return beta

def Beta_fast(args):
    mu_DS, sigma, z_hosts, Htemp = args
    
    # Pre-filter redshifts to calculate dl_array only for the ones in the range
    # This is more efficient than calculating all and then filtering
    dl_max = mu_DS + how_many_sigma * sigma
    dl_min = mu_DS - how_many_sigma * sigma
    
    # Estimate dl range for each z to pre-filter
    # This è un'approssimazione rapida per filtrare molti redshift fuori range
    z_max_est = z_from_dL_approx(dl_min, Htemp)
    z_min_est = z_from_dL_approx(dl_max, Htemp)
    
    # Pre-filter redshifts
    z_filtered = z_hosts[(z_hosts >= z_min_est * 0.9) & (z_hosts <= z_max_est * 1.1)]
    
    if len(z_filtered) == 0:
        return 0  # No hosts in range
        
    # Calcola dl_array solo per i redshift filtrati
    dl_array = Dl_z_vectorized(z_filtered, Htemp, Om0GLOB)
    
    # Filtra ulteriormente in base alla distanza luminosa
    dl_array = dl_array[(dl_array <= dl_max) & (dl_array >= dl_min)]
    
    beta = len(dl_array)
    return beta


# Parallelized function to compute beta in the pixel
def Beta2v0_pix(args):
    pix, mu_pix, sigma_pix, z_hosts, H0Grid = args
    
    if len(z_hosts) == 0:
        if debug:
            print('No hosts for this line of sight')
        return np.ones(len(H0Grid))
    
    if np.isnan(z_hosts).any():
        if debug:
            print('NONE in z_hosts')
        return np.ones(len(H0Grid))
    
    # Parallelizza anche il calcolo su H0Grid
    # Usiamo un pool diverso per ogni pixel per evitare nesting di multiprocessing
    # che causa problemi
    pixel_beta = np.ones(len(H0Grid))
    
    # Versione con loop ottimizzato (senza nesting di multiprocessing)
    # Questo evita overhead di creazione processi per pochi calcoli
    for j, h in enumerate(H0Grid):
        pixel_beta[j] = beta_inpix(mu_pix, sigma_pix, z_hosts, h)
    
    return pixel_beta

def beta_inpix(mu_DS, sigma, z_hosts, Htemp):
    """
    Compute the beta of H0 for a single pixel. Count how many possible hosts in the pixel
    between dl_max and dl_min.
    """
    # Calcola i limiti di distanza
    dl_max = mu_DS + how_many_sigma * sigma
    dl_min = mu_DS - how_many_sigma * sigma
    
    # Stima il range di redshift approssimato
    z_max_est = z_from_dL_approx(dl_min, Htemp)
    z_min_est = z_from_dL_approx(dl_max, Htemp)
    
    # Pre-filtra i redshift (approssimazione rapida)
    z_filtered = z_hosts[(z_hosts >= z_min_est * 0.9) & (z_hosts <= z_max_est * 1.1)]
    
    if len(z_filtered) == 0:
        return 0  # No hosts in range
        
    # Calculate dl_array only for the filtered redshifts
    dl_array = Dl_z_vectorized(z_filtered, Htemp, Om0GLOB)
    
    # Only print debug info if debug is enabled
    if debug:
        print("----------------------------------------------------------------")
        print('Debug beta_inpix of Beta2v0')
        print(f'dl_array before selection\n {dl_array}')
        print(f'mu_DS {mu_DS} Mpc Sigma {sigma} Mpc mu+-{how_many_sigma}*sigma {mu_DS+how_many_sigma*sigma} {mu_DS-how_many_sigma*sigma}\n Htemp= {Htemp}')
        sys.stdout.flush()
    
    # Filter by luminosity distance
    dl_array = dl_array[(dl_array <= dl_max) & (dl_array >= dl_min)]
    
    if debug:
        print(f'dl_array after dl_max, dl_min selection\n {dl_array}')
        print("----------------------------------------------------------------")
        sys.stdout.flush()
    
    if dl_array is None or np.isnan(dl_array).any():
        if debug:
            print(f"Warning: NaN detected in dl_array for Htemp={Htemp}")
            sys.stdout.flush()
        return 1

    if len(dl_array) == 0:
        return 0  # No hosts in range after filtering

    beta = len(dl_array)  # here we will add the weights
    return beta

    # Helper function for fast redshift estimation
def z_from_dL_approx(dL_val, H0):
    """
    Quick approximation of z from luminosity distance.
    Using simple relation z ≈ H0 * dL / c for small z.
    For higher accuracy, we scale by a factor based on cosmology.
    """
    # Simple approximation: z ≈ H0 * dL / c
    z_approx = H0 * dL_val / clight
    
    # Apply a correction factor based on typical cosmology
    # This improves accuracy for higher redshifts
    if z_approx < 0.1:
        return z_approx
    elif z_approx < 0.5:
        return z_approx * 0.9  # Correction for medium z
    else:
        return z_approx * 0.8  # Correction for higher z

if __name__=='__main__':
    debug = 0  # Set to 1 for verbose output
    print('Computing Beta for each event.')

    print(f"Files to process: {fname}")
    working_dir = os.getcwd()
    path = 'Results'

    # Ensure directory exists
    folder = os.path.join(path, runpath, 'Beta')
    os.makedirs(folder, exist_ok=True)
    print(f'\nData will be saved in {folder}')
    os.system('cp Beta2v0.py '+folder+'/beta-copy.py')

    # Read Galaxy Catalogue
    print('Reading Galaxy Catalogue--'+to_read)
    nside = 128
    hostcat = GalCat(to_read, nside).read_catalogue()
    mypixels = GalCat(to_read, nside).pixelizer()
    print('Reading catalogue completed')

    # Load GW Data
    print('Loading GW data')
    level = 0.9
    
    # Determine max CPUs to use
    max_cpus = multiprocessing.cpu_count()
    print(f'System has {max_cpus} CPUs available')
    
    # Process each event
    for name in fname:
        print(f'Processing {name}')
        DSs = GWskymap(os.path.join(MapPath, name), level=level)
        
        pix_selected = DSs.get_credible_region_pixels(level=level)
        nside = int(DSs.nside)
        skyprob = DSs.p_posterior
        allmu, allsigma = DSs.mu * 1000, DSs.sigma * 1000  # Convert to Mpc

        if np.isnan(allmu).any():
            print(f'There are NaN values in allmu of {name}')
            continue
            
        if np.isnan(allsigma).any():
            print(f'There are NaN values in allsigma of {name}')
            continue

        if(len(pix_selected) > pix_threshold):
            print(f'Skipping {name}, too many pixels ({len(pix_selected)})')
            continue
        
        print('DS data:')
        print(f'Event {name}')
        print(f'Area of DS: {DSs.area()} deg^2 at 90%')

        # Filter Galaxy Catalogue
        hostcat['Pixel'] = mypixels
        hostcat_filtered = hostcat[hostcat['Pixel'].isin(pix_selected)]
        
        # Pre-group by pixel to speed up filtering
        grouped_hostcat = hostcat_filtered.groupby('Pixel')['z']
    
        single_beta = np.ones(len(H0Grid))
        print(f'using beta {which_beta}')
        
        if which_beta == 'Beta2v0':
            # Prepare arguments for parallel processing
            pixel_args = [
                (pix, allmu[pix], allsigma[pix], grouped_hostcat.get_group(pix).values, H0Grid)
                for pix in pix_selected
                if pix in grouped_hostcat.groups
            ]

            if len(pixel_args) > 0:
                # Determine optimal CPU count and chunksize
                cpu = min(max_cpus, len(pixel_args))
                print(f'Using {cpu} CPUs')
                
                # Optimize chunksize - larger chunks reduce overhead but may cause imbalance
                chunksize = max(1, len(pixel_args) // (2 * cpu))
                
                # Use a shared memory array if available (multiprocessing.shared_memory in Python 3.8+)
                with multiprocessing.Pool(cpu) as pool:
                    results = list(pool.imap(Beta2v0_pix, pixel_args, chunksize=chunksize))
                
                # Sum results across all pixels
                single_beta = np.sum(results, axis=0)
                
                if np.sum(single_beta) == 0:
                    print("Warning: sum of beta is zero, using default of ones")
                    single_beta = np.ones(len(H0Grid))
            else:
                print("No Hosts found for this DS, skipping computation.")
                
            # Save results
            betaname = 'beta_' + name.split('.')[0]
            np.save(os.path.join(folder, betaname), single_beta)
            
        elif which_beta == 'Beta_fast':
            # Compute average mu and sigma weighted by skyprob
            mumean = np.sum(allmu[pix_selected] * skyprob[pix_selected]) / np.sum(skyprob[pix_selected])
            sigmamean = np.sum(allsigma[pix_selected] * skyprob[pix_selected]) / np.sum(skyprob[pix_selected])
            allz_for_beta = np.asarray(hostcat_filtered['z'])
            
            # For Beta_fast, parallelize over H0Grid
            cpu = min(max_cpus, len(H0Grid))
            print(f'Using {cpu} CPUs')
            chunksize = max(1, len(H0Grid) // (2 * cpu))
            
            args_list = [(mumean, sigmamean, allz_for_beta, h) for h in H0Grid]
            with multiprocessing.Pool(cpu) as pool:
                single_beta = pool.map(Beta_fast, args_list, chunksize=chunksize)
                
            np.save(os.path.join(folder, betaname), single_beta)
            
    print('All beta values saved')