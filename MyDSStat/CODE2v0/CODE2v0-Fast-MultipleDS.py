from Global import *
import numpy as np
import pandas as pd
from Global import *
import numpy as np
import pandas as pd
import os
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt
from numba import njit

from SkyMap import GWskymap
from GalaxyCat import GalCat

# Ottimizzazione della funzione di likelihood con Numba
@njit
def likelihood_line(mu_DS, dl, sigma):
    """Calcola la likelihood gaussiana per una distanza luminosa dato mu e sigma"""
    norm = 1 / (np.sqrt(2 * np.pi) * sigma)
    exponent = -((dl - mu_DS) ** 2) / (2 * sigma ** 2)
    # Usa exp con clipping per evitare underflow numerici
    # Quando exponent < -700, exp ritorna 0 nel float64
    if exponent < -700:
        return 0.0
    return norm * np.exp(exponent)

def LikeofH0_pixel(mu_DS, sigma, z_hosts, Htemp):
    """
    Compute the likelihood of H0 for a single pixel with pre-filtering.
    """
    if len(z_hosts) == 0:
        return 0.0
        
    if np.isnan(z_hosts).any():
        return 0.0
    
    # Calcola i limiti di distanza
    dl_max = mu_DS + how_many_sigma * sigma
    dl_min = mu_DS - how_many_sigma * sigma
    
    # Stima il range di redshift approssimato
    z_max_est = z_from_dL_approx(dl_min, Htemp)
    z_min_est = z_from_dL_approx(dl_max, Htemp)
    
    # Pre-filtra i redshift
    z_filtered = z_hosts[(z_hosts >= z_min_est * 0.9) & (z_hosts <= z_max_est * 1.1)]
    
    if len(z_filtered) == 0:
        return 0.0  # No hosts in range
    
    # Calcola dl_array solo per i redshift filtrati
    dl_array = Dl_z_vectorized(z_filtered, Htemp, Om0GLOB)
    
    # Filter distances
    mask = (dl_array <= dl_max) & (dl_array >= dl_min)
    dl_array = dl_array[mask]
    
    if len(dl_array) == 0:
        return 0.0

    # Calcola la likelihood per ogni distanza e somma
    likelihoods = likelihood_line(mu_DS, dl_array, sigma)  # Vettorizzato
    return np.sum(likelihoods)


# Parallelized function to compute the pixel likelihood
def compute_pixel_likelihood(args):
    pix, mu_pix, sigma_pix, z_hosts, H0Grid, angular_prob = args
    pixel_post = np.zeros(len(H0Grid))
    
    if len(z_hosts) == 0:
        return pixel_post
        
    if np.isnan(z_hosts).any():
        return pixel_post
    
    # Calcola likelihood per ogni H0
    for j, h in enumerate(H0Grid):
        pixel_post[j] = LikeofH0_pixel(mu_pix, sigma_pix, z_hosts, h) * angular_prob
    
    return pixel_post


if __name__=='__main__':
    print(f"Flagship params: H0 = {href}, Omega_M = {Om0GLOB}")
    print(f"Files to process: {fname}")
    print(f"Results will be saved in folder: {runpath}")
    
    working_dir = os.getcwd()
    path = 'Results'

    # Ensure directory exists
    folder = os.path.join(path, runpath)
    os.makedirs(folder, exist_ok=True)
    print(f'\nData will be saved in {folder}')
    
    # Copy script files for reference
    os.system('cp CODE2v0-Fast-MultipleDS.py '+folder+'/Script-copy.py')
    os.system('cp Global.py '+folder+'/Global-copy.py')

    # Initialize total posterior
    total_post = np.ones(len(H0Grid))
    
    # Read Galaxy Catalogue
    print('Reading Galaxy Catalogue--'+to_read)
    nside = 128
    hostcat = GalCat(to_read, nside).read_catalogue()
    mypixels = GalCat(to_read, nside).pixelizer()
    print('Reading catalogue completed')

    # Set log folder
    set_log_folder(folder)

    # Determine max CPUs to use
    max_cpus = multiprocessing.cpu_count()
    print(f'System has {max_cpus} CPUs available')
    
    # Process level
    level = 0.9
    if __name__=='__main__':
    print(f"Flagship params: H0 = {href}, Omega_M = {Om0GLOB}")
    print(f"Files to process: {fname}")
    print(f"Results will be saved in folder: {runpath}")
    
    working_dir = os.getcwd()
    path = 'Results'

    # Ensure directory exists
    folder = os.path.join(path, runpath)
    os.makedirs(folder, exist_ok=True)
    print(f'\nData will be saved in {folder}')
    
    # Copy script files for reference
    os.system('cp CODE2v0-Fast-MultipleDS.py '+folder+'/Script-copy.py')
    os.system('cp Global.py '+folder+'/Global-copy.py')

    # Initialize total posterior
    total_post = np.ones(len(H0Grid))
    
    # Read Galaxy Catalogue
    print('Reading Galaxy Catalogue--'+to_read)
    nside = 128
    hostcat = GalCat(to_read, nside).read_catalogue()
    mypixels = GalCat(to_read, nside).pixelizer()
    print('Reading catalogue completed')

    # Set log folder
    set_log_folder(folder)
    
    # Determine max CPUs to use
    max_cpus = multiprocessing.cpu_count()
    print(f'System has {max_cpus} CPUs available')
    
    # Process level
    level = 0.9
    
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
        print(f'Using {name}')
        print(f'Area of DS: {DSs.area()} deg^2 at 90%')

        # Filter Galaxy Catalogue (solo per i pixel rilevanti per questo evento)
        hostcat['Pixel'] = mypixels
        hostcat_filtered = hostcat[hostcat['Pixel'].isin(pix_selected)]
        
        # Pre-group by pixel to speed up filtering
        grouped_hostcat = hostcat_filtered.groupby('Pixel')['z']

        # Prepare arguments for pixel processing
        pixel_args = [
            (pix, allmu[pix], allsigma[pix], grouped_hostcat.get_group(pix).values, H0Grid, skyprob[pix])
            for pix in pix_selected
            if pix in grouped_hostcat.groups
        ]

        # Process pixels in parallel
        single_post = np.ones(len(H0Grid)) * 1e-10  # Valore di default molto piccolo
        
        if len(pixel_args) > 0:
            cpu = min(max_cpus, len(pixel_args))
            print(f'Using {cpu} CPUs')
            
            # Optimize chunksize for better load balancing
            chunksize = max(1, len(pixel_args) // (2 * cpu))
            
            with Pool(cpu) as pool:
                results = list(pool.imap(compute_pixel_likelihood, pixel_args, chunksize=chunksize))
            
            # Sum results across all pixels
            single_post = np.sum(results, axis=0)
            
            # Ensure likelihood is valid
            if np.all(single_post == 0):
                print("Warning: all likelihood values are zero, using small defaults")
                single_post = np.ones(len(H0Grid)) * 1e-10
        else:
            print("No Hosts found for this DS, using small defaults")
        
        # Save individual likelihood
        likename = 'like_' + name.split('.')[0]
        np.save(os.path.join(folder, likename), single_post)
        
        # Update total posterior
        total_post *= single_post
    
    # Save total posterior
    postname = 'Total_Posterior'
    np.save(os.path.join(folder, postname), total_post)
    
    # Plot results
    print('Plotting total likelihood')
    fig, ax = plt.subplots(1, figsize=(15, 10))
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.get_offset_text().set_fontsize(25)
    ax.grid(linestyle='dotted', linewidth='0.6')

    x = H0Grid
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_xlabel(r'$H_0(Km/s/Mpc)$', fontsize=30)
    ax.set_ylabel(r'$Posterior(H_0)$', fontsize=30)

    if np.min(x) < href < np.max(x):
        ax.axvline(x=href, color='k', linestyle='dashdot', label='H0=67')

    # Normalize posterior for plotting
    normalized_post = total_post / np.trapz(total_post, x)
    ax.plot(x, normalized_post, label='Total_posterior', linewidth=4, linestyle='solid')
    ax.legend(fontsize=13, ncol=2)
    
    plotname = 'Total_Like.pdf'
    plotpath = os.path.join(folder, plotname)
    plt.savefig(plotpath, format="pdf", bbox_inches="tight")
    plt.close()
