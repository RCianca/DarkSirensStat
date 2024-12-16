from Global import (
    href, Om0GLOB, clight, Dl_z_vectorized,Dl_z, fname, runpath
)
from SkyMap import GWskymap
from GalaxyCat import GalCat
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt
from numba import njit

#------------------Functions---------------------------------------------
#---------------------- Likelihood----------------------------------------
@njit
def likelihood_line(mu_DS, dl, sigma):
    norm = 1 / (np.sqrt(2 * np.pi) * sigma)
    body = np.exp(-((dl - mu_DS) ** 2) / (2 * sigma ** 2))
    return norm * body


def LikeofH0_pixel(mu_DS, sigma, z_hosts, Htemp):
    """
    Compute the likelihood of H0 for a single pixel.
    """
    if len(z_hosts) == 0:
        raise ValueError("z_hosts is empty")
    if np.isnan(z_hosts).any():
        raise ValueError("z_hosts contains NaN values")

    dl_array = Dl_z_vectorized(z_hosts, Htemp, Om0GLOB)  # Vectorized computation
    #begin mod speed up
    dl_array=dl_array[dl_array<=mu_DS+3.5*sigma]
    dl_array=dl_array[dl_array>=mu_DS-3.5*sigma]
    #end mod speed up
    if dl_array is None or np.isnan(dl_array).any():
        raise ValueError("Dl_z_vectorized returned None or NaN")

    likelihoods = likelihood_line(mu_DS, dl_array, sigma)  # Vectorized likelihood
    return np.sum(likelihoods)




# Parallelized function to compute the pixel likelihood
def compute_pixel_likelihood(args):
    pix, mu_pix, sigma_pix, z_hosts, H0Grid, angular_prob = args
    pixel_post = np.zeros(len(H0Grid))
    if len(z_hosts) == 0:
        #print('No hosts for this line of sight')
        return pixel_post
    if np.isnan(z_hosts).any():
        #print('NONE in z_hosts')
        return pixel_post
    #print('debug:found hosts')
    for j, h in enumerate(H0Grid):
        pixel_post[j] = LikeofH0_pixel(mu_pix, sigma_pix, z_hosts, h) * angular_prob
    
    return pixel_post




#########################################################################################

if __name__=='__main__':
    print(f"Flagship params: H0 = {href}, Omega_M = {Om0GLOB}")
    print(f"Files to process: {fname}")
    print(f"Results will be saved in folder: {runpath}")
    working_dir = os.getcwd()
    path = 'Results'
    #runpath = 'FirstBatch'

    # Ensure directory exists
    folder = os.path.join(path, runpath)
    os.makedirs(folder, exist_ok=True)
    print(f'\nData will be saved in {folder}')

    # H0 Grid
    H0min, H0max = 40, 100
    H0Grid = np.linspace(H0min, H0max, 1000)
    #DF_results = pd.DataFrame(columns=['Event', 'Likelihood'])
    total_post = np.ones(len(H0Grid))  # Total posterior

    # Read Galaxy Catalogue
    to_read = 'Uniform_paper_sampled_almostone.txt'
    print('Reading Galaxy Catalogue--'+to_read)
    nside = 128
    hostcat = GalCat(to_read, nside).read_catalogue()
    mypixels = GalCat(to_read, nside).pixelizer()
    print('Reading catalogue completed')

    # Load GW Data
    print('Loading GW data')
    #fname = ['GWtest07.fits', 'GWtest08.fits']
    MapPath = os.path.join(working_dir, 'Events/Uniform/TestRun00/')
    level = 0.9


    for name in fname:

        DSs = GWskymap(os.path.join(MapPath, name), level=level)
        print(f'DS name: {DSs.event_name}')
        print(f'Area of DS: {DSs.area()} deg^2 at 90%')
        
        pix_selected = DSs.get_credible_region_pixels(level=level)
        nside = int(DSs.nside)
        skyprob = DSs.p_posterior
        allmu, allsigma = DSs.mu * 1000, DSs.sigma * 1000  # Convert to Mpc

        if np.isnan(allmu).any():
            print('There are NaN values in allmu')
        if np.isnan(allsigma).any():
            print('There are NaN values in allsigma')

        mumean = np.sum(allmu * skyprob) / np.sum(skyprob)
        sigmamean = np.sum(allsigma * skyprob) / np.sum(skyprob)
        print(f'mu_pesato: {mumean} Mpc, sigma_pesato: {sigmamean} Mpc')


        print('DS data:')
        print('pix selected ={}'.format(len(pix_selected)))
        print('len dL={}'.format(len(allmu[pix_selected])))

        # Filter Galaxy Catalogue
        
        hostcat['Pixel'] = mypixels
        hostcat_filtered = hostcat[hostcat['Pixel'].isin(pix_selected)]

        ###Cross-Correlation#############################################    
    
        single_post=np.zeros(len(H0Grid))

        pixel_args = [
            (pix, allmu[pix], allsigma[pix], hostcat_filtered[hostcat_filtered['Pixel'] == pix]['z'].values, H0Grid, skyprob[pix])
            for pix in pix_selected
            if len(hostcat_filtered[hostcat_filtered['Pixel'] == pix]) > 0
        ]

            # Parallel computation
        #cpu = min(multiprocessing.cpu_count(), len(pixel_args))
        #print(f'Using {cpu} CPUs')
        #with Pool(cpu) as pool:
        #    results = list(pool.imap(compute_pixel_likelihood, pixel_args))

        if len(pixel_args) > 0:
            #cpu = min(multiprocessing.cpu_count(), len(pixel_args))
            cpu = multiprocessing.cpu_count()
            print(f'Using {cpu} CPUs')
            with Pool(cpu) as pool:
                results = list(pool.imap(compute_pixel_likelihood, pixel_args))
            single_post = np.sum(results, axis=0)
        else:
            print("No Hosts found for this DS, skipping computation.")
            results = []

        
        likename='like_'+name.split('.')[0]
        np.save(os.path.join(folder,likename),single_post)
        total_post *= single_post
        total_post+=0.000000001
        #DF_results = pd.concat(
        #    [DF_results, pd.DataFrame({'Event': [DSs.event_name], 'Likelihood': [single_post.tolist()]})],
        #    ignore_index=True
        #)

    # Save results
    #DF_results.to_csv(os.path.join(folder, 'GW01_10.csv'), index=False)

        ####################Plot###########################################################
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

    normalized_post = total_post / np.trapz(total_post, x)
    ax.plot(x, normalized_post, label='Total_posterior', linewidth=4, linestyle='solid')
    ax.legend(fontsize=13, ncol=2)
    plotname='Total_Like.pdf'
    plotpath=os.path.join(folder,plotname)
    plt.savefig(plotpath, format="pdf", bbox_inches="tight")
    plt.close()




    
