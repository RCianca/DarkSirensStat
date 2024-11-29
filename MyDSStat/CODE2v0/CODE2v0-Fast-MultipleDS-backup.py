import numpy as np
import pandas as pd
import healpy as hp

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from scipy import integrate
from scipy import interpolate
from scipy.optimize import fsolve
#from scipy.special import erfc
from profilehooks import profile 

from astropy.cosmology import FlatLambdaCDM

from os import mkdir
from os import listdir
from os.path import isfile, join

from multiprocessing import Pool
import multiprocessing
import time
from numba import njit
from tqdm import tqdm
import sys
#---------------------script import------------------------------------------
from SkyMap import GWskymap
from GalaxyCat import GalCat
from Global import *
#-----------------------Costants-----------------------------------------
href=67 #69
Om0GLOB=0.319
Xi0Glob =1.
clight = 2.99792458* 10**5#km/s
cosmoflag = FlatLambdaCDM(H0=href, Om0=Om0GLOB)
#------------------------------------------------------------------------
#------------------Functions---------------------------------------------

@njit
def likelihood_line(mu_DS, dl, sigma):
    norm = 1 / (np.sqrt(2 * np.pi) * sigma)
    body = np.exp(-((dl - mu_DS) ** 2) / (2 * sigma ** 2))
    return norm * body


def LikeofH0_pixel(mu_DS, sigma, z_hosts, Htemp):
    dl_array = Dl_z_vectorized(z_hosts, Htemp, Om0GLOB)  # Vectorized computation
    likelihoods = likelihood_line(mu_DS, dl_array, sigma)  # Vectorized likelihood
    return np.sum(likelihoods)


# def LikeofH0_pixel(mu_DS, sigma, z_hosts, Htemp):
#     to_sum = np.zeros(len(z_hosts))
#     for v in range(len(z_hosts)):
#         dl = Dl_z(z_hosts[v], Htemp, Om0GLOB)
#         to_sum[v] = likelihood_line(mu_DS, dl, sigma)
#     return np.sum(to_sum)

# Parallelized function to compute the pixel likelihood
def compute_pixel_likelihood(args):
    pix, mu_pix, sigma_pix, z_hosts, H0Grid, angular_prob = args
    pixel_post = np.zeros(len(H0Grid))
    # Print debug info to compare with sequential version
    #print(f'Processing pixel: {pix}')
    
    # Loop over H0Grid and compute the likelihood for each value of H0
    for j, h in enumerate(H0Grid):
        pixel_post[j] = LikeofH0_pixel(mu_pix, sigma_pix, z_hosts, h) * angular_prob
    
    return pixel_post


#########################################################################################

if __name__=='__main__':
     working_dir = os.getcwd()
    path = 'Results'
    runpath = 'FirstBatch'

    # Ensure directory exists
    folder = os.path.join(path, runpath)
    os.makedirs(folder, exist_ok=True)
    print(f'\nData will be saved in {folder}')

    # H0 Grid
    H0min, H0max = 40, 100
    H0Grid = np.linspace(H0min, H0max, 1000)
    DF_results = pd.DataFrame(columns=['Event', 'Likelihood'])
    total_post = np.ones(len(H0Grid))  # Total posterior

    # Read Galaxy Catalogue
    print('Reading Galaxy Catalogue')
    to_read = 'Uniform_paper.txt'
    nside = 128
    hostcat = GalCat(to_read, nside).read_catalogue()
    print('Reading catalogue completed')

    # Load GW Data
    print('Loading GW data')
    fname = ['GWtest07.fits', 'GWtest08.fits']
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
        mypixels = GalCat(to_read, nside).pixelizer()
        hostcat['Pixel'] = mypixels
        hostcat_filtered = hostcat[hostcat['Pixel'].isin(pix_selected)]

        ###Cross-Correlation#############################################    
    
        single_post=np.zeros(len(H0Grid))

        pixel_args = [
            (pix, allmu[pix], allsigma[pix], hostcat_filtered[hostcat_filtered['Pixel'] == pix]['z'].values, H0Grid, skyprob[pix])
            for pix in pix_selected
            if len(hostcat_filtered[hostcat_filtered['Pixel'] == pix]) > 0
        ]
        
        # pixel_args = []
        # for pix in pix_selected:
        #     pixel_galaxies = hostcat_filtered[hostcat_filtered['Pixel'] == pix]
        #     z_hosts = np.asarray(pixel_galaxies['z'])
        #     if len(z_hosts) > 0:
        #         # Pass only the pixel-specific values (allmu[pix], allsigma[pix], skyprob[pix])
        #         pixel_args.append((pix, allmu[pix], allsigma[pix], z_hosts, H0Grid, skyprob[pix]))

        # Use Pool to parallelize computation
    #     cpu=multiprocessing.cpu_count()
    #     print('using {} cpu'.format(cpu))
    #     with Pool(cpu) as pool:
    #         results = list(tqdm(pool.imap(compute_pixel_likelihood, pixel_args), total=len(pixel_args)))

    #     print('shape pix_selected {}  shape H0Grid {}'.format(np.shape(pix_selected),np.shape(H0Grid)))
    #     print('result shape {}'.format(np.shape(results)))

    #     for i, pixel_post in enumerate(results):
    #         #print(f"Parallel pixel_post for pixel {i}: {pixel_post}")
    #         single_post += pixel_post
    #     total_post += single_post
    #     # Append the event name and likelihood to DF_results
    #     DF_results = pd.concat(
    #         [DF_results, pd.DataFrame({'Event': [DSs.event_name], 'Likelihood': [single_post.tolist()]})],
    #         ignore_index=True
    #     )

    # DF_results.to_csv(folder + '/GW01_10.csv', index=False)

            # Parallel computation
        cpu = min(multiprocessing.cpu_count(), len(pixel_args))
        print(f'Using {cpu} CPUs')
        with Pool(cpu) as pool:
            results = list(tqdm(pool.imap(compute_pixel_likelihood, pixel_args, chunksize=50), total=len(pixel_args)))

        single_post = np.sum(results, axis=0)
        total_post += single_post
        DF_results = pd.concat(
            [DF_results, pd.DataFrame({'Event': [DSs.event_name], 'Likelihood': [single_post.tolist()]})],
            ignore_index=True
        )

    # Save results
    DF_results.to_csv(os.path.join(folder, 'GW01_10.csv'), index=False)

        ####################Plot###########################################################
    # Plot results
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

    plotpath = os.path.join(folder, 'MultyTest.pdf')
    plt.savefig(plotpath, format="pdf", bbox_inches="tight")
    plt.close()




    