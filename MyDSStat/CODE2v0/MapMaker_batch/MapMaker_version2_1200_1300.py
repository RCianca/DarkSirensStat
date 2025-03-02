import pandas as pd
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

from ligo.skymap.io import fits
import os
import sys


import gwfast.gwfastGlobals as glob
import gwfast
from gwfast.waveforms import IMRPhenomD,IMRPhenomHM
from gwfast.gwfastUtils import load_population

from tqdm import tqdm

import h5py
from multiprocessing import Pool
import multiprocessing
import pickle
from numba import jit

from Global import *

#################################################################################



def sample_multivariate_gaussian(mean, cov, num_samples):
    return np.random.multivariate_normal(mean, cov, num_samples)

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

####################################################################################################################################
if __name__=='__main__':

    folder='Uniform/TestRun01/'
    CAT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'
    SCRIPT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/'
    COV_SAVE_PATH='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Events/'+folder

    print('using {} CPU' .format(multiprocessing.cpu_count()))

    #-----------------------ORDERING OF THE VARIABLES--------------------------------------
    Cov_file='Cov_SNR_more_than_100_1200_1300.npy'
    Population='SNR_more_than_100_1200_1300.h5'
    tosave=load_population(COV_SAVE_PATH+Population)

    Allevents_DS_fromfile = pd.DataFrame.from_dict(tosave, orient='columns')
    print('Catalogue has shape {}'.format(Allevents_DS_fromfile.shape))
    print(Allevents_DS_fromfile.head(3))
    keys=list(Allevents_DS_fromfile.columns)
    print(keys)
    print('\nThis is the order of the paramers in the h5 file. The order is different from Cov file')
    print('Ordering of Cov variables is the same as IMRPhenomHM().ParNums')
    parameters=IMRPhenomHM().ParNums
    parameters_list=list(IMRPhenomHM().ParNums.keys())
    print('{}'.format(parameters_list))
    args=Allevents_DS_fromfile,parameters_list
    Allevents_DS=cat2parameter(args)
    keys = list(Allevents_DS.columns)
    #print('Performing permutation...\n Permuted keys are:')
    print(keys)
    #---------------------------------------------------------------------------------------
    allcov = np.load(COV_SAVE_PATH+Cov_file, allow_pickle=True)
    shift=1200
    for i in range(100):
        k=i+shift
        print(f"Generating map {k:02d}")

        # Select a different row for each map (you can modify this selection logic if needed)
        selected = i
        columns=Allevents_DS.columns
        # Construct mean vector and covariance matrix for the selected event
        mean = np.array(Allevents_DS.iloc[selected])
        cov = np.float64(allcov[:, :, selected])
        #print("Eigenvalues before permutation:", np.linalg.eigvalsh(cov))
        condition_number = np.linalg.cond(cov)
        #print("Condition number:", condition_number)
        if condition_number>10**12:
            epsilon = 1e-10 * np.trace(cov)
            cov += np.eye(cov.shape[0]) * epsilon
            print('condition number was too hight, used eigenvalues regularisation')
        #condition_number = np.linalg.cond(cov)
        #print("Condition number:", condition_number)       
        
        try:
            np.linalg.cholesky(cov)
            print('Cov Matrix is Cholesky approved')
        except:
            print('Cov not positive semi-defined')

        # Permutation and Cholesky decomposition
        args = mean, cov, parameters_list
        perm_mean, perm_cov, perm_keys = permutation(args)
        #print("Eigenvalues after permutation:", np.linalg.eigvalsh(perm_cov))
        #print("Condition number after permutation:", np.linalg.cond(perm_cov))
        #diag_cov=perm_cov.diagonal()#remove after test
        #perm_cov=np.diag(diag_cov)#remove after test
        L = np.linalg.cholesky(perm_cov)
        z = np.random.randn(10**8, len(perm_mean))
        samples = perm_mean + z @ L.T
        theta = samples[:, -2]
        phi = samples[:, -1]
        direct_dl=samples[:,0]
        theta_hp = np.mod(theta, np.pi) 
        phi_hp = np.mod(phi, 2 * np.pi) 

        # Healpix map generation
        nside = 128
        sky_map = np.zeros(hp.nside2npix(nside))
        #pixels = hp.ang2pix(nside, theta, phi,nest=True)
        pixels = hp.ang2pix(nside, theta_hp, phi_hp)
        np.add.at(sky_map, pixels, 1)
        sky_map = sky_map / np.sum(sky_map)

        # Compute the area of the 90% credible region
        all_pixels = np.arange(hp.nside2npix(nside))
        gw_area = compute_area(nside, all_pixels, sky_map, level=0.9)


        print('Number of unique pixels {}'.format(len(np.unique(pixels))))
        allsky=hp.nside2npix(nside)*hp.nside2pixarea(nside,degrees=True)
        print('Area GW 90%={} deg^2'.format(gw_area))
        print('Percentage of sky={}%'.format(100*gw_area/allsky))
        os.chdir(COV_SAVE_PATH)

        plt.figure(figsize=(12, 8))
        hp.mollview(sky_map, title=f'GWtest{k:02d}-skyprob', nest=False, hold=True)
        plt.savefig(f'GWtest{k:02d}.pdf')
        plt.close()      
        #theta_mean=perm_mean[-2]
        #phi_mean=perm_mean[-1]
        #mean_pix=hp.ang2pix(nside,theta_mean,phi_mean)
        #theta_DS, phi_DS = hp.pix2ang(nside,mean_pix)
        #DS_angs = np.zeros(2)
        #DS_angs[0] = theta_DS
        #DS_angs[1] = phi_DS   
        all_mu = np.zeros(hp.nside2npix(nside))
        all_std = np.zeros(hp.nside2npix(nside))
        unique_pixels = np.unique(pixels)
        luminosity_distance_samples = {}
        results = parallel_process_pixels(unique_pixels)
        for pix, mu, std,distance_sampled in results:
            all_mu[pix] = mu
            all_std[pix] = std
            luminosity_distance_samples[pix] = distance_sampled

        mod_postnorm = np.ones(hp.nside2npix(nside))

        # Save the map with an incremental name
        fname = f'GWtest{k:02d}.fits'
        dat = Table([sky_map, all_mu, all_std, mod_postnorm],
                    names=('PROB', 'DISTMU', 'DISTSIGMA', 'DISTNORM'))
        os.chdir(COV_SAVE_PATH)
        fits.write_sky_map(fname, dat, nest=False)
        print(f'Map {fname} saved')


