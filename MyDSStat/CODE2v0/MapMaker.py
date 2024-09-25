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
import pickle
from numba import jit


#################################################################################

def compute_area(nside,all_pixels,p_posterior,level=0.99):

    
    ''' Area of level% credible region, in square degrees.
        If level is not specified, uses current selection '''
    pixarea=hp.nside2pixarea(nside)
    return get_credible_region_pixels(all_pixels,p_posterior,level=level).size*pixarea*(180/np.pi)**2


def _get_credible_region_pth(p_posterior,level=0.99):
    '''
    Finds value minskypdf of rho_i that bouds the x% credible region , with x=level
    Then to select pixels in that region: self.all_pixels[self.p_posterior>minskypdf]
    '''
    prob_sorted = np.sort(p_posterior)[::-1]
    prob_sorted_cum = np.cumsum(prob_sorted)
    # find index of array which bounds the self.area confidence interval
    idx = np.searchsorted(prob_sorted_cum, level)
    minskypdf = prob_sorted[idx] #*skymap.npix

    #self.p[self.p]  >= minskypdf       
    return minskypdf

def get_credible_region_pixels(all_pixels, p_posterior, level=0.99):

    return all_pixels[p_posterior>_get_credible_region_pth(p_posterior,level=level)]

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
    perm = [dL_pos, theta_pos,phi_pos,tcoal_pos,psi_pos,iota_pos,eta_pos,phicoal_pos] + remaining_indices
    #mean_permuted = np.array(mean)[perm]
    mean_permuted = np.array(mean)[perm]
    cov_permuted = cov[np.ix_(perm, perm)]
    keys_permuted=list_perm(keys,perm)
    return mean_permuted,cov_permuted,keys_permuted


def process_pixel(args):
    pix = args
    pix=int(pix)
    if not isinstance(pix, int):
        raise TypeError(f"Expected integer for pixel, but got {type(pix)}")
        
        pix = int(pix)  # Explicitly cast to Python int
    
    # Get the fixed angles for this pixel
    theta_fixed, phi_fixed = hp.pix2ang(nside, pix)
    
    # Extract samples of all parameters for this pixel
    pixel_indices = np.where(pixels == pix)[0]
    samples_in_pixel = samples[pixel_indices]

    mu = np.mean(samples_in_pixel[:,0])
    std = np.std(samples_in_pixel[:,0])
    
    return pix, mu, std

####################################################################################################################################
if __name__=='__main__':

    folder='Uniform/TestRun00/'
    CAT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'
    SCRIPT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/'
    COV_SAVE_PATH='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Events/'+folder

    #-----------------------ORDERING OF THE VARIABLES--------------------------------------
    Cov_file='Cov_SNR_more_than_100_200.npy'
    Population='SNR_more_than_100_200.h5'
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
    print('Performing permutation...\n Permuted keys are:')
    print(keys)
    #---------------------------------------------------------------------------------------

    # construct the mean vector now only for one DS
    selected = 52
    mean = np.array(Allevents_DS.iloc[selected])
    allcov = np.load(COV_SAVE_PATH+Cov_file, allow_pickle=True)
    cov = np.float64(allcov[:, :, selected])

    #--------------------------CORE---------------------------------------------------------

    args=mean,cov,parameters_list
    perm_mean,perm_cov,perm_keys=permutation(args)

    print('Performed new permutation. Now order is\n{}'.format(perm_keys))

    try:
        np.linalg.cholesky(cov)
        print('Cov Matrix is Cholesky approved')
    except:
        print('Cov nont positive semi-defined')


    variances = np.diag(perm_cov)
    sigmas=np.sqrt(variances)
    #print(perm_keys)
    #print([f'{val:.4f}' for val in perm_mean])
    #print([f'{sigma:.4f}' for sigma in sigmas])
    print(' Theta Variance={:0.3f},sigma Theta={:0.3f}'.format(variances[1],sigmas[1]))
    print(' Phi Variance={:0.3f},sigma Phi={:0.3f}'.format(variances[2],sigmas[2]))

    #-------------sampling using Cholesky

    num_samples = 10**8
    #samples = np.random.multivariate_normal(perm_mean, perm_cov, num_samples)
    #print(np.shape(samples))

    # Cholesky
    L = np.linalg.cholesky(perm_cov)
    z = np.random.randn(num_samples, len(perm_mean))  # num_samples x num_dimensions
    samples = perm_mean + z @ L.T  
    theta = samples[:, 1]
    phi = samples[:, 2]
    print('Input mean values')
    print('theta={}, phi={}'.format(perm_mean[1],perm_mean[2]))
    print('Sampled mean values')
    print('theta={}, phi={}'.format(np.mean(theta),np.mean(phi)))
    print('Sampled std values')
    print('theta={}, phi={}'.format(np.std(theta),np.std(phi)))

    nside = 128
    npix=hp.nside2npix(nside)
    all_pixels=np.arange(npix)

    sky_map = np.zeros(hp.nside2npix(nside))
    pixels = hp.ang2pix(nside, theta, phi,nest=False)
    #pixels = hp.ang2pix(nside, theta_hp, phi_hp)
    # Increment the pixel values
    np.add.at(sky_map, pixels, 1)
    sky_map = sky_map / np.sum(sky_map)

    print('nside={}'.format(nside))
    print('Number of unique pixels {}'.format(len(np.unique(pixels))))
    gw_area=compute_area(nside,all_pixels,sky_map,level=0.9)
    allsky=hp.nside2npix(nside)*hp.nside2pixarea(nside,degrees=True)
    print('Area GW 90%={} deg^2'.format(gw_area))
    print('Percentage of sky={}%'.format(100*gw_area/allsky))

    #---Select only pixels in which sky_map is greater than 0 (unique_pixels) this can be refined by selection X% of the 2D probability

    unique_pixels=sky_map[sky_map>0]
    #pix99=get_credible_region_pixels(all_pixels,sky_map)
    #pix90=get_credible_region_pixels(all_pixels,sky_map,level=0.9)

    # Initialize arrays to store the mean and std of luminosity distance
    all_mu = np.zeros(hp.nside2npix(nside))
    all_std = np.zeros(hp.nside2npix(nside))

    # Initialize dictionary to store new luminosity distance arrays for each pixel
    luminosity_distance_samples = {}

    #--------------dL in each pixel---------------------------------------------------------------------

    

    # Specify the number of processors to use
    num_processors = 24
    print('computing the conditional probability for dL')
    #args = [(pix) for pix in unique_pixels]
    #with Pool(processes=num_processors) as pool:
    #    results = list(pool.imap(process_pixel, args))

    #from concurrent.futures import ProcessPoolExecutor

    #with ProcessPoolExecutor(max_workers=num_processors) as executor:
    #    results = list(executor.map(process_pixel, unique_pixels))

    results = [process_pixel(pix) for pix in unique_pixels]

    # Collect the results
    for pix, mu, std in results:
        if mu is not None and std is not None:
            all_mu[pix] = mu
            all_std[pix] = std

    mod_postnorm=np.ones(npix)
    print('saving skymap...')
    fname='GWtest00.fits'
    dat=Table([sky_map,all_mu,all_std,mod_postnorm],
          names=('PROB','DISTMU','DISTSIGMA','DISTNORM'))
    os.chdir(COV_SAVE_PATH)
    fits.write_sky_map(fname,dat, nest=False)
    print('map {} saved in {}'.format(fname,COV_SAVE_PATH))
