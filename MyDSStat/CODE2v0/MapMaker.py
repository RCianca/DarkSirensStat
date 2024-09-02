import pandas as pd
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

from ligo.skymap.io import fits
#from ligo.skymap.postprocess import find_greedy_credible_levels

import gwfast.gwfastGlobals as glob
import gwfast 

import os
import sys
#from tqdm import tqdm

import h5py
from multiprocessing import Pool
import pickle
from numba import jit

H0GLOB=67 #69
Om0GLOB=0.319
Xi0Glob =1.
clight = 2.99792458* 10**5#km/s
cosmoflag = FlatLambdaCDM(H0=H0GLOB, Om0=Om0GLOB)

#################################################################################

def area(nside,all_pixels,p_posterior,level=0.99):

    
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

def process_pixel(args):
    pix, nside, mean, cov, new_samples_per_pixel, samples, pixels = args
    
    # Get the fixed angles for this pixel
    theta_fixed, phi_fixed = hp.pix2ang(nside, pix)
    
    # Extract samples of all parameters for this pixel
    pixel_indices = np.where(pixels == pix)[0]
    samples_in_pixel = samples[pixel_indices]

    # Permutation order to move luminosity distance to index 0, theta to index 1, and phi to index 2
    perm = [2, 3, 4] + list(range(0, 2)) + list(range(5, len(mean)))

    # Apply the permutation to the mean and covariance matrix
    mean_permuted = np.array(mean)[perm]
    cov_permuted = cov[np.ix_(perm, perm)]

# Create the alpha vector with the fixed values and mean of other parameters
    alpha = np.zeros(len(mean_permuted) - 1)
    alpha[0] = theta_fixed
    alpha[1] = phi_fixed
    alpha[2:] = samples_in_pixel[:, [0, 1, 5, 6]].mean(axis=0)  # Use the mean of the other parameters in this pixel
    
    # Create a new mean vector excluding the luminosity distance mean (mean_permuted[0])
    mean_new = mean_permuted[1:]

    # Partition the permuted covariance matrix
    Sigma_xx = cov_permuted[1:, 1:]
    Sigma_xy = cov_permuted[1:, 0]
    Sigma_yx = cov_permuted[0, 1:]
    Sigma_yy = cov_permuted[0, 0]
    
    # Compute the conditional mean and covariance
    mu_cond = mean_permuted[0] + Sigma_yx @ np.linalg.inv(Sigma_xx) @ (alpha - mean_new)
    Sigma_cond = Sigma_yy - Sigma_yx @ np.linalg.inv(Sigma_xx) @ Sigma_xy
    
    # Handle negative Sigma_cond
    if Sigma_cond < 0:
        print(f"Warning: Negative Sigma_cond ({Sigma_cond}) encountered for pixel {pix}.")
        Sigma_cond = np.abs(Sigma_cond)  # Take the absolute value
        Sigma_cond = max(Sigma_cond, 1e-10)  # Ensure it is at least a small positive value
    
    # Sample from the conditional Gaussian distribution
    new_samples = np.random.normal(mu_cond, np.sqrt(Sigma_cond), new_samples_per_pixel)
    
    # Extract the new luminosity distances
    new_luminosity_distance = new_samples
    
    # Compute the mean and std of the luminosity distance
    mu = np.mean(new_luminosity_distance)
    std = np.std(new_luminosity_distance)
    
    return pix, mu, std, new_luminosity_distance

####################################################################################################################################
folder='Uniform/TestRun00/'
CAT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'
SCRIPT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/'
COV_SAVE_PATH='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Events/'+folder



Allevents_DS=gwfast.gwfastUtils.load_population(COV_SAVE_PATH+'SNR_more_than_50_100_fixed.h5')
keys=list(Allevents_DS.keys())
print(keys)
print('this is the order of the paramers. A permutation will be implemented.\nThe permutation will preserve the semi positivity')
# construct the mean vector now only for one DS
selectted=4
merge_param=np.array(list(newAllevents.values()))
mean=merge_param[:,selected]
cov=np.load(COV_SAVE_PATH+'Cov_SNR_more_than_50_100_fixed.npy',allow_pickle=True)
#now just for one covariance matrix
cov=np.float64(cov[:,:,selected])#4 select only that specific matrix

######################### CORE part ##################################à

# Number of samples
num_samples=100**5#5 is the num of dimension, 100 is the desired target of points
#num_samples = 100_000_000

# Sample from the multivariate Gaussian distribution
samples = sample_multivariate_gaussian(mean, cov, num_samples)

# Extract variables
#luminosity_distance = samples[:, 2]
angles = samples[:, 3:5]

# Ensure angles are within valid ranges
angles[:, 0] = np.mod(angles[:, 0], np.pi)  # theta in range [0, π]
angles[:, 1] = np.mod(angles[:, 1], 2 * np.pi)  # phi in range [0, 2π]

# Number of pixels in the sky map
nside = 64

# Create a HEALPix map
sky_map = np.zeros(hp.nside2npix(nside))
npix=hp.nside2npix(nside)
all_pixels=np.arange(npix)
# Convert angles to pixel indices
pixels = hp.ang2pix(nside, angles[:, 0], angles[:, 1])

# Increment the pixel values
np.add.at(sky_map, pixels, 1)

# Normalize the sky map
sky_map = sky_map / np.sum(sky_map)

# Get the array of good pixels where sky_map > 0
pix99=get_credible_region_pixels(all_pixels,sky_map)
pix90=get_credible_region_pixels(all_pixels,sky_map,level=0.9)
# Initialize arrays to store the mean and std of luminosity distance
all_mu = np.zeros(hp.nside2npix(nside))
all_std = np.zeros(hp.nside2npix(nside))

# Initialize dictionary to store new luminosity distance arrays for each pixel
luminosity_distance_samples = {}

new_samples_per_pixel = num_samples #35_000_000

# Prepare arguments for multiprocessing
args = [(pix, nside, mean, cov, new_samples_per_pixel, samples, pixels) for pix in pix99]

# Specify the number of processors to use
num_processors = 14

# Using multiprocessing Pool to parallelize the process
with Pool(processes=num_processors) as pool:
    # Using tqdm to add a progress bar
    results = list(tqdm(pool.imap(process_pixel, args), total=len(pix99)))

# Collect the results
for pix, mu, std, new_luminosity_distance in results:
    if mu is not None and std is not None:
        all_mu[pix] = mu
        all_std[pix] = std
        luminosity_distance_samples[pix] = new_luminosity_distance

mod_postnorm=np.ones(npix)
fname='GWtest00.fits'
dat=Table([sky_map,all_mu,all_std,mod_postnorm],
      names=('PROB','DISTMU','DISTSIGMA','DISTNORM'))
os.chdir(COV_SAVE_PATH)
fits.write_sky_map(fname,dat, nest=False)