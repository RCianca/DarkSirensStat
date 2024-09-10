import pandas as pd
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

from ligo.skymap.io import fits

import os
import sys



from tqdm import tqdm

import h5py
from multiprocessing import Pool
import pickle
from numba import jit

H0GLOB=67 #69
Om0GLOB=0.319
Xi0Glob =1.
clight = 2.99792458* 10**5#km/s
cosmoflag = FlatLambdaCDM(H0=H0GLOB, Om0=Om0GLOB)

# Declare global variables for large objects
mean = None
cov = None
samples = None
Allevents_DS = None

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

# Try Cholesky decomposition
def sample_multivariate_gaussian_cholesky(mean, cov, num_samples):
    # Perform Cholesky decomposition of the covariance matrix once
    L = np.linalg.cholesky(cov)

    # Generate standard normal samples, sample will be shaped like (num_sample,mean)
    z = np.random.randn(num_samples, len(mean))

    # Transform the samples using the Cholesky factor
    samples = mean + z @ L.T
    return samples

def save_multivariate_gaussian_batches(mean, cov, num_samples, batch_size, output_dir):
    num_batches = num_samples // batch_size

    # Generate samples in batches
    for i in range(num_batches):
        samples = sample_multivariate_gaussian_cholesky(mean, cov, batch_size)
        np.save(os.path.join(output_dir, f"samples_batch_{i + 1}.npy"), samples)
        del samples

    # Handle remaining samples if num_samples is not a multiple of batch_size
    remaining_samples = num_samples % batch_size
    if remaining_samples > 0:
        samples = sample_multivariate_gaussian_cholesky(mean, cov, remaining_samples)
        np.save(os.path.join(output_dir, "samples_batch_remaining.npy"), samples)
        del samples


def process_pixel(args):
    
    #global mean, cov, samples, Allevents_DS

    pix, nside,new_samples_per_pixel = args
    
    # Get the fixed angles for this pixel
    theta_fixed, phi_fixed = hp.pix2ang(nside, pix)
    
    # Extract samples of all parameters for this pixel
    pixel_indices = np.where(pixels == pix)[0]
    samples_in_pixel = samples[pixel_indices]

    # Determine the positions of 'dL', 'theta', and 'phi' in the columns of Allevents_DS
    columns = Allevents_DS.columns
    dL_pos = columns.get_loc('dL')
    theta_pos = columns.get_loc('theta')
    phi_pos = columns.get_loc('phi')

    # Create the permutation order with 'dL' first, 'theta' second, and 'phi' third
    remaining_indices = list(set(range(len(columns))) - {dL_pos, theta_pos, phi_pos})
    perm = [dL_pos, theta_pos, phi_pos] + remaining_indices

    # Apply the permutation to the mean and covariance matrix
    mean_permuted = np.array(mean)[perm]
    cov_permuted = cov[np.ix_(perm, perm)]

# Create the alpha vector with the fixed values and mean of other parameters
    alpha = np.zeros(len(mean_permuted) - 1)
    alpha[0] = theta_fixed
    alpha[1] = phi_fixed
    alpha[2:] = samples_in_pixel[:, remaining_indices].mean(axis=0)  # Use the mean of the other parameters in this pixel
    
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
    #new_luminosity_distance = new_samples
    
    # Compute the mean and std of the luminosity distance
    mu = np.mean(new_samples)
    std = np.std(new_samples)
    
    return pix, mu, std#, new_samples

####################################################################################################################################
def initialize_globals(catfile, covfile):
    global mean, cov, samples, Allevents_DS, pixels

    folder = 'Uniform/TestRun00/'
    COV_SAVE_PATH = '/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Events/' + folder

    # Load the data into global variables
    Allevents_DS = pd.read_csv(COV_SAVE_PATH + catfile, index_col=0)
    print(Allevents_DS.head(3))
    keys = list(Allevents_DS.columns)
    print(keys)
    print('This is the order of the parameters. A permutation will be implemented.\nThe permutation will preserve the semi-positivity.')

    selected = 4
    mean = np.array(Allevents_DS.iloc[selected])
    cov = np.load(COV_SAVE_PATH + covfile, allow_pickle=True)
    cov = np.float64(cov[:, :, selected])

    print(mean)
    print(len(mean), np.shape(cov))

    # Generate or load samples
    output_dir = COV_SAVE_PATH + "samples_batches/"
    if not os.path.exists(output_dir) or not os.listdir(output_dir):
        print("Generating new samples...")
        num_samples = 10**2  # Total number of samples to generate
        batch_size = 10**2  # Batch size for saving samples

        os.makedirs(output_dir, exist_ok=True)

        # Use the optimized function to save the samples in batches
        save_multivariate_gaussian_batches(mean, cov, num_samples, batch_size, output_dir)

    print("Loading samples from batches...")
    all_batches = [f for f in os.listdir(output_dir) if f.endswith(".npy")]
    samples = np.concatenate([np.load(os.path.join(output_dir, batch)) for batch in all_batches], axis=0)

    # Extract angles from the samples and convert them to pixel indices
    angles = samples[:, 3:5]
    angles[:, 0] = np.mod(angles[:, 0], np.pi)
    angles[:, 1] = np.mod(angles[:, 1], 2 * np.pi)

    # Compute HEALPix pixels
    nside = 64
    pixels = hp.ang2pix(nside, angles[:, 0], angles[:, 1])



######################### CORE part ##################################
DS_catalogue='SNR_more_than_50_100.csv'
Cov_file='Cov_SNR_more_than_50_100.npy'
initialize_globals(DS_catalogue,Cov_file)
angles = samples[:, 3:5]

# Ensure angles are within valid ranges
angles[:, 0] = np.mod(angles[:, 0], np.pi)  # theta in range [0, pi]
angles[:, 1] = np.mod(angles[:, 1], 2 * np.pi)  # phi in range [0, 2pi]

# Number of pixels in the sky map
nside = 64

# Create a HEALPix map
sky_map = np.zeros(hp.nside2npix(nside))
npix=hp.nside2npix(nside)
all_pixels=np.arange(npix)
# Convert angles to pixel indices
#pixels = hp.ang2pix(nside, angles[:, 0], angles[:, 1])

# Increment the pixel values
np.add.at(sky_map, pixels, 1)

# Normalize the sky map
sky_map = sky_map / np.sum(sky_map)

# Get the array of good pixels where sky_map > 0
print('computing credible regions')
pix99=get_credible_region_pixels(all_pixels,sky_map)
pix90=get_credible_region_pixels(all_pixels,sky_map,level=0.9)
# Initialize arrays to store the mean and std of luminosity distance
all_mu = np.zeros(hp.nside2npix(nside))
all_std = np.zeros(hp.nside2npix(nside))

# Initialize dictionary to store new luminosity distance arrays for each pixel
luminosity_distance_samples = {}

new_samples_per_pixel = 10**2#num_samples #35_000_000

# Prepare arguments for multiprocessing


# Specify the number of processors to use
num_processors = 24
print('computing the conditional probability for dL')
# Using multiprocessing Pool to parallelize the process
args = [(pix, nside, new_samples_per_pixel) for pix in pix99]
with Pool(processes=num_processors) as pool:
    # Using tqdm to add a progress bar
    #results = list(tqdm(pool.imap(process_pixel, args), total=len(pix99)))
    results = list(pool.imap(process_pixel, args))

# Collect the results
for pix, mu, std in results:
    if mu is not None and std is not None:
        all_mu[pix] = mu
        all_std[pix] = std
        #luminosity_distance_samples[pix] = new_luminosity_distance

mod_postnorm=np.ones(npix)
print('lenght of mean is {} and std is {}'.format(np.shape(all_mu),np.shape(all_std)))
fname='GWtest00.fits'
dat=Table([sky_map,all_mu,all_std,mod_postnorm],
      names=('PROB','DISTMU','DISTSIGMA','DISTNORM'))
os.chdir(COV_SAVE_PATH)
fits.write_sky_map(fname,dat, nest=False)