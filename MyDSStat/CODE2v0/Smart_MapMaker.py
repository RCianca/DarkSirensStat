import os
import sys

import copy
import numpy as np
import pandas as pd 
from astropy.cosmology import FlatLambdaCDM 

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
from numba import jit

from Global import *
###########################################################################################################################
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
folder='Uniform/TestRun01/'
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
allm1=np.asarray(DS_Cat['M1'])
allm2=np.asarray(DS_Cat['M2'])
allMc=np.asarray(DS_Cat['MC'])
allq=np.asarray(DS_Cat['q'])
alleta=allq/(1+allq)**2
allcos=np.asarray(DS_Cat['cos_iota'])
alliota=np.arccos(allcos)
allpsi=np.asarray(DS_Cat['psi'])/2
tGPS = np.array([1187008882.4])#arbitrario
#tGPS = np.array([1187508882.4])
allz=np.asarray(DS_Cat['z'])
allphi=np.asarray(DS_Cat['phi']) 
alltheta=np.asarray(DS_Cat['theta']) 
alldl=np.asarray(DS_Cat['Luminosity Distance'])/1000.0#servono i Gpc
alltcoal=np.asarray(DS_Cat['tcoal']) 
allPhicoal=np.asarray(DS_Cat['Phicoal']) 
allchi1z=np.asarray(DS_Cat['chi1z']) 
allchi2z=np.asarray(DS_Cat['chi2z']) 


tcoal=np.asarray(GPSt_to_LMST(tGPS, lat=40.516666666666666, long=9.416666666666666))
start=900
print('Start is {}'.format(start))
quanti=int(min(100,DS_Cat.shape[0]))
Allevents_DS = {'Mc':1*allMc[start:start+quanti]*(1+allz)[start:start+quanti],
            'eta':alleta[start:start+quanti],#alleta_tmp[2:3],
            'dL':alldl[start:start+quanti],
            'theta':alltheta[start:start+quanti],
            'phi':allphi[start:start+quanti],
            'iota':alliota[start:start+quanti],
            'psi':allpsi[start:start+quanti],
            'tcoal':alltcoal[start:start+quanti], # GMST is LMST computed at long = 0°
            'Phicoal':allPhicoal[start:start+quanti],
            'chi1z':allchi1z[start:start+quanti],
            'chi2z':allchi2z[start:start+quanti]
            #'chi2z':np.zeros(len(allMc))[0:1]
           }

#gwfast.gwfastUtils.save_data(COV_SAVE_PATH+'SNR_more_than_100_900_1000.h5', Allevents_DS)
totFET = myET.FisherMatr(Allevents_DS)
print('The computed Fisher matrix has shape %s'%str(totFET.shape))
#np.save(COV_SAVE_PATH+'Fish_SNR_more_than_100_900_1000',totFET)
totCov_ET, inversion_err_ET = CovMatr(totFET)
area_deg2=compute_localization_region(totCov_ET,totCov_ET)
if area_deg2<=100:
    np.save(COV_SAVE_PATH+'Cov_SNR_more_than_100_'+ num,totCov_ET)
