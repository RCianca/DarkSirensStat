import os
from os import listdir
from os.path import isfile, join

import sys

import copy
import numpy as np


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd())))
sys.path.append(SCRIPT_DIR)
import gwfast.gwfastGlobals as glob
import gwfast 
from gwfast.waveforms import IMRPhenomD_NRTidalv2, IMRPhenomHM, IMRPhenomD
from gwfast.signal import GWSignal
from gwfast.network import DetNet
from fisherTools import CovMatr, compute_localization_region, check_covariance, fixParams
from gwfastUtils import GPSt_to_LMST


import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from functools import partial


from multiprocessing import Pool
from numba import njit
from tqdm import tqdm

############################################################################

H0GLOB= 67#69.32#67.9 #69
Om0GLOB=0.319
Xi0Glob =1.
clight = 2.99792458* 10**5#km/s

cosmofast = FlatLambdaCDM(H0=H0GLOB, Om0=Om0GLOB)
H0=cosmofast.H(0).value
h=H0GLOB/100

CAT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'
SCRIPT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/DSCatalogueCreator/'
SAVE_PATH='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'
###########################################################################

os.chdir(CAT_FOLDER)
DS_Cat= pd.read_csv('DS_From_Parent_Uniform_Complete.txt')
dlmean=np.mean(DS_Cat['Luminosity Distance'])
print('dl media in Mpc')
print(dlmean)
dlmean=np.mean(np.array(DS_Cat['Luminosity Distance']/1000.0 ))#np.array(current_chunk['Luminosity Distance']/1000.0 )
print('dl media in Gpc')
print(dlmean)
os.chdir(SCRIPT_FOLDER)

ParNums = IMRPhenomHM().ParNums
print(ParNums)

# allm1=np.asarray(DS_Cat['M1'])
# allm2=np.asarray(DS_Cat['M2'])
# allMc=np.asarray(DS_Cat['MC'])
# allq=np.asarray(DS_Cat['q'])
# alleta=allq/(1+allq)**2
# allcos=np.asarray(DS_Cat['cos_iota'])
# alliota=np.arccos(allcos)
# allpsi=np.asarray(DS_Cat['psi'])/2
tGPS = np.array([1187008882.4])#arbitrario
# #tGPS = np.array([1187508882.4])
# allz=np.asarray(DS_Cat['z'])
# allphi=np.asarray(DS_Cat['phi'])
# alltheta=np.asarray(DS_Cat['theta'])
# alldl=np.asarray(DS_Cat['Luminosity Distance'])/1000#servono i Gpc
tcoal=np.asarray(GPSt_to_LMST(tGPS, lat=40.516666666666666, long=9.416666666666666))

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

chunksize = 1000  
total_length = len(DS_Cat)

# Prepare the output file path
output_file = os.path.join(SAVE_PATH, 'DS_From_Parent_Uniform_Complete_SNR.txt')

# Process the data in chunks
for start in range(0, total_length, chunksize):
    end = min(start + chunksize, total_length)
    
    # Slice the data for the current chunk
    current_chunk = DS_Cat.iloc[start:end].copy()
    print(current_chunk.head(3))
    # Convert necessary columns to numpy arrays
    mc_array = np.array(current_chunk['MC'] * (1 + current_chunk['z']))
    dl_array = np.array(current_chunk['Luminosity Distance'])/1000.0  # Convert to Gpc
    theta_array = np.array(current_chunk['theta'])
    phi_array = np.array(current_chunk['phi'])
    iota_array = np.arccos(np.array(current_chunk['cos_iota']))  # Convert cos(iota) to iota
    psi_array = np.array(current_chunk['psi']) / 2  # Apply the division to psi
    eta_array = np.array(current_chunk['q'] / (1 + current_chunk['q'])**2 )

    # Create the Allevents dictionary for the current chunk with numpy arrays
    Allevents = {
        'Mc': mc_array,
        'eta': eta_array,
        'dL': dl_array,
        'theta': theta_array,
        'phi': phi_array,
        'iota': iota_array,
        'psi': psi_array,
        'tcoal': 1 * tcoal * np.ones(len(current_chunk)),  # GMST is LMST computed at long = 0°
        'Phicoal': np.full(len(current_chunk), 0.0003),
        'chi1z': np.full(len(current_chunk), 0.00002),
        'chi2z': np.full(len(current_chunk), 0.00001)
    }

    # Compute the SNR for the current chunk
    SNR_ET = myET.SNR(Allevents)

    # Add the SNR column to the current chunk
    current_chunk['SNR'] = SNR_ET

    # Save the current chunk to the file
    mode = 'w' if start == 0 else 'a'
    header = True if start == 0 else False
    current_chunk.to_csv(output_file, header=header, index=False, mode=mode)

print(f"DataFrame with SNR saved to {output_file}")

# Return to the original directory
os.chdir(SCRIPT_FOLDER)