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

# Configure ET and the PSD
ETdet = {'ET': copy.deepcopy(glob.detectors).pop('ETS') }
print(ETdet)
ETdet['ET']['psd_path'] = os.path.join(glob.detPath, 'ET-0000A-18.txt')
mySignalsET = {}
for d in ETdet.keys():
    #print(d)
    mySignalsET[d] = GWSignal((IMRPhenomD()),
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
folder='Uniform/TestRun00/'
CAT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'
SCRIPT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/'
COV_SAVE_PATH='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Events/'+folder


H0GLOB= 67#67.9 #69
Om0GLOB=0.319
Xi0Glob =1.
cosmoeuclid = FlatLambdaCDM(H0=H0GLOB, Om0=Om0GLOB)

Allevents_DS=gwfast.gwfastUtils.load_population(COV_SAVE_PATH+'SNR_more_than_50_100.h5')
totFET = myET.FisherMatr(Allevents_DS)

keys=list(Allevents_DS.keys())
tofix_ParNums = [e for e in tofix_ParNums if e not in ('dL', 'phi','theta','Mc','eta')]
fixedFIM, newPars=fixParams(totFET,IMRPhenomD().ParNums,tofix_ParNums)

totCov_ET, inversion_err_ET = CovMatr(fixedFIM)

newAllevents = {key: Allevents_DS[key] for key in newPars}
gwfast.gwfastUtils.save_data(COV_SAVE_PATH+'SNR_more_than_50_100_fixed.h5', newAllevents)

np.save(COV_SAVE_PATH+'Cov_SNR_more_than_50_100_fixed',totCov_ET)
os.chdir(COV_SAVE_PATH)
#parname='param.txt'
#np.savetxt(parname,keys,fmt='%s')
#np.savetxt('mean_fixed.txt',mymu)