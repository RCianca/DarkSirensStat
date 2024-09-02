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

CAT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'
SCRIPT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/DSCatalogueCreator/'
COV_SAVE_PATH='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'

os.chdir(CAT_FOLDER)
DS_Cat= pd.read_csv('DS_From_Parent_Uniform_Complete.txt')
os.chdir(SCRIPT_FOLDER)

H0GLOB= 67#67.9 #69
Om0GLOB=0.319
Xi0Glob =1.
cosmoeuclid = FlatLambdaCDM(H0=H0GLOB, Om0=Om0GLOB)

ParNums = IMRPhenomD().ParNums
print(ParNums)

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
alldl=np.asarray(DS_Cat['Luminosity Distance'])/1000#servono i Gpc


tcoal=np.asarray(GPSt_to_LMST(tGPS, lat=40.516666666666666, long=9.416666666666666))

quanti=int(500)
Allevents_DS = {'Mc':1*allMc[0:quanti]*(1+allz)[0:quanti],
            'dL':np.ones(len(allMc))[0:quanti],#alldl[0:quanti],
            'theta':alltheta[0:quanti],
            'phi':allphi[0:quanti],
            'iota':alliota[0:quanti],
            'psi':allpsi[0:quanti],
            'tcoal':1*tcoal*np.ones(len(allMc))[0:quanti], # GMST is LMST computed at long = 0°
            'eta':alleta[0:quanti],#alleta_tmp[2:3],
            'Phicoal':np.zeros(len(allMc))[0:quanti],
            'chi1z':0.*np.ones(len(allMc))[0:quanti],
            'chi2z':0.00001*np.ones(len(allMc))[0:quanti]
            #'chi2z':np.zeros(len(allMc))[0:1]
           }

gwfast.gwfastUtils.save_data(COV_SAVE_PATH+'Allevents_from_Uniform_complete.h5', Allevents_DS)

totFET = myET.FisherMatr(Allevents_DS)
print('The computed Fisher matrix has shape %s'%str(totFET.shape))
totCov_ET, inversion_err_ET = CovMatr(totFET)
#name='Allevents_from_Uniform_complete'
np.save(COV_SAVE_PATH+'Allevents_from_Uniform_complete',totCov_ET)