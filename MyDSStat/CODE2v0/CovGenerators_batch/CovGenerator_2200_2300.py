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

ParNums = IMRPhenomD().ParNums
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
start=2200
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
#print('Allevents head')
#print(Allevents_DS.head(5))
print('Saving files')
gwfast.gwfastUtils.save_data(COV_SAVE_PATH+'SNR_more_than_100_2200_2300.h5', Allevents_DS)
totFET = myET.FisherMatr(Allevents_DS)
print('The computed Fisher matrix has shape %s'%str(totFET.shape))
np.save(COV_SAVE_PATH+'Fish_SNR_more_than_100_2200_2300',totFET)
totCov_ET, inversion_err_ET = CovMatr(totFET)
#name='Allevents_from_Uniform_complete'
np.save(COV_SAVE_PATH+'Cov_SNR_more_than_100_2200_2300',totCov_ET)
print('Number of computed Covs'+str(np.shape(totCov_ET)[2]))
