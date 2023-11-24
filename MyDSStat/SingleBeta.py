import numpy as np
#import healpy as hp
import pandas as pd
import os 
import statistics as stat
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm
from scipy import special
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import fsolve
import sys
from scipy import integrate
from scipy import interpolate

import os
from os import listdir
from os.path import isfile, join

href=67 #69
Om0GLOB=0.319
Xi0Glob =1.
clight = 2.99792458* 10**5#km/s
cosmoflag = FlatLambdaCDM(H0=href, Om0=Om0GLOB)

def E_z(z, H0, Om=Om0GLOB):
    return np.sqrt(Om*(1+z)**3+(1-Om))

def r_z(z, H0, Om=Om0GLOB):
    c = clight
    integrand = lambda x : 1/E_z(x, H0, Om)
    integral, error = integrate.quad(integrand, 0, z)
    return integral*c/H0

def Dl_z(z, H0, Om=Om0GLOB):
    return r_z(z, H0, Om)*(1+z)

def stat_weights(array_of_z):
    #alltheomega=w(array_of_z)
    temp=np.interp(array_of_z,z_bin,w_hist)
    return temp
def calculate_to_sum(args):
    z, x, mydlmin, mydlmax, s = args
    to_sum = np.zeros(len(x))
    for j in range(len(x)):
        dl = Dl_z(z, x[j])
        tmp = 0.5 * (special.erf((dl - mydlmin) / (np.sqrt(2) * s * dl)) - special.erf((mydlmax-dl) / (np.sqrt(2) * s * dl)))
        #tmp = 0.5 * (special.erfc((mydlmax-dl) / (np.sqrt(2) * s * dl)))
        to_sum[j] = tmp*stat_weights(z)
    return to_sum

def calculate_res(myallz, x, mydlmin, mydlmax, s):
    res = np.zeros(len(x))
    with Pool(processes=15) as pool:  # Adjust the number of processes as needed
        to_sum_list = list(tqdm(pool.imap(calculate_to_sum, [(z, x, mydlmin, mydlmax, s) for z in myallz]), total=len(myallz)))
    for to_sum in to_sum_list:
        res += to_sum
    return res

#----------------------------------------------------------------

path='results'
exist=os.path.exists(path)
if not exist:
    print('creating result folder')
    os.mkdir('results')
runpath='BetaErf15-check'
folder=os.path.join(path,runpath)
os.mkdir(folder)
print('\n data will be saved in '+folder)
H0min=60#30#55
H0max=76#140#85
x=np.linspace(H0min,H0max,1000)
NCORE=multiprocessing.cpu_count()-1#15
print('Using {} Cores\n' .format(NCORE))
#------------N(z)------------------------------------
z_bin=np.loadtxt('half_flag_bin.txt')
w_hist=np.loadtxt('half_flag_bin_weights.txt')
#---------------HostCat-----------------------------------------------
cat_name='half_flag.txt'# FullExplorer_big.txt#Uniform_for_half_flag
MyCat = pd.read_csv(cat_name, sep=" ", header=None)
colnames=['Ngal','Comoving Distance','Luminosity Distance','z','phi','theta','scattered DL']
MyCat.columns=colnames
allz=np.asarray(MyCat['z'])
dlmaxcat=10400#MyCat['scattered DL'].max()
dlmincat=MyCat['Luminosity Distance'].min()

z_sup=np.max(allz)
z_inf=np.min(allz)
#---------------------Beta---------------------------
s=0.15
mydlmax=dlmaxcat#15840.294579141924#Dl_z(zds_max,href,Om0GLOB)
mydlmin=dlmincat#5209.84979508345#Dl_z(zds_min,href,Om0GLOB)
print('Starting to compute beta')
myallz=allz
beta = calculate_res(myallz, x, mydlmin, mydlmax, s)
betapath=os.path.join(folder,runpath+'_betaErf.txt')
np.savetxt(betapath,beta)#allbetas
print('beta saved')