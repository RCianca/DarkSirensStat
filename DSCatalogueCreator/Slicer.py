import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import integrate
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.integrate import trapz
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import healpy as hp
from functools import partial

import os
from os import listdir
from os.path import isfile, join

from multiprocessing import Pool
import time
from numba import njit

import sys
from tqdm import tqdm

##############################################################################
H0GLOB= 67#69.32#67.9 #69
Om0GLOB=0.319
Xi0Glob =1.
clight = 2.99792458* 10**5#km/s

cosmofast = FlatLambdaCDM(H0=H0GLOB, Om0=Om0GLOB)
H0=cosmofast.H(0).value
h=H0GLOB/100

#geometrization of masses
Msun=(1.98892)*(10**30)
#solarmass_to_m=(constants.G*Msun)/((constants.c)**2)#G/c^2
#Mpc_to_m=3.08567758128*(10**22) #this will be used later
NCORE=24
###############################################################################

#--------------------GW--Rate-----------------------------------------

#--------Star Formation Rate-------------------------------
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
th=1000/cosmofast.H(0).value
alpha=14/th
def psi(x):
    ret= 0.015*(((1+x)**(2.7))/(1+(((1+x)/(2.9))**(5.6))))
    return ret
#-------------Phi----------------------------
#phi=lambda x:(1/((1+x)*(cosmofast.H(x).value)))
def phi(x):
    #N=14/((1/((1+0)*(cosmoglob.H(0).value))))
    #ret=N*(1/((1+x)*(cosmoglob.H(x).value)))
    ret=(1000/((1+x)*(cosmofast.H(x).value)))
    return ret
#--------------time inversion-------------------------
def inverted_time(x):
    #is the lookback_time use astropy
    ret=cosmofast.age(0).value-cosmofast.age(x).value
    return ret
#----------------Decay--------------------------------
def decay(x,z,tau):
    ret=np.exp(-((cosmofast.lookback_time(x).value-cosmofast.lookback_time(z).value)/(tau)))
    return ret
#--------------time difference-------------------------
def inversetimediff(x,z,tmin):
    timediff=alpha*(cosmofast.age(z).value-cosmofast.age(x).value)
    num=np.heaviside((alpha*cosmofast.age(z).value-alpha*cosmofast.age(x).value) -tmin,0.5)
    res=(num)/timediff
    return res
#-------------------Deriv of Comoving Volume-------------
def DvolDz(x):
    prefactor=4*np.pi*(((clight)/(cosmofast.H(0).value))**3)#Mpc^3
    dist=quad((cosmofast.inv_efunc),0,x)[0]
    ret=prefactor*cosmofast.inv_efunc(x)*(dist**2)
    return ret

def astrodiffvol(x):
    ret=cosmofast.differential_comoving_volume(x).value
    return ret

#------------------integrands------------------------------

def integrand_marr(x,z,tau):
    ret=psi(x)*(phi(x))*decay(x,z,tau)
    return ret
def totalrate(z,tmin):
    norm=50/(quad(integrand_marr,0,150,args=(0,tmin))[0]/tmin)
    littlerate=norm*quad(integrand_marr,z,150,args=(z,tmin))[0]/tmin
    ret=(littlerate*DvolDz(z))/(1+z)
    return ret
#########################################################################################################################
CAT_PATH='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'
THIS_DIR=os.getcwd()
os.chdir(CAT_PATH)
DS_From_Parent = pd.read_csv('DS_From_Parent_Uniform_Complete.txt')
os.chdir(THIS_DIR)

sliced=DS_From_Parent.head(1_000_000)

# Define the path to save the file
output_file = os.path.join(CAT_PATH, 'DS_From_Parent_Uniform_Complete-Sliced.txt')

# Save the DataFrame in chunks
chunksize = 1000  # Adjust the chunksize according to your memory capacity

# Write the first chunk with headers
sliced.iloc[:chunksize].to_csv(output_file, header=True, index=False, mode='w')

# Write the remaining chunks without headers
for i in range(chunksize, len(DS_From_Parent), chunksize):
    sliced.iloc[i:i+chunksize].to_csv(output_file, header=False, index=False, mode='a')

print(f"DataFrame saved to {output_file}")