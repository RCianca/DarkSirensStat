import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import integrate
from scipy.optimize import fsolve
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import healpy as hp

import os
from os import listdir
from os.path import isfile, join

from multiprocessing import Pool
import time
from numba import njit

import sys
from tqdm import tqdm
#################################################functions and constants#################################################
href=67 #69
Om0GLOB=0.319
Xi0Glob =1.
clight = 2.99792458* 10**5#km/s
cosmoflag = FlatLambdaCDM(H0=href, Om0=Om0GLOB)
@njit
def E_z(z, H0, Om=Om0GLOB):
    return np.sqrt(Om*(1+z)**3+(1-Om))

def r_z(z, H0, Om=Om0GLOB):
    c = clight
    integrand = lambda x : 1/E_z(x, H0, Om)
    integral, error = integrate.quad(integrand, 0, z)
    return integral*c/H0

def Dl_z(z, H0, Om=Om0GLOB):
    return r_z(z, H0, Om)*(1+z)
def z_dc(dc,h,om=Om0GLOB):
    func = lambda z :r_z(z, h, Om0GLOB) - dc
    zres = fsolve(func, 0.02)[0] 
    return zres
@njit
def phi2RA(phi):
    ret=np.rad2deg(phi)
    return ret
@njit
def theta2DEC(theta):
    ret=np.rad2deg(0.5*np.pi-theta)
    return ret
@njit
def RA2phi(RA):
    ret=np.deg2rad(RA)
    return ret
@njit
def DEC2theta(DEC):
    ret=0.5 * np.pi - np.deg2rad(DEC)
    return ret
##########################################################################################################################
#read the flagship
cat_data_path='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/data/GLADE/'
os.chdir(cat_data_path)
#all_event=os.listdir()
#print(all_event)
save_cat_path='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'

flag = pd.read_csv('GLADE_flagship.txt', sep=" ", header=None)
colnames=['numevent','GWGC_name','HyperLEDA_name','2MASS_XSC_name','SDSS-DR12Q_name','type_flag','right_ascension_degrees',
          'declination_degrees','dl','err_lum_dist','z','app_B_mag','err_app_B_mag','abs_B_mag','app_J_mag',
          'err_app_J_mag','app_H_mag','err_app_H_mag','app_K_mag','err_app_K_mag','lum_dist_flag','pec_vel_correction'
          ]
flag.columns=colnames

z_flag_min=flag['z'].min()
z_flag_max=flag['z'].max()
Nobj=flag.shape[0]
dl_flag_min=flag['dl'].min()
dl_flag_max=flag['dl'].max()
RA_flag_min=flag['right_ascension_degrees'].min()
RA_flag_max=flag['right_ascension_degrees'].max()
DEC_flag_min=flag['declination_degrees'].min()
DEC_flag_max=flag['declination_degrees'].max()
phi_flag_min=RA2phi(RA_flag_min)
phi_flag_max=RA2phi(RA_flag_max)
theta_flag_min=DEC2theta(DEC_flag_max)
theta_flag_max=DEC2theta(DEC_flag_min)

#denser bin
nbins=30
z_flag=np.asarray(flag['z'])
fig, ax = plt.subplots(1, figsize=(12,8)) #crea un tupla che poi è più semplice da gestire
ax.tick_params(axis='both', which='major', labelsize=14)
ax.yaxis.get_offset_text().set_fontsize(14)
ax.grid(linestyle='dotted', linewidth='0.6')#griglia in sfondo
colors=plt.cm.turbo(np.linspace(0.99,0.01,16))
(n, bins, patches)=ax.hist(z_flag,bins=nbins,range=(z_flag_min,z_flag_max))
delta_phi=phi_flag_max-phi_flag_min
theta_part=np.cos(theta_flag_min)-np.cos(theta_flag_max)
integrand=lambda x:clight*(cosmoflag.comoving_distance(x).value)**2/(cosmoflag.H(x).value)
density=np.zeros(len(n))
for i,j in enumerate (bins[:-1]):
    z_part=integrate.quad(integrand,bins[i],bins[i+1])[0]
    Volume=delta_phi*theta_part*z_part#integrand
    density[i]=n[i]/Volume
desity_max=np.max(density)
##########################################################################################
z_max_unif=2.3
z_min_unif=0.3
z_part=integrate.quad(integrand,z_min_unif,z_max_unif)[0]
tot_volume=delta_phi*theta_part*z_part
uniform_numb=int(3*desity_max*tot_volume)+1
dcom_min=cosmoflag.comoving_distance(z_min_unif).value
dcom_max=cosmoflag.comoving_distance(z_max_unif).value

cat_name='Uniform_paper.txt'


#------------------points generator------------------
u = np.random.uniform(0,1,size=uniform_numb) # uniform random vector of size nsamp
dc_gals_all     = np.cbrt((u*dcom_min**3)+((1-u)*dcom_max**3))
phi_gals   = np.random.uniform(phi_flag_min,phi_flag_max,uniform_numb)
theta_gals = np.arccos( np.random.uniform(np.cos(theta_flag_max),np.cos(theta_flag_min),uniform_numb) )
dc_gals=dc_gals_all[dc_gals_all>=dcom_min]
new_phi_gals=np.random.choice(phi_gals,len(dc_gals))
new_theta_gals=np.random.choice(theta_gals,len(dc_gals))

def uniform_volume(iterations):
#for i in tqdm(range(flagship.shape[0])):
#for i in range(iterations):
    i=iterations
    numevent=i
    phigal=new_phi_gals[i]
    thetagal=new_theta_gals[i]
    dc=dc_gals[i]
    #----------z----------------------
    zz=z_dc(dc,href,Om0GLOB)
    dl=Dl_z(zz,href,Om0GLOB)
    #----------row to append---------------------
    proxy_row={'Ngal':numevent,'Comoving Distance':dc,'Luminosity Distance':dl,
               'z':zz,'phi':phigal,'theta':thetagal
          }
    return proxy_row


numevent=int(0)
proxy_row={'Ngal':numevent,'Comoving Distance':0,'Luminosity Distance':0,
               'z':0,'phi':0,'theta':0
          }
colnames=['Ngal','Comoving Distance','Luminosity Distance','z','phi','theta']
uniform_scaled = pd.DataFrame(columns=colnames)

from tqdm import tqdm
arr=np.arange(0,len(dc_gals),dtype=int)
data=[]
tmp=[]
num_processors = 24  # Number of processors to match the requested CPUs

with Pool(num_processors) as p:
    tmp = p.map(uniform_volume, arr)

uniform_scaled=uniform_scaled.append(tmp, ignore_index=True)

os.chdir(save_cat_path)
uniform_scaled.to_csv(cat_name, header=None, index=None, sep=' ')