import numpy as np
import pandas as pd
import os 
from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import fsolve
from scipy import special
import multiprocessing
from multiprocessing import Pool
from scipy import integrate
from numba import njit

href=67 #69
Om0GLOB=0.319
Xi0Glob =1.
clight = 2.99792458* 10**5#km/s

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

def z_from_dl(h,dl):
    func = lambda z :Dl_z(z, h, Om0GLOB) -dl
    zmax = fsolve(func, 0.02)[0] 
    return zmax

def Vcat(zinf,zsup,h,Om=Om0GLOB):
    cosmo=FlatLambdaCDM(H0=h,Om0=Om0GLOB)
    vsup=cosmo.comoving_volume(zsup).value
    vinf=cosmo.comoving_volume(zinf).value
    volcat=vsup-vinf
    return volcat

def beta_mod(h,dlmin,dlmax,zsup,zinf):
    cosmo=FlatLambdaCDM(H0=h,Om0=Om0GLOB)
    zmax=z_from_dl(h,dlmax)
    zmin=z_from_dl(h,dlmin)
    v_sup=cosmo.comoving_volume(zsup).value
    v_inf=cosmo.comoving_volume(zinf).value
    denom=v_sup-v_inf
    if zmax<=zsup:
        v_max=cosmo.comoving_volume(zmax).value
    else:
        v_max=v_sup
    if zmin>=zinf:
        v_min=cosmo.comoving_volume(zmin).value
    else:
        v_min=v_inf
    tmp=(v_max-v_min)/(v_sup-v_inf)
    return tmp

def beta_comp(h,dlmin,dlmax,myz,zsup,zinf):
    vcat=len(myz)
    zmax=z_from_dl(h,dlmax)
    zmin=z_from_dl(h,dlmin)
    if zmax<=zsup:
        cut=myz[myz<=zmax]
        v_max=len(cut)
    else:
        v_max=len(myz)
    if zmin>=zinf:
        cut=myz[myz<=zmin]
        v_min=len(cut)
    else:
        v_min=len(cut)
    tmp=(v_max-v_min)/vcat
    return tmp


def W(args):
    dl, dlmin, dlmax, s=args
    denom=np.sqrt(2)*dl*s
    ret=(special.erf((dl-dlmin)/denom)-special.erf((dl-dlmax)/denom))*0.5
    return ret


def calculate_to_sum(args):
    z, x, mydlmin, mydlmax, s = args
    to_sum = np.zeros(len(x))
    for j in range(len(x)):
        dl = Dl_z(z, x[j])
        wargs=(dl,mydlmin,mydlmax,s)
        tmp = W(wargs)
        to_sum[j] = tmp#*stat_weights(z)*V_element(z,x[j])
    return to_sum

def calculate_res(myallz, x, mydlmin, mydlmax, s):
    res = np.zeros(len(x))
    with Pool(processes=NCORE) as pool:  # Adjust the number of processes as needed
        to_sum_list = list(tqdm(pool.imap(calculate_to_sum, [(z, x, mydlmin, mydlmax, s) for z in myallz]), total=len(myallz)))
    for to_sum in to_sum_list:
        res += to_sum
    return res


myds = pd.read_csv('Uniform_for_half_flag.txt', sep=" ", header=None)
colnames=['Ngal','Comoving Distance','Luminosity Distance','z','phi','theta','scattered DL']
myds.columns=colnames
myallz=np.linspace(1,100,10)#np.asarray(myds['z'])
z_sup=np.max(myallz)
z_inf=np.min(myallz)
dlmin_ds=8950
dlmax_ds=10400
x=np.linspace(60,76,1000)

vcat=np.zeros(len(x))
for i,h in enumerate(x):
    vcat[i]=Vcat(z_inf,z_sup,h)
#mybeta_mod=np.zeros(len(x))
#for i in tqdm(range(len(x))):
#    mybeta_mod[i]=beta_mod(x[i],dlmin_ds,dlmax_ds,z_sup,z_inf)

#mybeta_comp=np.zeros(len(x))
#for i in tqdm(range(len(x))):
#    mybeta_comp[i]=beta_comp(x[i],dlmin_ds,dlmax_ds,myallz,z_sup,z_inf)

NCORE=multiprocessing.cpu_count()
print('Using {} Cores\n' .format(NCORE))

s=0.15
beta_erf = calculate_res(myallz, x, dlmin_ds, dlmax_ds, s)
np.savetxt('beta_erf_Notebook_0.txt',beta_erf)
betatot=beta_erf/vcat
np.savetxt('beta_tot_Notebook_0.txt',betatot)
