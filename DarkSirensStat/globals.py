#
#    Copyright (c) 2021 Andreas Finke <andreas.finke@unige.ch>,
#                       Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by a modified BSD
#    license that can be found in the LICENSE file.



import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad
import sys
import healpy as hp
from scipy import interpolate
from scipy.signal import find_peaks

dirName = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

miscPath = os.path.join(dirName, 'data', 'misc')

baseGWPath= os.path.join(dirName, 'data', 'GW') 

metaPath= os.path.join(baseGWPath, 'metadata') 

detectorPath = os.path.join(baseGWPath, 'detectors')
#Raul: replaced all 70 with 67 with scale
scale=67
pivot=scale/100


###########################
# CONSTANTS
###########################

clight = 2.99792458* 10**5


O2BNS = ('GW170820',)
O3BNS = ('GW190425', )
O3BHNS = ('GW190426_152155', 'GW190426' )



##########################
##########################
# MASS DISTRIBUTIONS

BNS_gauss_mu = 1.35
BNS_gauss_sigma = 0.15

BNS_flat_Mmin = 1
BNS_flat_Mmax = 3

BBH_flat_Mmin = 5 # In source frame
BBH_flat_Mmax = 200


pow_law_Mmin = 5
pow_law_Mmax = 50


###########################



zRglob = 0.5

nGlob = 1.91
gammaGlob = 1.6


l_CMB, b_CMB = (263.99, 48.26)
v_CMB = 369

# Solar magnitude in B and K band
MBSun=5.498
MKSun=3.27

# Cosmologival parameters used in GLADE for z-dL conversion
H0GLADE=67
Om0GLADE=0.319

# Cosmologival parameters used for the analysis (RT minimal; table 2 of 2001.07619)
#H0GLOB=67.9 #69
H0GLOB=67
Om0GLOB=0.319
Xi0Glob =1.
cosmoglob = FlatLambdaCDM(H0=H0GLOB, Om0=Om0GLOB)

class PriorLimits:
    def __init__(self):
        self.H0min = 20
        self.H0max = 220
        self.Xi0min = 0.01
        self.Xi0max = 100

# Parameters of Schechter function in B band in units of 10^10 solar B band
# for h0=0.7
LBstar07 =2.45
phiBstar07  = 5.5 * 1e-3
alphaB07 =-1.07


# Parameters of Schechter function in K band in units of 10^10 solar K band
# for h0=0.7
LKstar07 = 10.56
phiKstar07 = 3.70 * 1e-3
alphaK07 =-1.02


###########################
###########################

from scipy.special import erfc, erfcinv

def sample_trunc_gaussian(mu = 1, sigma = 1, lower = 0, size = 1):

    sqrt2 = np.sqrt(2)
    Phialpha = 0.5*erfc(-(lower-mu)/(sqrt2*sigma))
    
    if np.isscalar(mu):
        arg = Phialpha + np.random.uniform(size=size)*(1-Phialpha)
        return np.squeeze(mu - sigma*sqrt2*erfcinv(2*arg))
    else:
        Phialpha = Phialpha[:,np.newaxis]
        arg = Phialpha + np.random.uniform(size=(mu.size, size))*(1-Phialpha)
        
        return np.squeeze(mu[:,np.newaxis] - sigma[:,np.newaxis]*sqrt2*erfcinv(2*arg))
    
def trunc_gaussian_pdf(x, mu = 1, sigma = 1, lower = 0):
#
#    if not np.isscalar(x) and not np.isscalar(mu):
#        x = x[np.newaxis, :]
#        if mu.ndim < 2:
#            mu = mu[:, np.newaxis]
#        if sigma.ndim < 2:
#            sigma = sigma[:, np.newaxis]
    
    #sigma=x*0.1
    Phialpha = 0.5*erfc(-(lower-mu)/(np.sqrt(2)*sigma))
    #print('sigma is {} len is {}'.format(sigma,len(sigma)))
    return np.where(x>0, 1/(np.sqrt(2*np.pi)*sigma)/(1-Phialpha) * np.exp(-(x-mu)**2/(2*sigma**2)) ,0.)
    
###########################
###########################

import multiprocessing

nCores = max(1,int(multiprocessing.cpu_count()/2)-1)

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nCores)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nCores)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


###########################
#--------------mygaus
def gauss(x,x0,sigma,norm=True):
        if norm:
            a=1/(sigma*np.sqrt(2*np.pi))
        else:
            a=1
        return a*np.exp(-(x-x0)**2/(2*sigma**2))  
###########################

    
def get_SchParams(Lstar, phiStar, h0):
        '''
        Input: Hubble parameter h0, values of Lstar, phiStar for h0=0.7
        Output: Schechter function parameters L_*, phi_* rescaled by h0
        '''
        #Lstar = Lstar*(h0/0.7)**(-2)
        #phiStar = phiStar*(h0/0.7)**(3)
        Lstar = Lstar*(h0/pivot)**(-2)
        phiStar = phiStar*(h0/pivot)**(3)
        return Lstar, phiStar



def get_SchNorm(phistar, Lstar, alpha, Lcut):
        '''
        
        Input:  - Schechter function parameters L_*, phi_*, alpha
                - Lilit of integration L_cut in units of 10^10 solar lum.
        
        Output: integrated Schechter function up to L_cut in units of 10^10 solar lum.
        '''
        from scipy.special import gammaincc
        from scipy.special import gamma
                
        norm= phistar*Lstar*gamma(alpha+2)*gammaincc(alpha+2, Lcut)
        return norm



def ra_dec_from_th_phi(theta, phi):
        ra = np.rad2deg(phi)
        dec = np.rad2deg(0.5 * np.pi - theta)
        return ra, dec

  
def th_phi_from_ra_dec(ra, dec):
    theta = 0.5 * np.pi - np.deg2rad(dec)
    phi = np.deg2rad(ra)
    return theta, phi

cosmo70GLOB = FlatLambdaCDM(H0=scale, Om0=Om0GLOB)

def dLGW(z, H0, Xi0, n):
    '''
    Modified GW luminosity distance
    '''
    cosmo=FlatLambdaCDM(H0=H0, Om0=Om0GLOB)
    return (cosmo.luminosity_distance(z).value)*Xi(z, Xi0, n=n) 
    
def Xi(z, Xi0, n):

    return Xi0+(1-Xi0)/(1+z)**n


zGridGLOB = np.logspace(start=-10, stop=5, base=10, num=1000)
dLGridGLOB = cosmo70GLOB.luminosity_distance(zGridGLOB).value
dcomGridGLOB = cosmo70GLOB.comoving_distance(zGridGLOB).value
HGridGLOB = cosmo70GLOB.H(zGridGLOB).value
from scipy import interpolate
dcom70fast = interpolate.interp1d(zGridGLOB, dcomGridGLOB, kind='cubic', bounds_error=False, fill_value=(0, np.NaN), assume_sorted=True)
dL70fast = interpolate.interp1d(zGridGLOB, dLGridGLOB, kind='cubic', bounds_error=False, fill_value=(0 ,np.NaN), assume_sorted=True)

H70fast = interpolate.interp1d(zGridGLOB, HGridGLOB, kind='cubic', bounds_error=False, fill_value=(scale ,np.NaN), assume_sorted=True)

#Raul: changed 70 into 67
def z_from_dLGW_fast(r, H0, Xi0, n):
    from scipy import interpolate
    #z2dL = interpolate.interp1d(dLGridGLOB/H0*70*Xi(zGridGLOB, Xi0, n=n), zGridGLOB, kind='cubic', bounds_error=False, fill_value=(0,np.NaN), assume_sorted=True)
    z2dL = interpolate.interp1d(dLGridGLOB/H0*scale*Xi(zGridGLOB, Xi0, n=n), zGridGLOB, kind='cubic', bounds_error=False, fill_value=(0,np.NaN), assume_sorted=True)
    return z2dL(r)




def z_from_dLGW(dL_GW_val, H0, Xi0, n):
    '''
    Returns redshift for a given luminosity distance dL_GW_val (in Mpc)
    
    Input:
        - dL_GW_val luminosity dist in Mpc
        - H0
        - Xi0: float. Value of Xi_0
        - n: float. Value of n

    '''   
    from scipy.optimize import fsolve
    #print(cosmo.H0)
    func = lambda z : dLGW(z, H0, Xi0, n=n) - dL_GW_val
    #z = fsolve(func, 0.77)
    z = fsolve(func, 0.77)
    return z[0]

def dVdcom_dVdLGW(z, H0, Xi0, n):
# D_com^2 d D_com = D_com^2 (d D_com/d D_L^{gw}) d D_L^{gw}

# d D_com / d D_L^{gw} = d D_com /dz * ( d D_L^{gw} / dz ) ^(-1)
# [with D_L^{gw} = (Xi0 + (1-Xi0)(1+z)**(-n)) (1+z) Dcom ]
# = c/H(z) * (  (Xi0 + (1-n) (1-Xi0)(1+z)**(-n)  ) D_com + (Xi0 + (1-Xi0)(1+z)**(-n)) (1+z) c/H(z)  )^(-1)
# = (  (Xi0 + (1-n) (1-Xi0)(1+z)**(-n)  ) D_com H /c + (Xi0 + (1-Xi0)(1+z)**(-n)) (1+z)  )^(-1)

# D_com^2 / D_L^{gw}^2 remains

    h7 = H0 / scale
    
    #dcom = cosmo70GLOB.comoving_distance(z).value/h7

    #H = cosmo70GLOB.H(z).value*h7
    
    dcom = dcom70fast(z) / h7
    H  = H70fast(z) * h7
    
    dLGWsq_over_dcomsq = ((1+z)*Xi(z, Xi0=Xi0, n=n))**2
    
    jac = 1 / (H*(Xi0 + (1-n)*(1-Xi0)*(1+z)**(-n))*dcom/clight + (Xi0+(1-Xi0)*(1+z)**(-n))*(1+z) )
    
    jac /= dLGWsq_over_dcomsq
    
    return jac


def s(z, Xi0, n):
    return (1+z)*Xi(z, Xi0, n)


def sPrime(z, Xi0, n):
    return Xi(z, Xi0, n)-n*(1-Xi0)/(1+z)**n

def j(z):
    '''
    Dimensioneless Jacobian of comoving volume. Does not depend on H0
    '''
    return cosmoglob.differential_comoving_volume(z).value*(cosmoglob.H0.value/clight)**3


def uu(z):
    '''
    Dimensionless comoving distance. Does not depend on H0
    '''
    return scale/clight*FlatLambdaCDM(H0=scale, Om0=Om0GLOB).comoving_distance(z).value

def ddL_dz(z, H0, Xi0, n):
    '''
    Jacobian d(DL)/dz
    '''
    return (sPrime(z, Xi0, n)*uu(z)+s(z, Xi0, n)/E(z))*(clight/H0)

def E(z):
    '''
    E(z). Does not depend on H0
    '''
    return FlatLambdaCDM(H0=scale, Om0=Om0GLOB).efunc(z)





class Logger(object):
    
    def __init__(self, fname):
        self.terminal = sys.__stdout__
        self.log = open(fname, "w+")
        self.log.write('--------- LOG FILE ---------\n')
        print('Logger created log file: %s' %fname)
        #self.write('Logger')
       
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

    def close(self):
        self.log.close
        sys.stdout = sys.__stdout__
        
    def isatty(self):
        return False
        
        
def hpx_downgrade_idx(hpx_array, nside_out=1024):
    #Computes the list of explored indices in a hpx array for the chosen nside_out
    arr_down = hp.ud_grade(hpx_array, nside_out)
    return np.where(arr_down>0.)[0] 


def hav(theta):
    return (np.sin(theta/2))**2
def haversine(phi, theta, phi0, theta0):
    return np.arccos(1 - 2*(hav(theta-theta0)+hav(phi-phi0)*np.sin(theta)*np.sin(theta0)))


def get_norm_posterior(lik_inhom, lik_hom, beta, grid, prior=None):
    tot_post=(lik_inhom+lik_hom)/beta
    #mytot_post =(lik_inhom+lik_hom)/beta
    #maxima,_= find_peaks(mytot_post,height=0)
    #if len(maxima)>1:
    #    first_index=maxima[0]
    #    last_index=maxima[-1]
    #    mymin=np.min(mytot_post[first_index:last_index])
    #    tot_post=np.where(mytot_post>mymin, mytot_post ,mymin)
    #else:
    #    tot_post=mytot_post
    if prior is not None:
        tot_post*=prior 
        
    norm = np.trapz( tot_post, grid)
    
    post =tot_post/norm
    return post, lik_inhom/beta/norm, lik_hom/beta/norm

#-----------------Raul:Fancy stuff----------------
    #RC: Homogeneous Malmquist effect. not a refined implementation but still    
def malm_homogen(d,err):
    #exponent=(err**2)*(7/2)
    #ret=1*np.exp(-exponent)
    ret=d/err
    return ret

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
#ratex=np.loadtxt('/home/rciancarella/DarkSirensStat/DSCatalogueCreator/zz_comovingrate.txt')
#ratey=np.loadtxt('/home/rciancarella/DarkSirensStat/DSCatalogueCreator/comovingrate.txt')
myrate=1#interpolate.interp1d(ratex,ratey,kind='cubic',fill_value='extrapolate')

    
#zz=np.loadtxt('/home/rciancarella/DarkSirensStat/data/GLADE/myzz.txt')
#ww=np.loadtxt('/home/rciancarella/DarkSirensStat/data/GLADE/myweights.txt')
#----------------------------------------------------------------------------------
#zz=np.loadtxt('/home/rciancarella/DarkSirensStat/data/GLADE/myzz_autoconsistent.txt')
#ww=np.loadtxt('/home/rciancarella/DarkSirensStat/data/GLADE/myweights_autoconsistent.txt')
#-----------------------------------------------------------------------------------
#zz=np.loadtxt('/home/rciancarella/DarkSirensStat/data/GLADE/myzz_autoconsistent_halved.txt')
#ww=np.loadtxt('/home/rciancarella/DarkSirensStat/data/GLADE/myweights_autoconsistent_halved.txt')
#-----------------------------------------------------------------------------------
#zz=np.loadtxt('/home/rciancarella/DarkSirensStat/data/GLADE/myzz_autoconsistent_halved_10bins.txt')
#ww=np.loadtxt('/home/rciancarella/DarkSirensStat/data/GLADE/myweights_autoconsistent_halved_10bins.txt')
#-------------------catgw23_run30-----------------------------------------------------
statw=0
#zz=np.loadtxt('/home/rciancarella/DarkSirensStat/data/GLADE/catgw23_run30_z.txt')
#if statw==1:
#    ww=np.loadtxt('/home/rciancarella/DarkSirensStat/data/GLADE/catgw23_run30_w.txt')
#else:
#    ww=np.ones(len(zz))
#stat_weights=interpolate.interp1d(zz,ww,kind='cubic',fill_value='extrapolate')