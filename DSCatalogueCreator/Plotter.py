import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy import integrate
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.integrate import trapz

import pandas as pd

from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

from ligo.skymap.io import fits
#from astropy.io import fits
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
##################################################################################
#-----------rejection-stuff------------
def sample(g,xmin,xmax):
    x = np.linspace(xmin,xmax,1000000)
    y = g(x)                        # probability density function, pdf
    cdf_y = np.cumsum(y)            # cumulative distribution function, cdf
    cdf_y = cdf_y/cdf_y.max()       # takes care of normalizing cdf to 1.0
    inverse_cdf = interpolate.interp1d(cdf_y,x,fill_value="extrapolate")# this is a function
    #inverse_cdf = np.interp(0,cdf_y,x) #this works but it is not a function
    return inverse_cdf
def return_samples(f,xmin,xmax,N=1000000):
    # let's generate some samples according to the chosen pdf, f(x)
    uniform_samples = np.random.random(int(N))       
    required_samples = sample(f,xmin,xmax)(uniform_samples)
    return required_samples

def S(x, m_min, dm):
    x = np.asarray(x)
    s = np.ones_like(x)
    s[x < m_min] = 0
    mask = (m_min <= x) & (x < m_min + dm)
    
    exp_values = dm / (x[mask] - m_min) + dm / (x[mask] - m_min - dm)
    
    # Avoid overflow by capping exp_values
    large_mask = exp_values > 600  # Value beyond which exp() will overflow
    safe_exp_values = np.where(large_mask, 600, exp_values)
    
    s[mask] = (np.exp(safe_exp_values) + 1) ** (-1)
    return s

def tonorm(x,alpha):
    return x**(-alpha)

def PowerLawPlusPeak(m1, m_min, m_max, lamb, alpha, mu, sigma_m, dm):
    c=(alpha-1)*m_min**(alpha-1)
    powlaw = (1 - lamb) * c*(m1 ** (-alpha))
    gauss = lamb * np.exp(-(m1 - mu) ** 2 / (2 * sigma_m ** 2)) / (np.sqrt(2 * np.pi) * sigma_m)
    tmp = S(m1, m_min, dm) * np.float64((powlaw + gauss))
    return tmp
###################################################################################################

mass_plott=0
qplot=0
SNR_plot=0
sky_plot=1
betaplot=0



CAT_PATH='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'
THIS_DIR=os.getcwd()
os.chdir(CAT_PATH)
DS_From_Parent = pd.read_csv('DS_From_Parent_Uniform_Complete_SNR.txt')
os.chdir(THIS_DIR)

#colnames=['Ngal','Comoving Distance','Luminosity Distance','z','phi','theta','DS','M1','M2','MC','q','cos_iota','psi','tcoal','Phicoal','chiz1','chiz2']
#DS_From_Parent.columns=colnames

# Display the first few rows of the new DataFrame to verify
print(DS_From_Parent.head())
print(DS_From_Parent.columns)



if mass_plott==1:
    m_min = 4.59#np.float64(4.59)
    m_max=86
    lamb=0.1
    alpha=2.63
    mu_m=33.07
    sigma_m=5.7
    dm = 4.82#np.float64(4.82)

    PowerLawPlusPeak_with_params = partial(PowerLawPlusPeak, m_min=m_min,
                                            m_max=m_max,
                                            lamb=lamb,
                                            alpha=alpha,
                                            mu=mu_m,
                                            sigma_m=sigma_m,
                                            dm=dm)

    m1_values=np.linspace(m_min,m_max,1000000)
    p_m1_values=PowerLawPlusPeak_with_params(m1_values)
    norm=np.trapz(p_m1_values,m1_values)
    p_m1_values=p_m1_values/norm
    fig, ax = plt.subplots(figsize=(15,10))
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.get_offset_text().set_fontsize(25)
    # Plot the histogram of the sampled probabilities
    ax.hist(DS_From_Parent['M1'], bins=100, density=True, alpha=0.6, color='orange', label='Sampled $M_1$')

    # Plot the theoretical distribution
    ax.plot(m1_values, p_m1_values, label='Theoretical $p(m_1)$', color='teal',linewidth=3)

    # Labels and title
    ax.set_xlabel('$M_1$', fontsize=15)
    ax.set_ylabel('$P(M_1)$', fontsize=15)
    plt.title('Theoretical Distribution vs Sampled Histogram', fontsize=18)

    # Set log scale
    #ax.set_yscale('log')

    # Legends
    ax.legend(loc='upper right',prop={'size': 15})

    # Show plot
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('m1_extracted_fromsave.png')

    #########################################################################ààà
    fig, ax = plt.subplots(figsize=(15,10))
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.get_offset_text().set_fontsize(25)
    # Plot the histogram of the sampled probabilities
    ax.hist(DS_From_Parent['M2'], bins=100, density=True, alpha=0.6, color='orange', label='Sampled $M_2$')

    # Plot the theoretical distribution
    #ax.plot(m1_values, p_m1_values, label='Theoretical $p(m_1)$', color='teal',linewidth=3)

    # Labels and title
    ax.set_xlabel('$M_2$', fontsize=15)
    ax.set_ylabel('$P(M_2)$', fontsize=15)
    plt.title('Sampled Histogram', fontsize=18)

    # Set log scale
    #ax.set_yscale('log')

    # Legends
    ax.legend(loc='upper right',prop={'size': 15})

    # Show plot
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('m2_extracted_fromsave.png')

    #################################################################################àà
    fig, ax = plt.subplots(figsize=(15,10))
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.get_offset_text().set_fontsize(25)
    # Plot the histogram of the sampled probabilities
    ax.hist(DS_From_Parent['MC'], bins=100, density=True, alpha=0.6, color='orange', label='Sampled $M_c$')

    # Plot the theoretical distribution
    #ax.plot(m1_values, p_m1_values, label='Theoretical $p(m_1)$', color='teal',linewidth=3)

    # Labels and title
    ax.set_xlabel('$mc$', fontsize=15)
    ax.set_ylabel('$P(M_c)$', fontsize=15)
    plt.title('Sampled Histogram', fontsize=18)

    # Set log scale
    #ax.set_yscale('log')

    # Legends
    ax.legend(loc='upper right',prop={'size': 15})

    # Show plot
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('mc_extracted_fromsave.png')

##############################################################################################################################################
if qplot==1:
    fig, ax = plt.subplots(figsize=(15,10))
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.get_offset_text().set_fontsize(25)
    # Plot the histogram of the sampled probabilities
    ax.hist(DS_From_Parent['q'], bins=100, density=True, alpha=0.6, color='orange', label='Sampled $q$')

    # Plot the theoretical distribution
    #ax.plot(m1_values, p_m1_values, label='Theoretical $p(m_1)$', color='teal',linewidth=3)

    # Labels and title
    ax.set_xlabel('$q$', fontsize=15)
    ax.set_ylabel('$P(q)$', fontsize=15)
    plt.title('Sampled Histogram', fontsize=18)

    # Set log scale
    #ax.set_yscale('log')

    # Legends
    ax.legend(loc='upper right',prop={'size': 15})

    # Show plot
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('q_extracted_fromsave.png')

###################################################################################################################################à
# CAT_PATH='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'
# THIS_DIR=os.getcwd()
# os.chdir(CAT_PATH)
# DS_From_Parent_Initial = pd.read_csv('DS_From_Parent_Uniform.txt')
# os.chdir(THIS_DIR)

# print('---------------------initial shape----------------------------------- ')
# print(DS_From_Parent_Initial.columns)
# print(DS_From_Parent_Initial.shape)
# print('Mean initial M1={},M2={},MC={}'.format(np.mean(DS_From_Parent_Initial['M1']),np.mean(DS_From_Parent_Initial['M2']),np.mean(DS_From_Parent_Initial['MC'])))
# print(DS_From_Parent_Initial['M1'][:15],DS_From_Parent_Initial['M2'][:15],DS_From_Parent_Initial['MC'][:15])
# print('---------------------shape Completed----------------------------------')
# print(DS_From_Parent.columns)
# print(DS_From_Parent.shape)
# print('Mean Completed M1={},M2={},MC={}'.format(np.mean(DS_From_Parent['M1']),np.mean(DS_From_Parent['M2']),np.mean(DS_From_Parent['MC'])))
# print(DS_From_Parent['M1'][:15],DS_From_Parent['M2'][:15],DS_From_Parent['MC'][:15])
############################################################################################################################################
if SNR_plot==1:
    total=DS_From_Parent.shape[0]
    fig, ax = plt.subplots(figsize=(15,10))
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.get_offset_text().set_fontsize(25)
    # Plot the histogram of the sampled probabilities
    ax.hist(DS_From_Parent['SNR']/total, bins=50, density=False, alpha=0.6, color='teal', label='$SNR-ET-T$')

   
    # Labels and title
    ax.set_xlabel('$SNR$', fontsize=15)
    ax.set_ylabel('$P(SNR)$', fontsize=15)
    plt.title('Distribution of Measured SNR ', fontsize=18)

    # Set log scale
    #ax.set_yscale('log')

    # Legends
    ax.legend(loc='upper right',prop={'size': 15})

    # Show plot
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('SNR_fromsave.png')
    ##################################################################################################################à
    fig, ax = plt.subplots(figsize=(15,10))
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.get_offset_text().set_fontsize(25)
    # Plot the histogram of the sampled probabilities
    ax.hist(DS_From_Parent['SNR'][DS_From_Parent['SNR']<=300]/total, bins=50, density=False, alpha=0.6, color='teal', label='$SNR-ET-T$')

   
    # Labels and title
    ax.set_xlabel('$SNR$', fontsize=15)
    ax.set_ylabel('$P(SNR)$', fontsize=15)
    plt.title('Distribution of Measured SNR-zoomed', fontsize=18)

    # Set log scale
    #ax.set_yscale('log')

    # Legends
    ax.legend(loc='upper right',prop={'size': 15})

    # Show plot
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('SNR_fromsave_zoom.png')
    ###############################################################################################################################
    fig, ax = plt.subplots(figsize=(15,10))
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.get_offset_text().set_fontsize(25)
    # Plot the histogram of the sampled probabilities
    ax.hist(DS_From_Parent['SNR'][DS_From_Parent['SNR']<=50]/total, bins=50, density=False, alpha=0.6, color='teal', label='$SNR-ET-T$')

   
    # Labels and title
    ax.set_xlabel('$SNR$', fontsize=15)
    ax.set_ylabel('$P(SNR)$', fontsize=15)
    plt.title('Distribution of Measured SNR-zoomed', fontsize=18)

    # Set log scale
    #ax.set_yscale('log')

    # Legends
    ax.legend(loc='upper right',prop={'size': 15})

    # Show plot
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('SNR_fromsave_zoom-0-50.png')
    ###############################################################################################################################
    fig, ax = plt.subplots(figsize=(15,10))
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.get_offset_text().set_fontsize(25)
    # Plot the histogram of the sampled probabilities
    ax.hist(DS_From_Parent['SNR'][(DS_From_Parent['SNR']<=100)&(DS_From_Parent['SNR']>=50)]/total, bins=50, density=False, alpha=0.6, color='teal', label='$SNR-ET-T$')

   
    # Labels and title
    ax.set_xlabel('$SNR$', fontsize=15)
    ax.set_ylabel('$P(SNR)$', fontsize=15)
    plt.title('Distribution of Measured SNR-zoomed', fontsize=18)

    # Set log scale
    #ax.set_yscale('log')

    # Legends
    ax.legend(loc='upper right',prop={'size': 15})

    # Show plot
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('SNR_fromsave_zoom-50-100.png')
    ###################################################################################################################################
    fig, ax = plt.subplots(figsize=(15,10))
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.get_offset_text().set_fontsize(25)
    # Plot the histogram of the sampled probabilities
    ax.hist(DS_From_Parent['SNR'][(DS_From_Parent['SNR']<=300)&(DS_From_Parent['SNR']>=100)]/total, bins=50, density=False, alpha=0.6, color='teal', label='$SNR-ET-T$')

   
    # Labels and title
    ax.set_xlabel('$SNR$', fontsize=15)
    ax.set_ylabel('$P(SNR)$', fontsize=15)
    plt.title('Distribution of Measured SNR-zoomed', fontsize=18)

    # Set log scale
    #ax.set_yscale('log')

    # Legends
    ax.legend(loc='upper right',prop={'size': 15})

    # Show plot
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('SNR_fromsave_zoom-100-300.png')
###################################################################################################################################################


if sky_plot == 1:
    folder = 'Uniform/TestRun00/'
    GW_data_path = '/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Events/' + folder
    fname = 'GWtest00.fits'
    print(fname)
    Data, metadata = fits.read_sky_map(GW_data_path + fname, nest=False, distances=True)
    
    sky_map = Data[0]
    mu = Data[1]
    sigma = Data[2]
    posteriorNorm = Data[3]

    # Plot the sky map
    hp.mollview(sky_map, title="Sky Map", coord=['C'], unit='Prob')
    hp.graticule()

    # Save the sky map as a PNG file
    plt.savefig(GW_data_path + "Sky_Map.png", dpi=300)
    plt.close()

##################################################################################################################################
if betaplot == 1:
    folder = 'Uniform/TestRun00/'
    GW_data_path = '/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Events/' + folder

    betaUnif = np.load(GW_data_path+'betaUnif.npy')
    betaUnif_pix = np.load(GW_data_path+'betaUnif_pix.npy')
    betaUnif_pix_dl=np.load(GW_data_path+'betaUnif_pix_dl.npy')

    H0min=40#30#55
    H0max=100#140#85
    H0Grid=np.linspace(H0min,H0max,1000)
    


    fig, ax = plt.subplots(1, figsize=(15,10)) #crea un tupla che poi è più semplice da gestire
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.get_offset_text().set_fontsize(25)
    ax.grid(linestyle='dotted', linewidth='0.6')#griglia in sfondo

    x=H0Grid
    xmin=np.min(x)
    xmax=np.max(x)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r'$H_0(Km/s/Mpc)$', fontsize=30)
    #ax.set_ylabel(r'$P(H_0)$', fontsize=20)
    ax.set_ylabel(r'$\beta (H_0)$', fontsize=30)

    Mycol='teal'
    ax.plot(x,betaUnif/np.trapz(betaUnif,x),label='betaUnif',color=Mycol,linewidth=4,linestyle='solid')
    ax.legend(fontsize=13, ncol=2) 

    plotpath=os.path.join(GW_data_path,'betaUnif.pdf')
    plt.savefig(plotpath, format="pdf", bbox_inches="tight")


    fig, ax = plt.subplots(1, figsize=(15,10)) #crea un tupla che poi è più semplice da gestire
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.get_offset_text().set_fontsize(25)
    ax.grid(linestyle='dotted', linewidth='0.6')#griglia in sfondo

    x=H0Grid
    xmin=np.min(x)
    xmax=np.max(x)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r'$H_0(Km/s/Mpc)$', fontsize=30)
    #ax.set_ylabel(r'$P(H_0)$', fontsize=20)
    ax.set_ylabel(r'$\beta (H_0)$', fontsize=30)

    Mycol='teal'
    ax.plot(x,betaUnif_pix/np.trapz(betaUnif_pix,x),label='betaUnif_pix',color=Mycol,linewidth=4,linestyle='solid')
    ax.legend(fontsize=13, ncol=2) 

    plotpath=os.path.join(GW_data_path,'betaUnif_pix.pdf')
    plt.savefig(plotpath, format="pdf", bbox_inches="tight")

    fig, ax = plt.subplots(1, figsize=(15,10)) #crea un tupla che poi è più semplice da gestire
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.get_offset_text().set_fontsize(25)
    ax.grid(linestyle='dotted', linewidth='0.6')#griglia in sfondo

    x=H0Grid
    xmin=np.min(x)
    xmax=np.max(x)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r'$H_0(Km/s/Mpc)$', fontsize=30)
    #ax.set_ylabel(r'$P(H_0)$', fontsize=20)
    ax.set_ylabel(r'$\beta (H_0)$', fontsize=30)

    Mycol='teal'
    ax.plot(x,betaUnif_pix_dl/np.trapz(betaUnif_pix_dl,x),label='betaUnif_pix_dl',color=Mycol,linewidth=4,linestyle='solid')
    ax.legend(fontsize=13, ncol=2) 

    plotpath=os.path.join(GW_data_path,'betaUnif_pix_dl.pdf')
    plt.savefig(plotpath, format="pdf", bbox_inches="tight")