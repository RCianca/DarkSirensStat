import numpy as np
import pandas as pd
import healpy as hp

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from scipy import integrate
from scipy import interpolate
from scipy.optimize import fsolve
#from scipy.special import erfc

from astropy.cosmology import FlatLambdaCDM

import os
from os import listdir
from os.path import isfile, join

from multiprocessing import Pool
import multiprocessing
import time
from numba import njit
from tqdm import tqdm
import sys
#---------------------script import------------------------------------------
from SkyMap import GWskymap
from GalaxyCat import GalCat
#-----------------------Costants-----------------------------------------
href=67 #69
Om0GLOB=0.319
Xi0Glob =1.
clight = 2.99792458* 10**5#km/s
cosmoflag = FlatLambdaCDM(H0=href, Om0=Om0GLOB)
#------------------------------------------------------------------------
#------------------Functions---------------------------------------------
def z_from_dL(dL_val):
    '''
    Returns redshift for a given luminosity distance dL (in Mpc)'''
    
    func = lambda z :cosmoflag.luminosity_distance(z).value - dL_val
    z = fsolve(func, 0.02)
    return z[0]

def E_z(z, H0, Om=Om0GLOB):
    return np.sqrt(Om*(1+z)**3+(1-Om))

def r_z(z, H0, Om=Om0GLOB):
    c = clight
    integrand = lambda x : 1/E_z(x, H0, Om)
    integral, error = integrate.quad(integrand, 0, z)
    return integral*c/H0

def Dl_z_vett(z, H0, Om=Om0GLOB):
    c = clight
    integral = np.zeros_like(z)  # Array vuoto per salvare i risultati

    for i, z_val in enumerate(z):
        integrand = lambda x: 1 / E_z(x, H0, Om)
        integral[i], error = integrate.quad(integrand, 0, z_val)
        integral[i]=integral[i]*(1+z_val)

    return integral * c / H0

def Dl_z(z, H0, Om=Om0GLOB):
    return r_z(z, H0, Om)*(1+z)

def z_from_dcom(dc_val):
    '''
    Returns redshift for a given comoving distance dc (in Mpc)'''
    
    func = lambda z :cosmoflag.comoving_distance(z).value - dc_val
    z = fsolve(func, 0.02)
    return z[0]

def h_of_z_dl(z,dl):
    func = lambda h :Dl_z(z, h, Om0GLOB) -dl
    heq = fsolve(func, 30)[0] 
    return heq


def likelihood_line(mu_DS,dl,sigma):
    norm=1/(np.sqrt(2*np.pi)*sigma)
    body=np.exp(-((dl-mu_DS)**2)/(2*sigma**2))
    ret=norm*body
    return ret
def LikeofH0_pixel(mu_DS,sigma,z_hosts,Htemp):
    #----------computing sum
    to_sum=np.zeros(len(z_hosts))
    for y in range(len(z_hosts)):
        dl = Dl_z(z_hosts[y], Htemp, Om0GLOB)
        to_sum[y]=likelihood_line(mu_DS,dl,sigma)
    
    tmp=np.sum(to_sum)
    return tmp




#########################################################################################

if __name__=='__main__':

    #######################GW-Events#####################################################
    print('Loading GW data')
    fname='GWtest00.fits'# importare la lista da un config
    working_dir=os.getcwd()
    MapPath=working_dir+'/Events/Uniform/TestRun00/'
    level=0.3
    DSs=GWskymap(MapPath+fname,level=level)
    print('test GWskymap class\nPrinting some info')
    print('DS name {}'.format(DSs.event_name))
    print('Area of DS is {} deg^2 at 90%'.format(DSs.area()))
    pix_selected=DSs.get_credible_region_pixels(level=level)
    nside=int(DSs.nside)
    print('nside is {}'.format(nside))
    skyprob=DSs.p_posterior
    allmu=DSs.mu*1000#servono in Mpc 
    allsigma=DSs.sigma*1000
    print(allmu)
    print(allsigma)
    if np.isnan(allmu).any():
        print('There are NaN in allmu')
    if np.isnan(allsigma).any():
        print('There are NaN in allsigma')
    mumean=(np.sum(allmu*skyprob))/np.sum(skyprob)
    print('mu_pesato= {} Mpc'.format(mumean))
    thetas,phis=hp.pix2ang(nside,pix_selected)
    print('DS data:')
    print('pix selected ={}'.format(len(pix_selected)))
    print('len dL={}'.format(len(allmu[pix_selected])))
    #########################Galaxy-Catalogue#############################################
    print('Reading Galaxy Catalogue')
    #reading the catalogue and selecting the pixel
    to_read='Uniform_paper.txt'
    hostcat=GalCat(to_read,nside).read_catalogue()
    print('Reading catalogue completed')
    mypixels=GalCat(to_read,nside).pixelizer()
    hostcat['Pixel']=mypixels
    mask = hostcat['Pixel'].isin(pix_selected)
    hostcat_filtered=hostcat[mask]
    #print('hostcat shape {}'.format(len(z_hosts)))
    ###Cross-Correlation#############################################
    Event_dict = {
        'Event': [],
        'Likelihood': np.array([]),
        'beta': np.array([]),
        'posterior': np.array([])
    }

    H0min=40#30#55
    H0max=100#140#85
    H0Grid=np.linspace(H0min,H0max,1000)

    
    total_post=np.zeros(len(H0Grid))# this will be the total for all the events
    single_post=np.zeros(len(H0Grid))
    #pixel_post=np.zeros(len(H0Grid))
    #for n, name in enumerate(fname)#loop sugli eventi, da fare dopo che si decide la struttura dei dati
    Event_dict['Event'].append('GWtest00')
    #for k, name in enumerate(eventlist):
    #print('shape pix_selected {}'.format(np.shape(pix_selected)))
    for i, pix in enumerate(pix_selected):
        pixel_galaxies = hostcat_filtered[hostcat_filtered['Pixel'] == pix]
        z_hosts = np.asarray(pixel_galaxies['z'])
        print('pix {}'.format(pix))
        print('hosts\n {}'.format(z_hosts))
        #z_hosts=z_hosts[-10:-1]
        if len(z_hosts) == 0:
            continue  # Skip this pixel if no galaxies are present
        pixel_post=np.zeros(len(H0Grid))
        angular_prob = skyprob[pix]#now is the same for each gal in the pix, can be computed apart and multiply
        for j, h in enumerate(H0Grid):
            #print(f'pix={pix}, mu_pix={allmu[pix]}, sigma_pix={allsigma[pix]}, angular_prob={angular_prob}')
            pixel_post[j] = LikeofH0_pixel(allmu[pix], allsigma[pix], z_hosts, h)*angular_prob
    
            # Sum the pixel posterior into the total posterior
        single_post = single_post + pixel_post
            ###Still need beta!

    #TO DO: Pensare ad un modo efficiente di salvare le cose, un dizionario dovrebbe andare. Chiavi:nome evento, posterior evento likelihood evento, beta evento
    #       Il plotter poi leggerà il dizionario e il codice deve salvare il dizionario, abbiamo visto che torna utile salvarsi ogni evento

    np.save(MapPath+'DSs_dictionary.npy', Event_dict) 

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
    ax.set_ylabel(r'$Posterior(H_0)$', fontsize=30)
    if xmin<href<xmax:
        ax.axvline(x = href, color = 'k', linestyle='dashdot',label = 'H0=67')

    Mycol='teal'
    ax.plot(x,single_post/np.trapz(single_post,x),label='Total_posterior',color=Mycol,linewidth=4,linestyle='solid')
    ax.legend(fontsize=13, ncol=2) 

    plotpath=os.path.join(MapPath+'GWtest00_lesspix.pdf')
    plt.savefig(plotpath, format="pdf", bbox_inches="tight")




    