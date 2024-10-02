import numpy as np
import pandas as pd
import healpy as hp

import matplotlib.pyplot as plt


from scipy import integrate
from scipy import interpolate
from scipy.optimize import fsolve


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

href=67 #69
Om0GLOB=0.319
Xi0Glob =1.
clight = 2.99792458* 10**5#km/s
cosmoflag = FlatLambdaCDM(H0=href, Om0=Om0GLOB)

def UniformBeta(hostcat_filtered,pix_selected,H0Grid):
    beta=np.ones(len(H0Grid))
    z_hosts=[]
    for i, pix in enumerate(pix_selected):
        pixel_galaxies = hostcat_filtered[hostcat_filtered['Pixel'] == pix]
        z_hosts.append(pixel_galaxies['z'].values)

    z_hosts = np.hstack(z_hosts)
    zh_min=np.min(z_hosts)
    zh_max=np.max(z_hosts)
    for j,h in enumerate(H0Grid):
        cosmo=FlatLambdaCDM(H0=h, Om0=Om0GLOB)
        integrand=lambda x:clight*(cosmo.comoving_distance(x).value)**2/(cosmo.H(x).value)
        num=integrate.quad(integrand,zh_min,zh_max)[0]
        integrand=lambda x:clight*(cosmo.comoving_distance(x).value)**2/(cosmo.H(x).value)
        norm=integrate.quad(integrand,0,20)[0]
        beta[j]=num/norm 
    return beta


def UniformBeta_pixel(hostcat_filtered,pix_selected,nside,H0Grid):
    beta=np.ones(len(H0Grid))
    for j,h in enumerate(H0Grid):
        cosmo=FlatLambdaCDM(H0=h, Om0=Om0GLOB)
        for i, pix in enumerate(pix_selected):
            pixel_galaxies = hostcat_filtered[hostcat_filtered['Pixel'] == pix]
            z_hosts = np.asarray(pixel_galaxies['z'])
            zh_min=np.min(z_hosts)
            zh_max=np.max(z_hosts)
            integrand=lambda x:clight*(cosmo.comoving_distance(x).value)**2/(cosmo.H(x).value)
            num=integrate.quad(integrand,zh_min,zh_max)[0]
            integrand=lambda x:clight*(cosmo.comoving_distance(x).value)**2/(cosmo.H(x).value)
            norm=integrate.quad(integrand,0,20)[0]
            allskyvol=num/norm

            beta[j]=allskyvol/hp.nside2npix(nside)
    return beta

###############################Fare anche in dl con le sigma per ogni pix#####################################
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

def UniformBeta_pixel_dl(allmu,allsigma,zmin_cat,zmax_cat,pix_selected,nside,H0Grid):
    beta=np.ones(len(H0Grid))

    for j,h in enumerate(H0Grid):
        cosmo=FlatLambdaCDM(H0=h, Om0=Om0GLOB)
        for i, pix in enumerate(pix_selected):
            how_many_sigma=3.5
            mu=allmu[pix]
            s=allsigma[pix]
            func = lambda z :Dl_z(z, h, Om0GLOB) - (mu+s*how_many_sigma*mu)
            zMax = fsolve(func, 0.02)[0] 
            func = lambda z :Dl_z(z, h, Om0GLOB) - (mu-s*how_many_sigma*mu)
            zmin = fsolve(func, 0.02)[0]
            zMax=min(zMax,zmax_cat)
            zmin=max(zmin,zmin_cat)

            integrand=lambda x:clight*(cosmo.comoving_distance(x).value)**2/(cosmo.H(x).value)
            num=integrate.quad(integrand,zmin,zMax)[0]
            integrand=lambda x:clight*(cosmo.comoving_distance(x).value)**2/(cosmo.H(x).value)
            norm=integrate.quad(integrand,0,20)[0]
            allskyvol=num/norm

            beta[j]=allskyvol/hp.nside2npix(nside)
    return beta

if __name__=='__main__':
    from GalaxyCat import GalCat
    from SkyMap import GWskymap


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


    print('Reading Galaxy Catalogue')
    #reading the catalogue and selecting the pixel
    to_read='Uniform_paper.txt'
    hostcat=GalCat(to_read,nside).read_catalogue()
    print('Reading catalogue completed')
    mypixels=GalCat(to_read,nside).pixelizer()
    hostcat['Pixel']=mypixels
    mask = hostcat['Pixel'].isin(pix_selected)
    hostcat_filtered=hostcat[mask]

    zmin_cat=np.min(hostcat_filtered['z'])
    zmax_cat=np.max(hostcat_filtered['z'])


    print('computing betas')
    H0min=40#30#55
    H0max=100#140#85
    H0Grid=np.linspace(H0min,H0max,1000)

    betaUnif=UniformBeta(hostcat_filtered,pix_selected,H0Grid)
    betaUnif_pix=UniformBeta_pixel(hostcat_filtered,pix_selected,nside,H0Grid)
    betaUnif_pix_dl=UniformBeta_pixel_dl(allmu,allsigma,zmin_cat,zmax_cat,pix_selected,nside,H0Grid)

    np.save('betaUnif',betaUnif)
    np.save('betaUnif_pix',betaUnif_pix)
    np.save('betaUnif_pix_dl',betaUnif_pix_dl)

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
    ax.plot(x,betaUnif,label='betaUnif',color=Mycol,linewidth=4,linestyle='solid')
    ax.legend(fontsize=13, ncol=2) 

    plotpath=os.path.join('betaUnif.pdf')
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
    ax.plot(x,betaUnif_pix,label='betaUnif_pix',color=Mycol,linewidth=4,linestyle='solid')
    ax.legend(fontsize=13, ncol=2) 

    plotpath=os.path.join('betaUnif_pix.pdf')
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
    ax.plot(x,betaUnif_pix_dl,label='betaUnif_pix_dl',color=Mycol,linewidth=4,linestyle='solid')
    ax.legend(fontsize=13, ncol=2) 

    plotpath=os.path.join('betaUnif_pix_dl.pdf')
    plt.savefig(plotpath, format="pdf", bbox_inches="tight")