import numpy as np
import pandas as pd
import healpy as hp


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
        z_hosts.append(pixel_galaxies['z'])

    z_hosts=np.asarray(z_hosts)
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