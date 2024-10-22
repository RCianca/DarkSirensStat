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

from multiprocessing import Pool, Array
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

def z_from_dL(dL_val):
    '''
    Returns redshift for a given luminosity distance dL (in Mpc)'''
    
    func = lambda z :cosmoflag.luminosity_distance(z).value - dL_val
    z = fsolve(func, 0.02)
    return z[0]

def compute_area(nside,all_pixels,p_posterior,level=0.99):

    
    ''' Area of level% credible region, in square degrees.
        If level is not specified, uses current selection '''
    pixarea=hp.nside2pixarea(nside)
    return get_credible_region_pixels(all_pixels,p_posterior,level=level).size*pixarea*(180/np.pi)**2


def _get_credible_region_pth(p_posterior,level=0.99):
    '''
    Finds value minskypdf of rho_i that bouds the x% credible region , with x=level
    Then to select pixels in that region: self.all_pixels[self.p_posterior>minskypdf]
    '''
    prob_sorted = np.sort(p_posterior)[::-1]
    prob_sorted_cum = np.cumsum(prob_sorted)
    # find index of array which bounds the self.area confidence interval
    idx = np.searchsorted(prob_sorted_cum, level)
    minskypdf = prob_sorted[idx] #*skymap.npix

    #self.p[self.p]  >= minskypdf       
    return minskypdf

def get_credible_region_pixels(all_pixels, p_posterior, level=0.99):

    return all_pixels[p_posterior>_get_credible_region_pth(p_posterior,level=level)]
