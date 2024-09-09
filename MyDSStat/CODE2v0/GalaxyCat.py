import pandas as pd
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

import os
import sys

from tqdm import tqdm

import h5py
from multiprocessing import Pool
import pickle
from numba import jit


H0GLOB=67 #69
Om0GLOB=0.319
Xi0Glob =1.
clight = 2.99792458* 10**5#km/s
cosmoflag = FlatLambdaCDM(H0=H0GLOB, Om0=Om0GLOB)



class GalCat:
    """docstring for ClassName"""
    def __init__(self,catname, nside):
        self.catname = catname
        self.catname_noext=catname.split('.')[0]
        self.catpath=os.path.join(os.getcwd(),'Catalogues/GalaxyCatalogue/Uniform',catname)
        self.maskpath=os.path.join(os.getcwd(),'Catalogues/GalaxyCatalogue/Uniform')
        self.maskname=self.catname_noext+str(nside)+'.npy'
        self.nside=nside
        self.prevpath='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/Uniform_paper.txt'


    def read_catalogue(self):
        print("Reading host catalogue {}".format(self.catname))
        hostcat=pd.read_csv(self.prevpath)#will be catpath
        colnames = ['Ngal', 'Comoving Distance', 'Luminosity Distance', 'z', 'phi', 'theta']
        hostcat.columns=colnames
        return hostcat
    def pixelizer(self):
        if not os.path.exists(os.path.join(self.maskpath,self.maskname)):
            print("Generating pixel mask for host catalogue {}".format(self.catname))
            hostcat=pd.read_csv(self.prevpath)#will be catpath
            colnames = ['Ngal', 'Comoving Distance', 'Luminosity Distance', 'z', 'phi', 'theta']
            hostcat.columns=colnames
            print('showing head of {}'.format(self.catname))
            print(hostcat.head(5))
            Alltheta=hostcat['theta']
            Allphi=hostcat['phi']
            Allpixels=hp.ang2pix(self.nside,Alltheta,Allphi)
            print(len(Allpixels),len(Alltheta))
            np.save(os.path.join(self.maskpath,self.maskname),Allpixels)
            print('Pixel mask saved as {} in folder {}'.format(self.maskname,self.maskpath))

        else:
            print('Loading pixel mask for host catalogue {}'.format(self.prevpath))
            Allpixels=np.load(os.path.join(self.maskpath,self.maskname))
        return Allpixels


if __name__=='__main__':
    #work in progress
    to_read='Uniform_paper.txt'
    #mycat=GalCat(to_read,64)
    mypixels=GalCat(to_read,64).pixelizer()
