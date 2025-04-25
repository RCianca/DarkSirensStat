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
    """
    Class for handling galaxy catalogs with support for both relative and absolute paths.
    
    Parameters:
    -----------
    catname : str
        Catalog filename
    nside : int, optional
        HEALPix nside parameter (default: None, which becomes 128)
    absolute_path : str, optional
        Absolute path to the catalog file. If provided, this path will be used directly.
        If None (default), the path will be constructed using os.getcwd() and standard folders.
    """
    def __init__(self, catname, nside=None, absolute_path=None):
        self.catname = catname
        self.catname_noext = catname.split('.')[0]
        
        # Handle absolute path case
        if absolute_path is not None:
            if os.path.isdir(absolute_path):
                # If absolute_path is a directory, join it with catname
                self.catpath = os.path.join(absolute_path, catname)
                self.maskpath = absolute_path
            elif os.path.isfile(absolute_path):
                # If absolute_path is a file (complete path to catalog)
                self.catpath = absolute_path
                self.maskpath = os.path.dirname(absolute_path)
                # Update catname if necessary
                if os.path.basename(absolute_path) != catname:
                    self.catname = os.path.basename(absolute_path)
                    self.catname_noext = self.catname.split('.')[0]
        else:
            # Original behavior - use relative paths from current directory
            self.catpath = os.path.join(os.getcwd(), 'Catalogues/GalaxyCatalogue/Uniform', catname)
            self.maskpath = os.path.join(os.getcwd(), 'Catalogues/GalaxyCatalogue/Uniform')
        
        self.maskname = self.catname_noext + '_' + str(nside if nside is not None else 128) + '.npy'
        self.nside = 128 if nside is None else nside

    def read_catalogue(self):
        """Read and return the galaxy catalog as a pandas DataFrame."""
        print(f"Reading host catalogue {self.catname} from {self.catpath}")
        hostcat = pd.read_csv(self.catpath)
        colnames = ['Ngal', 'Comoving Distance', 'Luminosity Distance', 'z', 'phi', 'theta']
        hostcat.columns = colnames
        return hostcat
        
    def pixelizer(self):
        """Create or load a pixel mask for the galaxy catalog."""
        mask_full_path = os.path.join(self.maskpath, self.maskname)
        
        if not os.path.exists(mask_full_path):
            print(f"Generating pixel mask for host catalogue {self.catname}")
            hostcat = pd.read_csv(self.catpath)
            colnames = ['Ngal', 'Comoving Distance', 'Luminosity Distance', 'z', 'phi', 'theta']
            hostcat.columns = colnames
            print(f'Showing head of {self.catname}')
            print(hostcat.head(3))
            Alltheta = hostcat['theta'].to_numpy()
            Allphi = hostcat['phi'].to_numpy()
            Allpixels = hp.ang2pix(self.nside, Alltheta, Allphi)
            print(f'{len(Allpixels)}, {len(Alltheta)}')
            np.save(mask_full_path, Allpixels)
            print(f'Pixel mask saved as {self.maskname} in folder {self.maskpath}')
        else:
            print(f'Loading pixel mask for host catalogue {self.catname}')
            Allpixels = np.load(mask_full_path)
        
        return Allpixels


if __name__ == '__main__':
    # Example usage with relative path (original behavior)
    to_read = 'Uniform_paper.txt'
    mypixels = GalCat(to_read, 128).pixelizer()
    
    # Example usage with absolute path
    # absolute_path = '/path/to/your/catalog/Uniform_paper.txt'
    # mypixels = GalCat('Uniform_paper.txt', 128, absolute_path=absolute_path).pixelizer()
    
    # Or directly use the full path as absolute_path
    # mypixels = GalCat('', 128, absolute_path=absolute_path).pixelizer()