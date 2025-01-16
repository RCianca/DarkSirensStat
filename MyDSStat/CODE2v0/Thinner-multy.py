import pandas as pd
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

from ligo.skymap.io import fits
import os
import sys


import gwfast.gwfastGlobals as glob
import gwfast
from gwfast.gwfastUtils import load_population

from tqdm import tqdm

import h5py
from multiprocessing import Pool
import multiprocessing
import pickle
from numba import jit

from GalaxyCat import GalCat
from Global import *

#################################################################################

def list_perm(lista,permutazione):
    tmp=[]
    for e in permutazione:
        tmp.append(lista[e])
    return tmp

def cat2parameter(args):
    catalogue, keys = args
    
    missing_columns = [key for key in keys if key not in catalogue.columns]
    if missing_columns:
        raise ValueError(f"Some keys are missing in the DataFrame: {missing_columns}")
    
    # Reorder the catalogue according to the order in keys
    catalogue_permuted = catalogue[keys]
    
    return catalogue_permuted



####################################################################################################################################
if __name__=='__main__':

    folder='Uniform/TestRun00/'
    CAT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'
    SCRIPT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/'
    COV_SAVE_PATH='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Events/'+folder
    output_path='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Catalogues/GalaxyCatalogue/Uniform/'

    print('using {} CPU' .format(multiprocessing.cpu_count()))

    #-----------------------load the galaxy catalogue and the GW event--------------------------------------
    print('Reading Galaxy Catalogue')
    #reading the catalogue and selecting the pixel
    to_read='Uniform_paper.txt'
    nside=128
    hostcat=GalCat(to_read).read_catalogue()
    Population='SNR_more_than_100_200.h5'
    tosave=load_population(COV_SAVE_PATH+Population)
    Allevents_DS = pd.DataFrame.from_dict(tosave, orient='columns')
    print(list(Allevents_DS.columns))
    selected=np.arange(0,Allevents_DS.shape[0])
    temp_df = pd.DataFrame()
# Iterate through selected indices and filter entries
    for k in selected:
        DS_dl = Allevents_DS.iloc[k]['dL'] * 1000
        DS_theta = Allevents_DS.iloc[k]['theta']
        DS_phi = Allevents_DS.iloc[k]['phi']

        print('DS info:')
        print(DS_dl, DS_theta, DS_phi)

        # Filter hostcat based on the given properties
        i = hostcat[((hostcat['Luminosity Distance'] == DS_dl) &
                     (hostcat['theta'] == DS_theta) &
                     (hostcat['phi'] == DS_phi))].index

        # Append the removed entries to the temporary DataFrame
        temp_df = pd.concat([temp_df, hostcat.loc[i]])

        # Drop the selected entries from the host catalog
        hostcat = hostcat.drop(i)

    # Sample the remaining entries in the host catalog
    hostcat_sampled = hostcat.sample(frac=0.001, replace=False, random_state=42)

    # Add back the removed entries to the sampled catalog
    hostcat_sampled = pd.concat([hostcat_sampled, temp_df], ignore_index=True)

    print('Tail of the sampled catalog:')
    print(hostcat_sampled.tail(3))
    print(hostcat_sampled.iloc[-1]['Luminosity Distance'])

    # Save the sampled catalog
    name='Uniform_paper_sampled_frac001-host.txt'
    temp_df.to_csv(os.path.join(output_path,name), index=False)
    print(f'Sampled catalog saved to {output_path}')

