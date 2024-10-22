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
from gwfast.waveforms import IMRPhenomD,IMRPhenomHM
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

    print('using {} CPU' .format(multiprocessing.cpu_count()))

    #-----------------------load the galaxy catalogue and the GW event--------------------------------------
    print('Reading Galaxy Catalogue')
    #reading the catalogue and selecting the pixel
    to_read='Uniform_paper.txt'
    hostcat=GalCat(to_read,nside).read_catalogue()
    print(hostcat.columns)
    Population='SNR_more_than_100_200.h5'
    tosave=load_population(COV_SAVE_PATH+Population)
    Allevents_DS = pd.DataFrame.from_dict(tosave, orient='columns')
    selected=52
    Allevents_DS=Allevents_DS.iloc[selected]
    print(Allevents_DS.columns)
    #---------------------------------------------------------------------------------------
    DS_host=hostcat[hostcat['z']==Allevents_DS['z']]
    DS_host=DS_host[DS_host['theta']==Allevents_DS['theta']]
    DS_host=DS_host[DS_host['phi']==Allevents_DS['phi']]
    print(DS_host.shape)