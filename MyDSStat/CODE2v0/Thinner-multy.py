import pandas as pd
import numpy as np
import multiprocessing
import os
import glob

import gwfast
from gwfast.gwfastUtils import load_population

from GalaxyCat import GalCat
from Global import *

from multiprocessing import shared_memory

# Global variable for shared access
hostcat_shared_mem = None
hostcat_shape = None
hostcat_columns = None

def init_worker(shared_name, shape, columns):
    global hostcat_shared_mem, hostcat_shape, hostcat_columns
    hostcat_shared_mem = shared_memory.SharedMemory(name=shared_name)
    hostcat_shape = shape
    hostcat_columns = columns

def process_event(k):
    global hostcat_shared_mem, hostcat_shape, hostcat_columns

    DS_dl = Allevents_DS.iloc[k]['dL'] * 1000
    DS_theta = Allevents_DS.iloc[k]['theta']
    DS_phi = Allevents_DS.iloc[k]['phi']

    hostcat_array = np.ndarray(hostcat_shape, dtype=np.float64, buffer=hostcat_shared_mem.buf)
    hostcat = pd.DataFrame(hostcat_array, columns=hostcat_columns)

    matched_rows = hostcat[
        np.isclose(hostcat['Luminosity Distance'], DS_dl, atol=1e-5) &  
        np.isclose(hostcat['theta'], DS_theta, atol=1e-5) &
        np.isclose(hostcat['phi'], DS_phi, atol=1e-5)
    ]
    
    if matched_rows.empty:
        print(f"Warning: No match found for event {k} (dL={DS_dl}, theta={DS_theta}, phi={DS_phi})")
    
    return matched_rows

if __name__ == '__main__':
    folder = 'Uniform/TestRun03/'
    COV_SAVE_PATH = '/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Events/' + folder
    output_path = '/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Catalogues/GalaxyCatalogue/Uniform/'

    print(f'Using {multiprocessing.cpu_count()} CPUs')

    print('Reading Galaxy Catalogue')
    to_read = 'Uniform_paper.txt'
    hostcat = GalCat(to_read).read_catalogue()
    print('Number of hosts in {}'.format(to_read))

    Allevents_DS = pd.DataFrame()
    for file_name in glob.glob(os.path.join(COV_SAVE_PATH, 'SNR_more_than_100_*.h5')):
        tosave = load_population(file_name)
        tmp = pd.DataFrame.from_dict(tosave, orient='columns')
        Allevents_DS = pd.concat([Allevents_DS, tmp], ignore_index=True)

    selected = np.arange(0, Allevents_DS.shape[0])
    Host_in_cat = hostcat.shape[0]
    Density_cat = 0.00171
    Density_version1 = 0.000286
    Nhost = min(int(Host_in_cat * Density_version1 / Density_cat), Host_in_cat)

    hostcat_array = hostcat.to_numpy(dtype=np.float64)
    hostcat_columns = list(hostcat.columns)
    hostcat_shape = hostcat_array.shape

    shm = shared_memory.SharedMemory(create=True, size=hostcat_array.nbytes)
    shared_hostcat = np.ndarray(hostcat_shape, dtype=np.float64, buffer=shm.buf)
    shared_hostcat[:] = hostcat_array[:]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=init_worker, initargs=(shm.name, hostcat_shape, hostcat_columns)) as pool:
        results = pool.map(process_event, selected)

    temp_df = pd.concat(results, ignore_index=True)
    shm.close()
    shm.unlink()

    #print(f"Shape of hostcat before removal: {hostcat.shape[0]}")
    print(f"Shape of extracted entries (temp_df): {temp_df.shape[0]}")

    hostcat = hostcat.loc[~hostcat.index.isin(temp_df.index)]

    print(f"Shape of hostcat after removal: {hostcat.shape[0]}")

    Nhost = min(Nhost, hostcat.shape[0])  # Adjust Nhost to available entries

    hostcat_sampled = hostcat.sample(n=Nhost, replace=False, random_state=42)
    print(f"Shape of hostcat after dilution: {hostcat_sampled.shape[0]}")

    hostcat_sampled = pd.concat([hostcat_sampled, temp_df], ignore_index=True)
    print(f"Shape of hostcat after contact: {hostcat_sampled.shape[0]}")
        
    output_filename = 'Uniform_paper_sampled_density_of_version_one_testrun03.txt'
    hostcat_sampled.to_csv(os.path.join(output_path, output_filename), index=False)
    print(f'Sampled catalog saved to {output_path}')
