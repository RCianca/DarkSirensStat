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

# Global variables for shared access
complete_hostcat_shared_mem = None
complete_hostcat_shape = None
complete_hostcat_columns = None

target_hostcat_shared_mem = None
target_hostcat_shape = None
target_hostcat_columns = None

def init_worker(complete_shared_name, complete_shape, complete_columns, 
                target_shared_name, target_shape, target_columns):
    """Initialize the worker with shared memory references."""
    global complete_hostcat_shared_mem, complete_hostcat_shape, complete_hostcat_columns
    global target_hostcat_shared_mem, target_hostcat_shape, target_hostcat_columns
    
    complete_hostcat_shared_mem = shared_memory.SharedMemory(name=complete_shared_name)
    complete_hostcat_shape = complete_shape
    complete_hostcat_columns = complete_columns
    
    target_hostcat_shared_mem = shared_memory.SharedMemory(name=target_shared_name)
    target_hostcat_shape = target_shape
    target_hostcat_columns = target_columns

def process_event(k):
    """Process a single event and find matches in the complete catalog."""
    global complete_hostcat_shared_mem, complete_hostcat_shape, complete_hostcat_columns
    global target_hostcat_shared_mem, target_hostcat_shape, target_hostcat_columns

    DS_dl = Allevents_DS.iloc[k]['dL'] * 1000
    DS_theta = Allevents_DS.iloc[k]['theta']
    DS_phi = Allevents_DS.iloc[k]['phi']

    # Create DataFrame from shared memory for complete catalog
    complete_hostcat_array = np.ndarray(complete_hostcat_shape, dtype=np.float64, 
                                        buffer=complete_hostcat_shared_mem.buf)
    complete_hostcat = pd.DataFrame(complete_hostcat_array, columns=complete_hostcat_columns)

    # Create DataFrame from shared memory for target catalog
    target_hostcat_array = np.ndarray(target_hostcat_shape, dtype=np.float64, 
                                      buffer=target_hostcat_shared_mem.buf)
    target_hostcat = pd.DataFrame(target_hostcat_array, columns=target_hostcat_columns)

    # Find matches in complete catalog
    matched_rows = complete_hostcat[
        np.isclose(complete_hostcat['Luminosity Distance'], DS_dl, atol=1e-6) &  
        np.isclose(complete_hostcat['theta'], DS_theta, atol=1e-6) &
        np.isclose(complete_hostcat['phi'], DS_phi, atol=1e-6)
    ]
    
    if matched_rows.empty:
        print(f"Warning: No match found for event {k} (dL={DS_dl}, theta={DS_theta}, phi={DS_phi})")
        return None
    
    # Check which matched rows are not already in target catalog
    rows_to_add = []
    for _, row in matched_rows.iterrows():
        # Check if this row already exists in target catalog
        exists_in_target = False
        for _, target_row in target_hostcat.iterrows():
            if (np.isclose(row['Luminosity Distance'], target_row['Luminosity Distance'], atol=1e-5) and
                np.isclose(row['theta'], target_row['theta'], atol=1e-5) and
                np.isclose(row['phi'], target_row['phi'], atol=1e-5)):
                exists_in_target = True
                break
        
        if not exists_in_target:
            rows_to_add.append(row)
    
    return pd.DataFrame(rows_to_add) if rows_to_add else None

if __name__ == '__main__':
    folder = 'Uniform/TestRun03/'
    COV_SAVE_PATH = '/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Events/' + folder
    
    # Define catalog paths
    complete_cat_input_path = '/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Catalogues/GalaxyCatalogue/Uniform/'
    target_cat_input_path = '/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Catalogues/GalaxyCatalogue/Uniform_nflag/'
    output_path = '/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Catalogues/GalaxyCatalogue/Uniform_nflag/'
    output_cat_name = 'Unif_nz_flag_filled.txt'

    print(f'Using {multiprocessing.cpu_count()} CPUs')

    # Read the complete catalog
    print('Reading Complete Galaxy Catalogue')
    complete_catalog_name = 'Uniform_paper.txt'
    complete_catalog_path = os.path.join(complete_cat_input_path, complete_catalog_name)
    complete_hostcat = GalCat(complete_catalog_name,nside=128,absolute_path=os.path.join(complete_cat_input_path, complete_catalog_name)).read_catalogue()
    print(f'Number of hosts in complete catalog: {complete_hostcat.shape[0]}')

    # Read the target catalog
    print('Reading Target Galaxy Catalogue')
    target_catalog_name = 'Unif_nz_flag.txt'
    target_catalog_path = os.path.join(target_cat_input_path, target_catalog_name)
    
    if os.path.exists(target_catalog_path):
        #target_hostcat = pd.read_csv(target_catalog_path)
        target_hostcat = GalCat(target_catalog_name,nside=128,absolute_path=target_catalog_path).read_catalogue()
        print(f'Number of hosts in target catalog: {target_hostcat.shape[0]}')
    else:
        print(f'Target catalog {target_catalog_path} does not exist. Creating new empty catalog.')
        target_hostcat = pd.DataFrame(columns=complete_hostcat.columns)

    # Load DS events
    Allevents_DS = pd.DataFrame()
    for file_name in glob.glob(os.path.join(COV_SAVE_PATH, 'SNR_more_than_100_*.h5')):
        tosave = load_population(file_name)
        tmp = pd.DataFrame.from_dict(tosave, orient='columns')
        Allevents_DS = pd.concat([Allevents_DS, tmp], ignore_index=True)
    
    print(f'Number of DS events: {Allevents_DS.shape[0]}')
    selected = np.arange(0, Allevents_DS.shape[0])

    # Convert catalogs to numpy arrays for shared memory
    complete_hostcat_array = complete_hostcat.to_numpy(dtype=np.float64)
    complete_hostcat_columns = list(complete_hostcat.columns)
    complete_hostcat_shape = complete_hostcat_array.shape

    # Ensure target catalog has the same columns as complete catalog
    for col in complete_hostcat.columns:
        if col not in target_hostcat.columns:
            target_hostcat[col] = pd.Series(dtype=np.float64)
    
    target_hostcat_array = target_hostcat.to_numpy(dtype=np.float64)
    target_hostcat_columns = list(target_hostcat.columns)
    target_hostcat_shape = target_hostcat_array.shape

    # Create shared memory for both catalogs
    complete_shm = shared_memory.SharedMemory(create=True, size=complete_hostcat_array.nbytes)
    shared_complete_hostcat = np.ndarray(complete_hostcat_shape, dtype=np.float64, buffer=complete_shm.buf)
    shared_complete_hostcat[:] = complete_hostcat_array[:]

    target_shm = shared_memory.SharedMemory(create=True, size=target_hostcat_array.nbytes)
    shared_target_hostcat = np.ndarray(target_hostcat_shape, dtype=np.float64, buffer=target_shm.buf)
    shared_target_hostcat[:] = target_hostcat_array[:]

    # Process events in parallel
    with multiprocessing.Pool(
        processes=multiprocessing.cpu_count(), 
        initializer=init_worker, 
        initargs=(complete_shm.name, complete_hostcat_shape, complete_hostcat_columns,
                 target_shm.name, target_hostcat_shape, target_hostcat_columns)
    ) as pool:
        results = pool.map(process_event, selected)

    # Clean up shared memory
    complete_shm.close()
    complete_shm.unlink()
    target_shm.close()
    target_shm.unlink()

    # Combine non-None results
    rows_to_add = [df for df in results if df is not None]
    if rows_to_add:
        rows_to_add_df = pd.concat(rows_to_add, ignore_index=True)
        print(f"Found {rows_to_add_df.shape[0]} new rows to add to target catalog")
        
        # Add to target catalog
        updated_target_catalog = pd.concat([target_hostcat, rows_to_add_df], ignore_index=True)
        print(f"Updated target catalog now has {updated_target_catalog.shape[0]} rows")
        
        # Save updated target catalog
        output_catalog_path = os.path.join(output_path, output_cat_name)
        updated_target_catalog.to_csv(output_catalog_path, index=False)
        print(f'Updated target catalog saved to {output_catalog_path}')
        
        # Verify all matches were added correctly
        print("Verifying all matches were added to target catalog...")
        original_target_size = target_hostcat.shape[0]
        new_target_size = updated_target_catalog.shape[0]
        expected_new_rows = sum(df.shape[0] for df in rows_to_add if df is not None)
        
        if new_target_size - original_target_size == expected_new_rows:
            print("Verification successful: All matched rows were added correctly.")
        else:
            print(f"Verification failed: Expected to add {expected_new_rows} rows, but added {new_target_size - original_target_size}.")
    else:
        print("No new rows to add to target catalog")