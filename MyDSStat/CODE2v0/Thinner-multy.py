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
        np.isclose(hostcat['Luminosity Distance'], DS_dl, atol=1e-6) &  
        np.isclose(hostcat['theta'], DS_theta, atol=1e-6) &
        np.isclose(hostcat['phi'], DS_phi, atol=1e-6)
    ]
    
    if matched_rows.empty:
        print(f"Warning: No match found for event {k} (dL={DS_dl}, theta={DS_theta}, phi={DS_phi})")
    
    return matched_rows

def verify_entries_in_catalog(original_matches, final_catalog):
    """
    Verify that all matches are present in the final catalog.
    
    Args:
        original_matches (DataFrame): The matched entries that should be in the final catalog
        final_catalog (DataFrame): The final catalog to check
        
    Returns:
        bool: True if all matches are in the final catalog, False otherwise
    """
    all_found = True
    match_count = 0
    
    # Create a verification function to check if a row exists in the final catalog
    def row_exists_in_catalog(row, catalog):
        matches = catalog[
            np.isclose(catalog['Luminosity Distance'], row['Luminosity Distance'], atol=1e-5) &  
            np.isclose(catalog['theta'], row['theta'], atol=1e-5) &
            np.isclose(catalog['phi'], row['phi'], atol=1e-5)
        ]
        return not matches.empty
    
    # Check each original match
    for _, row in original_matches.iterrows():
        if row_exists_in_catalog(row, final_catalog):
            match_count += 1
        else:
            all_found = False
            print(f"ERROR: Match not found in final catalog: (dL={row['Luminosity Distance']}, theta={row['theta']}, phi={row['phi']})")
    
    print(f"Verification complete: {match_count}/{len(original_matches)} matches found in final catalog")
    
    return all_found

if __name__ == '__main__':
    folder = 'Uniform/TestRun03/'
    COV_SAVE_PATH = '/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Events/' + folder
    output_path = '/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Catalogues/GalaxyCatalogue/Uniform/'

    print(f'Using {multiprocessing.cpu_count()} CPUs')

    print('Reading Galaxy Catalogue')
    to_read = 'Uniform_paper.txt'
    hostcat = GalCat(to_read).read_catalogue()
    print(f'Number of hosts in {to_read}: {hostcat.shape[0]}')

    Allevents_DS = pd.DataFrame()
    for file_name in glob.glob(os.path.join(COV_SAVE_PATH, 'SNR_more_than_100_*.h5')):
        tosave = load_population(file_name)
        tmp = pd.DataFrame.from_dict(tosave, orient='columns')
        Allevents_DS = pd.concat([Allevents_DS, tmp], ignore_index=True)
    
    print(f'Found {Allevents_DS.shape[0]} Dark Siren events')

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

    # Save original matches for verification
    original_matches = temp_df.copy()
    
    print(f"Shape of hostcat before removal: {hostcat.shape[0]}")
    print(f"Shape of extracted entries (temp_df): {temp_df.shape[0]}")

    # Create a copy of the matches with a unique ID for verification
    temp_df_with_ids = temp_df.copy()
    temp_df_with_ids['_match_id'] = np.arange(len(temp_df_with_ids))
    
    # Remove matches from hostcat
    hostcat = hostcat.loc[~hostcat.index.isin(temp_df.index)]
    print(f"Shape of hostcat after removal: {hostcat.shape[0]}")

    Nhost = min(Nhost, hostcat.shape[0])  # Adjust Nhost to available entries

    # Sample from the remaining hosts
    hostcat_sampled = hostcat.sample(n=Nhost, replace=False, random_state=42)
    print(f"Shape of hostcat after dilution: {hostcat_sampled.shape[0]}")

    # Add matched entries back
    hostcat_sampled = pd.concat([hostcat_sampled, temp_df], ignore_index=True)
    print(f"Shape of hostcat after concat: {hostcat_sampled.shape[0]}")
    
    # Verify matches are in the final catalog
    print("Verifying that all matched entries are in the final catalog...")
    verification_result = verify_entries_in_catalog(original_matches, hostcat_sampled)
    
    if verification_result:
        print("VERIFICATION PASSED: All matched entries found in final catalog")
    else:
        print("VERIFICATION FAILED: Some matched entries are missing from final catalog")
        
    # Additional sanity check - the final count should be exactly Nhost + temp_df.shape[0]
    expected_final_count = Nhost + temp_df.shape[0]
    if hostcat_sampled.shape[0] == expected_final_count:
        print(f"Count verification PASSED: Final catalog has {hostcat_sampled.shape[0]} entries, which matches expected count")
    else:
        print(f"Count verification FAILED: Final catalog has {hostcat_sampled.shape[0]} entries, expected {expected_final_count}")
    print(list(hostcat_sampled.columns))    
    output_filename = 'Uniform_paper_sampled_density_of_version_one_testrun03.txt'
    hostcat_sampled.to_csv(os.path.join(output_path, output_filename), index=False)
    print(f'Sampled catalog saved to {os.path.join(output_path, output_filename)}')
    
    # Save a backup of just the dark siren hosts for reference
    ds_hosts_filename = 'dark_siren_hosts_testrun03.txt'
    original_matches.to_csv(os.path.join(output_path, ds_hosts_filename), index=False)
    print(f'Dark Siren hosts saved to {os.path.join(output_path, ds_hosts_filename)}')