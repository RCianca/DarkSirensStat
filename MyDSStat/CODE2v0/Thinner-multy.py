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
    """ Initialize the shared memory for hostcat in each worker process """
    global hostcat_shared_mem, hostcat_shape, hostcat_columns
    
    hostcat_shared_mem = shared_memory.SharedMemory(name=shared_name)
    hostcat_shape = shape
    hostcat_columns = columns

def process_event(k):
    """ Worker function to filter hostcat based on event properties """
    global hostcat_shared_mem, hostcat_shape, hostcat_columns

    DS_dl = Allevents_DS.iloc[k]['dL'] * 1000
    DS_theta = Allevents_DS.iloc[k]['theta']
    DS_phi = Allevents_DS.iloc[k]['phi']

    # Reconstruct the DataFrame from shared memory
    hostcat_array = np.ndarray(hostcat_shape, dtype=np.float64, buffer=hostcat_shared_mem.buf)
    hostcat = pd.DataFrame(hostcat_array, columns=hostcat_columns)

    # Use np.isclose() for floating-point tolerance
    matched_rows = hostcat[
        np.isclose(hostcat['Luminosity Distance'], DS_dl, atol=1e-7) &  
        np.isclose(hostcat['theta'], DS_theta, atol=1e-6) &
        np.isclose(hostcat['phi'], DS_phi, atol=1e-6)
    ]
    
    # **Logging Issues**
    if matched_rows.empty:
        print(f"Warning: No match found for event {k} (dL={DS_dl}, theta={DS_theta}, phi={DS_phi})")
    elif len(matched_rows) > 1:
        print(f"Warning: Multiple matches found for event {k}, taking the first one.")

    # Take the first match or return empty row
    return matched_rows.iloc[:1] if not matched_rows.empty else pd.DataFrame(columns=hostcat_columns)

if __name__ == '__main__':
    folder = 'Uniform/TestRun00/'
    CAT_FOLDER = '/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'
    COV_SAVE_PATH = '/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Events/' + folder
    output_path = '/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Catalogues/GalaxyCatalogue/Uniform/'

    print(f'Using {multiprocessing.cpu_count()} CPUs')

    # Load the galaxy catalog
    print('Reading Galaxy Catalogue')
    to_read = 'Uniform_paper.txt'
    hostcat = GalCat(to_read).read_catalogue()

    # Load all population files
    Allevents_DS = pd.DataFrame()
    for file_name in glob.glob(os.path.join(COV_SAVE_PATH, 'SNR_more_than_100_*.h5')):
        tosave = load_population(file_name)
        More_population = os.path.basename(file_name)
        tmp = pd.DataFrame.from_dict(tosave, orient='columns')
        Allevents_DS = pd.concat([Allevents_DS, tmp], ignore_index=True)
        print('Loaded population {}'.format(More_population))

    print(list(Allevents_DS.columns))
    
    selected = np.arange(0, Allevents_DS.shape[0])
    Host_in_cat = hostcat.shape[0]
    Density_cat = 0.00171
    Density_version1 = 0.000286
    Nhost = int(Host_in_cat * Density_version1 / Density_cat)

    # Convert hostcat to NumPy array for shared memory
    hostcat_array = hostcat.to_numpy(dtype=np.float64)
    hostcat_columns = list(hostcat.columns)
    hostcat_shape = hostcat_array.shape

    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=hostcat_array.nbytes)
    shared_hostcat = np.ndarray(hostcat_shape, dtype=np.float64, buffer=shm.buf)
    shared_hostcat[:] = hostcat_array[:]  # Copy data to shared memory

    # Use multiprocessing to process events in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=init_worker, initargs=(shm.name, hostcat_shape, hostcat_columns)) as pool:
        results = pool.map(process_event, selected)

    # Convert results back to DataFrame
    temp_df = pd.concat(results, ignore_index=True)

    # Free shared memory
    shm.close()
    shm.unlink()

    print(f"Shape of hostcat before removal: {hostcat.shape[0]}")
    print(f"Shape of extracted entries (temp_df): {temp_df.shape[0]}")
    
    # Remove matched rows from hostcat
    hostcat = hostcat.drop(temp_df.index)

    print(f"Shape of hostcat after removal: {hostcat.shape[0]}")

    # Sample the remaining hostcat
    hostcat_sampled = hostcat.sample(n=Nhost, replace=False, random_state=42)

    print(f"Shape of hostcat_sampled before concatenation: {hostcat_sampled.shape[0]}")
    # Mark extracted entries in temp_df
    temp_df['is_extracted'] = True
    hostcat_sampled['is_extracted'] = False
    # Extract the actual extracted entries for validation
    last_entries = hostcat_sampled[hostcat_sampled['is_extracted'] == True].drop(columns=['is_extracted'])

    # Reset index for proper comparison
    temp_df_sorted = temp_df.reset_index(drop=True)
    last_entries_sorted = last_entries.reset_index(drop=True)

    # **Ensure Both DataFrames Have the Same Column Order and Drop 'is_extracted'**
    temp_df_sorted = temp_df_sorted.drop(columns=['is_extracted'], errors='ignore')  # Drop safely if exists
    last_entries_sorted = last_entries_sorted[temp_df_sorted.columns]  # Align column order

    # Ensure both DataFrames have only numeric values
    temp_df_sorted = temp_df_sorted.select_dtypes(include=[np.number])
    last_entries_sorted = last_entries_sorted.select_dtypes(include=[np.number])

    # Check for NaN or Inf values before comparison
    if temp_df_sorted.isnull().values.any() or last_entries_sorted.isnull().values.any():
        print("Warning: NaN values found in the data. This may cause validation failure.")

    if not np.isfinite(temp_df_sorted.to_numpy()).all() or not np.isfinite(last_entries_sorted.to_numpy()).all():
        print("Warning: Infinite values found in the data. This may cause validation failure.")

    # Convert to NumPy before comparison
    if np.allclose(temp_df_sorted.to_numpy(), last_entries_sorted.to_numpy(), atol=1e-6):
        print("Validation Passed: Extracted entries match after sorting (within numerical tolerance).")
    else:
        print("Validation Failed: Extracted entries do not match even after sorting!")
        print("Possible numerical precision issue.")

        # Debugging Output
        print("\nFirst few rows of temp_df_sorted:")
        print(temp_df_sorted.head())

        print("\nFirst few rows of last_entries_sorted:")
        print(last_entries_sorted.head())

        print("\nDifference between DataFrames:")
        print(temp_df_sorted.to_numpy() - last_entries_sorted.to_numpy())

    # **Save the correctly updated `hostcat_sampled` instead of `temp_df`**
    output_filename = 'Uniform_paper_sampled_density_of_version_one.txt'
    hostcat_sampled.to_csv(os.path.join(output_path, output_filename), index=False)
    print(f'Sampled catalog saved to {output_path}')

