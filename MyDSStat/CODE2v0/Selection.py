from Global import *

from SkyMap import GWskymap
import glob
import numpy as np
import pandas as pd
import os
import multiprocessing
from multiprocessing import Pool

if __name__ == '__main__':
    #print(f"Files to process: {fname}")
    working_dir = os.getcwd()
    path = 'DS_th'
    save_path = os.path.join(path, 'th_200')

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Load GW Data
    print('Loading GW data')
    level = 0.9

    saved_events = []  # List for events that meet the condition

    fits_files = sorted(glob.glob(os.path.join(MapPath, "*.fits")))

    for name in fits_files: #fname:
        DSs = GWskymap(os.path.join(MapPath, name), level=level)
        pix_selected = DSs.get_credible_region_pixels(level=level)
        nside = int(DSs.nside)
        skyprob = DSs.p_posterior
        allmu, allsigma = DSs.mu * 1000, DSs.sigma * 1000  # Convert to Mpc

        if np.isnan(allmu).any():
            print(f'There are NaN values in allmu of {os.path.basename(name)}')
        if np.isnan(allsigma).any():
            print(f'There are NaN values in allsigma of {os.path.basename(name)}')

        if len(pix_selected) < pix_threshold:  # Save the event if pix_selected is less than the threshold
            print('DS data:')
            print(f'Using {os.path.basename(name)}')
            print(f'Area of DS: {DSs.area()} deg^2 at 90%')
            saved_events.append(os.path.basename(name))
        else:
            print(f'Skipping {os.path.basename(name)}, too many pixels')

    # Save the valid event names as a NumPy array
    print('Number of files in folder= {}'.format(len(fits_files)))
    print('Number of selected events= {}'.format(len(saved_events)))
    np.save(os.path.join(save_path, 'saved_events.npy'), np.array(saved_events))
    print(f"Saved events stored in {os.path.join(save_path, 'saved_events.npy')}")
