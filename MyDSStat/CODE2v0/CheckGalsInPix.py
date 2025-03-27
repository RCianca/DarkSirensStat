#from Global import (
#    href, Om0GLOB, clight, start,stop,InputEvents, runpath,to_read,pix_threshold,MapPath,H0min,H0max
#)
from Global import *
#from scipy.integrate import quad, quad_vec,simpson
from SkyMap import GWskymap
from GalaxyCat import GalCat
import numpy as np
import pandas as pd
import os
import numpy as np
from SkyMap import GWskymap
from GalaxyCat import GalCat
from Global import MapPath, to_read

# Load galaxy catalog once (outside loop)
nside = 128
catalog = GalCat(to_read, nside)
hostcat = catalog.read_catalogue()
host_pixels = catalog.pixelizer()
hostcat['Pixel'] = host_pixels

# Get list of fits files
fits_files = [f for f in os.listdir(MapPath) if f.endswith('.fits')]

# Loop over each GW map
for fits_file in fits_files:
    full_path = os.path.join(MapPath, fits_file)

    # Load GW map
    gw_map = GWskymap(full_path, level=0.9)
    
    # Get pixels in credible region
    gw_pixels = gw_map.get_credible_region_pixels(level=0.9)
    n_pixels = len(gw_pixels)
    allmu, allsigma = gw_map.mu * 1000, gw_map.sigma * 1000  # Convert to Mpc
    allmu[gw_pixels]
    allsigma[gw_pixels]


    # Get nside resolution
    map_nside = gw_map.nside

    # Count galaxies within GW pixels
    galaxies_in_region = hostcat[hostcat['Pixel'].isin(gw_pixels)].shape[0]


    # Output results
    print(f"Results for {fits_file}:")
    print(f"  Number of pixels in 90% credible region: {n_pixels}")
    print(f"  GW map nside resolution: {map_nside}")
    print(f"  Num of Galaxies within these pixels: {galaxies_in_region}\n")
    if galaxies_in_region==0:
        print("--------------------------")
        print(" NO GALAXIES DETECTED")
        print("--------------------------")

    # New check for zero values in mu and sigma
    zero_mu_pixels = np.where(allmu == 0)[0]
    if zero_mu_pixels.size > 0:
        print(f'Error: mu is zero for {zero_mu_pixels.size} pixels in event {fits_file}')

    zero_sigma_pixels = np.where(allsigma == 0)[0]
    if zero_sigma_pixels.size > 0:
        print(f'Error: sigma is zero for {zero_sigma_pixels.size} pixels in event {fits_file}')

    # Existing NaN checks
    if np.isnan(allmu).any():
        print(f'There are NaN values in allmu of {fits_file}')

    if np.isnan(allsigma).any():
        print(f'There are NaN values in allsigma of {fits_file}')