import numpy as np
from Global import *
import os

import multiprocessing
from multiprocessing import Pool

from GalaxyCat import GalCat
from SkyMap import GWskymap


#for now is a function to use in CODE2v0-Fast-MultipleDS, so all args will be passed by the script. Used as main, you need to give the arguments

# Parallelized function to compute beta in the pixel
def beta2v0_pix(args):
    pix, mu_pix, sigma_pix, z_hosts, H0Grid = args
    pixel_beta = np.ones(len(H0Grid))
    if len(z_hosts) == 0:
        #print('No hosts for this line of sight')
        return pixel_beta
    if np.isnan(z_hosts).any():
        #print('NONE in z_hosts')
        return pixel_beta
    #print('debug:found hosts')
    for j, h in enumerate(H0Grid):
        pixel_beta[j] = beta_inpix(mu_pix, sigma_pix, z_hosts, h)
    
    return pixel_beta

def beta_inpix(mu_DS, sigma, z_hosts, Htemp):
    """
    Compute the beta of H0 for a single pixel. Count how many possible hosts in the pixel
    between dl_max and dl_min.
    """

    dl_array = Dl_z_vectorized(z_hosts, Htemp, Om0GLOB)  # Vectorized computation
    #begin mod speed up
    dl_array=dl_array[dl_array<=mu_DS+4.5*sigma]
    dl_array=dl_array[dl_array>=mu_DS-4.5*sigma]
    
    if dl_array is None or np.isnan(dl_array).any():
        raise ValueError("Dl_z_vectorized returned None or NaN")

    beta = len(dl_array)# here we will add the weights
    return beta

if __name__=='__main__':
    print('Computing Beta for each event.')

    fname = InputEvents(start,stop)
    print(f"Files to process: {fname}")
    #print(f"Results will be saved in the beta folder inside: {runpath}")
    working_dir = os.getcwd()
    path = 'Results'
    #runpath = 'FirstBatch'

    # Ensure directory exists
    folder = os.path.join(path, runpath,'Beta')
    os.makedirs(folder, exist_ok=True)
    print(f'\nData will be saved in {folder}')
    os.system('cp Beta2v0.py '+folder+'/beta-copy.py')

    # H0 Grid
    H0Grid = np.linspace(H0min, H0max, 1000)

    # Read Galaxy Catalogue
    
    print('Reading Galaxy Catalogue--'+to_read)
    nside = 128
    hostcat = GalCat(to_read, nside).read_catalogue()
    mypixels = GalCat(to_read, nside).pixelizer()
    print('Reading catalogue completed')

    # Load GW Data
    print('Loading GW data')
    #MapPath = os.path.join(working_dir, 'Events/Uniform/TestRun00/')
    level = 0.9

    for name in fname:

        DSs = GWskymap(os.path.join(MapPath, name), level=level)
        #print(f'DS name: {DSs.event_name}')
        #print(f'Area of DS: {DSs.area()} deg^2 at 90%')
        
        pix_selected = DSs.get_credible_region_pixels(level=level)
        nside = int(DSs.nside)
        skyprob = DSs.p_posterior
        allmu, allsigma = DSs.mu * 1000, DSs.sigma * 1000  # Convert to Mpc

        if np.isnan(allmu).any():
            print(f'There are NaN values in allmu of {name}')
        if np.isnan(allsigma).any():
            print(f'There are NaN values in allsigma of {name}')

        if(len(pix_selected)>pix_threshold): #this can be relaxed, now is a test run. Such confition should be implemented in the skymap generator to have different quality sets
            print(f'Skipping {name}, too many pixels')
            
        else:    
            print('DS data:')
            print(f'Using {name}')
            print(f'Area of DS: {DSs.area()} deg^2 at 90%')

            # Filter Galaxy Catalogue
            
            hostcat['Pixel'] = mypixels
            hostcat_filtered = hostcat[hostcat['Pixel'].isin(pix_selected)]
            # Pre-group by pixel to speed up filtering
            grouped_hostcat = hostcat_filtered.groupby('Pixel')['z'] # Remove if creates problem. This shoud group already hostcat and avoid to do it in pixel_args

    
        
            single_beta=np.ones(len(H0Grid))

            pixel_args = [
                (pix, allmu[pix], allsigma[pix], grouped_hostcat.get_group(pix).values, H0Grid)
                for pix in pix_selected
                if pix in grouped_hostcat.groups
            ] # This is the new version with goupby. If not working rmove also groupby above

            if len(pixel_args) > 0:
                #cpu = min(multiprocessing.cpu_count(), len(pixel_args))
                cpu = multiprocessing.cpu_count()
                print(f'Using {cpu} CPUs')
                chunksize = max(1, len(pixel_args) // (2 * cpu))
                with Pool(cpu) as pool:
                    results = list(pool.imap(beta2v0_pix, pixel_args, chunksize=chunksize))# better chunksize should improve time
                single_beta = np.sum(results, axis=0)
                if np.sum(single_beta)==0:
                    single_beta=np.ones(len(H0Grid))
                if np.isnan(single_beta).any():
                    single_beta=np.ones(len(H0Grid))
                print('some NaN in singlepost, skipped\n')
            else:
                print("No Hosts found for this DS, skipping computation.")
                results = []
            betaname='beta_'+name.split('.')[0]
            np.save(os.path.join(folder,betaname),single_beta)
            #total_post *= single_post
print('All beta Saved')