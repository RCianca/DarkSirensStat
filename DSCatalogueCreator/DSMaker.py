import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from functools import partial
import os
import h5py
from multiprocessing import Pool
from scipy.optimize import fsolve
import time

# Constants and global variables
H0GLOB = 67
Om0GLOB = 0.319
Xi0Glob = 1.
clight = 2.99792458 * 10**5  # km/s

cosmofast = FlatLambdaCDM(H0=H0GLOB, Om0=Om0GLOB)
H0 = cosmofast.H(0).value
h = H0GLOB/100

# Geometrization of masses
Msun = (1.98892) * (10**30)
NCORE = 24

# Path configuration
CAT_PATH = '/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'
THIS_DIR = os.getcwd()

# Read the parent catalog
os.chdir(CAT_PATH)
Parent_Catalogue = pd.read_csv('TrueFlag_half.txt', sep=' ', header=None)
os.chdir(THIS_DIR)

# Rename columns
colnames = ['Ngal', 'Comoving Distance', 'Luminosity Distance', 'z', 'phi', 'theta']
Parent_Catalogue.columns = colnames
print(Parent_Catalogue.columns)
print(Parent_Catalogue.head())

# Plot histogram of redshift distribution
n, bins, patches = plt.hist(x=Parent_Catalogue['z'], bins=50, color='teal',
                            alpha=0.7, rwidth=1, density=False)
plt.grid(axis='y', alpha=0.75)

label_fontsize = 15
title_fontsize = 18

plt.xlabel('Redshift', fontsize=label_fontsize)
plt.ylabel('N(z)', fontsize=label_fontsize)
plt.title('N(z)-Uniform', fontsize=title_fontsize)
plt.yscale('log')
plt.xscale('log')
#plt.savefig('NzFlag_log_log.png')

# Test the uniform distribution
position = []
volume = []
numobj = []
alldc = np.asarray(Parent_Catalogue['Comoving Distance'])
Nbis = 15
step = (np.max(alldc) - np.min(alldc)) / Nbis
start = np.min(alldc)

for i in range(Nbis):
    dcsup = step / 2 + start + (step) * i
    position.append(dcsup)
    tmp = alldc[alldc < dcsup]
    numobj.append(len(tmp))
    volume.append(dcsup**3 - start**3)  # We also sum the missed volume from 0 to zmin

position = np.asarray(position)
volume = np.asarray(volume)
numobj = np.asarray(numobj)
volume = volume / np.min(volume)
norm = numobj[0]
volume = volume * norm

plt.figure(figsize=(15, 10))
label_fontsize = 15
title_fontsize = 18
plt.xscale('log')
plt.yscale('log')
plt.title('Galaxy Distribution')
plt.scatter(position / np.max(position), numobj / np.max(numobj), s=100, marker='+', c='k', zorder=10)
plt.plot(position / np.max(position), volume / np.max(volume), color='g')
plt.xlabel('$dc$')
plt.ylabel('# of object in a sphere')
plt.grid(axis='y', alpha=0.75)

# Load the Branchesi distribution (Dark Sirens data) from h5 file
os.chdir(CAT_PATH)
file = h5py.File('18321_1yrCatalogBBH.h5', 'r')
data_dict = {}
keys = file.keys()
# Extract each dataset and store in the dictionary
for key in keys:
    if key in file:
        data_dict[key] = file[key][:]
    else:
        print(f"Warning: Key '{key}' not found in the h5 file")
file.close()
# Create a pandas DataFrame from the dictionary
DS_Branchesi = pd.DataFrame(data_dict)
os.chdir(THIS_DIR)

# Print the columns to help with debugging
#print("Branchesi catalog columns:", DS_Branchesi.columns)
#print("First few rows of Branchesi catalog:")
#print(DS_Branchesi.head())


z_bins = np.linspace(DS_Branchesi['z'].min(), DS_Branchesi['z'].max(), 201)
hist, bin_edges = np.histogram(DS_Branchesi['z'], bins=z_bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Create interpolation function from the histogram
# Handle possible zeros in the histogram by adding a small constant
hist_for_interp = hist + 1e-10  # Add small constant to avoid zero values
Branchesi_dist = interpolate.interp1d(bin_centers, hist_for_interp, kind='cubic', fill_value='extrapolate')

# Create redshift bins
nbins_rate = 151
z_min = Parent_Catalogue['z'].min()
z_max = Parent_Catalogue['z'].max()
step_rate = (z_max - z_min) / nbins_rate
arr_of_redshift = np.linspace(z_min, z_max, nbins_rate)

# Calculate bin centers
shift = (arr_of_redshift[1] - arr_of_redshift[0]) / 2
arr_central_bin_value = arr_of_redshift[:-1] + shift

# Calculate number of Dark Sirens in each bin using the Branchesi distribution
Numb_DS_of_z = np.zeros(len(arr_of_redshift) - 1)
for i in range(len(arr_central_bin_value)):
    # If the bin center is within the range of the Branchesi data
    if (arr_central_bin_value[i] >= bin_centers.min() and 
        arr_central_bin_value[i] <= bin_centers.max()):
        Numb_DS_of_z[i] = Branchesi_dist(arr_central_bin_value[i])
    else:
        # For values outside the range, use the nearest edge value
        if arr_central_bin_value[i] < bin_centers.min():
            Numb_DS_of_z[i] = Branchesi_dist(bin_centers.min())
        else:
            Numb_DS_of_z[i] = Branchesi_dist(bin_centers.max())

# Plot the Branchesi distribution
plt.figure(figsize=(12, 8))
plt.bar(bin_centers, hist, width=(bin_edges[1]-bin_edges[0]), alpha=0.6, color='blue', label='Original Histogram')
z_plot = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
plt.plot(z_plot, Branchesi_dist(z_plot), 'r-', linewidth=2, label='Interpolated Distribution')
plt.xlabel('Redshift (z)', fontsize=14)
plt.ylabel('Number of Dark Sirens', fontsize=14)
plt.title('Branchesi Dark Sirens Distribution', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('Branchesi_distribution.png')

# Apply angular scaling factor
radphimin = Parent_Catalogue['phi'].min()
radphimax = Parent_Catalogue['phi'].max()
radthetamin = Parent_Catalogue['theta'].min()
radthetamax = Parent_Catalogue['theta'].max()
angular_part = (radphimax - radphimin) * (-(np.cos(radthetamax) - np.cos(radthetamin)))

# Scale the number of Dark Sirens
Numb_DS_of_z = Numb_DS_of_z * (angular_part/4*np.pi) * 8000  # Scaling factor to increase population (angular_part/4*np.pi) ~ 1/8
Numb_DS_of_z_int = np.around(Numb_DS_of_z)

# Plot the number of Dark Sirens vs redshift
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
ax.set_xlabel('redshift')
ax.set_ylabel('$number of GW(z)$')
ax.plot(arr_central_bin_value, Numb_DS_of_z, label='Numb Of GWs', color='teal')
plt.grid(axis='y', alpha=0.75)
plt.grid(axis='x', alpha=0.75)
plt.title('N(z)-DS', fontsize=title_fontsize)
plt.savefig('NumDSvsz.png')

# Assign DS values to the catalog
Parent_Catalogue['DS'] = 0

def assign_ds_values(index, catalogue, arr_of_redshift, num_ds_arr):
    bin_min, bin_max = arr_of_redshift[int(index)], arr_of_redshift[int(index) + 1]
    num_ds = int(num_ds_arr[int(index)])
    entries_in_bin = catalogue[(catalogue['z'] >= bin_min) & (catalogue['z'] < bin_max)]
    if len(entries_in_bin) == 0:
        return []

    # Sample without replacement to avoid duplication
    if num_ds > len(entries_in_bin):
        num_ds = len(entries_in_bin)
    sampled_indices = entries_in_bin.sample(n=num_ds, replace=False).index.tolist()
    return sampled_indices  # Return the indices for updating

# Function to update the DataFrame after processing
def update_catalogue(indices, catalogue):
    for idx in indices:
        catalogue.loc[idx, 'DS'] = 1

# Split the DataFrame into smaller chunks to avoid memory issues
chunks = np.array_split(Parent_Catalogue, NCORE)

# Generate the arguments for the pool
args = [(i, chunk, arr_of_redshift, Numb_DS_of_z_int) for chunk in chunks for i in range(nbins_rate - 1)]

with Pool(NCORE) as pool:
    # Use starmap to parallelize the assign_ds_values function
    result_indices = pool.starmap(assign_ds_values, args)

# Flatten the list of indices
result_indices = [item for sublist in result_indices for item in sublist if item]

# Update the original DataFrame
update_catalogue(result_indices, Parent_Catalogue)

print(Parent_Catalogue.head(10))

# Plot histogram of extracted Dark Sirens
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
ax.set_xlabel('redshift')
ax.set_ylabel('number of $GW(z)$')
n, bins, patches = plt.hist(x=Parent_Catalogue[Parent_Catalogue['DS'] == 1].z, bins=50, color='teal',
                            alpha=0.7, rwidth=1, density=False)

plt.grid(axis='y', alpha=0.75)
plt.grid(axis='x', alpha=0.75)
plt.title('N(z)-DS-extracted', fontsize=title_fontsize)
plt.savefig('NumDSvsz_extracted.png')

# Create DS_From_Parent DataFrame
DS_From_Parent = Parent_Catalogue[Parent_Catalogue['DS'] == 1].copy()

# Add additional columns with specified values
num_entries = DS_From_Parent.shape[0]

DS_From_Parent['M1'] = 0
DS_From_Parent['M2'] = 0
DS_From_Parent['MC'] = 0
DS_From_Parent['q'] = 0
DS_From_Parent['cos_iota'] = np.random.uniform(-1, 1, num_entries)
DS_From_Parent['psi'] = np.random.uniform(0, 2 * np.pi, num_entries)
DS_From_Parent['tcoal'] = 0
DS_From_Parent['Phicoal'] = 0
DS_From_Parent['chiz1'] = 0
DS_From_Parent['chiz2'] = 0

# Save catalog up to now
os.chdir(CAT_PATH)
DS_From_Parent.to_csv('DS_From_Parent_Half_Flag.txt', header=True, index=False)
os.chdir(THIS_DIR)

# Mass sampling functions
def S(x, m_min, dm):
    x = np.asarray(x)
    s = np.ones_like(x)
    s[x < m_min] = 0
    mask = (m_min <= x) & (x < m_min + dm)
    
    exp_values = dm / (x[mask] - m_min) + dm / (x[mask] - m_min - dm)
    
    # Avoid overflow by capping exp_values
    large_mask = exp_values > 700  # Value beyond which exp() will overflow
    safe_exp_values = np.where(large_mask, 700, exp_values)
    
    s[mask] = (np.exp(safe_exp_values) + 1) ** (-1)
    return s

def tonorm(x, alpha):
    return x**(-alpha)

def PowerLawPlusPeak(m1, m_min, m_max, lamb, alpha, mu, sigma_m, dm):
    c = (alpha-1) * m_min**(alpha-1)
    powlaw = (1 - lamb) * c * (m1 ** (-alpha))
    gauss = lamb * np.exp(-(m1 - mu) ** 2 / (2 * sigma_m ** 2)) / (np.sqrt(2 * np.pi) * sigma_m)
    tmp = S(m1, m_min, dm) * np.float64((powlaw + gauss))
    return tmp

def sample(g, xmin, xmax):
    x = np.linspace(xmin, xmax, 1000000)
    y = g(x)                        # probability density function, pdf
    cdf_y = np.cumsum(y)            # cumulative distribution function, cdf
    cdf_y = cdf_y/cdf_y.max()       # takes care of normalizing cdf to 1.0
    inverse_cdf = interpolate.interp1d(cdf_y, x, fill_value="extrapolate")  # this is a function
    return inverse_cdf

def return_samples(f, xmin, xmax, N=1000000):
    # Generate samples according to the chosen pdf, f(x)
    uniform_samples = np.random.random(int(N))       
    required_samples = sample(f, xmin, xmax)(uniform_samples)
    return required_samples

# Mass distribution parameters
m_min = 4.59
m_max = 86
lamb = 0.1
alpha = 2.63
mu_m = 33.07
sigma_m = 5.7
dm = 4.82

PowerLawPlusPeak_with_params = partial(PowerLawPlusPeak, m_min=m_min,
                                       m_max=m_max,
                                       lamb=lamb,
                                       alpha=alpha,
                                       mu=mu_m,
                                       sigma_m=sigma_m,
                                       dm=dm)

# Number of samples
num_samples = DS_From_Parent.shape[0] * 100
# Sample the probability distribution
m1_samples = np.loadtxt('m1masses_paper.txt')
DS_From_Parent['M1'] = np.random.choice(m1_samples, size=DS_From_Parent.shape[0])

# Plot M1 distribution
m1_values = np.linspace(m_min, m_max, 1000000)
p_m1_values = PowerLawPlusPeak_with_params(m1_values)
norm = np.trapz(p_m1_values, m1_values)
p_m1_values = p_m1_values / norm

fig, ax = plt.subplots(figsize=(15, 10))
ax.tick_params(axis='both', which='major', labelsize=25)
ax.yaxis.get_offset_text().set_fontsize(25)
# Plot the histogram of the sampled probabilities
ax.hist(DS_From_Parent['M1'], bins=100, density=True, alpha=0.6, color='orange', label='Sampled $M_1$')

# Plot the theoretical distribution
ax.plot(m1_values, p_m1_values, label='Theoretical $p(m_1)$', color='teal', linewidth=3)

# Labels and title
ax.set_xlabel('$m_1$', fontsize=15)
ax.set_ylabel('$P(m_1)$', fontsize=15)
plt.title('Theoretical Distribution vs Sampled Histogram', fontsize=18)

# Legends
ax.legend(loc='upper right', prop={'size': 15})

# Show plot
plt.grid(axis='y', alpha=0.75)
plt.savefig('m1_extracted.png')

print(DS_From_Parent.columns)
print(DS_From_Parent.head(5))

# Define the path to save the file
output_file = os.path.join(CAT_PATH, 'DS_From_Parent_Half_Flag_Branchesi_Complete.txt')

# Save the DataFrame in chunks
chunksize = 1000  # Adjust the chunksize according to your memory capacity

# Write the first chunk with headers
DS_From_Parent.iloc[:chunksize].to_csv(output_file, header=True, index=False, mode='w')

# Write the remaining chunks without headers
for i in range(chunksize, len(DS_From_Parent), chunksize):
    DS_From_Parent.iloc[i:i+chunksize].to_csv(output_file, header=False, index=False, mode='a')

print(f"DataFrame saved to {output_file}")
os.chdir(THIS_DIR)

# Assign q (mass ratio) values
b = 1.26
chunksize = 1000  # Adjust according to memory capacity

# Function to calculate p_q
def p_q(q, m1, m_min, dm, b):
    return S(m1 * q, m_min, dm) * q ** b

# Function to calculate q and M2 for a given m1
def calculate_q_and_m2(m1, m_min, dm, b):
    p_q_with_params = partial(p_q, m1=m1, m_min=m_min, dm=dm, b=b)
    q_samples = return_samples(p_q_with_params, 0, 1, 1)
    q = q_samples[0]
    M2 = m1 * q
    return q, M2

# Wrapper function to unpack arguments
def calculate_q_and_m2_wrapper(args):
    return calculate_q_and_m2(*args)
    
# Function to calculate chirp mass Mc
def Mc(m1, m2):
    num = (m1 * m2) ** (3 / 5)
    denom = (m1 + m2) ** (1 / 5)
    return num / denom

# Process the data in chunks
for start in range(0, len(DS_From_Parent), chunksize):
    end = min(start + chunksize, len(DS_From_Parent))
    chunk = DS_From_Parent.iloc[start:end].copy()

    # Prepare arguments for multiprocessing
    args = [(m1, m_min, dm, b) for m1 in chunk['M1']]

    # Use multiprocessing Pool to calculate q and M2
    with Pool(NCORE) as pool:
        results = pool.map(calculate_q_and_m2_wrapper, args)

    # Assign q and M2 to the chunk DataFrame
    chunk['q'], chunk['M2'] = zip(*results)

    # Calculate chirp mass
    chunk['MC'] = Mc(chunk['M1'], chunk['M2'])

    # Save the chunk to the file
    mode = 'w' if start == 0 else 'a'
    header = True if start == 0 else False
    chunk.to_csv(output_file, header=header, index=False, mode=mode)

print(f"DataFrame saved to {output_file}")
os.chdir(THIS_DIR)