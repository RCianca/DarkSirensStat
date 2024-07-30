import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import integrate
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.integrate import trapz
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import healpy as hp
from functools import partial

import os
from os import listdir
from os.path import isfile, join

from multiprocessing import Pool
import time
from numba import njit

import sys
from tqdm import tqdm

##############################################################################
H0GLOB= 67#69.32#67.9 #69
Om0GLOB=0.319
Xi0Glob =1.
clight = 2.99792458* 10**5#km/s

cosmofast = FlatLambdaCDM(H0=H0GLOB, Om0=Om0GLOB)
H0=cosmofast.H(0).value
h=H0GLOB/100

#geometrization of masses
Msun=(1.98892)*(10**30)
#solarmass_to_m=(constants.G*Msun)/((constants.c)**2)#G/c^2
#Mpc_to_m=3.08567758128*(10**22) #this will be used later
NCORE=24
###############################################################################

#--------------------GW--Rate-----------------------------------------

#--------Star Formation Rate-------------------------------
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
th=1000/cosmofast.H(0).value
alpha=14/th
def psi(x):
    ret= 0.015*(((1+x)**(2.7))/(1+(((1+x)/(2.9))**(5.6))))
    return ret
#-------------Phi----------------------------
#phi=lambda x:(1/((1+x)*(cosmofast.H(x).value)))
def phi(x):
    #N=14/((1/((1+0)*(cosmoglob.H(0).value))))
    #ret=N*(1/((1+x)*(cosmoglob.H(x).value)))
    ret=(1000/((1+x)*(cosmofast.H(x).value)))
    return ret
#--------------time inversion-------------------------
def inverted_time(x):
    #is the lookback_time use astropy
    ret=cosmofast.age(0).value-cosmofast.age(x).value
    return ret
#----------------Decay--------------------------------
def decay(x,z,tau):
    ret=np.exp(-((cosmofast.lookback_time(x).value-cosmofast.lookback_time(z).value)/(tau)))
    return ret
#--------------time difference-------------------------
def inversetimediff(x,z,tmin):
    timediff=alpha*(cosmofast.age(z).value-cosmofast.age(x).value)
    num=np.heaviside((alpha*cosmofast.age(z).value-alpha*cosmofast.age(x).value) -tmin,0.5)
    res=(num)/timediff
    return res
#-------------------Deriv of Comoving Volume-------------
def DvolDz(x):
    prefactor=4*np.pi*(((clight)/(cosmofast.H(0).value))**3)#Mpc^3
    dist=quad((cosmofast.inv_efunc),0,x)[0]
    ret=prefactor*cosmofast.inv_efunc(x)*(dist**2)
    return ret

def astrodiffvol(x):
    ret=cosmofast.differential_comoving_volume(x).value
    return ret

#------------------integrands------------------------------

def integrand_marr(x,z,tau):
    ret=psi(x)*(phi(x))*decay(x,z,tau)
    return ret
def totalrate(z,tmin):
    norm=50/(quad(integrand_marr,0,150,args=(0,tmin))[0]/tmin)
    littlerate=norm*quad(integrand_marr,z,150,args=(z,tmin))[0]/tmin
    ret=(littlerate*DvolDz(z))/(1+z)
    return ret
#########################################################################################################################
CAT_PATH='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'
THIS_DIR=os.getcwd()
os.chdir(CAT_PATH)
Parent_Catalogue = pd.read_csv('Uniform_paper.txt', header=None)
os.chdir(THIS_DIR)
#print(Parent_Catalogue.shape)
#print(Parent_Catalogue.head(5))
colnames=['Ngal','Comoving Distance','Luminosity Distance','z','phi','theta']
Parent_Catalogue.columns=colnames
print(Parent_Catalogue.columns)
print(Parent_Catalogue.head())

n, bins, patches = plt.hist(x=Parent_Catalogue['z'], bins=50, color='teal',
                            alpha=0.7, rwidth=1,density=False)
plt.grid(axis='y', alpha=0.75)

label_fontsize = 15
title_fontsize = 18

#plt.xlabel('Redshift', fontsize=label_fontsize)
#plt.ylabel('N(z)', fontsize=label_fontsize)
#plt.title('N(z)-Uniform', fontsize=title_fontsize)

plt.savefig('NzUnif.png')

n, bins, patches = plt.hist(x=Parent_Catalogue['z'], bins=50, color='teal',
                            alpha=0.7, rwidth=1,density=False)
plt.grid(axis='y', alpha=0.75)

plt.xlabel('Redshift', fontsize=label_fontsize)
plt.ylabel('N(z)', fontsize=label_fontsize)
plt.title('N(z)-Uniform', fontsize=title_fontsize)
plt.yscale('log')
plt.xscale('log')
plt.savefig('NzUnif_log_log.png')

#-----------------test the uniform distribution
position=[]
volume=[]
numobj=[]
alldc=np.asarray(Parent_Catalogue['Comoving Distance'])
Nbis=15
step=(np.max(alldc)-np.min(alldc))/Nbis
start=np.min(alldc)
for i in range(Nbis):
    dcsup=step/2 +start+(step)*i
    position.append(dcsup)
    tmp=alldc[alldc<dcsup]
    numobj.append(len(tmp))
    volume.append(dcsup**3-start**3)# We also sum the missed volume from 0 to zmin
position=np.asarray(position)
volume=np.asarray(volume)
numobj=np.asarray(numobj)
volume=volume/np.min(volume)
norm=numobj[0]
volume=volume*norm

plt.figure(figsize=(15,10))
#n, bins, patches = plt.hist(x=numobj,grid=True, bins=Num, rwidth=0.9,color='#607c8e')
label_fontsize = 15
title_fontsize = 18
plt.xscale('log')
plt.yscale('log')
plt.title('Galaxy Distribution')
plt.scatter(position/np.max(position),numobj/np.max(numobj),s=100, marker='+', c='k', zorder=10 )

plt.plot(position/np.max(position),volume/np.max(volume),color='g')

plt.xlabel('$dc$')
plt.ylabel('# of object in a sphere')
plt.grid(axis='y', alpha=0.75)
plt.savefig('TestVolume.png')
#############################################################################################################à

########### How many DS= Rate X Volume
z_plot_max=Parent_Catalogue['z'].max()+0.5
tpar=5
norm=50/totalrate(0.220,5)#just from the plot on the paper
ZZ=np.linspace(Parent_Catalogue['z'].min(),Parent_Catalogue['z'].max(),1000)
totalratetoplot=np.zeros(len(ZZ))

def calculate_totalrate(i, ZZ, norm, tpar):
    return norm * totalrate(ZZ[i], tpar)


with Pool(NCORE) as pool:
    args = [(i, ZZ, norm, tpar) for i in range(len(ZZ))]
    results = pool.starmap(calculate_totalrate, args)

for i in range(len(results)):
    totalratetoplot[i] = results[i]

#for i in range(len(ZZ)):
#    totalratetoplot[i]=norm*totalrate(ZZ[i],tpar)

rate_interpol=interpolate.interp1d(ZZ,totalratetoplot)

tempx=np.linspace(ZZ.min(),ZZ.max(),1500)
fig,ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
ax.set_xlabel('redshift')
ax.set_ylabel('$R[Mpc^{-3}]$')
ax.plot(tempx,rate_interpol(tempx),label='RateGW',color='teal')
plt.grid(axis='y', alpha=0.75)
plt.grid(axis='x', alpha=0.75)
plt.savefig('rate_paper.png')

nbins_rate=151
step_rate=(Parent_Catalogue['z'].max()-Parent_Catalogue['z'].min())/nbins_rate
arr_of_redshift=np.linspace(Parent_Catalogue['z'].min(),Parent_Catalogue['z'].max(),nbins_rate)

Numb_DS_of_z=np.zeros(len(arr_of_redshift)-1)
for i in range(nbins_rate-1):
    Numb_DS_of_z[i]=quad(rate_interpol,arr_of_redshift[i],arr_of_redshift[i+1])[0]

#-----now assign an integer value, remember to sometimes put more ds so use round
radphimin=Parent_Catalogue['phi'].min()
radphimax=Parent_Catalogue['phi'].max()
radthetamin=Parent_Catalogue['theta'].min()
radthetamax=Parent_Catalogue['theta'].max()
angular_part=(radphimax-radphimin)*(-(np.cos(radthetamax)-np.cos(radthetamin)))
#print(angular_part)
Numb_DS_of_z=Numb_DS_of_z*angular_part*1000#this is just a factor to create a more populated catalogue
Numb_DS_of_z_int=np.around(Numb_DS_of_z)

half_bin=(arr_of_redshift[1]-arr_of_redshift[0])/2
centerbins=arr_of_redshift[:-1]+half_bin
fig,ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
ax.set_xlabel('redshift')
ax.set_ylabel('$number of GW(z)$')
ax.plot(centerbins,Numb_DS_of_z,label='Numb Of GWs',color='teal')
plt.grid(axis='y', alpha=0.75)
plt.grid(axis='x', alpha=0.75)
plt.title('N(z)-DS', fontsize=title_fontsize)
plt.savefig('NumDSvsz.png')

###########We now extract uniformly DSs in each bin ##################################

Parent_Catalogue['DS'] = 0

# def assign_ds_values(index, catalogue, arr_of_redshift, num_ds_arr):
#     bin_min, bin_max = arr_of_redshift[index], arr_of_redshift[index + 1]
#     num_ds = num_ds_arr[index]
#     entries_in_bin = catalogue[(catalogue['z'] >= bin_min) & (catalogue['z'] < bin_max)]
#     if len(entries_in_bin) == 0:
#         return

#     # Sample without replacement to avoid duplication
#     sampled_indices = entries_in_bin.sample(n=num_ds, replace=False).index
#     catalogue.loc[sampled_indices, 'DS'] = 1

# with Pool(NCORE) as pool:
#     # Generate the arguments for the pool
#     args = [(i, Parent_Catalogue, arr_of_redshift, Numb_DS_of_z_int) for i in range(nbins_rate - 1)]
#     # Use starmap to parallelize the assign_ds_values function
#     pool.starmap(assign_ds_values, args)

def assign_ds_values(index, catalogue, arr_of_redshift, num_ds_arr):
    bin_min, bin_max = arr_of_redshift[int(index)], arr_of_redshift[int(index) + 1]
    num_ds = int(num_ds_arr[int(index)])
    entries_in_bin = catalogue[(catalogue['z'] >= bin_min) & (catalogue['z'] < bin_max)]
    if len(entries_in_bin) == 0:
        return []

    # Sample without replacement to avoid duplication
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

n, bins, patches = plt.hist(x=Parent_Catalogue[Parent_Catalogue['DS']==1].z, bins=50, color='teal',
                            alpha=0.7, rwidth=1,density=False)
plt.grid(axis='y', alpha=0.75)

plt.xlabel('Redshift', fontsize=label_fontsize)
plt.ylabel('N(z)', fontsize=label_fontsize)
plt.title('N(z)-DS', fontsize=title_fontsize)
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

#save catalogue uo to now 
os.chdir(CAT_PATH)
DS_From_Parent.to_csv('DS_From_Parent_Uniform.txt', header=None, index=False)
os.chdir(THIS_DIR)
############################extract masses now and assign##########################################

#-----------rejection-stuff------------
def sample(g,xmin,xmax):
    x = np.linspace(xmin,xmax,1000000)
    y = g(x)                        # probability density function, pdf
    cdf_y = np.cumsum(y)            # cumulative distribution function, cdf
    cdf_y = cdf_y/cdf_y.max()       # takes care of normalizing cdf to 1.0
    inverse_cdf = interpolate.interp1d(cdf_y,x,fill_value="extrapolate")# this is a function
    #inverse_cdf = np.interp(0,cdf_y,x) #this works but it is not a function
    return inverse_cdf
def return_samples(f,xmin,xmax,N=1000000):
    # let's generate some samples according to the chosen pdf, f(x)
    uniform_samples = np.random.random(int(N))       
    required_samples = sample(f,xmin,xmax)(uniform_samples)
    return required_samples

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

def tonorm(x,alpha):
    return x**(-alpha)

def PowerLawPlusPeak(m1, m_min, m_max, lamb, alpha, mu, sigma_m, dm):
    c=(alpha-1)*m_min**(alpha-1)
    powlaw = (1 - lamb) * c*(m1 ** (-alpha))
    gauss = lamb * np.exp(-(m1 - mu) ** 2 / (2 * sigma_m ** 2)) / (np.sqrt(2 * np.pi) * sigma_m)
    tmp = S(m1, m_min, dm) * np.float64((powlaw + gauss))
    return tmp
###################################################################################################

# Display the first few rows of the new DataFrame to verify
print(DS_From_Parent.head())

m_min = 4.59#np.float64(4.59)
m_max=86
lamb=0.1
alpha=2.63
mu_m=33.07
sigma_m=5.7
dm = 4.82#np.float64(4.82)

PowerLawPlusPeak_with_params = partial(PowerLawPlusPeak, m_min=5,
                                       m_max=m_max,
                                       lamb=lamb,
                                       alpha=alpha,
                                       mu=mu_m,
                                       sigma_m=sigma_m,
                                       dm=dm)

# Number of samples
num_samples = DS_From_Parent.shape[0]*100
# Sample the probability distribution
m1_samples = return_samples(PowerLawPlusPeak_with_params, m_min, m_max, num_samples)
np.savetxt('m1masses_paper.txt',m1_samples)

# Generate theoretical distribution for plotting
#m1_values = np.linspace(m_min+0.001, m_max, 10000)
#p_m1_values = PowerLawPlusPeak_with_params(m1_values)
#p_m1_values=p_m1_values/np.trapz(p_m1_values,m1_values)

#fig, ax = plt.subplots(figsize=(15,10))
#ax.tick_params(axis='both', which='major', labelsize=25)
#ax.yaxis.get_offset_text().set_fontsize(25)
## Plot the histogram of the sampled probabilities
#ax.hist(m1_samples, bins=100, density=True, alpha=0.6, color='orange', label='Sampled $M_1$')

## Plot the theoretical distribution
#ax.plot(m1_values[m1_values>7], p_m1_values[m1_values>7], label='Theoretical $p(m_1)$', color='teal',linewidth=3)

## Labels and title
#ax.set_xlabel('$m_1$', fontsize=15)
#ax.set_ylabel('$P(m_1)$', fontsize=15)
#plt.title('Theoretical Distribution vs Sampled Histogram', fontsize=18)

## Set log scale
#ax.set_yscale('log')

## Legends
#ax.legend(loc='upper right',prop={'size': 15})

## Show plot
#plt.grid(axis='y', alpha=0.75)

################################Assign m1#############################################
DS_From_Parent['M1'] = np.random.choice(num_samples, size=DS_From_Parent.shape[0])

fig, ax = plt.subplots(figsize=(15,10))
ax.tick_params(axis='both', which='major', labelsize=25)
ax.yaxis.get_offset_text().set_fontsize(25)
# Plot the histogram of the sampled probabilities
ax.hist(DS_From_Parent['M1'], bins=100, density=True, alpha=0.6, color='orange', label='Sampled $M_1$')

# Plot the theoretical distribution
ax.plot(m1_values, p_m1_values, label='Theoretical $p(m_1)$', color='teal',linewidth=3)

# Labels and title
ax.set_xlabel('$m_1$', fontsize=15)
ax.set_ylabel('$P(m_1)$', fontsize=15)
plt.title('Theoretical Distribution vs Sampled Histogram', fontsize=18)

# Set log scale
ax.set_yscale('log')

# Legends
ax.legend(loc='upper right',prop={'size': 15})

# Show plot
plt.grid(axis='y', alpha=0.75)
plt.savefig('m1_extracted.png')
##################################Assign q###########################################
def p_q(q, m1, m_min=m_min, dm=dm, b=b):
    return S(m1*q, m_min, dm) * q**b


#p_q_with_params = partial(p_q, m1=m1, m_min=m_min, dm=dm, b=b)
#q_samples = return_samples(p_q_with_params, 0, 1, 1)

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

# Prepare arguments for the pool
args = [(m1, m_min, dm, b) for m1 in DS_From_Parent['M1']]

# Use multiprocessing Pool to speed up the computation
with Pool(NCORE) as pool:
    results = pool.map(calculate_q_and_m2_wrapper, args)

# Extract q and M2 from results and assign to the DataFrame
DS_From_Parent['q'], DS_From_Parent['M2'] = zip(*results)
print(DS_From_Parent.head())
os.chdir(CAT_PATH)
DS_From_Parent.to_csv('DS_From_Parent_Uniform_Complete.txt', header=None, index=False)
os.chdir(THIS_DIR)