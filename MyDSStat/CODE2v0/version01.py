import numpy as np
import pandas as pd
import healpy as hp

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from scipy import integrate
from scipy import interpolate
from scipy.optimize import fsolve
#from scipy.special import erfc

from astropy.cosmology import FlatLambdaCDM

import os
from os import listdir
from os.path import isfile, join

from multiprocessing import Pool
import multiprocessing
import pickle
import time
import h5py
from numba import njit
from tqdm import tqdm
import sys

import gwfast.gwfastGlobals as glob
import gwfast
from gwfast.waveforms import IMRPhenomD,IMRPhenomHM
from gwfast.gwfastUtils import load_population


#---------------------script import------------------------------------------
from SkyMap import GWskymap
from GalaxyCat import GalCat
#-----------------------Costants-----------------------------------------
href=67 #69
Om0GLOB=0.319
Xi0Glob =1.
clight = 2.99792458* 10**5#km/s
cosmoflag = FlatLambdaCDM(H0=href, Om0=Om0GLOB)
#-----------------Functions--------------------------
def uniform_volume(iterations):
#for i in tqdm(range(flagship.shape[0])):
#for i in range(iterations):
    i=iterations
    numevent=i
    phigal=new_phi_gals[i]
    thetagal=new_theta_gals[i]
    dc=dc_gals[i]
    #----------z----------------------
    zz=z_from_dcom(dc)
    dl=Dl_z(zz,href,Om0GLOB)#dc*(1+zz)
    #----------row to append---------------------
    proxy_row={'Ngal':numevent,'Comoving Distance':dc,'Luminosity Distance':dl,
               'z':zz,'phi':phigal,'theta':thetagal
          }
    return proxy_row

def z_from_dL(dL_val):
    '''
    Returns redshift for a given luminosity distance dL (in Mpc)'''
    
    func = lambda z :cosmoflag.luminosity_distance(z).value - dL_val
    z = fsolve(func, 0.02)
    return z[0]

def z_of_h_dl(h,dl):
    func = lambda z :Dl_z(z, h, Om0GLOB) -dl
    zmax = fsolve(func, 0.02)[0] 
    return zmax

def z_from_dcom(dc_val):
    '''
    Returns redshift for a given comoving distance dc (in Mpc)'''
    
    func = lambda z :cosmoflag.comoving_distance(z).value - dc_val
    z = fsolve(func, 0.02)
    return z[0]

@njit
def sphere_uncorr_gauss(x,y,mux,muy,sigx,sigy):
    #correlation is 0 so is a multiplication of two gaussians
    #x is theta, y is phi
    #meanvec=(mux,muy)
    meanvec=np.asarray((np.sin(mux)*np.cos(muy),np.sin(mux)*np.sin(muy),np.cos(mux)))
    norm=np.sqrt(np.sum(meanvec**2))
    meanvec=meanvec/norm
    
    #var=(x,y)
    var=np.asarray((np.sin(x)*np.cos(y),np.sin(x)*np.sin(y),np.cos(x)))
    #norm=np.sqrt(np.dot(var,var))
    norm=np.sqrt(np.sum(var**2))
    var=var/norm
    
    diff=meanvec-var
    diff_len=np.sqrt(np.sum(diff**2))
    #xfactor=((x-mux)/sigx)**2
    #yfactor=((y*(1-np.sin(y))-muy*(1-np.sin(muy)))/sigy)**2
    #yfactor=((y-muy)/sigy)**2
    #norm=2*np.pi*sigx*sigy
    factor=((diff_len)/sigy)**2
    ret=np.exp(-(1/2)*(factor))#/norm
    #ret=np.exp(-1/2*(xfactor+yfactor))
    return ret

#----------VonMisesFisher_Sampler-----------------------------------
def random_VMF ( mu , kappa , size = None ) :
# https://hal.science/hal-04004568/
# parse input parameters
    n = 1 if size is None else np.product(size)
    shape = () if size is None else tuple(np.ravel(size))
    mu = np.asarray(mu)
    mu = mu/np.linalg.norm(mu)
    (d ,) = mu.shape
    # zcomponent:radial samples perpendicular to mu
    z = np.random.normal(0,1,(n,d))
    z /= np.linalg.norm(z,axis=1,keepdims=True)
    z = z - (z@mu[:,None])*mu[None, :]
    z /= np.linalg.norm(z,axis =1,keepdims=True )
    # sample angles ( in cos and sin form )
    cos = random_VMF_angle(d,kappa,n)
    sin = np.sqrt(1-cos ** 2)
    # combine angles with the z component
    x = z * sin[ : , None] + cos[ : , None] * mu[None , : ]
    return x .reshape((*shape,d))
def random_VMF_angle(d:int , k:float, n:int ) :
    alpha = (d - 1)/2
    t0 = r0 = np.sqrt(1 + (alpha / k)** 2) - alpha/k
    log_t0 = k*t0+(d-1)*np.log(1 - r0*t0)
    found = 0
    out = [ ]
    while found < n :
        m = min(n,int(( n - found ) * 1.5 ))
        t = np.random.beta(alpha,alpha,m )
        t = 2 * t - 1
        t = (r0 + t) / (1 + r0*t)
        log_acc = k*t +(d - 1)*np.log (1 - r0 * t) - log_t0
        t = t[np.random.random(m) < np.exp( log_acc )]
        out.append(t)
        found += len(out[ - 1 ])
    return np.concatenate(out) [:n]
#------------------------------------------------------------------

@njit
def E_z(z, H0, Om=Om0GLOB):
    return np.sqrt(Om*(1+z)**3+(1-Om))

def r_z(z, H0, Om=Om0GLOB):
    c = clight
    integrand = lambda x : 1/E_z(x, H0, Om)
    integral, error = integrate.quad(integrand, 0, z)
    return integral*c/H0

def Dl_z_vett(z, H0, Om=Om0GLOB):
    c = clight
    integral = np.zeros_like(z)  # Array vuoto per salvare i risultati

    for i, z_val in enumerate(z):
        integrand = lambda x: 1 / E_z(x, H0, Om)
        integral[i], error = integrate.quad(integrand, 0, z_val)
        integral[i]=integral[i]*(1+z_val)

    return integral * c / H0

def Dl_z(z, H0, Om=Om0GLOB):
    return r_z(z, H0, Om)*(1+z)

def z_from_dcom(dc_val):
    '''
    Returns redshift for a given comoving distance dc (in Mpc)'''
    
    func = lambda z :cosmoflag.comoving_distance(z).value - dc_val
    z = fsolve(func, 0.02)
    return z[0]



#--------------------Posterior Function----------------------------


# def singlebetaline(iterator):
#     i=iterator
#     Htemp=H0Grid[i]
#     func = lambda z :Dl_z(z, Htemp, Om0GLOB) -mydlmax#(mu+s*how_many_sigma*mu)#mydlmax
#     zMax = fsolve(func, 0.02)[0]    
#     func = lambda z :Dl_z(z, Htemp, Om0GLOB) -mydlmin#(mu-s*how_many_sigma*mu)#mydlmin
#     zmin = fsolve(func, 0.02)[0]
#     tmp=allz[allz>=zmin] # host with z>z_min
#     tmp=tmp[tmp<=zMax]  # host with z_min<z<z_max  
#     gal_invol=(len(tmp))
#     if gal_invol==0:
#         gal_invol=gal_invol+1

#     ret=gal_invol#/gal_incat
#     return ret

def new_angular_prob(mean_vec,gal_vec,inv_perm_cov):
    exponent=-(1/2)*((mean_vec-gal_vec).T)@inv_perm_cov@((mean_vec-gal_vec))
    #norm=
    return np.exp(exponent)


@njit
def likelihood_line(mu,dl,sigmamean):
    norm=1/(np.sqrt(2*np.pi)*sigmamean)
    body=np.exp(-((dl-mu)**2)/(2*sigmamean**2))
    ret=norm*body
    return ret

#@njit
#def likelihood_line(mu_DS, dl, sigma):
#    norm = 1 / (np.sqrt(2 * np.pi) * sigma)
#    body = np.exp(-((dl - mu_DS) ** 2) / (2 * sigma ** 2))
#    return norm * body


def LikeofH0(iterator):
    i=iterator
    Htemp=H0Grid[i]
    #----------computing sum
    to_sum=np.zeros(len(z_gals))
    for j in range(len(z_gals)):
        #dl=FlatLambdaCDM(H0=Htemp, Om0=Om0GLOB).luminosity_distance(z_gals[j]).value
        dl = Dl_z(z_gals[j], Htemp, Om0GLOB)
        #a=0.01
        vec=hp.ang2vec(new_theta_gals[j],new_phi_gals[j])
        template_gal[:3]=vec
        gal_vec=template_gal 
        angular_prob=new_angular_prob(mean_vec,gal_vec,inv_perm_cov)
        to_sum[j]=likelihood_line(mu,dl,sigmamean)*angular_prob#*stat_weights(z_gals[j])
        
    tmp=np.sum(to_sum)#*norm
    #denom_cat=allz[allz<=20]
    #denom=np.sum(w(denom_cat))
    return tmp#/denom 


def LikeofH0_pixel(mu_DS, sigma, z_hosts, Htemp):
    to_sum = np.zeros(len(z_hosts))
    for v in range(len(z_hosts)):
        dl = Dl_z(z_hosts[v], Htemp, Om0GLOB)
        to_sum[v] = likelihood_line(mu_DS, dl, sigma)
    return np.sum(to_sum)

# Parallelized function to compute the pixel likelihood
def compute_pixel_likelihood(args):
    pix, mu_pix, sigma_pix, z_hosts, H0Grid, angular_prob = args
    pixel_post = np.zeros(len(H0Grid))
    # Print debug info to compare with sequential version
    #print(f'Processing pixel: {pix}')
    
    # Loop over H0Grid and compute the likelihood for each value of H0
    for j, h in enumerate(H0Grid):
        pixel_post[j] = LikeofH0_pixel(mu_pix, sigma_pix, z_hosts, h) * angular_prob
    
    return pixel_post

def list_perm(lista,permutazione):
    tmp=[]
    for e in permutazione:
        tmp.append(lista[e])
    return tmp

def permutation(args):
    mean,cov,keys=args
    
    dL_pos = keys.index('dL')
    theta_pos = keys.index('theta')
    phi_pos = keys.index('phi')
    iota_pos=keys.index('iota')
    eta_pos=keys.index('eta')
    phicoal_pos=keys.index('Phicoal')
    tcoal_pos=keys.index('tcoal')
    psi_pos=keys.index('psi')
    remaining_indices = list(set(range(len(keys))) - {dL_pos, theta_pos,phi_pos
                                                    ,tcoal_pos,psi_pos,iota_pos,eta_pos,phicoal_pos})
    perm = [dL_pos, theta_pos,phi_pos,tcoal_pos,psi_pos,iota_pos,eta_pos,phicoal_pos] + remaining_indices
    #mean_permuted = np.array(mean)[perm]
    mean_permuted = np.array(mean)[perm]
    cov_permuted = cov[np.ix_(perm, perm)]
    keys_permuted=list_perm(keys,perm)
    return mean_permuted,cov_permuted,keys_permuted

def cat2parameter(args):
    catalogue, keys = args
    
    missing_columns = [key for key in keys if key not in catalogue.columns]
    if missing_columns:
        raise ValueError(f"Some keys are missing in the DataFrame: {missing_columns}")
    
    # Reorder the catalogue according to the order in keys
    catalogue_permuted = catalogue[keys]
    
    return catalogue_permuted
###########################################################################################################
#----------------------Main-------------------------------------------------------------------------------#
if __name__=='__main__':
    folder='Uniform/TestRun00/'
    CAT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/'
    SCRIPT_FOLDER='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/'
    COV_SAVE_PATH='/storage/DATA-03/astrorm3/Users/rcianca/DarkSirensStat/MyDSStat/CODE2v0/Events/'+folder
    #######################GW-Events#####################################################
    print('Loading GW data')
    fname='oldGWtest00.fits'# importare la lista da un config
    #working_dir=os.getcwd()
    level=0.9
    DSs=GWskymap(COV_SAVE_PATH+fname,level=level)
    print('test GWskymap class\nPrinting some info')
    print('DS name {}'.format(DSs.event_name))
    print('Area of DS is {} deg^2 at 90%'.format(DSs.area()))
    pix_selected=DSs.get_credible_region_pixels(level=level)
    nside=int(DSs.nside)
    print('nside is {}'.format(nside))
    skyprob=DSs.p_posterior
    allmu=DSs.mu*1000#servono in Mpc 
    allsigma=DSs.sigma*1000
    print(allmu)
    print(allsigma)
    if np.isnan(allmu).any():
        print('There are NaN in allmu')
    if np.isnan(allsigma).any():
        print('There are NaN in allsigma')
    mumean=(np.sum(allmu*skyprob))/np.sum(skyprob)
    sigmamean=(np.sum(allsigma*skyprob))/np.sum(skyprob)
    print('mu_pesato= {} Mpc'.format(mumean))
    thetas,phis=hp.pix2ang(nside,pix_selected)
    print('DS data:')
    print('pix selected ={}'.format(len(pix_selected)))
    print('len dL={}'.format(len(allmu[pix_selected])))
    #########################Galaxy-Catalogue#############################################
    print('Reading Galaxy Catalogue')
    #reading the catalogue and selecting the pixel
    to_read='Uniform_paper.txt'
    hostcat=GalCat(to_read,nside).read_catalogue()
    print('Reading catalogue completed')
    mypixels=GalCat(to_read,nside).pixelizer()
    hostcat['Pixel']=mypixels
    mask = hostcat['Pixel'].isin(pix_selected)
    hostcat_filtered=hostcat[mask]
    #print('hostcat shape {}'.format(len(z_hosts)))
    ############################Reading Cov Matrix######################################
    Cov_file='Cov_SNR_more_than_100_200.npy'
    Population='SNR_more_than_100_200.h5'
    tosave=load_population(COV_SAVE_PATH+Population)
    allcov = np.load(COV_SAVE_PATH+Cov_file, allow_pickle=True)
    Allevents_DS_fromfile = pd.DataFrame.from_dict(tosave, orient='columns')

    parameters_list=list(IMRPhenomHM().ParNums.keys())
    args=Allevents_DS_fromfile,parameters_list
    Allevents_DS=cat2parameter(args)
    selected=52
    mean = np.array(Allevents_DS.iloc[selected])
    cov = np.float64(allcov[:, :, selected])
    args = mean, cov, parameters_list
    perm_mean, perm_cov, perm_keys = permutation(args)
    print(perm_keys)
    inv_perm_cov=np.linalg.inv(perm_cov)
    simple_mean=perm_mean
    simple_mean[3:]=0
    template_gal=simple_mean
    ###Cross-Correlation#############################################
    Event_dict = {
        'Event': [],
        'Likelihood': np.array([]),
        'beta': np.array([]),
        'posterior': np.array([])
    }

    H0min=40
    H0max=100
    H0Grid=np.linspace(H0min,H0max,1000)

    
    total_post=np.zeros(len(H0Grid))# this will be the total for all the events
    single_post=np.zeros(len(H0Grid))

    NCORE=multiprocessing.cpu_count()
    print('using {} cpu'.format(NCORE))
    arr=np.arange(0,len(H0Grid),dtype=int)
    My_Like=np.zeros(len(H0Grid))
    template_gal=simple_mean
    NumDS=1
    print('start for loop on DS')
    for i in tqdm(range(NumDS)):
        DS_phi=Allevents_DS.iloc[selected]['phi']
        DS_theta=Allevents_DS.iloc[selected]['theta']
        mu=mumean
        tmp=hp.ang2vec(DS_theta,DS_phi)
        simple_mean[0:3]=tmp
        mean_vec=simple_mean
        zz=z_from_dL(mu)
        z_gals=np.asarray(hostcat_filtered['z'])
        new_phi_gals=np.asarray(hostcat_filtered['phi'])
        new_theta_gals=np.asarray(hostcat_filtered['theta'])
        #tmp=hp.ang2vec(new_theta_gals,new_phi_gals)
        #gal_vec=
        with Pool(NCORE) as p:
            My_Like=p.map(LikeofH0, arr)

#############################################################################################
    np.save(COV_SAVE_PATH+'event_data_version1.npy',My_Like)

    fig, ax = plt.subplots(1, figsize=(15,10)) #crea un tupla che poi è più semplice da gestire
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.get_offset_text().set_fontsize(25)
    ax.grid(linestyle='dotted', linewidth='0.6')#griglia in sfondo


    x=H0Grid
    xmin=np.min(x)
    xmax=np.max(x)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r'$H_0(Km/s/Mpc)$', fontsize=30)
    #ax.set_ylabel(r'$P(H_0)$', fontsize=20)
    ax.set_ylabel(r'$Posterior(H_0)$', fontsize=30)
    if xmin<href<xmax:
        ax.axvline(x = href, color = 'k', linestyle='dashdot',label = 'H0=67')

    Mycol='teal'
    ax.plot(x,My_Like/np.trapz(My_Like,x),label='Total_posterior',color=Mycol,linewidth=4,linestyle='solid')
    ax.legend(fontsize=13, ncol=2) 

    plotpath=os.path.join(COV_SAVE_PATH+'GWtest00_pool_oldversion.pdf')
    plt.savefig(plotpath, format="pdf", bbox_inches="tight")
