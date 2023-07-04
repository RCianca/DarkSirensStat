import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from scipy import integrate
from scipy import interpolate
from scipy.optimize import fsolve
#from scipy.special import erfc

import math

from astropy.cosmology import FlatLambdaCDM

import os
from os import listdir
from os.path import isfile, join

from multiprocessing import Pool
import time
from numba import njit
from tqdm import tqdm
import sys

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

def Mises_Fisher(theta,phi,DS_theta,DS_phi,conc):
    meanvec=hp.ang2vec(DS_theta,DS_phi)
    meanvec=np.asarray(meanvec,dtype=np.float128)
    norm=np.sqrt(np.dot(meanvec,meanvec))
    meanvec=meanvec/norm
    
    var=hp.ang2vec(theta,phi)
    var=np.asarray(var,dtype=np.float128)
    norm=np.sqrt(np.dot(var,var))
    var=var/norm
    
    factor=np.dot(conc*var,meanvec)
    factor=np.float128(factor)
    #Normalization is futile, we will devide by the sum
    #fullnorm=conc/(2*np.pi*(np.exp(conc)-np.exp(-conc)))
    ret=np.float128(np.exp(factor))#/fullnorm
    #ret=factor
    return ret

def z_from_dL(dL_val):
    '''
    Returns redshift for a given luminosity distance dL (in Mpc)'''
    
    func = lambda z :cosmoflag.luminosity_distance(z).value - dL_val
    z = fsolve(func, 0.02)
    return z[0]
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
    meanvec=np.asarray((1,mux,muy))
    norm=np.sqrt(np.sum(meanvec**2))
    meanvec=meanvec/norm
    
    #var=(x,y)
    var=np.asarray((1,x,y))
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

def gaussian_prob_distribution_on_sphere(phi, theta, phi0, theta0, radius, sigma):
    # If input is a single point, convert to a list
    if isinstance(phi, float) or isinstance(phi, int):
        phi = np.array([phi])
        theta = np.array([theta])

    # Calculate the distance from each point to the fixed point
    distances = np.zeros_like(phi)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    dx = x - radius * np.sin(phi0) * np.cos(theta0)
    dy = y - radius * np.sin(phi0) * np.sin(theta0)
    dz = z - radius * np.cos(phi0)
    distances = np.sqrt(dx*dx + dy*dy + dz*dz)

    # Calculate the Gaussian probability distribution for each point
    prob = 1.0/(sigma*np.sqrt(2*np.pi)) * np.exp(-distances*distances/(2*sigma*sigma))

    # If input was a single point, return a single probability value
    if len(prob) == 1:
        return prob[0]

    return prob

@njit
def E_z(z, H0, Om):
    return np.sqrt(Om*(1+z)**3+(1-Om))

def r_z(z, H0, Om):
    c = clight
    integrand = lambda x : 1/E_z(x, H0, Om)
    integral, error = integrate.quad(integrand, 0, z)
    return integral*c/H0

def r_z_vett(z, H0, Om):
    c = clight
    integral = np.zeros_like(z)  # Array vuoto per salvare i risultati

    for i, z_val in enumerate(z):
        integrand = lambda x: 1 / E_z(x, H0, Om)
        integral[i], error = integrate.quad(integrand, 0, z_val)

    return integral * c / H0

def Dl_z(z, H0, Om):
    return r_z(z, H0, Om)*(1+z)

def z_from_dcom(dc_val):
    '''
    Returns redshift for a given comoving distance dc (in Mpc)'''
    
    func = lambda z :cosmoflag.comoving_distance(z).value - dc_val
    z = fsolve(func, 0.02)
    return z[0]



#--------------------Posterior Function----------------------------
@njit
def truncated_part(x, mu, k):
    #x is the dl of the gal
    #mu is the pos of the DS
    #lower set to zero. This should move the DS
        #https://stats.stackexchange.com/questions/2037/estimating-mean-and-st-dev-of-a-truncated-gaussian-curve-without-spike
    wall=dlmaxcat
    sigma=k*true_mu
    Phialpha = 0.5*math.erfc(-(x-wall)/(np.sqrt(2)*sigma))
    #Phialpha = 0.5*(1+math.erf((-mu)/(np.sqrt(2)*sigma)))
    #print('sigma is {} len is {}'.format(sigma,len(sigma)))
    return np.where(x>0, (1/((np.sqrt(2*np.pi)*sigma)*(1-Phialpha+0.000001))) * np.exp(-(x-mu)**2/(2*sigma**2)) ,0.)
@njit
def likelihood_line(mu,dl,k):
    sigma=k*true_mu
    norm=1/(np.sqrt(2*np.pi)*sigma)
    body=np.exp(-((dl-mu)**2)/(2*sigma**2))
    ret=norm*body
    return ret

def LikeofH0(iterator):
    i=iterator
    Htemp=H0Grid[i]
    #norm=integrate.quad(lambda x:  FlatLambdaCDM(H0=Htemp, Om0=Om0GLOB).differential_comoving_volume(x).value, 0, 10 )[0]
    #----------computing sum
    to_sum=np.zeros(len(z_gals))
    for j in range(len(z_gals)):
        #dl=FlatLambdaCDM(H0=Htemp, Om0=Om0GLOB).luminosity_distance(z_gals[j]).value
        dl = Dl_z(z_gals[j], Htemp, Om0GLOB)
        #a=0.01
        angular_prob=sphere_uncorr_gauss(new_phi_gals[j],new_theta_gals[j],DS_phi,DS_theta,sigma_phi,sigma_theta)
        to_sum[j]=likelihood_line(mu,dl,s)*angular_prob#*stat_weights(z_gals[j])#w(z_gals[j])#*(1+z_gals[j])
        #to_sum[j]=truncated_part(dl,mu,s)*angular_prob#*stat_weights(z_gals[j])#w(z_gals[j])#*(1+z_gals[j])
        
    tmp=np.sum(to_sum)#*norm
    #denom_cat=allz[allz<=20]
    #denom=np.sum(w(denom_cat))
    return tmp#/denom

@njit
def beta_line(galaxies,z0,z1,zmax):
    denom=len(galaxies[(galaxies>=z0)&(galaxies<=z1)])
    num=len(galaxies[galaxies<=zmax])
    ret=num/denom
    return ret
@njit
def stat_weights(array_of_z):
    #alltheomega=w(array_of_z)
    temp=np.interp(array_of_z,z_bin,w_hist)
    return temp
def multibetaline(iterator):
    i=iterator
    Htemp=H0Grid[i]
    cosmo=FlatLambdaCDM(H0=Htemp, Om0=Om0GLOB)
    func = lambda z :Dl_z(z, Htemp, Om0GLOB) -(mu+s*how_many_sigma*mu)#10188.4#
    zMax = fsolve(func, 0.02)[0] 
    #zMax=min(zMax,zmax_cat)
    
    func = lambda z :Dl_z(z, Htemp, Om0GLOB) - (mu-s*how_many_sigma*mu)
    zmin = fsolve(func, 0.02)[0]
    #zmin=max(zmin,zmin_cat)

    tmp=allz[allz>=zmin]
    tmp=tmp[tmp<=zMax]
    #---------------------------Volume-------------------------------------------------
    #colnames=['Ngal','Comoving Distance','Luminosity Distance','z','phi','theta']
    #phimax=np.max(sliced_for_beta['phi'])
    #phimin=np.min(sliced_for_beta['phi'])
    #thetamax=np.max(sliced_for_beta['theta'])
    #thetamin=np.min(sliced_for_beta['theta'])
    #delta_phi=phimax-phimin
    #theta_part=np.cos(thetamin)-np.cos(thetamax)
    #integrand=lambda x:clight*(cosmo.comoving_distance(x).value)**2/(cosmo.H(x).value)
    #z_part=integrate.quad(integrand,zmin,zMax)[0]

    #Volume=delta_phi*theta_part*z_part#integrand
    gal_invol=(len(tmp))
    
    #func = lambda z :Dl_z(z, href, Om0GLOB) -(mu+s*how_many_sigma*mu)#10188.4#
    #zMax_fix = fsolve(func, 0.02)[0] 
    #zM=min(zMax_fix,zmax_cat)
    
    #func = lambda z :Dl_z(z, Htemp, Om0GLOB) - (mu-s*how_many_sigma*mu)
    #zmin_fix = fsolve(func, 0.02)[0]
    #zm=max(zmin_fix,zmin_cat)    
    
    #integrand=lambda x:clight*(cosmo.comoving_distance(x).value)**2/(cosmo.H(x).value)
    #Tot_z_part=integrate.quad(integrand,0,20)[0]
    gal_incat=len(allz[allz<=20])
    if gal_invol==0:
        gal_invol=gal_invol+1

    ret=gal_invol/gal_incat
    return ret
@njit
def sum_stat_weights(array_of_z):
    #alltheomega=w(array_of_z)
    num=np.sum(np.interp(array_of_z,z_bin,w_hist))
    return num
def multibetaline_stat(iterator):
    i=iterator
    Htemp=H0Grid[i]
    func = lambda z :Dl_z(z, Htemp, Om0GLOB)# -(mu+s*how_many_sigma*mu)#10188.4#
    zMax = fsolve(func, 0.02)[0] 
    func = lambda z :Dl_z(z, Htemp, Om0GLOB)# - (mu-s*how_many_sigma*mu)
    zmin = fsolve(func, 0.02)[0]
    #z_gals Try to use the cone no cuts :default is allz:
    #Next: z_cone and z_gals to reduce time---se if the volume is equal :define z_cone:
    tmp=allz[allz>=zmin]
    tmp=tmp[tmp<=zMax]
    
    #phimax=np.max(sliced_for_beta['phi'])
    #phimin=np.min(sliced_for_beta['phi'])
    #thetamax=np.max(sliced_for_beta['theta'])
    #thetamin=np.min(sliced_for_beta['theta'])
    #delta_phi=phimax-phimin
    #theta_part=np.cos(thetamin)-np.cos(thetamax)
    #integrand=lambda x:clight*(r_z(x, Htemp, Om0GLOB))**2/(Htemp)
    #z_part=integrate.quad(integrand,zmin,zMax)[0]

    Volume=delta_phi*theta_part*z_part#integrand
    #gal_invol=len(tmp)
    num=sum_stat_weights(tmp)*Volume
    #denom_cat=allz[allz<=20]
    vol_denom=denom*delta_phi*theta_part*Tot_z_part
    #gal_incat=len(allz[allz<=20])
    if num==0:
        num=num+1

    ret=num/denom
    return ret



def vol_beta(iterator):
    i=iterator
    Htemp=H0Grid[i]
    cosmo=FlatLambdaCDM(H0=Htemp, Om0=Om0GLOB)
    func = lambda z :Dl_z(z, Htemp, Om0GLOB) - (mu+s*how_many_sigma*mu)
    zMax = fsolve(func, 0.02)[0] 
    func = lambda z :Dl_z(z, Htemp, Om0GLOB) - (mu-s*how_many_sigma*mu)
    zmin = fsolve(func, 0.02)[0] 
    
    zMax=min(zMax,zmax_cat)
    zmin=max(zmin,zmin_cat)
    
    integrand=lambda x:clight*(cosmo.comoving_distance(x).value)**2/(cosmo.H(x).value)
    num=integrate.quad(integrand,zmin,zMax)[0]
    integrand=lambda x:clight*(cosmo.comoving_distance(x).value)**2/(cosmo.H(x).value)
    norm=integrate.quad(integrand,0,20)[0]  
    
    return num/norm
def testbeta(iterator):
    i=iterator
    Htemp=H0Grid[i]
    cosmo=FlatLambdaCDM(H0=Htemp, Om0=Om0GLOB)
    func = lambda z :Dl_z(z, Htemp, Om0GLOB) - (mu+s*how_many_sigma*mu)
    zMax = fsolve(func, 0.02)[0] 
    func = lambda z :Dl_z(z, Htemp, Om0GLOB) - (mu-s*how_many_sigma*mu)
    zmin = fsolve(func, 0.02)[0] 
    
    tmp=allz[allz>=zmin]
    tmp=tmp[tmp<=zMax]
    #numerator
    numtosum=r_z_vett(tmp, Htemp, Om0GLOB)
    numtosum=numtosum**2
    numtosum=numtosum*tmp
    allh=cosmo.H(tmp).value
    num=np.sum(numtosum/allh)
    #denom
    #normtosum=r_z_vett(allz, Htemp, Om0GLOB)
    #normtosum=normtosum**2
    #normtosum=normtosum*allz
    #norm=np.sum(normtosum)/Htemp
    return num#/norm
###########################################################################################################
#----------------------Main-------------------------------------------------------------------------------#
###########################################################################################################
#------------------trigger---------------------
generation=0
read=1
DS_read=0
save=1
#----------------------------------------------
path='results'
exist=os.path.exists(path)
if not exist:
    print('creating result folder')
    os.mkdir('results')
runpath='P0_dirac_28'
folder=os.path.join(path,runpath)
os.mkdir(folder)
print('data will be saved in '+folder)
H0min=55#30
H0max=85#140
H0Grid=np.linspace(H0min,H0max,1000)
NCORE=15
#z_bin=np.loadtxt('weights_bin.txt')
#w_hist=np.loadtxt('weights.txt')
#w_hist=np.loadtxt('weights_halved.txt')
#w=interpolate.CubicSpline(z_bin,w_hist,extrapolate='None')
cat_name='FullExplorer_big.txt'


if generation==1:
#------------------points generator------------------
    nsamp=1000000#6500000+2156000
    z_inf_cat=0.05#0.79
    z_sup_cat=2.5#2

    dcom_min=cosmoflag.comoving_distance(z_inf_cat).value
    dcom_max=cosmoflag.comoving_distance(z_sup_cat).value
    dl_min=cosmoflag.luminosity_distance(z_inf_cat).value
    dl_max=cosmoflag.luminosity_distance(z_sup_cat).value


    #---------angular stuff------------------
    radius_deg= np.sqrt(10/np.pi)
    sigma90=radius_deg/np.sqrt(2)
    sigma_deg=sigma90/1.5
    circle_deg=6*sigma_deg
    sigma_theta=np.radians(sigma_deg)
    sigma_phi=np.radians(sigma_deg)
    radius_rad=np.radians(circle_deg)
    #----------------------------------------------------------------
    phi_min=0
    phi_max=np.pi/2
    theta_min=0
    theta_max=np.pi/2


    print('Cosmology: Flat Universe. H0={}, OmegaM={}'.format(href,Om0GLOB))
    print('Parameters:\nH0_min={}, H0_max={}'.format(H0min,H0max))
    print('Catalogue:\nz_min={}, z_max={},\nphi_min={}, phi_max={}, theta_min={}, theta_max={}'.format(z_inf_cat,z_sup_cat,phi_min,phi_max,theta_min,theta_max))


    u = np.random.uniform(0,1,size=nsamp) # uniform random vector of size nsamp
    dc_gals_all     = np.cbrt((u*dcom_min**3)+((1-u)*dcom_max**3))
    phi_gals   = np.random.uniform(phi_min,phi_max,nsamp)
    theta_gals = np.arccos( np.random.uniform(np.cos(theta_max),np.cos(theta_min),nsamp) )
    dc_gals=dc_gals_all[dc_gals_all>=dcom_min]
    #num=np.arange(len(dc_gals))
    #z_gals=np.zeros(len(dc_gals))
    #dl_gals=np.zeros(len(dc_gals))
# need to use pool here
    #for i in tqdm(range(len(dc_gals))):
    #    z=z_from_dcom(dc_gals[i])
    #    z_gals[i]=z
    #   dl_gals[i]=Dl_z(z,href,Om0GLOB)
    new_phi_gals=np.random.choice(phi_gals,len(dc_gals))
    new_theta_gals=np.random.choice(theta_gals,len(dc_gals))
    numevent=int(0)
    proxy_row={'Ngal':numevent,'Comoving Distance':0,'Luminosity Distance':0,
                   'z':0,'phi':0,'theta':0
              }
    colnames=['Ngal','Comoving Distance','Luminosity Distance','z','phi','theta']
    MyCat = pd.DataFrame(columns=colnames)
    arr=np.arange(0,len(dc_gals),dtype=int)
    data=[]
    tmp=[]
    print('Generating the catalogue using pool, please wait...')
    with Pool(NCORE) as p:
        tmp=p.map(uniform_volume, arr)
    MyCat=MyCat.append(tmp, ignore_index=True)

    #MyCat = pd.DataFrame(columns=colnames)
    #MyCat['Ngal']=num
    #MyCat['Comoving Distance']=dc_gals
    #MyCat['Luminosity Distance']=dl_gals
    #MyCat['z']=z_gals
    #MyCat['phi']=new_phi_gals
    #MyCat['theta']=new_theta_gals
    print('Saving '+cat_name)
    MyCat.to_csv(cat_name, header=None, index=None, sep=' ')
    del tmp
    del data
    del u    
    del dc_gals
    del phi_gals
    del theta_gals
    del new_phi_gals
    del new_theta_gals
    del dc_gals_all
#------------------------Reading the catalogue----------------------------------
if read==1:
    #cat_name='FullExplorer.txt'
    print('Reading the catalogue: ' + cat_name)
    MyCat = pd.read_csv(cat_name, sep=" ", header=None)
    colnames=['Ngal','Comoving Distance','Luminosity Distance','z','phi','theta']
    MyCat.columns=colnames
    allz=np.asarray(MyCat['z'])
    Min_Box_dl=MyCat['Luminosity Distance'].min()
    Max_Box_dl=MyCat['Luminosity Distance'].max()
    #---------angular stuff------------------
    radius_deg= np.sqrt(10/np.pi)
    sigma90=radius_deg/np.sqrt(2)
    sigma_deg=sigma90/1.5
    circle_deg=6*sigma_deg
    sigma_theta=np.radians(sigma_deg)
    sigma_phi=np.radians(sigma_deg)
    radius_rad=np.radians(circle_deg)
    #----------------------------------------------------------------
    phi_min=MyCat['phi'].min()
    phi_max=MyCat['phi'].max()
    theta_min=MyCat['theta'].min()
    theta_max=MyCat['theta'].max()
    print('Cosmology: Flat Universe. H0={}, OmegaM={}'.format(href,Om0GLOB))
    print('Parameters:\nH0_min={}, H0_max={}'.format(H0min,H0max))
    print('Catalogue:\nz_min={}, z_max={},\nphi_min={}, phi_max={}, theta_min={}, theta_max={}'.format(np.min(allz),np.max(allz),phi_min,phi_max,theta_min,theta_max))
    print('Number of galaxies={}'.format(len(allz)))
#################################DS control room#########################################

if DS_read==1:
    #name=os.path.join(folder,'catname')#move to te right folder
    source_folder='RC_UNIF_FLAG'
    data_path=os.path.join(path,source_folder)
    print('reading an external DS catalogue from '+source_folder)
    sample = pd.read_csv(data_path+'/'+source_folder+'_DSs.txt', sep=" ", header=None)
    colnames=['Ngal','Comoving Distance','Luminosity Distance','z','phi','theta']
    sample.columns=colnames
    
    if save==1:
        cat_name=os.path.join(folder,runpath+'_DSs.txt')
        #cat_name=runpath+'_DSs.txt'
        print('Saving '+cat_name)
        sample.to_csv(cat_name, header=None, index=None, sep=' ')

    ds_z=np.asarray(sample['z'])
    ds_dl=np.asarray(sample['Luminosity Distance'])
    ds_phi=np.asarray(sample['phi'])
    ds_theta=np.asarray(sample['theta'])
    NumDS=len(ds_z)
else:
    NumDS=150#150
    #Selezionare in dcom non z: implementa anche Dirac 
    zds_max=1.75#1.42#1.02
    zds_min=0.3#1.38#0.98
    
    mydlmax=Dl_z(zds_max,href,Om0GLOB)
    mydcmax=mydlmax/(1+zds_max)
    mydlmin=Dl_z(zds_min,href,Om0GLOB)
    mydcmin=mydlmin/(1+zds_min)
    #-----------------------------------------------------------------------------
    cutted=MyCat[MyCat['Comoving Distance']<=mydcmax]
    cutted=cutted[cutted['Comoving Distance']>=mydcmin]
    cutted=cutted[cutted['phi']<= phi_max-10*sigma_phi]
    cutted=cutted[cutted['phi']>= phi_min+10*sigma_phi]
    cutted=cutted[cutted['theta']<= theta_max-10*sigma_theta]
    cutted=cutted[cutted['theta']>= theta_min+10*sigma_theta]

    sample=cutted.sample(NumDS) #This is the DS cat

    if save==1:
        cat_name=os.path.join(folder,runpath+'_DSs.txt')
        #cat_name=runpath+'_DSs.txt'
        print('Saving '+cat_name)
        sample.to_csv(cat_name, header=None, index=None, sep=' ')

    ds_z=np.asarray(sample['z'])
    ds_dl=np.asarray(sample['Luminosity Distance'])
    ds_phi=np.asarray(sample['phi'])
    ds_theta=np.asarray(sample['theta'])
###################################################################################
#---------------------Start analysis--------------------------------------
#some global stuff###########################################
zmax_cat=np.max(allz)
zmin_cat=np.min(allz)
integrand=lambda x:clight*(r_z(x, href, Om0GLOB))**2/(href)
Tot_z_part=integrate.quad(integrand,0,20)[0]
##############################################################
arr=np.arange(0,len(H0Grid),dtype=int)
beta=np.zeros(len(H0Grid))
My_Like=np.zeros(len(H0Grid))
dlsigma=0.1
how_many_sigma=3.5
fullrun=[]
allbetas=[]
s=dlsigma

#---------------------USE WHEN YOU HAVE A N(z)
#denom_cat=allz[allz<=20]
#denom=np.sum(np.interp(denom_cat,z_bin,w_hist))
###################################Likelihood##################################################
for i in tqdm(range(NumDS)):
    DS_phi=ds_phi[i]
    tmp=MyCat[MyCat['phi']<=DS_phi+3.5*sigma_phi]
    tmp=tmp[tmp['phi']>=DS_phi-3.5*sigma_phi]
    DS_theta=ds_theta[i]
    tmp=tmp[tmp['theta']<=DS_theta+3.5*sigma_theta]
    tmp=tmp[tmp['theta']>=DS_theta-3.5*sigma_theta]
    true_mu=ds_dl[i]
    mu= np.random.normal(loc=true_mu, scale=true_mu*s, size=None)#ds_dl[i]#
    dsz=ds_z[i]
    dlrange=s*mu*how_many_sigma
    tmp=tmp[tmp['Luminosity Distance']<=mu+dlrange]
    tmp=tmp[tmp['Luminosity Distance']>=mu-dlrange]
    sliced_for_beta=tmp
    z_gals=np.asarray(tmp['z'])
    new_dl_gals=np.asarray(tmp['Luminosity Distance'])
    new_phi_gals=np.asarray(tmp['phi'])
    new_theta_gals=np.asarray(tmp['theta'])
    #print(tmp.shape[0])
    with Pool(NCORE) as p:
        My_Like=p.map(LikeofH0, arr)
        beta=p.map(multibetaline, arr)
    My_Like=np.asarray(My_Like)
    fullrun.append(My_Like)
    beta=np.asarray(beta)
    allbetas.append(beta)

#############################################################################################
##############################BETA#################################################################
#with Pool(14) as p:
#    beta=p.map(just_vol_beta, arr)
#beta=np.asarray(beta)
###################################################################################################
###########################Saving Results & posterior##############################################
betapath=os.path.join(folder,runpath+'_beta.txt')
np.savetxt(betapath,allbetas)#allbetas
print('Beta Saved')
fullrunpath=os.path.join(folder,runpath+'_fullrun.txt')
np.savetxt(fullrunpath,fullrun)
fullrun_beta=[]#fullrun/beta#[]
print('All likelihood Saved')
for i in range(NumDS):
    fullrun_beta.append(fullrun[i]/allbetas[i])
combined=[]
for i in range(len(fullrun_beta)):
    #combined=combined+post[i]
    if i==0:
        combined.append(fullrun_beta[i]*1)
    else:
        num=np.float128(combined[i-1]*(fullrun_beta[i]*1))
        combined.append(num)

postpath=os.path.join(folder,runpath+'_totpost.txt')
np.savetxt(postpath,combined[-1])
print('posterior saved')
grid=os.path.join(folder,runpath+'_H0grid.txt')
np.savetxt(grid,H0Grid)
print('H0 grid saved')
os.system('cp postcalculator.py '+folder+'/run_postcalculator.py')

import matplotlib.pyplot as plt
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

Mycol='navy'
ax.plot(x,combined[-1]/np.trapz(combined[-1],x),label='Total_posterior',color=Mycol,linewidth=4,linestyle='solid')
ax.legend(fontsize=13, ncol=2) 
newdist=(combined[-1])/np.trapz(combined[-1],x)
mean=np.trapz(x*newdist,x)/np.trapz(newdist,x)
std=np.sqrt(np.trapz(newdist*(x-mean)**2,x)/np.trapz(newdist,x))
plt.figtext(0.75,0.6,'Mean={:0.2f}'.format(mean),fontsize=18,c=Mycol)
plt.figtext(0.75,0.55,'Std={:0.2f}'.format(std),fontsize=18, c=Mycol)
print('mean={},std={} std/mean={}%'.format(mean,std,100*std/mean))
plotpath=os.path.join(folder,runpath+'_PostTot.pdf')
plt.savefig(plotpath, format="pdf", bbox_inches="tight")
