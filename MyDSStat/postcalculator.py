import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from scipy import integrate
from scipy.optimize import fsolve

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
#----------------------Functions----------------------------------------
def z_from_dL(dL_val):
    '''
    Returns redshift for a given luminosity distance dL (in Mpc)'''
    
    func = lambda z :cosmoflag.luminosity_distance(z).value - dL_val
    z = fsolve(func, 0.77)
    return z[0]
def z_from_dcom(dc_val):
    '''
    Returns redshift for a given comoving distance dc (in Mpc)'''
    
    func = lambda z :cosmoflag.comoving_distance(z).value - dc_val
    z = fsolve(func, 0.77)
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
    ret=np.exp(-(1/2)*(factor))/norm
    #ret=np.exp(-1/2*(xfactor+yfactor))
    return ret

@njit
def E_z(z, H0, Om):
    return np.sqrt(Om*(1+z)**3+(1-Om))

def r_z(z, H0, Om):
    c = clight
    integrand = lambda x : 1/E_z(x, H0, Om)
    integral, error = integrate.quad(integrand, 0, z)
    return integral*c/H0

def Dl_z(z, H0, Om):
    return r_z(z, H0, Om)*(1+z)



@njit
def normalisation(galaxies,zmax):
    #is a beta after some simplification with the likelihood
    #cat must be complete, omega must be a constant we set that all to 1
    ret=len(galaxies[galaxies<=zmax])
    return ret


#--------------------Posterior Function----------------------------
@njit
def beta_line(galaxies,z0,z1,zmax):
    denom=len(galaxies[(galaxies>=z0)*(galaxies<=z1)])
    num=len(galaxies[galaxies<=zmax])
    ret=num/denom
    return ret
@njit
def likelihood_line(mu,dl,k=0.1):
    sigma=k*mu
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
        dl = Dl_z(z_gals[j], Htemp, Om0GLOB)              #angular_prob=sphere_uncorr_gauss(new_phi_gals[j],new_theta_gals[j],DS_phi,DS_theta,sigma_phi,sigma_theta)
        to_sum[j]=likelihood_line(mu,dl,s)*1#*norm
    tmp=np.sum(to_sum)#*norm
    return tmp

def multibeta(iterator):
    i=iterator
    Htemp=H0Grid[i]
    #cosmo=FlatLambdaCDM(H0=Htemp, Om0=Om0GLOB)
    func = lambda z :Dl_z(z, Htemp, Om0GLOB) - betaHomdMax
    zmax = fsolve(func, 0.5)[0]
    Num=normalisation(z_gals,zmax)
    if Num==0:
        Num=Num+1
    return Num
def multibetaline(iterator):
    i=iterator
    Htemp=H0Grid[i]
    #cosmo=FlatLambdaCDM(H0=Htemp, Om0=Om0GLOB)
    func = lambda z :Dl_z(z, Htemp, Om0GLOB) - betaHomdMax
    z1 = fsolve(func, 0.5)[0]
    ret=beta_line(z_gals,0,z1,10)
    return ret

def vol_beta(iterator):
    i=iterator
    Htemp=H0Grid[i]
    cosmo=FlatLambdaCDM(H0=Htemp, Om0=Om0GLOB)
    func = lambda z :Dl_z(z, Htemp, Om0GLOB) - betaHomdMax
    zMax = fsolve(func, 0.5)[0] 
    func = lambda z :Dl_z(z, Htemp, Om0GLOB) - dl_min_beta
    zmin = fsolve(func, 0.2)[0]  
    norm = integrate.quad(lambda x: cosmo.differential_comoving_volume(x).value,zmin,200)[0]
    num = integrate.quad(lambda x:cosmo.differential_comoving_volume(x).value,zmin,zMax)[0]
    return num/norm

###########################################################################################################
#----------------------Main-------------------------------------------------------------------------------#
###########################################################################################################
#------------------trigger---------------------
generation=1
read=1
DS_read=0
save=1
#----------------------------------------------
path='results'
exist=os.path.exists(path)
if not exist:
    print('creating result folder')
    os.mkdir('results')
runpath='CatGeneration'
folder=os.path.join(path,runpath)
os.mkdir(folder)
print('data will be saved in '+folder)
H0min=55#30
H0max=85#140
H0Grid=np.linspace(H0min,H0max,1000)
nsamp=6500000+2156000
z_inf_cat=0.85#0.79
z_sup_cat=1.15#2

dcom_min=cosmoflag.comoving_distance(z_inf_cat).value
dcom_max=cosmoflag.comoving_distance(z_sup_cat).value
dl_min=cosmoflag.luminosity_distance(z_inf_cat).value
dl_max=cosmoflag.luminosity_distance(z_sup_cat).value

betaHomdMax=Dl_z(z_sup_cat,np.min(H0Grid),Om0GLOB)
dl_min_beta=Dl_z(z_inf_cat,np.max(H0Grid),Om0GLOB)
#---------angular stuff------------------
radius_deg= np.sqrt(10/np.pi)
sigma90=radius_deg/np.sqrt(2)
sigma_deg=sigma90/1.5
circle_deg=6*sigma_deg
sigma_theta=np.radians(sigma_deg)
sigma_phi=np.radians(sigma_deg)
radius_rad=np.radians(circle_deg)
#print('Sigma phi={},Sigma theta={}'.format(sigma_phi,sigma_phi))
#print('Sigma phi={}°,Sigma theta={}°'.format(sigma_deg,sigma_deg))
#----------------------------------------------------------------
phi_min=0
phi_max=np.pi/2
theta_min=0
theta_max=np.pi/2
#print('phi min={},phi Max={}'.format(phi_min,phi_max))
#print('theta min={},theta Max={}'.format(theta_min,theta_max))

print('Cosmology: Flat Universe. H0={}, OmegaM={}'.format(href,Om0GLOB))
print('Parameters:\nH0_min={}, H0_max={}'.format(H0min,H0max))
print('Catalogue:\nz_min={}, z_max={},\nphi_min={}, phi_max={}, theta_min={}, theta_max={}'.format(z_inf_cat,z_sup_cat,phi_min,phi_max,theta_min,theta_max))

if generation==1:
#------------------points generator------------------
    u = np.random.uniform(0,1,size=nsamp) # uniform random vector of size nsamp
    dc_gals_all     = np.cbrt((u*0**3)+((1-u)*dcom_max**3))
    #dc_gals_all     = (u*dcom_min)+(1-u)*dcom_max
    phi_gals   = np.random.uniform(phi_min,phi_max,nsamp)
    theta_gals = np.arccos( np.random.uniform(np.cos(theta_max),np.cos(theta_min),nsamp) )
    dc_gals=dc_gals_all[dc_gals_all>=dcom_min]
    #print(len(dc_gals_all),len(dc_gals),len(dc_gals)/len(dc_gals_all))
    num=np.arange(len(dc_gals))
    z_gals=np.zeros(len(dc_gals))
    dl_gals=np.zeros(len(dc_gals))

    for i in tqdm(range(len(dc_gals))):
        z=z_from_dcom(dc_gals[i])
        z_gals[i]=z
        dl_gals[i]=Dl_z(z,href,Om0GLOB)
    new_phi_gals=np.random.choice(phi_gals,len(dc_gals))
    new_theta_gals=np.random.choice(theta_gals,len(dc_gals))

    colnames=['Ngal','Comoving Distance','Luminosity Distance','z','phi','theta']
    MyCat = pd.DataFrame(columns=colnames)
    MyCat['Ngal']=num
    MyCat['Comoving Distance']=dc_gals
    MyCat['Luminosity Distance']=dl_gals
    MyCat['z']=z_gals
    MyCat['phi']=new_phi_gals
    MyCat['theta']=new_theta_gals
    cat_name='FullExplorer.txt'
    print('Saving '+cat_name)
    MyCat.to_csv(cat_name, header=None, index=None, sep=' ')
#------------------------Reading the catalogue----------------------------------
if read==1:
    print('Reading the catalogue ')
    MyCat = pd.read_csv('FullExplorer.txt', sep=" ", header=None)
    colnames=['Ngal','Comoving Distance','Luminosity Distance','z','phi','theta']
    MyCat.columns=colnames
#################################DS control room#########################################

if DS_read==1:
    #name=os.path.join(folder,'catname')#move to te right folder
    sample = pd.read_csv(folder+'_DSs.txt.txt', sep=" ", header=None)
    colnames=['Ngal','Comoving Distance','Luminosity Distance','z','phi','theta']
    sample.columns=colnames
    ds_z=np.asarray(sample['z'])
    ds_dl=np.asarray(sample['Luminosity Distance'])
    ds_phi=np.asarray(sample['phi'])
    ds_theta=np.asarray(sample['theta'])
else:
    NumDS=1
    zds_max=1.05
    zds_min=0.95

    cutted=MyCat[MyCat['z']<=zds_max]
    cutted=cutted[cutted['z']>=zds_min]
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
arr=np.arange(0,len(H0Grid),dtype=int)
beta=np.zeros(len(H0Grid))
My_Like=np.zeros(len(H0Grid))
dlsigma=0.1
fullrun=[]
s=dlsigma
###################################Likelihood##################################################
for i in tqdm(range(NumDS)):
    DS_phi=ds_phi[i]
    tmp=MyCat[MyCat['phi']<=DS_phi+5.5*sigma_phi]
    tmp=tmp[tmp['phi']>=DS_phi-5.5*sigma_phi]
    DS_theta=ds_theta[i]
    tmp=tmp[tmp['theta']<=DS_theta+5.5*sigma_theta]
    tmp=tmp[tmp['theta']>=DS_theta-5.5*sigma_theta]
    mu=ds_dl[i]
    dlrange=s*mu*5
    tmp=tmp[tmp['Luminosity Distance']<=mu+dlrange]
    tmp=tmp[tmp['Luminosity Distance']>=mu-dlrange]
    
    z_gals=np.asarray(tmp['z'])
    new_dl_gals=np.asarray(tmp['Luminosity Distance'])
    new_phi_gals=np.asarray(tmp['phi'])
    new_theta_gals=np.asarray(tmp['theta'])
    #print(tmp.shape[0])
    with Pool(14) as p:
        My_Like=p.map(LikeofH0, arr)
    My_Like=np.asarray(My_Like)
    fullrun.append(My_Like)
    #############################################################################################
##############################BETA#################################################################
with Pool(14) as p:
    beta=p.map(vol_beta, arr)
beta=np.asarray(beta)
###################################################################################################
###########################Saving Results & posterior##############################################
betapath=os.path.join(folder,runpath+'_beta.txt')
np.savetxt(betapath,beta)
print('Beta Saved')
fullrunpath=os.path.join(folder,runpath+'_fullrun.txt')
np.savetxt(fullrunpath,fullrun)
print('All likelihood Saved')
fullrun_beta=fullrun/beta
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
os.system('cp postcalculator.py '+folder)
