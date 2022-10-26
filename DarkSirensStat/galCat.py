#
#    Copyright (c) 2021 Andreas Finke <andreas.finke@unige.ch>,
#                       Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by a modified BSD
#    license that can be found in the LICENSE file.

####
# This module contains a abstract classes to handle a galaxy catalogue
####
from config import EM
from config import forcePcopl
from config import myredshift
from config import mysigz
import pandas as pd
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from copy import deepcopy

from globals import *
from keelin import bounded_keelin_3_discrete_probabilities


class GalCat(ABC):
    
    def __init__(self, foldername, completeness, useDirac, verbose, **kwargs):
        
        self._path = os.path.join(dirName, 'data', foldername)
          
        self._nside = 128
        self._useDirac = useDirac
        self.data = pd.DataFrame()
        self.verbose = verbose
        
        self.load(**kwargs)
        
        self.selectedData = self.data
        
        self._completeness = deepcopy(completeness)
        self._completeness.verbose = verbose
        self._completeness.compute(self.data, useDirac)
        self.nonorm=0
    def get_data(self):
        return self.selectedData
        
    def select_area(self, pixels, nside):
        if self.verbose:
            print("Restricting area of the catalogue to %s pixels with nside=%s" %(pixels.shape[0], nside))
        pixname = "pix" + str(nside)
        
        if not pixname in self.data:
            self.data.loc[:, pixname] = hp.ang2pix(nside, self.data.theta, self.data.phi)

        mask = self.data.isin({pixname: pixels}).any(1)

        self.selectedData = self.data[mask]
        if self.verbose:
            print('Selected Galaxies')
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                print(self.selectedData)
            print('%s galaxies kept' %self.selectedData.shape[0])
        
    def set_z_range_for_selection(self, zMin, zMax):
        if self.verbose:
            print("Setting z range of the catalogue between %s, %s" %(np.round(zMin,3), np.round(zMax,3)))
        self.selectedData = self.selectedData[(self.selectedData.z >= zMin) & (self.selectedData.z < zMax)]
        if self.verbose:
            print('%s galaxies kept' %self.selectedData.shape[0])
            #Raul:selected data must have 2D-sky coordinate. Work with it
            
    def count_selection(self):
        return self.selectedData.shape[0]
        
    @abstractmethod
    def load(self):
        pass
    
    def completeness(self, theta, phi, z, oneZPerAngle=False):
        if not forcePcopl:
            return self._completeness.get(theta, phi, z, oneZPerAngle) + 1e-9
        else:
            #print('using forcePcopl in completeness')
            return np.ones(theta.shape)
            #return self._completeness.get(theta, phi, z, oneZPerAngle) + 1e-9


    def group_correction(self, df, df_groups, which_z='z_cosmo'):
        '''
        Corrects cosmological redshift in heliocentric frame 
        for peculiar velocities in galaxies 
        inside the group galaxy catalogue in  arXiv:1705.08068, table 2
        To be applied BEFORE changing to CMB frame
        
        Inputs: df - dataframe to correct
                df groups - dataframe of group velocities
                which_z - name of column to correct
        
        Output : df, but with the column given by which_z corrected for peculiar velocities
                    in the relevant cases and a new column named which_z+'_or' 
                    with the original redshift
        
        '''
        
        

        #print('Correcting %s for group velocities...' %which_z)

#        df_groups.loc[:, 'isInCat'] = df_groups['PGC'].isin(df['PGC'])
#        print(df_groups)
#        df_groups.set_index(keys=['PGC1'], drop=False, inplace=True)
#        groups = df_groups.groupby(level=0)
#        #groups = df_groups.groupby('PGC1')
#
#        isModified = np.zeros(len(df), dtype=int)
#
#        for grname, gr in groups:
#
#            if gr.isInCat.any():
#                print('y')
#                galnames = gr['PGC']
#                mask = df['PGC'].isin(galnames)
#
#                z_group = gr.HRV.mean()/clight
#
#                df.loc[mask, which_z] = z_group
#                isModified = isModified | mask.to_numpy().astype(int)
#
#        df.loc[:, 'group_correction'] = isModified
       
        df.loc[:, which_z+'_or'] = df[which_z].values
        zs = df.loc[df['PGC'].isin(df_groups['PGC'])][['PGC', which_z]]
        z_corr_arr = []
        #z_group_arr = []
        for PGC in zs.PGC.values:
                #z_or=zs.loc[zs['PGC']==PGC][which_z_correct].values[0]
            PGC1=df_groups[df_groups['PGC']==PGC]['PGC1'].values[0]
                #print(PGC1)

            z_group = df_groups[df_groups['PGC1']== PGC1].HRV.mean()/clight


            z_corr_arr.append(z_group)
        z_corr_arr=np.array(z_corr_arr)

        df.loc[df['PGC'].isin(df_groups['PGC']), which_z] = z_corr_arr
        correction_flag_array = np.where(df[which_z+'_or'] != df[which_z], 1, 0)
        df.loc[:, 'group_correction'] = correction_flag_array

        
    
    def CMB_correction(self, df, which_z='z_cosmo'):
        
        '''
        Gives cosmological redshift in CMB frame starting from heliocentric
        
        Inputs: df - dataframe to correct
                which_z - name of column to correct
        
        Output : df,  with a new column given by 
                which_z +'_CMB'
        
        '''
        
        #print('Correcting %s for CMB reference frame...' %which_z)
        
        v_gal = clight*df[which_z].values
        phi_CMB, dec_CMB = gal_to_eq(np.radians(l_CMB), np.radians(b_CMB))
        theta_CMB =  0.5 * np.pi - dec_CMB
                        
        delV = v_CMB*(np.sin(df.theta)*np.sin(theta_CMB)*np.cos(df.phi-phi_CMB) +np.cos(df.theta)*np.cos(theta_CMB))
            
        v_corr = v_gal+delV  # at first order in v/c ...
            
        z_corr = v_corr/clight
        df.loc[:,which_z+'_CMB'] = z_corr
  
    def include_vol_prior(self, df):
        batchSize = 10000
        nBatches = max(int(len(df)/batchSize), 1)
         
        if self.verbose:
            print("Computing galaxy posteriors...")
          
        from keelin import convolve_bounded_keelin_3
        from astropy.cosmology import FlatLambdaCDM
        fiducialcosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
        zGrid = np.linspace(0, 1.4*np.max(df.z_upperbound), 500)
        jac = fiducialcosmo.comoving_distance(zGrid).value**2 / fiducialcosmo.H(zGrid).value
        
        from scipy import interpolate
        func = interpolate.interp1d(zGrid, jac, kind='cubic')
        
        def convolve_batch(df, func, batchId, nBatches): 
            N = len(df)
            # actual batch size, different from batchSize only due to integer rounding 
            n = int(N/nBatches) 
            start = n*batchId
            stop = n*(batchId+1)
            if batchId == nBatches-1:
                stop = N 

            batch = df.iloc[start:stop]

            if self.verbose:
                if batchId % 100 == 0:
                    print("Batch " + str(batchId) + " of " + str(nBatches) )
            
            ll = batch.z_lowerbound.to_numpy()
            l  = batch.z_lower.to_numpy()
            m  = batch.z.to_numpy()
            u  = batch.z_upper.to_numpy()
            uu = batch.z_upperbound.to_numpy()
               
            return convolve_bounded_keelin_3(func, 0.16, l, m, u, ll, uu, N=1000)

        res = np.vstack(parmap(lambda b: convolve_batch(df, func, b, nBatches), range(nBatches)))
        
        mask = (res[:,0] >= res[:,1]) | (res[:,1] >= res[:,2]) | (res[:,2] >= res[:,3]) | (res[:,3] >= res[:,4]) | (res[:,0] < 0)
      
        if self.verbose: 
            print('Removing ' + str( np.sum(mask) ) + ' galaxies with unfeasible redshift pdf after r-squared prior correction.' )

        df.z_lowerbound = res[:, 0]
        df.z_lower = res[:, 1]
        df.z = res[:, 2]
        df.z_upper = res[:, 3]
        df.z_upperbound = res[:, 4]

        df = df[~mask]

        return df 

    
    
def gal_to_eq(l, b):
    '''
    input: galactic coordinates (l, b) in radians
    returns equatorial coordinates (RA, dec) in radians
    
    https://en.wikipedia.org/wiki/Celestial_coordinate_system#Equatorial_↔_galactic
    '''
    
    l_NCP = np.radians(122.93192)
    
    del_NGP = np.radians(27.128336)
    alpha_NGP = np.radians(192.859508)
    
    
    RA = np.arctan((np.cos(b)*np.sin(l_NCP-l))/(np.cos(del_NGP)*np.sin(b)-np.sin(del_NGP)*np.cos(b)*np.cos(l_NCP-l)))+alpha_NGP
    dec = np.arcsin(np.sin(del_NGP)*np.sin(b)+np.cos(del_NGP)*np.cos(b)*np.cos(l_NCP-l))
    
    return RA, dec



    
class GalCompleted(object):
    
    def __init__(self, completionType = 'mix', **kwargs):
    
        self._galcats = []
        self._catweights = []
        
        self._additive = False
        self._multiplicative = False
        
        if completionType == 'add':
            self._additive = True
        elif completionType == 'mult':
            self._multiplicative = True
    
    
    def add_cat(self, cat, weight = 1):
        self._galcats.append(cat)
        self._catweights.append(weight)
        
        
    def total_completeness(self, theta, phi, z, oneZPerAngle=False):
    
        # sums completenesses of all catalogs, taking into account the additional
        # catalog weights
        
        res = 0
        for c, w in zip(self._galcats, self._catweights):
            res += w*c.completeness(theta, phi, z, oneZPerAngle)
        
        return res
        #return sum(list(map(lambda c: c.completeness, self._galcats)))
    
    def select_area(self, pixels, nside):
        for c in self._galcats:
            c.select_area(pixels, nside)
            
    def set_z_range_for_selection(self, zMin, zMax):
        for c in self._galcats:
            c.set_z_range_for_selection(zMin, zMax)
            
    def distance(self,theta,phi,z):
        #Raul:function to compute distance in the selected volume. NB only for small z-z0
        dist=np.sqrt(z**2+theta**2+phi**2)
        return dist
    
    def count_selection(self):
        return [c.count_selection() for c in self._galcats]

    def get_inhom_contained(self, zGrid, nside):
        ''' return pixels : array N_galaxies
        
                    weights: array N_galaxies x len(zGrid)
        '''
        
        allpixels = []
        allweights = []
        #nudeweights=[]
        
        # iterate through catalogs and add results to lists
        
        catweightTotal = 0
        for c, w in zip(self._galcats, self._catweights):
        
            # shorthand
            d = c.get_data()
            if EM==1:
                #Raul: given a point in space, find nearest N host
                distances=np.zeros(d.shape[0])
                theta_ref=1.5844686277555844#3.44616
                phi_ref=1.8377089838869982#-0.408084 
                z_ref= myredshift #0.0098 #0.015410590744339491
                if (d.shape[0]>1):
                    for index, row in d.iterrows():
                        distances[index]=np.sqrt((row['theta']-theta_ref)**2+(row['phi']-phi_ref)**2+(row['z']-z_ref)**2)
                    d['distances']=distances
                    d=d.sort_values(["distances"], ascending=True) 
                    d=d.head(1)
                #print('d={}'.format(d))
                
            pixname = "pix" + str(nside)
            # compute this only once
            if not pixname in c.get_data():
                d.loc[:, pixname] = hp.ang2pix(nside, d.theta.to_numpy(), d.phi.to_numpy())

            # pixels are already known
            allpixels.append(d[pixname].to_numpy())
            #print('allpixels={}'.format(allpixels))
            #print('d[pixname]={}'.format(d[pixname]))
            # keelin weights. N has to be tuned for speed vs quality
            # for each gal, on zGrid
            weights = bounded_keelin_3_discrete_probabilities(zGrid, 0.16, d.z_lower, d.z, d.z_upper, d.z_lowerbound, d.z_upperbound, N=40, P=0.99999)
            if weights.ndim == 1:
                weights = weights[np.newaxis, :]
            trigger=np.all((weights==0))
            #if self.nonorm==1:
                #temp=weights.flatten()
                #s=np.sum(temp)
                #print(s)
                #print('\n')
            weights *= d.w[:, np.newaxis]
            
            #temp=weights/temp
            if self._additive and (len(self._galcats) == 1): # no need to evaluate completeness...
                catweightTotal = w
                
            else:
            
                # completeness eval for each gal, on grid - same shape as weights
                completeness = c.completeness(d.theta.to_numpy(), d.phi.to_numpy(), zGrid)
                    
                # multiplicative completion
                weights /= completeness
                # downweighted for low completeness
                weights *= self.confidence(completeness)
                
                # catalog weighting based also on completeness
                catweight = w*np.mean(completeness)
                weights *= catweight
                catweightTotal += catweight
            
            # normalize in case different goals are used for different catalogs, to make them comparable
            weights /= c._completeness._comovingDensityGoal
   
            allweights.append(weights)
            
        allweights = np.vstack(allweights)
        #nudeweights=np.vstack(nudeweights)
        allweights /= catweightTotal
        #Raul:Printing the norm for weights
        #return np.squeeze(np.vstack(allpixels)), np.vstack(allweights)
        return np.hstack(allpixels), allweights, catweightTotal
    

    def get_inhom(self, nside):
        '''
        returns pixels, redshifts and weights of all galaxies (redshift medians) in the selection, ignoring galaxy redshift errror pdfs
        
        returns:
        pixels :   array nGal
        redshifts: array nGal
        weights:   array nGal
    
        '''
        
        allpixels = []
        allredshifts = []
        allweights = []
        catweightTotal = 0
        
        for c, w in zip(self._galcats, self._catweights):
        
            # shorthand
            d = c.get_data()
            
            pixname = "pix" + str(nside)
            
            # compute this only once
            if not pixname in c.get_data():
                d.loc[:, pixname] = hp.ang2pix(nside, d.theta.to_numpy(), d.phi.to_numpy())

            allpixels.append(d[pixname].to_numpy())
            
            
            weights = d.w.to_numpy().copy() # Get intrinsic weigts, this can be separately set to one
            print('Galcat.py: (loaded) weights={}'.format(weights))
            #Raul:print it then remove or make it possible to set weights to one
            redshifts = d.z.to_numpy()
            allredshifts.append(redshifts)
            
            if self._additive and (len(self._galcats) == 1): # no need to evaluate completeness...
                catweightTotal = w
                
            else:
           
                # completeness eval for each gal if forcePcompl==1 this is one
                completeness = c.completeness(d.theta.to_numpy(), d.phi.to_numpy(), redshifts, oneZPerAngle = True)
                   
                # multiplicative completion
                weights /= completeness #see eq 3.70 is the division
                #print('Galcat.py: completeness={}, new weights(/=completeness)={}'.format(completeness,weights))
                # downweighted for low completeness
                weights *= self.confidence(completeness)# 3.71 with smooth step instead of theta. If forcePcompl, this is one if we have forcePcompl
                #print('Galcat.py: self.confidence(completeness)={}, new weights(=*self..)={}'.format(self.confidence(completeness),weights))
               
                # catalog weighting based also on completeness
                catweight = w*np.mean(completeness)
                weights *= catweight #volume is linked to the number of gals, so this is the integral in 3.70
                #print('Galcat.py: catweight={}, new weights={}'.format(catweight,weights))
                catweightTotal += catweight
           
            # normalize in case different goals are used for different catalogs, to make them comparable
            weights /= c._completeness._comovingDensityGoal # Raul: why we need this, this look like nbar, the expected number of gal in a volume
  
            allweights.append(weights)
            #print('last mod weights={}'.format(weights))
           
        allweights = np.hstack(allweights)
        print('allpixels={}'.format(allpixels))
        print('allweights={}'.format(allweights))
        print('Galcat.py: catweightTotal={}'.format(catweightTotal))
        print('Galcat.py: c._completeness._comovingDensityGoal={}'.format(c._completeness._comovingDensityGoal))
                   
        return np.hstack(allpixels), np.hstack(allredshifts), allweights, catweightTotal
    
    def eval_inhom(self, Omega, z):
        '''
        For the future if we had posterior samples
        '''
        pass
    
    def eval_hom(self, theta, phi, z, MC=True):
        '''
        Homogeneous completion part. Second term in 2.59
        '''
        if MC:
            assert(len(theta) == len(z))
        
        ret = np.zeros(len(theta))
        catweightTotal = 0
        
        for c, w in zip(self._galcats, self._catweights):
            
        
            # completeness eval for each point
            completeness = c.completeness(theta, phi, z, oneZPerAngle = True)
            
            
            # how much of homogeneous stuff to add - note in case of additive completion, confidence returns its argument, and we have 1 - completeness, the correct prefactor in that case
            
            retcat = (1-self.confidence(completeness))
            
            # catalog weighting based also on completeness
            catweight = w*np.mean(completeness)
            retcat *= catweight
            catweightTotal += catweight
            
            ret += retcat
        
        # for catalog averaging (3)
        ret /= catweightTotal
        return ret
        
        
    def confidence(self, compl):
    
        if self._multiplicative:
            return 1
        elif self._additive:
            return compl
        else: #interpolation between multiplicative and additive
            confpower = 0.05
            complb = np.clip(compl, a_min=2e-3, a_max=1)
            return np.exp(confpower*(1-1/complb))
