#
#    Copyright (c) 2021 Andreas Finke <andreas.finke@unige.ch>
#
#    All rights reserved. Use of this source code is governed by a modified BSD
#    license that can be found in the LICENSE file.


####
# This module contains objects to compute the completeness of a galaxy catalogue
####

from abc import ABC, abstractmethod

from keelin import bounded_keelin_3_discrete_probabilities_between
from globals import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import healpy as hp

class Completeness(ABC):
    
    def __init__(self, comovingDensityGoal, **kwargs):
        self._computed = False
        self.verbose = None
        self._comovingDensityGoal = comovingDensityGoal
       
    def compute(self, galdata, useDirac = False): 
        if self.verbose:
                print('Computing completeness')
        self.compute_implementation(galdata, useDirac)
        self._computed = True
    
    @abstractmethod
    def compute_implementation(self, galdata, useDirac):
        pass
        
    @abstractmethod
    def zstar(self, theta, phi):
        pass
     
    def get(self, theta, phi, z, oneZPerAngle=False):
        assert(self._computed)
        if np.isscalar(z):
            return np.where(z > self.zstar(theta, phi), self.get_at_z_implementation(theta, phi, z), 1)
        else:
            if oneZPerAngle:
                assert(not np.isscalar(theta))
                assert(len(z) == len(theta))
                close = z < self.zstar(theta, phi)
                ret = np.zeros(len(z))
                ret[~close] = self.get_many_implementation(theta[~close], phi[~close], z[~close])
                ret[close] = 1
                return ret
            else:
                if not np.isscalar(theta):
                    if len(z) == len(theta):
                        print('Completeness::get: number of redshift bins and number of angles agree but oneZPerAngle is not True. This may be a coincidence, or indicate that the flag should be True')
                
                    ret = self.get_implementation(theta, phi, z)
                    close = z[np.newaxis, :] < self.zstar(theta, phi)[:, np.newaxis]
                    return np.where(close, 1, ret)
                else:
                    ret = self.get_implementation(theta, phi, z)
                    close = z < self.zstar(theta, phi)
                    return np.where(close, 1, ret)
                    
    
    # z is scalar (theta, phi may or may not be)
    # useful to make maps at fixed z and for single points
    @abstractmethod
    def get_at_z_implementation(self, theta, phi, z):
        pass
        
    # z, theta, phi are vectors of same length
    @abstractmethod
    def get_many_implementation(self, theta, phi, z):
        pass
    
    # z is grid, we want matrix: for each theta,phi compute *each* z
    @abstractmethod
    def get_implementation(self, theta, phi, z):
        pass
        
  
    
class SkipCompleteness(Completeness):
    
    def __init__(self, **kwargs):
        Completeness.__init__(self, comovingDensityGoal=1, **kwargs)
        
        
    def zstar(self, theta, phi):
        if np.isscalar(theta):
            return 0
        return np.zeros(theta.size)
        
    def compute_implementation(self, galdata, useDirac):
        if self.verbose:
            print("SkipCompleteness: nothing to compute")
        
    def get_at_z_implementation(self, theta, phi, z):
        if np.isscalar(theta):
            return 1
        return np.ones(theta.size)
        
    def get_implementation(self, theta, phi, z):
        return np.ones((theta.size, z.size))
    
    def get_many_implementation(self, theta, phi, z):
        return np.ones(theta.size)
        


    
    
class SuperpixelCompleteness(Completeness):
    
    def __init__(self, comovingDensityGoal, angularRes, zRes, interpolateOmega, **kwargs):
    
        assert(angularRes < 7)
        self._nside = 2**angularRes
        self._npix  = hp.nside2npix(self._nside)
        self._pixarea = hp.nside2pixarea(self._nside)
        self._zRes = zRes
        self.zedges = None
        self.zcenters = None
        
        self._interpolateOmega = interpolateOmega
        self._zstar = None
        
        self._map = np.zeros((self._npix, zRes), dtype=np.float)
        
        from astropy.cosmology import FlatLambdaCDM
        
        self._fiducialcosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
        
        Completeness.__init__(self, comovingDensityGoal, **kwargs)
        
    def zstar(self, theta, phi):
        if self._interpolateOmega:
            return hp.get_interp_val(self._zstar, theta, phi)
        else:
            return self._zstar[hp.ang2pix(self._nside, theta, phi)]
        
    
    def compute_implementation(self, galdata, useDirac):
    
        coarseden = self._map.copy()
        zmax = 1.5*np.quantile(galdata.z.to_numpy(), 0.9)
        self.zedges  = np.linspace(0, zmax, self._zRes+1)
        
        z1 = self.zedges[:-1]
        z2 = self.zedges[1:]
        
        self.zcenters = 0.5*(z1 + z2)
        
        galdata.loc[:,'pix'] = hp.ang2pix(self._nside, galdata.theta, galdata.phi)
        galdata.set_index(keys=['pix'], drop=False, inplace=True)
        
        # need to iterate through pixels...
 #       print('Preparing... ')
        #catparts = []
#        for i in np.arange(self._npix):
#            if (i % np.int(self._npix/30)) == 0:
#                print('#', end='', flush=True)
#            if i in galdata.index:
#                galpixel = galdata.loc[[i]]
#                catparts.append(galpixel)
#            else:
#                catparts.append(pd.DataFrame())
                
        if self.verbose:
            print('Computing in parallel... ', flush=True)
        #from multiprocessing import Pool
        #p = Pool(2)
        #from functools import partial
        def g(galpixelgroups, i):
            try:
                galpixel = galpixelgroups.get_group(i)
            except KeyError as e:
                return np.zeros(len(z1))
                
#            if len(galpixel) == 0:
#                return np.zeros(len(z1))
            if useDirac:
                res, _ = np.histogram(a=galpixel.z.to_numpy(), bins=self.zedges, weights=galpixel.w.to_numpy())
                return res.astype(float)
            else:
                weights = bounded_keelin_3_discrete_probabilities_between(self.zedges, 0.16, galpixel.z_lower, galpixel.z, galpixel.z_upper, galpixel.z_lowerbound, galpixel.z_upperbound, N=100)
                # if there is 1 galaxy only, weights won't be a matrix - fix
                if weights.ndim == 1:
                    weights = weights[np.newaxis, :]
                
                return np.sum(weights * galpixel.w[:, np.newaxis], axis=0)
          
        gr = galdata.groupby(level=0)
        coarseden = np.vstack( parmap(lambda i : g(gr, i), range(self._npix)) )
        
        if self.verbose:
            print(' Almost done!')
        
        vol = self._pixarea * (self._fiducialcosmo.comoving_distance(z2).value**3 - self._fiducialcosmo.comoving_distance(z1).value**3)/3
   
        coarseden /= vol
        self._map = coarseden/self._comovingDensityGoal
        if self.verbose:
            print('Final computations for completeness...')
        zFine = np.linspace(0, zmax, 3000)
        zFine = zFine[::-1]
        evals = self.get_implementation(*hp.pix2ang(self._nside, np.arange(self._npix)), zFine)
       
        # argmax returns "first" occurence of maximum, which is True in a boolean array. we search starting at large z due to the flip
        idx = np.argmax(evals >= 1, axis=1)
        # however, if all enries are False, argmax returns 0, which would be the largest redshift, while we want 0 in that case
        self._zstar = np.where(idx == 0, 0, zFine[idx])
        if self.verbose:
            print('Completeness done.')

    # z is a number
    def get_at_z_implementation(self, theta, phi, z):
    
        from scipy import interpolate
    
        # if only one point, things are pretty clear
        if np.isscalar(theta):
        
            valsAtZs = np.zeros(self._zRes)
        
            if self._interpolateOmega:
                for i in np.arange(self._zRes):
                    valsAtZs[i] = hp.get_interp_val(self._map[:,i], theta, phi)
            else:
                valsAtZs = self._map[hp.ang2pix(self._nside, theta, phi), :]
        
            f = interpolate.interp1d(self.zcenters, valsAtZs, kind='linear', bounds_error=False, fill_value=(1,0))
            
            return f(z)
        
        # many angles, one z
        else:
            
            # the performance-critical case is when theta, phi are long arrays of points and interpolation in Omega is on. Thus, first interpolate maps in redshift, then interpolate in Omega in the single resulting map. For no interpolation in Omega this is still better when Omega is much longer than the map resolution, since only one long lookup is needed (versus interpolating the values for each Omega in various maps in redshift)
       
            f = interpolate.interp1d(self.zcenters, self._map, kind='linear', bounds_error=False, fill_value=(1,0))
                
            self._buf = f(z)
                
            if self._interpolateOmega:
                return hp.get_interp_val(self._buf, theta, phi)
            else:
                return self._buf[hp.ang2pix(self._nside, theta, phi)]
     
     
     
    def get_many_implementation(self, theta, phi, z):
    
        tensorProductThresh = 4000
    
        # even though we only need to compute len(z) results, for this number small enough < tensorProductThresh, we can do a O(len(z)^2) but loopfree algorithm. At 4000 the O(4000^2) is still faster on my machine but not that much anymore. Larger matrices perhaps take too much memory...
        
        if (len(z) < tensorProductThresh):
            
            res = self.get_implementation(theta, phi, z)
            return np.diag(res)
        
        
        # for many points, a loop is almost necessary - only if many points have same pix32 coordinate we could do something better (if no angular interpolation)
        
        ret = np.zeros(len(z))
        for i, (thetai, phii, zi) in enumerate(zip(theta,phi,z)):
            
            ret[i] = self.get_at_z_implementation(thetai, phii, zi)
            
        return ret
   
    def get_implementation(self, theta, phi, z):
     
         # here theta, phi, z are all arrays.
         # angular interpolation could be done at each z bin, to make one vector of values in z for each galaxy, to be interpolated at each individual z. Since the z are different for each galaxy, scipy.interpolate does not vectorize here (neither does numpy.interp)
         # interpolating in z first would mean making a map for each galaxy, which is even worse.
         # therefore, we do not support interpolation in angle here!

        from scipy import interpolate
             
        f = interpolate.interp1d(self.zcenters, self._map, kind='linear', bounds_error=False, fill_value=(1,0))
             
        # different z will be different columns
        # each column is the interpolated map
        buf = f(z)
             
        if self._interpolateOmega:
            ret = np.zeros((len(theta), len(z)))
            # interpolate map by map
            for i in np.arange(len(z)):
                ret[:,i] = hp.get_interp_val(buf[:,i], theta, phi)
            return ret
        else:
            return buf[hp.ang2pix(self._nside, theta, phi), :]


class MaskCompleteness(Completeness):

    def __init__(self, comovingDensityGoal, zRes, nMasks=2, **kwargs):
      
        assert(nMasks >= 1)
        self._nMasks = nMasks
        self._zRes = zRes
        
        # for the resolution of the masks
        self._nside = 32
        self._npix  = hp.nside2npix(self._nside)
        self._pixarea = hp.nside2pixarea(self._nside)
        # integer mask
        self._mask = None
        
        # will be lists!
        self.zedges = []
        self.zcenters = []
        self.areas = []
        self._compl = []
        self._interpolators = []
        # this one actually a np array
        self._zstar = []
          
        from astropy.cosmology import FlatLambdaCDM
          
        self._fiducialcosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
          
        Completeness.__init__(self, comovingDensityGoal, **kwargs)
          
    def zstar(self, theta, phi):
        return self._zstar[ self._mask[hp.ang2pix(self._nside, theta, phi)] ]
          
      
    def compute_implementation(self, galdata, useDirac):
    
        ### MAKE MASKS ###
        
        # the only feature we use is number of galaxies in a pixel
        X = np.zeros((self._npix, 1))
        avgweight = np.mean(galdata.completenessGoal.to_numpy())
        
        galdata.loc[:,'pix'] = hp.ang2pix(self._nside, galdata.theta.to_numpy(), galdata.phi.to_numpy())
        galdata.set_index(keys=['pix'], drop=False, inplace=True)
        
#        # need to iterate through pixels...
#        print('Preparing... ')
#        for i in np.arange(self._npix):
#            if (i % np.int(self._npix/30)) == 0:
#                print('#', end='', flush=True)
#            if i in galdata.index:
#                X[i,0] = len(galdata.loc[[i]])
#            else:
#                X[i,0] = 0
        #foo = galdata.groupby(level=0).size()
        foo = galdata.groupby(level=0).completenessGoal.sum()/avgweight
        X[foo.index.to_numpy(), 0] = foo.to_numpy()
        
        # this improves the clustering (at least for GLADE)
        X = np.log(X+10)
        
        from sklearn import cluster
        if self.verbose:
            print('Making masks...')
        clusterer = cluster.AgglomerativeClustering(self._nMasks, linkage='ward')
        self._mask = clusterer.fit(X).labels_.astype(np.int)
        if self.verbose:
            print('Preparing further... ')
        
      #  catparts = []


        galdata.loc[:,'component'] = self._mask[galdata['pix'].to_numpy()]
        galdata.set_index(keys=['component'], drop=False, inplace=True)
        gr = galdata.groupby(level=0)
        
        for i in np.arange(self._nMasks):
            #if i in galdata.index:
            try:
                galcomp = gr.get_group(i)
                
                #galcomp = galdata.loc[[i]]
                #catparts.append(galcomp)
        
                #zmax = 1.0001*np.max(galcomp.z.to_numpy())
                zmax = 1.5*np.quantile(galcomp.z.to_numpy(), 0.9)
                self.zedges.append(np.linspace(0, zmax, self._zRes+1))
                z1 = self.zedges[i][:-1]
                z2 = self.zedges[i][1:]
                self.zcenters.append(0.5*(z1 + z2))
                self.areas.append(np.sum(self._mask == i)*self._pixarea)
            except KeyError as e: # label i was never put into the map, or it is the component without any galaxies. fill in some irrelevant but non-breaking stuff
                #catparts.append(pd.DataFrame())
                self.zedges.append(np.array([0,1,2]))
                self.zcenters.append(np.array([0.5,1.5]))
                self.areas.append([1,1])
        if self.verbose:
            print('Computing in parallel... ', flush=True)
       
        def g(galgroups, maskId, batchId, nBatches):
       
            zedges = self.zedges[maskId]
            zcenters = self.zcenters[maskId]
            
            try:
                gals = galgroups.get_group(maskId)
            #if len(galpixel) == 0:
            except KeyError as e:
                return np.zeros(len(zedges)-1)
            
            N = len(gals)
            n = int(N/nBatches)
            start = n*batchId
            stop = n*(1+batchId) 
            if batchId == nBatches-1:
                stop = N

            batch = gals.iloc[start:stop]

            if useDirac:
                res, _ = np.histogram(a=batch.z.to_numpy(), bins=zedges, weights=batch.completenessGoal.to_numpy())
                return res.astype(float)
            else:
                weights = bounded_keelin_3_discrete_probabilities_between(zedges, 0.16, batch.z_lower, batch.z, batch.z_upper, batch.z_lowerbound, batch.z_upperbound, N=100)
                
                # if there is 1 galaxy only, weights won't be a matrix - fix
                if weights.ndim == 1:
                    weights = weights[np.newaxis, :]
                
                return np.sum(weights * batch.completenessGoal[:, np.newaxis], axis=0)
                
        coarseden = []  
        nBatches = int(60.*len(galdata)/1000000.)
        if self.verbose:
            print('Batch number: ' + str(nBatches))
            
        for i in np.arange(self._nMasks):   
            coarseden.append(sum(parmap(lambda b : g(gr, maskId=i, batchId=b, nBatches=nBatches), range(nBatches))))
       
        if self.verbose:
            print('Final computations for completeness...')
       
        if self._comovingDensityGoal == 'auto':
            maxden = 0 
            for i in np.arange(self._nMasks):

                z1 = self.zedges[i][:-1]
                z2 = self.zedges[i][1:]
                vol = self.areas[i] * (self._fiducialcosmo.comoving_distance(z2).value**3 - self._fiducialcosmo.comoving_distance(z1).value**3)/3
                # too small volumes can be spurious. Ignore these by reducing the density so they won't be picked 
                vol[vol < 10000] = 1e30
                nearden = np.max(coarseden[i]/vol)
                
                print(maxden, nearden, z1, z2, vol)

                if nearden > maxden:
                    import copy
                    maxden = copy.copy(nearden)

            self._comovingDensityGoal = maxden

            if self.verbose:
                print('Comoving density goal is set to ' + str(self._comovingDensityGoal) )


        for i in np.arange(self._nMasks):
            z1 = self.zedges[i][:-1]
            z2 = self.zedges[i][1:]
            vol = self.areas[i] * (self._fiducialcosmo.comoving_distance(z2).value**3 - self._fiducialcosmo.comoving_distance(z1).value**3)/3
        
            coarseden[i] /= vol
            
            from scipy import interpolate
            # first, make a 1000 pt linear interpolation
           
            zmax = self.zedges[i][-1]*1.001
            zFine = np.linspace(0, zmax, 1000)
            
            coarseden_interp = np.interp(zFine, self.zcenters[i], coarseden[i], right=None)
            
            # now filter our fluctuations.
            from scipy.signal import savgol_filter
            # with 251 points (must be odd) there are effectively 4 independent points left
            # to describe the decay in the intervall adjusted to the part of the mask
            coarseden_filtered = np.zeros(coarseden_interp.shape)
            n=0
            coarseden_filtered[n:] = savgol_filter(coarseden_interp[n:], 251, 2)
            
            # build a quadratic interpolator. A subseet of points is enough (will be faster to evaluate).
            coarseden_filtered_sampled = np.interp(self.zcenters[i], zFine, coarseden_filtered)
            
            # save this just in case
            self._compl.append(coarseden_filtered_sampled.copy()/self._comovingDensityGoal)
        
            # interpolator
            if self.zcenters[i].size > 3:
                self._interpolators.append(interpolate.interp1d(self.zcenters[i], self._compl[i], kind='linear', bounds_error=False, fill_value=(1,0)))
            else:
                self._interpolators.append(lambda x: np.squeeze(np.zeros(np.atleast_1d(x).shape)))
        
        
            # find the point where the interpolated result crosses 1 for the last time
            zFine = np.linspace(0, self.zcenters[i][-1]*0.999, 10000)
            zFine = zFine[::-1]
            evals = self._interpolators[i](zFine)
        
            # argmax returns "first" occurence of maximum, which is True in a boolean array. we search starting at large z due to the flip
            idx = np.argmax(evals >= 1)
            # however, if all enries are False, argmax returns 0, which would be the largest redshift, while we want 0 in that case
            # if all entries are True, we want indeed the largest 
            if idx == 0:
                if evals[1] < 1:
                    self._zstar.append(0)
                    if self.verbose:
                        print("Catalog nowhere overcomplete in region {}.".format(i))
                else:
                    self._zstar.append(zFine[idx])
                    if self.verbose:
                        print("Warning: overcomplete catalog {} region even at largest redshift {}.".format(i, zmax))
            else:
                self._zstar.append(zFine[idx])
                if self.verbose:
                    print("Catalog overcomplete in region {} up to redshift {}.".format(i, zFine[idx]))

            
        self._zstar = np.array(self._zstar)

        if self.verbose:
            print('Completeness done.')
        
        
        
    # z is a number
    def get_at_z_implementation(self, theta, phi, z):

        from scipy import interpolate

        # if only one point, things are pretty clear
        if np.isscalar(theta):
            
            component = self._mask[hp.ang2pix(self._nside, theta, phi)]
            return self._interpolators[component](z)
            
        else:
            components = self._mask[hp.ang2pix(self._nside, theta, phi)]
            
            ret = np.zeros(len(theta))
            
            for i in np.arange(self._nMasks):
                # select arguments in this component
                compMask = (components == i)
                
                if np.sum(compMask) > 0:
                    
                    # do a single calculation
                    
                    res = self._interpolators[i](z)
                    
                    # put it into all relevant outputs
                    ret[compMask] = res
                    
            return ret
                
    def get_many_implementation(self, theta, phi, z):
        
        tensorProductThresh = 4000 # copied from SuperpixelCompletenss, ideally recheck
        
        if (len(z) < tensorProductThresh):
        
            res = self.get_implementation(theta, phi, z)
            return np.diag(res)
        
        
        ret = np.zeros(len(z))
        
#        for i, (thetai, phii, zi) in enumerate(zip(theta,phi,z)):
#
#            ret[i] = self.get_at_z_implementation(thetai, phii, zi)
#
#        return ret

        # no need to loop through points in the MaskCompletness case!
        
        components = self._mask[hp.ang2pix(self._nside, theta, phi)]

        for i in np.arange(self._nMasks):
            # select arguments in this component
            compMask = (components == i)
            
            if np.sum(compMask) > 0:
                
                # res is a vector here
                res = self._interpolators[i](z[compMask])
                
                # put it into all relevant outputs
                ret[compMask] = res
                
        return ret
        
    def get_implementation(self, theta, phi, z):
        
        from scipy import interpolate
        
        components = self._mask[hp.ang2pix(self._nside, theta, phi)]
        
        if not np.isscalar(theta):
            ret = np.zeros((len(theta), len(z)))
        else:
            ret = np.zeros(len(z))
        
        for i in np.arange(self._nMasks):
            # select arguments in this component
            compMask = (components == i)
            
            if np.sum(compMask) > 0:
                
                # do a single calculation
                
                res = self._interpolators[i](z)
                
                # put it into all relevant outputs
                if not np.isscalar(theta):
                    ret[compMask, :] = res
                else:
                    ret = res
                
        return ret
 






class LoadCompleteness(SuperpixelCompleteness):
    
    def __init__(self, fname, comovingDensityGoal, interpolateOmega, **kwargs):
        import os
        filepath = os.path.join(dirName, 'data', 'GLADE', fname)
        
        self.filepath = filepath

        f = open(filepath)
        header = f.readline()
        f.close()
        
        nside = int(header.split(',')[3].split('=')[1])
        angularRes = int(np.log2(nside)+1e-5)
        zRes = int(header.split(',')[4].split('=')[1])
        
        #print(nside, 2**angularRes, zRes)
        
        SuperpixelCompleteness.__init__(self, comovingDensityGoal, angularRes, zRes, interpolateOmega, **kwargs)
        
        zcentermin = float(header.split(',')[1].split('=')[1])
        zcentermax = float(header.split(',')[2].split('=')[1])
        
        self.zcenters = np.linspace(zcentermin, zcentermax, zRes)
        deltaz = self.zcenters[1]-self.zcenters[0]
        self.zMin = self.zcenters[0] - deltaz*0.5
        self.zMax = self.zcenters[-1] + deltaz*0.5
        
        self.zedges = np.linspace(self.zMin, self.zMax, zRes + 1)
        

    

    def compute_implementation(self, galdata, useDirac):

        z1 = self.zedges[:-1]
        z2 = self.zedges[1:]

        # LOAD MAP
        if self.verbose:
            print('Loading copleteness from %s...' %self.filepath)
        self._map = np.loadtxt(self.filepath).T
        #print(self._map[:2])

        if self.verbose:
            print('Final computations for completeness...')
        zFine = np.linspace(0, self.zMax, 3000)
        zFine = zFine[::-1]
        evals = self.get_implementation(*hp.pix2ang(self._nside, np.arange(self._npix)), zFine)

        # argmax returns "first" occurence of maximum, which is True in a boolean array. we search starting at large z due to the flip
        idx = np.argmax(evals >= 1, axis=1)
        # however, if all enries are False, argmax returns 0, which would be the largest redshift, while we want 0 in that case
        self._zstar = np.where(idx == 0, 0, zFine[idx])
        #print(self._zstar[:5])
        if self.verbose:
            print('Completeness done.')
