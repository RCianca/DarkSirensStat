#
#    Copyright (c) 2021 Andreas Finke <andreas.finke@unige.ch>,
#                       Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by a modified BSD
#    license that can be found in the LICENSE file.


####
# This module contains a class to handle GW-galaxy correlation and compute the likelihood
####
from config import forcePcopl
from config import EM
from config import myredshift
from config import mysigz
from globals import *
import pandas as pd
from copy import deepcopy
from scipy.special import erf
from scipy import stats


class GWgal(object):

    
    def __init__(self, GalCompleted, GWevents,
                 eventSelector, lamb = 1,
                 MC = True, nHomSamples=1000, 
                 galRedshiftErrors = True, 
                 zR=zRglob,
                 verbose=False):
                 
        self.lamb = lamb
        
        self.eventSelector=eventSelector
        self.gals = GalCompleted
        self.GWevents = GWevents
        self.selectedGWevents= deepcopy(GWevents)

        self._galRedshiftErrors = galRedshiftErrors
        self.verbose=verbose
        
        self.nHomSamples = nHomSamples
        self.MC=MC
        self.zR=zR
        self.nGals={}
        
        
        self._get_avgPcompl()
        self._select_events()
        
        for eventName in GWevents.keys():
            self.select_gals_event(eventName)
            self.nGals[eventName] = self.gals.count_selection()
        # Note on the generalization. Eventually we should have a dictionary
        # {'GLADE': glade catalogue, 'DES': ....}
        # and a dictionary {'GW event': 'name of the catalogue to use'}
        # The function _inhom_lik will use the appropriate catalogue according to
        # this dictionary
        
        # Completeness needs a name or something to know if we use 
        # multiplicative, homogeneous or no completion
        #if self.verbose:
        #    print('\n --- GW events: ')
        #    for event in GWevents.keys():
        #        print(event)
    
    
    def _select_events(self):
        
            self.selectedGWevents = { eventName:self.GWevents[eventName] for eventName in self.GWevents.keys() if self.eventSelector.is_good_event(self.GWevents[eventName]) }
        #print('Selected GW events with Pc_Av>%s or Pc_event>%s. Events: %s' %(completnessThreshAvg, completnessThreshCentral, str(list(self.selectedGWevents.keys()))))
            if self.verbose:
                print('Selected GW events: %s' %( str(list(self.selectedGWevents.keys()))))
    
    
    def select_gals_event(self,eventName):
        self.gals.select_area(self.GWevents[eventName].selected_pixels, self.GWevents[eventName].nside)
        self.gals.set_z_range_for_selection( *self.GWevents[eventName].get_z_lims())
    
    def _get_summary(self):
        
        self.summary = pd.DataFrame.from_dict({'name': [self.GWevents[eventName].event_name for eventName in self.GWevents.keys()],
         'Omega_degSq': [self.GWevents[eventName].area() for eventName in self.GWevents.keys()],
         'dL_Mpc': [self.GWevents[eventName].dL for eventName in self.GWevents.keys()],
        'dLlow_Mpc':[self.GWevents[eventName].dLmin for eventName in self.GWevents.keys()],
        'dLup_Mpc':[self.GWevents[eventName].dLmax for eventName in self.GWevents.keys()],
        'z_event':[self.GWevents[eventName].zfiducial for eventName in self.GWevents.keys()],
        'zLow':[self.GWevents[eventName].zmin for eventName in self.GWevents.keys()],
        'zUp':[self.GWevents[eventName].zmax for eventName in self.GWevents.keys()],
         'Vol_mpc3':[self.GWevents[eventName].volCom for eventName in self.GWevents.keys()],
         'nGal':[self.nGals[ eventName] if eventName in self.selectedGWevents.keys() else '--' for eventName in self.GWevents.keys()],
         'Pc_Av': [self.PcAv[eventName] for eventName in self.GWevents.keys()],
         'Pc_event': [self.PEv[eventName] for eventName in self.GWevents.keys()]})
        
        
        
        
        
        
        
    def _get_avgPcompl(self):
        if self.verbose:
            print('Computing <P_compl>...')
            #Raul: forced pcompl
            if forcePcopl==1:
                print('<P_compl> forced to 1')

        PcAv={}
        PEv = {}
        #from scipy.integrate import quad
        for eventName in self.GWevents.keys():
            #self.GWevents[eventName].adap_z_grid(H0GLOB, Xi0Glob, nGlob, zR=self.zR)
            zGrid = np.linspace(self.GWevents[eventName].zmin, self.GWevents[eventName].zmax, 100)
            
            if self.GWevents[eventName].selected_pixels.size==0:
                #Pcomp=np.zeros(zGrid.shape)
                PcAv[eventName] = 0.
            else:
                Pcomp = np.array([self.gals.total_completeness( *self.GWevents[eventName].find_theta_phi(self.GWevents[eventName].selected_pixels), z).sum() for z in zGrid])
                vol = self.GWevents[eventName].areaRad*np.trapz(cosmoglob.differential_comoving_volume(zGrid).value, zGrid) #quad(lambda x: cosmoglob.differential_comoving_volume(x).value, self.GWevents[eventName].zmin,  self.GWevents[eventName].zmax)
                _PcAv = np.trapz(Pcomp*cosmoglob.differential_comoving_volume(zGrid).value, zGrid)*self.GWevents[eventName].pixarea/vol
                if forcePcopl==1:
                    _PcAv=1
                PcAv[eventName] = _PcAv
            
            
            _PEv = self.gals.total_completeness( *self.GWevents[eventName].find_event_coords(polarCoords=True), self.GWevents[eventName].zfiducial)
            mytheta,myphi=self.GWevents[eventName].find_theta_phi(self.GWevents[eventName].selected_pixels)
            mytheta=np.mean(mytheta)
            myphi=np.mean(myphi)
            myascension,mydeclination=ra_dec_from_th_phi(mytheta, myphi)
            if forcePcopl==1:
                    _PEv=1
            PEv[eventName] = _PEv
            if self.verbose:
                print('<P_compl> for %s = %s; Completeness at (z_event, Om_event): %s' %(eventName, np.round(_PcAv, 3), np.round(_PEv, 3))) 
                print('Skylocation of event %s: theta=%s, phi=%s Right Asc=%s dec=%s'%(eventName,mytheta,myphi,myascension,mydeclination))
        self.PcAv = PcAv
        self.PEv = PEv
        
    
    
    def get_lik(self, H0s, Xi0s, n=nGlob):
        '''
        Computes likelihood for all events
        Returns dictionary {event_name: L_cat, L_comp }
        '''
        ret = {}
        H0s = np.atleast_1d(H0s)
        Xi0s = np.atleast_1d(Xi0s)
        for eventName in self.selectedGWevents.keys():
            if self.verbose:
                print('-- %s' %eventName)
            
            #self.gals.select_area(self.selectedGWevents[eventName].selected_pixels, self.selectedGWevents[eventName].nside)
            #self.nGals[eventName] = self.gals.set_z_range_for_selection( *self.selectedGWevents[eventName].get_z_lims(), return_count=True)
            self.select_gals_event(eventName)
            
            Linhom = np.ones((H0s.size, Xi0s.size))
            Lhom   = np.ones((H0s.size, Xi0s.size))
            Linhomnude = np.ones((H0s.size, Xi0s.size))
            weigts = np.ones((H0s.size, Xi0s.size))
            weights_norm = np.ones((H0s.size, Xi0s.size))
            #no_norm=np.ones((H0s.size, Xi0s.size))
            for i in np.arange(H0s.size):
            
                for j in np.arange(Xi0s.size):
                    temp=self._inhom_lik(eventName=eventName, H0=H0s[i], Xi0=Xi0s[j], n=n)
                    
           
                    #Linhom[i,j] = self._inhom_lik(eventName=eventName, H0=H0s[i], Xi0=Xi0s[j], n=n)
                    Linhom[i,j]=temp[0]
                    Lhom[i,j] = self._hom_lik(eventName=eventName, H0=H0s[i], Xi0=Xi0s[j], n=n)
                    Linhomnude[i,j]=temp[1]
                    weigts[i,j]=temp[2]
                    weights_norm=temp[3]
                    #no_norm=temp[4]

            ret[eventName] = (np.squeeze(Linhom), np.squeeze(Lhom),np.squeeze(Linhomnude),np.squeeze(weigts),np.squeeze(weights_norm))
            
        return ret  
    
    def _inhom_lik(self, eventName, H0, Xi0, n):
        '''
        Computes likelihood with p_cat for one event
        Output:
        '''
        
        if self._galRedshiftErrors:
        
            # Convolution with z errors
            
            rGrid = self._get_rGrid(eventName, minPoints=20)

            zGrid = z_from_dLGW_fast(rGrid, H0=H0, Xi0=Xi0, n=n)
            #pos=int(len(zGrid)/2)
            #zz=zGrid[pos]
            zz=np.mean(zGrid)
            #print(zz)
            pixels, weights, norm= self.gals.get_inhom_contained(zGrid, self.selectedGWevents[eventName].nside )
            #np.savetxt('pixels.txt',pixels)
            weights *= myrate(zz)*(1+zGrid[np.newaxis, :])**(self.lamb-1)     
            my_skymap = self.selectedGWevents[eventName].likelihood_px(rGrid[np.newaxis, :], pixels[:, np.newaxis])

             
        else: # use Diracs
            
            pixels, zs, weights, norm, no_norm_weights =  self.gals.get_inhom(self.selectedGWevents[eventName].nside)
            
            rs = dLGW(zs, H0=H0, Xi0=Xi0, n=n)
            
            weights *= (1+zs)**(self.lamb-1)
            
            my_skymap = self.selectedGWevents[eventName].likelihood_px(rs, pixels)

        #Raul: Try to add EM info
        if (EM==1):
            LL = np.sum(my_skymap*weights*stats.norm.pdf(zGrid,loc=myredshift,scale=mysigz ))
            #LL = np.sum(my_skymap*weights*gauss(zGrid,myredshift,mysigz,norm=False))
            #Raul:Only for one test
            #LL = np.sum(my_skymap*stats.norm.pdf(zGrid,loc=redshift,scale=sigz))
        else:
            LL = np.sum(my_skymap*weights*myrate(zGrid))
        #LL = 0#np.sum(self.gauss(zz,0.0098,0.0004))
        sky_to_return=np.sum(my_skymap)
        weights_to_return=np.sum(weights)
        norm_to_return=np.sum(norm)
        #pure_to_return=np.sum(no_norm_weights)
        
        return LL, sky_to_return, weights_to_return, norm_to_return
    
    def _hom_lik(self, eventName, H0, Xi0, n):
        #modificato da Raul
        #print('sono in GWgal.py:_hom_lik. self.MC={}'.format(self.MC))
        if self.MC:
            #print('Eseguo likelihood MC')
            return self._hom_lik_MC(eventName, H0, Xi0, n)
        else:
            return self._hom_lik_trapz(eventName, H0, Xi0, n)
        
        
    
    def _hom_lik_trapz(self, eventName, H0, Xi0, n):
        
        zGrid = self.selectedGWevents[eventName].adap_z_grid(H0, Xi0, n, zR=self.zR)
        
        #self.gals.eval_hom(theta, phi, z) #glade._completeness.get( *myGWgal.selectedGWevents[ename].find_theta_phi(pxs), z)
        
        pxs = self.selectedGWevents[eventName].get_credible_region_pixels()
        th, ph = self.selectedGWevents[eventName].find_theta_phi(pxs)
        
        integrand_grid = np.array([ j(z)*(1+z)**(self.lamb-1)*(self.gals.eval_hom(th, ph, z, MC=False))*self.selectedGWevents[eventName].likelihood_px( dLGW(z, H0, Xi0, n), pxs) for z in zGrid])
        
        integral = np.trapz(integrand_grid.sum(axis=1), zGrid)
        den = (70/clight)**3
        LL = integral*self.selectedGWevents[eventName].pixarea/den
        
        return LL
    
    
    def _hom_lik_MC(self, eventName, H0, Xi0, n):
        '''
        Computes likelihood homogeneous part for one event
        '''
        
        theta, phi, r = self.selectedGWevents[eventName].sample_posterior(nSamples=self.nHomSamples)
        
        z = z_from_dLGW_fast(r, H0=H0, Xi0=Xi0, n=n)
        zz=np.mean(z)
        #print(zz)
        
        # the prior is nbar in comoving volume, so it transforms if we integrate over D_L^{gw}
        # nbar D_com^2 d D_com = nbar D_com^2 (d D_com/d D_L^{gw}) d D_L^{gw}
        
        # we put a D_L^{gw}^2 into sampling from the posterior instead from the likelihood, and divide the jacobian by it.
       
        jac = dVdcom_dVdLGW(z, H0=H0, Xi0=Xi0, n=n)
         
        # MC integration

        #Raul:Here add the EM-mock, maybe try expect?

        if (EM==1):
            #def temp(x):
            #    toreturn=(H0/70)**3*jac*(1+x)**(self.lamb-1)*self.gals.eval_hom(theta, phi, x)
            #    return toreturn
            
            #LL= np.mean(stats.norm.expect(toreturn,loc=0.0098,scale=0.0004))
            thetatoput=1.5844686277555844
            phitoput=1.8377089838869982
            #Raul:not elegant but it will work
            theta=np.where(theta!=0,thetatoput,theta)
            theta=np.where(theta==0,thetatoput,theta)
            phi=np.where(phi!=0,phitoput,phi)
            phi=np.where(phi==0,phitoput,phi)
            
            #LL = 0
            #LL = (H0/70)**3*np.mean(jac*(1+z)**(self.lamb-1)*self.gals.eval_hom(theta, phi, z)*stats.norm.pdf(z,loc=myredshift,scale=mysigz))
            #LL = (H0/70)**3*np.mean(jac*(1+z)**(self.lamb-1)*self.gals.eval_hom(theta, phi, z))

        else:
            LL = (H0/70)**3*np.mean(myrate(z)*jac*(1+z)**(self.lamb-1)*self.gals.eval_hom(theta, phi, z))
        #LL=(H0/70)**3*np.mean(self.gauss(z,0.0098, 0.0004))
        return LL
    
    
    def _get_rGrid(self, eventName, minPoints=50):
    
        lower, upper = self.selectedGWevents[eventName].dLmin, self.selectedGWevents[eventName].dLmax,
        
        nPoints = np.int(minPoints*(upper-lower)/self.selectedGWevents[eventName].sigmadL)
        
        return np.linspace(lower, upper, nPoints)
    
    
