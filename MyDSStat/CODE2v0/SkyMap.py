import os
import numpy as np
import healpy as hp

from ligo.skymap.io import fits




class GWskymap:
    def __init__(self,event_name,level=None):
        self.event_name=event_name
        self.read_event(event_name)
        if level==None:
            self.level=0.99
        else:
            self.level=level
        
    def read_event(self,event_name):
        skymap, metadata = fits.read_sky_map(event_name, nest=None, distances=True)
        self.p_posterior= skymap[0]
        self.mu= skymap[1]
        self.sigma = skymap[2]
        self.posteriorNorm= skymap[3]        
        self.npix = len(self.p_posterior)
        self.nside=hp.npix2nside(self.npix)
        self.metadata=metadata
        #self.all_pixels=np.arange(npix)
        #return p_posterior,mu,sigma,posteriorNorm,nside,npix,metadata 

    def get_credible_region_pixels(self,level):
        all_pixels=np.arange(self.npix)
        return all_pixels[self.p_posterior>self._get_credible_region_pth(self.level)]            
    
    def area(self):
        level=self.level
        nside=self.nside
        ''' Area of level% credible region, in square degrees.
            If level is not specified, uses current selection '''
        pixarea=hp.nside2pixarea(nside)
        return self.get_credible_region_pixels(level=level).size*pixarea*(180/np.pi)**2
    
    def _get_credible_region_pth(self,level):
        '''
        Finds value minskypdf of rho_i that bouds the x% credible region , with x=level
        Then to select pixels in that region: self.all_pixels[self.p_posterior>minskypdf]
        '''

        prob_sorted = np.sort(self.p_posterior)[::-1]
        prob_sorted_cum = np.cumsum(prob_sorted)
        # find index of array which bounds the self.area confidence interval
        idx = np.searchsorted(prob_sorted_cum, self.level)
        minskypdf = prob_sorted[idx] #*skymap.npix

        #self.p[self.p]  >= minskypdf       
        return minskypdf
    


#------------------------------------------------For-Test-------------------------------------------------------


if __name__=='__main__':
	#work in progress
    fname='GWtest00.fits'
    DSs=GWskymap(fname)
    print('test GWskymap class\nPrinting some info')
    print('DS name {}'.format(DSs.event_name))
    print('Area of DS is {} deg^2'.format(DSs.area()))
    mypix=DSs.get_credible_region_pixels(level=0.99)
    print(type(mypix))
    print(np.shape(mypix))