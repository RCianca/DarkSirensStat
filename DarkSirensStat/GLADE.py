#
#    Copyright (c) 2021 Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by a modified BSD
#    license that can be found in the LICENSE file.


import pandas as pd
import healpy as hp
import numpy as np

import os, os.path


####
# This module contains a class to handle the GLADE catalogue
####

from globals import *
from galCat import GalCat

from astropy.cosmology import FlatLambdaCDM
cosmoGLADE = FlatLambdaCDM(H0=H0GLADE, Om0=Om0GLADE)  # the values used by GLADE

class GLADE(GalCat):
    
    def __init__(self, foldername, compl, useDirac, #finalData = None,
                 subsurveysIncl = ['GWGC', 'HYPERLEDA', 'TWOMASS', 'SDSS'], 
                 subsurveysExcl = [], 
                 verbose=True,
                 **kwargs):
        
        self._subsurveysIncl = subsurveysIncl
        self._subsurveysExcl = subsurveysExcl
        #self._finalData = finalData
        
        assert(set(subsurveysExcl).isdisjoint(subsurveysIncl))
        assert(len(subsurveysIncl) > 0)

        GalCat.__init__(self, foldername, compl, useDirac, verbose, **kwargs)
        
    
    def load(self, band=None, band_weight=None,
                   Lcut=0,
                   zMax = 100,
                   z_flag=None,
                   drop_z_uncorr=False,
                   get_cosmo_z=True, #cosmo=None, 
                   pos_z_cosmo=True,
                   drop_no_dist=False,
                   group_correct=True,
                   which_z_correct = 'z_cosmo',
                   CMB_correct=True,
                   which_z='z_cosmo_CMB',
                   galPosterior = True,
                   err_vals='GLADE',
                   drop_HyperLeda2=True, 
                   colnames_final = ['theta','phi','z','z_err', 'z_lower', 'z_lowerbound', 'z_upper', 'z_upperbound', 'w', 'completenessGoal']):
        
        if band_weight is not None:
            assert band_weight==band
       
        loaded = False
        computePosterior = False
        posteriorglade = os.path.join(self._path, 'posteriorglade.csv')
        if self.verbose:
            print(posteriorglade)
        if galPosterior:
            from os.path import isfile
            if isfile(posteriorglade):
                if self.verbose:
                    print("Directly loading final data ")
                df = pd.read_csv(os.path.join(self._path, 'posteriorglade.csv'))
                loaded=True
            else:
                computePosterior=True
                loaded=False
                
            
        if not loaded:
            #gname='GLADE_2.4.txt'on
            #gname='GLADE_flagship.txt'
            #gname='GLADE_flagship_host.txt'
            
            #gname='GLADE_flagship_host_alone.txt'
            #gname='GLADE_flagship_two_host_alone.txt'
            #gname='GLADE_flagship_two_host_alone_nofodder.txt'
            #gname='GLADE_flagship_20rand_twohost.txt'
            #gname='GLADE_flagship_100rand_twohost.txt'
            #gname='GLADE_flagship_500rand_twohost.txt'
            #gname='GLADE_flagship_500rand_sixhost.txt'
            #gname='GLADE_flagship_500rand_sixhost_05moved.txt'
            #gname='GLADE_flagship_5000rand_sixhost.txt'
            #gname='GLADE_flagship_500000rand_sixhost.txt'
            #gname='GLADE_flagship_5000000rand_sixhost.txt'
            
            #------------Fine num_gal increment----------
            #gname='GLADE_flagship_100incr.txt'
            #gname='GLADE_flagship_200incr.txt'
            #gname='GLADE_flagship_300incr.txt'
            #gname='GLADE_flagship_400incr.txt'
            #gname='GLADE_flagship_500incr.txt'
            #gname='GLADE_flagship_600incr.txt'
            #gname='GLADE_flagship_700incr.txt'
            #gname='GLADE_flagship_800incr.txt'
            #gname='GLADE_flagship_900incr.txt'
            #gname='GLADE_flagship_1000incr.txt'
            #gname='GLADE_flagship_1500incr.txt'
            #gname='GLADE_flagship_2000incr.txt'
            #gname='GLADE_flagship_2500incr.txt'
            #gname='GLADE_flagship_3000incr.txt'
            #gname='GLADE_flagship_3500incr.txt'
            #gname='GLADE_flagship_4000incr.txt'
            #gname='GLADE_flagship_4500incr.txt'
            #gname='GLADE_flagship_5000incr.txt'
            #-------------cone increment----------------
            #gname='GLADE_flagship_cone05_10incr.txt'
            #gname='GLADE_flagship_cone05_100incr.txt'
            #gname='GLADE_flagship_cone05_500incr.txt'
            #gname='GLADE_flagship_cone05_1000incr.txt'
            #gname='GLADE_flagship_cone05_2000incr.txt'
            #gname='GLADE_flagship_cone05_3000incr.txt'
            #gname='GLADE_flagship_cone05_4000incr.txt'
            #gname='GLADE_flagship_cone05_5000incr.txt'
            #gname='GLADE_flagship_cone05_6000incr.txt'
            #gname='GLADE_flagship_cone05_7000incr.txt'
            #gname='GLADE_flagship_cone05_8000incr.txt'
            #gname='GLADE_flagship_cone05_9000incr.txt'
            #gname='GLADE_flagship_cone05_10000incr.txt'
            #---------------------------------------------------------
            #gname='GLADE_flagship_lineof03.txt'
            #gname='GLADE_flagship_lineof03_offline1.txt'
            #gname='GLADE_flagship_lineof03_offline2.txt'
            #gname='GLADE_flagship_lineof05.txt'
            #gname='GLADE_flagship_lineof05_offline1.txt'
            #gname='GLADE_flagship_lineof05_offline1-1.txt'
            #gname='GLADE_flagship_lineof05_offline2.txt'
            #gname='GLADE_flagship_lineof07.txt'
            #gname='GLADE_flagship_angulardisp05.txt'
            #gname='GLADE_flagship_angulardisp05_newsig.txt'
            #--------------Volume---------------------------
            #gname='GLADE_flagship_volume_host.txt'
            #gname='GLADE_flagship_volume_cone_10_host.txt'
            #gname='GLADE_flagship_volume_cone_100_host.txt'
            #gname='GLADE_flagship_volume_cone_1000_host.txt'
            #gname='GLADE_flagship_volume_cone_3000_host.txt'
            #gname='GLADE_flagship_volume_cone_6000_host.txt'
            #gname='GLADE_flagship_volume_cone_9000_host.txt'
            #gname='GLADE_flagship_volume_cone_12000_host.txt'
            #gname='GLADE_flagship_volume_cone_15000_host.txt'
            #gname='GLADE_flagship_volume_cone_18000_host.txt'
            #gname='GLADE_flagship_cone13_host.txt'
            #gname='GLADE_flagship_cone14_host.txt'
            #gname='GLADE_flagship_cone15_host.txt'
            #gname='GLADE_flagship_cone16_host.txt'
            #-----------------volume comoving-------------------------
            #gname='GLADE_flagship_cone15_host_dc.txt'
            #gname='GLADE_flagship_cone16_host_dc.txt'
            #gname='GLADE_flagship_cone17_host_dc.txt'
            #gname='GLADE_flagship_cone19_host_dc.txt'
            #gname='gw2220xx_host.txt'
            #gname='gw23xxxx_host.txt'
            #gname='gw23xxxx_host_flag.txt'
            #gname='gw24xxxx_host_flag.txt'
            #gname='gw25xxxx_host_flag.txt'
            #gname='gw26xxxx_host_flag.txt'
            #gname='gw27xxxx_host_flag.txt'
            #gname='gw27xxxx_host_flag_dc.txt'
            #gname='host_of_GW23xxxx.txt'
            #gname='host_of_GW24xxxx.txt'
            #------------Catalogue with stat_weights--------------------------
            #gname='diluted_flag_host.txt'   #100%
            #gname='uniform_comoving_father_of_diluted_host.txt' #100%
            #gname='diluted_flag.txt'
            #gname='uniform_comoving_father_of_diluted.txt'
            #------------------GW28------------------------------------------
            #gname='diluted_flag_GW28_comp100_host.txt'   #100%
            #gname='uniform_comoving_father_of_diluted_GW28_comp100_host.txt' #100%
            #gname='selected_autoconsistent_host.txt'   #100%
            #gname='selected_autoconsistent_halved_host.txt'   #8.3%
            #gname='selected_autoconsistent_halved_10bins_host.txt'   #8.3%
            #gname='uniform_comoving_autoconsistent_host.txt' #100%
            #--------------------GW29-------------------------------------
            #gname='selected_autoconsistent_halved_host_29.txt'   #8.3%
            #gname='selected_autoconsistent_halved_10bins_host.txt'   #8.3% do it
            #gname='uniform_comoving_autoconsistent_host_29.txt' #100% 
            #--------------------GW30-------------------------------------
            #gname='selected_autoconsistent_halved_host_30.txt'   #8.3%
            #gname='uniform_comoving_autoconsistent_host_30.txt' #100%
            #gname='gw23cat_uniform_30_host.txt'
            #gname='gw23cat_uniform_30_1oct_host.txt' #deltadl1%
            #gname='gw23cat_uniform_31_1oct_host.txt' #deltadl5%
            #gname='gw23cat_uniform_32_1oct_host.txt' #deltadl10%
            #gname='gw23cat_uniform_30_1oct_halved_host.txt'
            #gname='gw23cat_uniform_32_1oct_halved_host.txt'
            #--------------------GW33-------------------------------------
            gname='uniform_comoving_autoconsistent.txt' #100%
            #------------------minimalExample----------------------------
            #gname='MinimalCat.txt' #100%
            #gname='MockGlade.txt' #100%
            #gname='genova_uniform_samezasminimal.txt' #100%
            #-------------line-----------------------------------------
            #gname='host_of_GW2216xx.txt'
            #gname='GLADE_line16_50_host.txt'
            #gname='GLADE_line16_10_host.txt'
            #gname='GLADE_line16_113.txt'
            #gname='GLADE_line18_11.txt'
            #gname='GLADE_line18_20.txt'
            #gname='GLADE_line18_31.txt'
            #gname='GLADE_line18_50.txt'
            #gname='GLADE_line18_20_allhost.txt'
            #------------cone_increment--------------------------------
            #gname='GLADE_flagship_cone18_dc_10_host.txt'
            #gname='GLADE_flagship_cone18_dc_20_host.txt'
            #gname='GLADE_flagship_cone18_dc_30_host.txt'
            #gname='GLADE_flagship_cone18_dc_50_host.txt'
            #gname='GLADE_flagship_cone18_dc_100_host.txt'
            #gname='GLADE_flagship_cone18_dc_500_host.txt'
            #gname='GLADE_flagship_cone18_dc_1000_host.txt'
            
            #gname='GLADE_flagship_cone18_dc_10_hostdrop8.txt'
            #gname='GLADE_flagship_cone18_dc_20_hostdrop8.txt'
            #gname='GLADE_flagship_cone18_dc_30_hostdrop8.txt'
            #gname='GLADE_flagship_cone18_dc_50_hostdrop8.txt'
            #gname='GLADE_flagship_cone18_dc_100_hostdrop8.txt'
            #gname='GLADE_flagship_cone18_dc_500_hostdrop8.txt'
            #gname='GLADE_flagship_cone18_dc_1000_hostdrop8.txt'
            #gname='puppet_host.txt'
            #gname='puppet_host02.txt'
            #gname='puppet_host03.txt'
            #gname='puppet_line_host.txt'
            #gname='gw221800_single.txt'
            #gname='gw221809_single.txt'
            #gname='gw221800_single_z2.txt'
            
            #gname='GLADE_flagship_two_host_samezdiffang.txt'
            #gname='GLADE_fakeBS_170817.txt'
            #gname='GLADE_fakeBS.txt'
            #gname='GLADE_fakeBS68.txt'
            #gname='GLADE_fakeBS54.txt'
            #gname='GLADE_fakeBS40.txt'
            #gname='GLADE_fakeBS20.txt'
            #gname='GLADE_fakehost.txt'
            #gname='GLADE_fakehost_single.txt'
            #gname='GLADE_fakehost_single_170817.txt'
            #gname='GLADE_spectro_1.txt'
            #gname='GLADE_photo_1.txt'
            #gname='GLADE_trimmed_1.txt'
            #gname='GLADE_trimmed_5.txt'
            #gname='GLADE_trimmed_10.txt'
            #gname='GLADE_trimmed_15.txt'
            #gname='GLADE_trimmed_20.txt'
            #gname='GLADE_trimmed_50.txt'
            #gname='GLADE_zpluserr.txt'
            groupname='Galaxy_Group_Catalogue.csv'
            filepath_GLADE = os.path.join(self._path, gname)
            filepath_groups = os.path.join(miscPath, groupname)



            
            
            # ------ LOAD CATALOGUE
            if self.verbose:
                print('Loading GLADE from %s...' %filepath_GLADE)
            df = pd.read_csv(filepath_GLADE, sep=" ", header=None, low_memory=False)

            colnames = ['PGC', 'GWGC_name', 'HYPERLEDA_name', 'TWOMASS_name', 'SDSS_name', 'flag1', 'RA', 'dec',
                        'dL', 'dL_err', 'z', 'B', 'B_err', 'B_Abs', 'J', 'J_err', 'H', 'H_err', 'K', 'K_err',
                        'flag2', 'flag3'
                       ]
            
            df.columns=colnames
            print('Catalogue loaded')
             
            # ------  SELECT SUBSURVEYS
            
            
            # if object is named in a survey (not NaN), it is present in a survey
            for survey in ['GWGC', 'HYPERLEDA', 'TWOMASS', 'SDSS']:
                # new column named suvey that will be True or False
                # if object is in or not
                
                #copy the name column
                df.loc[:,survey] = df[survey + '_name']
                
                # NaN to False
                df[survey] = df[survey].fillna(False)
                # not False (i.e. a name) to True
                df.loc[df[survey] != False, survey] = True

          
            # include only objects that are contained in at least one of the surveys listed in _subsurveysIncl (is non-empty list)
            mask = df[self._subsurveysIncl[0]] == True
            for incl in self._subsurveysIncl[1:]:
                mask = mask | (df[incl] == True)
            df = df.loc[mask]
            
            # explicitely exclude objects if they are contained in some survey(s) in _subsurveysExcl
            # (may be necessary if an object is in multiple surveys)
            for excl in self._subsurveysExcl:
                df = df.loc[df[excl] == False]
            
            
            # ------ Add theta, phi for healpix in radians
            
            df.loc[:,"theta"] = np.pi/2 - (df.dec*np.pi/180)
            df.loc[:,"phi"]   = df.RA*np.pi/180
      
            
            
            
            or_dim = df.shape[0] # ORIGINAL LENGTH OF THE CATALOGUE
            if self.verbose:
                print('N. of objects: %s' %or_dim)
            
           
                
            
            # ------ Select parts of the catalogue
                    
            if z_flag is not None:
                df=  df[df.flag2 != z_flag ]
                if self.verbose:
                    print('Dropping galaxies with flag2=%s...' %z_flag)
                    print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
                
            if drop_z_uncorr:
                df=df[df['flag3']==1]
                if self.verbose:
                    print('Keeping only galaxies with redshift corrected for peculiar velocities...')
                
                    print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
              
                
            if drop_no_dist:
                df=df[df.dL.notna()==True]
                if self.verbose:
                    print('Keeping only galaxies with known value of luminosity distance...')
                    print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
            
            if drop_HyperLeda2:
                df=df.drop(df[(df['HYPERLEDA_name'].isna()) & (df['flag2']==2)].index)
                if self.verbose:
                    print("Dropping galaxies with HyperLeda name=null and flag2=2...")
                    print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.00%}".format(df.shape[0]/or_dim)+' of total' )
            
            
            
            # ------ Add z corrections
            
            
            if get_cosmo_z:
                if self.verbose:
                    print('Computing cosmological redshifts from given luminosity distance with H0=%s, Om0=%s...' %(cosmoGLADE.H0, cosmoGLADE.Om0))

                
                z_max = df[df.dL.notna()]['z'].max() +0.01
                z_min = max(0, df[df.dL.notna()]['z'].min() - 1e-05)
                if self.verbose:
                    print('Interpolating between z_min=%s, z_max=%s' %(z_min, z_max))
                z_grid = np.linspace(z_min, z_max, 200000)
                dL_grid = cosmoGLADE.luminosity_distance(z_grid).value
                
                if not drop_no_dist:
                    dLvals = df[df.dL.notna()]['dL']
                    if self.verbose:
                        print('%s points have valid entry for dist' %dLvals.shape[0])
                    zvals = df[df.dL.isna()]['z']
                    if self.verbose:
                        print('%s points have null entry for dist, correcting original redshift' %zvals.shape[0])
                    z_cosmo_vals = np.where(df.dL.notna(), np.interp( df.dL , dL_grid, z_grid), df.z)
                else:
                    z_cosmo_vals = np.interp( df.dL , dL_grid, z_grid)
                
                df.loc[:,'z_cosmo'] = z_cosmo_vals
                    
                
                
                
                if not CMB_correct and not group_correct and pos_z_cosmo:
                    if self.verbose:
                        print('Keeping only galaxies with positive cosmological redshift...')
                    df = df[df.z_cosmo >= 0]
                    if self.verbose:
                        print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
            #Raul: turn off correction with mockGW
            group_correct=0
            print(bcolors.WARNING + "Warning: No group correction" + bcolors.ENDC)
            if group_correct:
                if not get_cosmo_z:
                    raise ValueError('To apply group corrections, compute cosmological redshift first')
                if self.verbose:
                    print('Loading galaxy group catalogue from %s...' %filepath_groups)
                df_groups =  pd.read_csv(filepath_groups)
                self.group_correction(df, df_groups, which_z=which_z_correct)
                
          
            
            if CMB_correct:
                if not get_cosmo_z:
                    raise ValueError('To apply CMB corrections, compute cosmological redshift first')

                self.CMB_correction(df, which_z=which_z_correct)
                if pos_z_cosmo:
                    if self.verbose:
                        print('Keeping only galaxies with positive redshift in the colums %s...' %which_z)
                    df = df[df[which_z ]>= 0]
                    if self.verbose:
                        print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
            
            if which_z!='z':
                if self.verbose:
                    print('Renaming column %s to z. This will be used in the analysis.' %which_z)
                df = df.drop(columns='z')
                df.rename(columns={which_z:'z'}, inplace=True)
            # From now on, the relevant column for redshift, including all corrections, will be 'z'
            
            # ------ Potentially drop large z
            
            df = df[df.z < zMax]
            
            
            # ------ Add z errors
            
            #print(bcolors.WARNING + "not adding errors" + bcolors.ENDC)
            eq_err=1
            if err_vals is not None:
                if self.verbose:
                    print('Adding errors on z with %s values' %err_vals)
                if err_vals=='GLADE':
                    if eq_err==1:
                        print(bcolors.WARNING + "Warning: Equal Error Run" + bcolors.ENDC)
                        scales = np.where(df['flag2'].values==3, 0.001, 0.001)
                    else:
                        scales = np.where(df['flag2'].values==3, (1+df.z)*1e-03, (1+df.z)*1e-03)
                    #print('Scales is {}'.format(scales))
                elif err_vals=='const_perc':
                    scales=np.where(df['flag2'].values==3, df.z/100, df.z/10)
                elif err_vals=='const':
                    scales = np.full(df.shape[0], 200/clight)
                else:
                    raise ValueError('Enter valid choice for err_vals. Valid options are: GLADE, const, const_perc . Got %s' %err_vals)
                
                # restrict error to <=z itself. otherwise for z very close to 0 input is infeasible for keelin distributions, which would break things silently
                df.loc[:, 'z_err'] = np.minimum(scales, df.z.to_numpy())#original
                #df.loc[:, 'z_err'] = 0.002
                df.loc[:, 'z_lowerbound'] = df.z - 3*df.z_err
                df.loc[df.z_lowerbound < 0, 'z_lowerbound'] = 0
                df.loc[:, 'z_lower'] = df.z - df.z_err
                df.loc[df.z_lower < 0.5*df.z, 'z_lower'] = 0.5*df.z
                df.loc[:, 'z_upper'] = df.z + df.z_err
                df.loc[:, 'z_upperbound'] = df.z + 3*df.z_err
                #Raul:all numbers are divided by 2. See local code to restore the values
            
                # ------ Estimate galaxy posteriors with contant-in-comoving prior
                
                if computePosterior:
                    
                    self.include_vol_prior(df)

        # ------ End if not use precomputed table
        #        Always be able to still chose the weighting and cut.
        
        if band=='B' or band_weight=='B':
            add_B_lum=True
            add_K_lum=False
        elif band=='K' or band_weight=='B':
            add_B_lum=False
            add_K_lum=True
        else:
            add_B_lum=False
            add_K_lum=False
            
            
        # ------ Add B luminosity
        
        if add_B_lum:
            if self.verbose:
                print('Computing total luminosity in B band...')
            # add  a column for B-band luminosity
            my_dist=cosmoGLADE.luminosity_distance(df.z.values).value
            df.loc[:,"B_Abs_corr"]=df.B_Abs-5*np.log10(my_dist)+5*np.log10(df.dL.values)
            BLum = df.B_Abs_corr.apply(lambda x: TotLum(x, MBSun))
            df.loc[:,"B_Lum"] =BLum
            df = df.drop(columns='B_Abs')
            # df = df.drop(columns='B') don't assume it's here
            #print('Done.')
        
        
        # ------ Add K luminosity
        
        if add_K_lum:
            if self.verbose:
                print('Computing total luminosity in K band...')
            my_dist=cosmoGLADE.luminosity_distance(df.z.values).value
            df.loc[:,"K_Abs"]=df.K-5*np.log10(my_dist)-25
            KLum = df.K_Abs.apply(lambda x: TotLum(x, MKSun))
            df.loc[:,"K_Lum"]=KLum
            # df = df.drop(columns='K_Abs') don't assume it's here
            df = df.drop(columns='K')
        
        
        # ------ Apply cut in luminosity
        if band is not None:
            col_name=band+'_Lum'
            if band=='B':
                Lstar=LBstar07
            elif band=='K':
                Lstar=LKstar07
            
            L_th = Lcut*Lstar
            if self.verbose:
                print('Applying cut in luminosity in %s-band. Selecting galaxies with %s>%s x L_* = %s' %(band, col_name, Lcut, np.round(L_th,2)))
                print('L_* in %s band is L_*=%s' %(band, np.round(Lstar,5)))
            or_dim = df.shape[0]
            df = df[df[col_name]>L_th]
            if self.verbose:
                print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
            if self.verbose:
                print('Using %s-band to compute completeness.')
            band_vals = df.loc[:, col_name].values   
        else:
            if self.verbose:
                print('No cut in luminosity applied ' )
            if self.verbose:
                print('Using number counts to compute completeness.')
            band_vals = np.ones(df.shape[0])
        
        
        df.loc[:, 'completenessGoal'] = band_vals
        
        
         
        # ------ Add 'w' column for weights
        if band_weight is not None:
            w_name=band_weight+'_Lum'
            w = df.loc[:, w_name].values
            if self.verbose:
                print('Using %s for weighting' %col_name)
        else:
            w = np.ones(df.shape[0])
            if self.verbose:
                print('Using weights =1 .')
            
        df.loc[:, 'w'] = w
        
        
        # ------ Keep only some columns
        
        if colnames_final is not None:
            if self.verbose:
                print('Keeping only columns: %s' %colnames_final)
            df = df[colnames_final]
       
        # ------ Add pixel column. Note that not providing nest parameter to ang2pix defaults to nest=True, which has to be set in GW too!
        df.loc[:,"pix"  + str(self._nside)]   = hp.ang2pix(self._nside, df.theta.to_numpy(), df.phi.to_numpy())
        
        # ------
        if self.verbose:
            print('GLADE loaded.')
        
        
        self.data = self.data.append(df, ignore_index=True)
            
      

   
def TotLum(x, MSun): 
    return 10**(-0.4* (x+25-MSun))
