from globals import H0GLOB, Xi0Glob

####################################
# Set here the parameters of the run 
####################################


# --------------------------------------------------------------
# INFERENCE SETUP
# --------------------------------------------------------------
do_inference=True

## Variable for the inference:  H0 or Xi0
goalParam = 'H0'

## Output folder name
#fout = 'ET_newsigma'
fout = 'GW36_testmalm_11'
#Malmquist param
Malm_delta=0.15

#only to some  test, then remove 
delta=1#(0.318639,0.674490,0.977925,0.994458,1.281552,1.644854,1.959964,2)
forcePcopl=1
rate=0
## Specify the exponent of the redshift distribution , p(z) = dV/dz* (1+z)^(lamb-1)
lamb=1
EM=0
#mock BNS dl
#68 54 40 20 170817
myredshift=0.003692 #0.0155368 #0.0120821 #0.00862733 #0.003692 #0.0098 
mysigz= 0.3528418527151356 #0.0010199 #0.000851667 #0.000694791 #0.000511387 #0.0004 #testconst=0.3528418527151356
## Prior limits
Xi0min =  Xi0Glob # 0.3 
Xi0max =  Xi0Glob # 10
H0min =   30 # H0GLOB
H0max =    130


## Number of points for posterior grid
nPointsPosterior = 1000

verbose=True

# --------------------------------------------------------------
# GW DATA OPTIONS
# --------------------------------------------------------------


## Select dataset : O2, O3
observingRun = 'O3'

## Select BBH or BNS
eventType='BBH'

detector='ET'
#detector='vanilla'

print(detector)

## Specify which mass distribution to use. Options: O2, O3, NS-flat, NS-gauss
massDist='O3'


## Specify parameters of the broken power law model
#Raul: Using new parameter, to recover look at config original ora at some old runs
gamma1=1.5
gamma2=5.0
betaq=0.28
mMin=5
mMax=80
deltam=0.39

# Only one between mBreak and b should be given
b=None
mBreak=40


# How to select credible region in redshift, 'skymap' or 'header'
zLimSelection='skymap'

## Names of events to analyse. If None, all events in the folder will be used
#subset_names = ['GW221501'] #['GW190924_021846'] #['GW190924_021846'] #['GW190425',] #['GW190814']
#------------------------Pre-selected-runs----------------------------------
#-----------------------------------------GW36xxxx----------------------------------------
#subset_names=['GW360000','GW360001','GW360002','GW360003','GW360004','GW360005','GW360006','GW360007','GW360008','GW360009','GW360010','GW360011','GW360012','GW360013','GW360014','GW360015','GW360016','GW360017','GW360018','GW360019','GW360020','GW360021','GW360022','GW360023','GW360024','GW360025','GW360026','GW360027','GW360028','GW360029']#00
#subset_names=['GW360030','GW360031','GW360032','GW360033','GW360034','GW360035','GW360036','GW360037','GW360038','GW360039','GW360040','GW360041','GW360042','GW360043','GW360044','GW360045','GW360046','GW360047','GW360048','GW360049','GW360050','GW360051','GW360052','GW360053','GW360054','GW360055','GW360056','GW360057','GW360058']#01
#subset_names=['GW360059','GW360060','GW360061','GW360062','GW360063','GW360064','GW360065','GW360066','GW360067','GW360068','GW360069','GW360070','GW360071','GW360072','GW360073','GW360074','GW360075','GW360076','GW360077','GW360078','GW360079','GW360080','GW360081','GW360082','GW360083','GW360084','GW360085','GW360086','GW360087']#02
#subset_names=['GW360088','GW360089','GW360090','GW360091','GW360092','GW360093','GW360094','GW360095','GW360096','GW360097','GW360098','GW360099','GW360100','GW360101','GW360102','GW360103','GW360104','GW360105','GW360106','GW360107','GW360108','GW360109','GW360110','GW360111','GW360112','GW360113','GW360114','GW360115','GW360116']#03
#subset_names=['GW360117','GW360118','GW360119','GW360120','GW360121','GW360122','GW360123','GW360124','GW360125','GW360126','GW360127','GW360128','GW360129','GW360130','GW360131','GW360132','GW360133','GW360134','GW360135','GW360136','GW360137','GW360138','GW360139','GW360140','GW360141','GW360142','GW360143','GW360144','GW360145']#04
#subset_names=['GW360146','GW360147','GW360148','GW360149','GW360150','GW360151','GW360152','GW360153','GW360154','GW360155','GW360156','GW360157','GW360158','GW360159','GW360160','GW360161','GW360162','GW360163','GW360164','GW360165','GW360166','GW360167','GW360168','GW360169','GW360170','GW360171','GW360172','GW360173','GW360174']#05
#subset_names=['GW360175','GW360176','GW360177','GW360178','GW360179','GW360180','GW360181','GW360182','GW360183','GW360184','GW360185','GW360186','GW360187','GW360188','GW360189','GW360190','GW360191','GW360192','GW360193','GW360194','GW360195','GW360196','GW360197','GW360198','GW360199','GW360200','GW360201','GW360202','GW360203']#06
#subset_names=['GW360204','GW360205','GW360206','GW360207','GW360208','GW360209','GW360210','GW360211','GW360212','GW360213','GW360214','GW360215','GW360216','GW360217','GW360218','GW360219','GW360220','GW360221','GW360222','GW360223','GW360224','GW360225','GW360226','GW360227','GW360228','GW360229','GW360230','GW360231','GW360232']#07
#subset_names=['GW360233','GW360234','GW360235','GW360236','GW360237','GW360238','GW360239','GW360240','GW360241','GW360242','GW360243','GW360244','GW360245','GW360246','GW360247','GW360248','GW360249','GW360250','GW360251','GW360252','GW360253','GW360254','GW360255','GW360256','GW360257','GW360258','GW360259','GW360260','GW360261']#08
#subset_names=['GW360262','GW360263','GW360264','GW360265','GW360266','GW360267','GW360268','GW360269','GW360270','GW360271','GW360272','GW360273','GW360274','GW360275','GW360276','GW360277','GW360278','GW360279','GW360280','GW360281','GW360282','GW360283','GW360284','GW360285','GW360286','GW360287','GW360288','GW360289','GW360290']#09
#subset_names=['GW360291','GW360292','GW360293','GW360294','GW360295','GW360296','GW360297','GW360298','GW360299','GW360300','GW360301','GW360302','GW360303','GW360304','GW360305','GW360306','GW360307','GW360308','GW360309','GW360310','GW360311','GW360312','GW360313','GW360314','GW360315','GW360316','GW360317','GW360318','GW360319']#10
subset_names=['GW360320','GW360321','GW360322','GW360323','GW360324','GW360325','GW360326','GW360327','GW360328','GW360329','GW360330','GW360331','GW360332','GW360333','GW360334','GW360335','GW360336','GW360337','GW360338','GW360339','GW360340','GW360341']#11
#-----------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------


## Select events based on completeness at the nominal position
select_events=True

## Threshold in probability at position of the event, for event selection
completnessThreshCentral=0.7



## Confidence region for GW skymaps
level = 0.99
std_number=5 #if none, it is computed from level


# --------------------------------------------------------------
# GALAXY CATALOGUE OPTIONS
# --------------------------------------------------------------

## Galaxy catalogue
catalogue='GLADE'

# Check if events fall in DES or GWENS footprint
do_check_footprint=False


## Luminosity cut and band. 
# Band should be None if we use number counts
Lcut=0.6
# Band for lum cut
band= 'K' # B, K, or None . 
# Average galaxy density in comoving volume, used if band='None'. A number, or 'auto' (only for mask completeness) 
Nbar = 'auto'
# Band for lum weights
band_weight = None  # B, K, or None . 



## Use of galaxy redshift errors
galRedshiftErrors = False

## Use of galaxy posteriors, i.e. convolve the likelihood in redshift with a prior p(z) = dV_c/dz
galPosterior = False


# --------------------------------------------------------------
# COMPLETENESS AND COMPLETION OPTIONS
# --------------------------------------------------------------

## Completeness. 'load', 'pixel' main mi corregge con pix, 'mask', 'skip'
completeness = 'mask'
# path of completeness file if completeness='load' and using GLADE
completeness_path = 'hpx_B_zmin0p01_zmax0p25_nside32_npoints25.txt'
# Options for SuperPixelCompleteness
angularRes, interpolateOmega = 4, False
zRes = 30
# nMasks for mask completeness. 2 for DES/GWENS, >5 for GLADE
nMasks = 9
#
plot_comp=False


## Type of completion: 'mult' , 'add' or 'mix' 
completionType = 'mix'
# Use MC integration or not in the computation of additive completion
MChom=True
# N. of homogeneous MC samples
nHomSamples=15000



# --------------------------------------------------------------
# BETA OPTIONS
# --------------------------------------------------------------

## Which beta to use. 'fit', 'MC', 'hom', 'cat'
which_beta = 'hom'

# only used when which_beta='hom'. If 'scale', use individually SNR rescaled dmax estimate. If 'flat' use d of event. If a number use that for all events. 
betaHomdMax =15978.6 #600 roughly O3, for ET use this now or 15000 or 15978.6
#betaHomMax = 425.7 # O2 


# Max redshift  of the region R,  if beta is 'fit'
zR = 20 
# n of MC samples for beta MC
nSamplesBetaMC= 250000
nUseCatalogBetaMC = True
SNRthresh=100

# Use SNR at all orders or 1st order approximation.
# SNR at all orders is computed from a pre-computed grid
fullSNR=True
