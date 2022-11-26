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
fout = 'Flag_volume_cone15_5_00DS'

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
gamma1=1.05
gamma2=5.17
betaq=0.28
mMin=2.22
mMax=86.16
deltam=0.39

# Only one between mBreak and b should be given
b=None
mBreak=36.7


# How to select credible region in redshift, 'skymap' or 'header'
zLimSelection='skymap'

## Names of events to analyse. If None, all events in the folder will be used
#subset_names = ['GW221501'] #['GW190924_021846'] #['GW190924_021846'] #['GW190425',] #['GW190814']
#------------------------Pre-selected-runs----------------------------------
#subset_names = ['GW221500','GW221501','GW221502','GW221503','GW221504','GW221505','GW221506','GW221507','GW221508','GW221509']

subset_names = ['GW221520','GW221521','GW221522','GW221523','GW221524']

#subset_names = ['GW221400','GW221421','GW221441','GW221402','GW221423','GW221443','GW221404','GW221425','GW221445','GW221406']#10 ma non in linea
#subset_names = ['GW221459','GW221408','GW221405','GW221420','GW221427','GW221429','GW221401','GW221434','GW221449','GW221447']#10 ma non in linea
#subset_names = ['GW221457','GW221455','GW221453','GW221452','GW221450','GW221448','GW221439','GW221438','GW221437','GW221435']#10 non in linea
#subset_names = ['GW221433','GW221432','GW221430','GW221418','GW221416','GW221414','GW221412','GW221411','GW221409','GW221428']#10
#subset_names = ['GW221402','GW221403','GW221404','GW221405','GW221406','GW221407','GW221408','GW221409','GW221410','GW221411','GW221412','GW221413','GW221414','GW221415','GW221416']#15
#subset_names = ['GW221405','GW221406','GW221407','GW221408','GW221409','GW221410','GW221411','GW221412','GW221413','GW221414']#10
#subset_names = ['GW221408','GW221409','GW221410','GW221411','GW221412']#5
#subset_names =  None

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
betaHomdMax =1000 #600 roughly O3 #1000 for ET  
#betaHomMax = 425.7 # O2 


# Max redshift  of the region R,  if beta is 'fit'
zR = 10
# n of MC samples for beta MC
nSamplesBetaMC= 250000
nUseCatalogBetaMC = True
SNRthresh=100

# Use SNR at all orders or 1st order approximation.
# SNR at all orders is computed from a pre-computed grid
fullSNR=True
