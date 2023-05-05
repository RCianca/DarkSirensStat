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
#fout = 'GW99-None_00'
fout = 'GW97-genova_Pdet1InSearchArea_00'
#betaHomdMax =40000
#Malmquist param
#CDMtest #1.15229 #1.03942 #1.00032
#Malmtest #1.03886 #1.00993 #1.000399880059968
Malm_delta=1.15229 

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
H0min =   55 # H0GLOB
H0max =    85


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
#------------------------------------------GWtest-------------------------------------------
#subset_names=['GW960000']
#------------------------------------------GW97-------------------------------------------
subset_names=['GW970000','GW970001','GW970002','GW970003','GW970004','GW970005','GW970006','GW970007','GW970008','GW970009','GW970010','GW970011','GW970012','GW970013','GW970014','GW970015','GW970016','GW970017','GW970018','GW970019','GW970020','GW970021','GW970022','GW970023','GW970024','GW970025','GW970026','GW970027','GW970028','GW970029','GW970030','GW970031','GW970032','GW970033','GW970034','GW970035','GW970036','GW970037','GW970038']
#subset_names=['GW970039','GW970040','GW970041','GW970042','GW970043','GW970044','GW970045','GW970046','GW970047','GW970048','GW970049','GW970050','GW970051','GW970052','GW970053','GW970054','GW970055','GW970056','GW970057','GW970058','GW970059','GW970060','GW970061','GW970062','GW970063','GW970064','GW970065','GW970066','GW970067','GW970068','GW970069','GW970070','GW970071','GW970072','GW970073','GW970074','GW970075','GW970076']
#subset_names=['GW970077','GW970078','GW970079','GW970080','GW970081','GW970082','GW970083','GW970084','GW970085','GW970086','GW970087','GW970088','GW970089','GW970090','GW970091','GW970092','GW970093','GW970094','GW970095','GW970096','GW970097','GW970098','GW970099','GW970100','GW970101','GW970102','GW970103','GW970104','GW970105','GW970106','GW970107','GW970108','GW970109','GW970110','GW970111','GW970112','GW970113','GW970114']
#subset_names=['GW970115','GW970116','GW970117','GW970118','GW970119','GW970120','GW970121','GW970122','GW970123','GW970124','GW970125','GW970126','GW970127','GW970128','GW970129','GW970130','GW970131','GW970132','GW970133','GW970134','GW970135','GW970136','GW970137','GW970138','GW970139','GW970140','GW970141','GW970142','GW970143','GW970144','GW970145','GW970146','GW970147','GW970148','GW970149','GW970150','GW970151','GW970152']
#subset_names=['GW970153','GW970154','GW970155','GW970156','GW970157','GW970158','GW970159','GW970160','GW970161','GW970162','GW970163','GW970164','GW970165','GW970166','GW970167','GW970168','GW970169','GW970170','GW970171','GW970172','GW970173','GW970174','GW970175','GW970176','GW970177','GW970178','GW970179','GW970180','GW970181','GW970182','GW970183','GW970184','GW970185','GW970186','GW970187','GW970188','GW970189','GW970190']
#subset_names=['GW970191','GW970192','GW970193','GW970194','GW970195','GW970196','GW970197','GW970198','GW970199','GW970200','GW970201','GW970202','GW970203','GW970204','GW970205','GW970206','GW970207','GW970208','GW970209','GW970210','GW970211','GW970212','GW970213','GW970214','GW970215','GW970216','GW970217','GW970218','GW970219','GW970220','GW970221','GW970222','GW970223','GW970224','GW970225','GW970226','GW970227','GW970228']
#subset_names=['GW970229','GW970230','GW970231','GW970232','GW970233','GW970234','GW970235','GW970236','GW970237','GW970238','GW970239','GW970240','GW970241','GW970242','GW970243','GW970244','GW970245','GW970246','GW970247','GW970248','GW970249','GW970250','GW970251','GW970252','GW970253','GW970254','GW970255','GW970256','GW970257','GW970258','GW970259','GW970260','GW970261','GW970262','GW970263','GW970264','GW970265','GW970266']
#subset_names=['GW970267','GW970268','GW970269','GW970270','GW970271','GW970272','GW970273','GW970274','GW970275','GW970276','GW970277','GW970278','GW970279','GW970280','GW970281','GW970282','GW970283','GW970284','GW970285','GW970286','GW970287','GW970288','GW970289','GW970290','GW970291','GW970292','GW970293','GW970294','GW970295','GW970296','GW970297','GW970298','GW970299','GW970300']
#-----------------------------------------------------------------------------------------



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
