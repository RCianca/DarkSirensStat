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
fout = 'GW99-genova_NI_03'
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
#--------------------------------GW99------------------------------------------------
#subset_names=['GW990000','GW990001','GW990002','GW990003','GW990004','GW990005','GW990006','GW990007','GW990008','GW990009','GW990010','GW990011','GW990012','GW990013','GW990014','GW990015','GW990016','GW990017','GW990018','GW990019','GW990020','GW990021','GW990022','GW990023','GW990024','GW990025','GW990026','GW990027','GW990028','GW990029','GW990030','GW990031','GW990032','GW990033','GW990034','GW990035','GW990036','GW990037','GW990038']
#subset_names=['GW990039','GW990040','GW990041','GW990042','GW990043','GW990044','GW990045','GW990046','GW990047','GW990048','GW990049','GW990050','GW990051','GW990052','GW990053','GW990054','GW990055','GW990056','GW990057','GW990058','GW990059','GW990060','GW990061','GW990062','GW990063','GW990064','GW990065','GW990066','GW990067','GW990068','GW990069','GW990070','GW990071','GW990072','GW990073','GW990074','GW990075','GW990076']
#subset_names=['GW990077','GW990078','GW990079','GW990080','GW990081','GW990082','GW990083','GW990084','GW990085','GW990086','GW990087','GW990088','GW990089','GW990090','GW990091','GW990092','GW990093','GW990094','GW990095','GW990096','GW990097','GW990098','GW990099','GW990100','GW990101','GW990102','GW990103','GW990104','GW990105','GW990106','GW990107','GW990108','GW990109','GW990110','GW990111','GW990112','GW990113','GW990114']
subset_names=['GW990115','GW990116','GW990117','GW990118','GW990119','GW990120','GW990121','GW990122','GW990123','GW990124','GW990125','GW990126','GW990127','GW990128','GW990129','GW990130','GW990131','GW990132','GW990133','GW990134','GW990135','GW990136','GW990137','GW990138','GW990139','GW990140','GW990141','GW990142','GW990143','GW990144','GW990145','GW990146','GW990147','GW990148','GW990149','GW990150','GW990151','GW990152']
#subset_names=['GW990153','GW990154','GW990155','GW990156','GW990157','GW990158','GW990159','GW990160','GW990161','GW990162','GW990163','GW990164','GW990165','GW990166','GW990167','GW990168','GW990169','GW990170','GW990171','GW990172','GW990173','GW990174','GW990175','GW990176','GW990177','GW990178','GW990179','GW990180','GW990181','GW990182','GW990183','GW990184','GW990185','GW990186','GW990187','GW990188','GW990189','GW990190']
#subset_names=['GW990191','GW990192','GW990193','GW990194','GW990195','GW990196','GW990197','GW990198','GW990199','GW990200','GW990201','GW990202','GW990203','GW990204','GW990205','GW990206','GW990207','GW990208','GW990209','GW990210','GW990211','GW990212','GW990213','GW990214','GW990215','GW990216','GW990217','GW990218','GW990219','GW990220','GW990221','GW990222','GW990223','GW990224','GW990225','GW990226','GW990227','GW990228']
#subset_names=['GW990229','GW990230','GW990231','GW990232','GW990233','GW990234','GW990235','GW990236','GW990237','GW990238','GW990239','GW990240','GW990241','GW990242','GW990243','GW990244','GW990245','GW990246','GW990247','GW990248','GW990249','GW990250','GW990251','GW990252','GW990253','GW990254','GW990255','GW990256','GW990257','GW990258','GW990259','GW990260','GW990261','GW990262','GW990263','GW990264','GW990265','GW990266']
#subset_names=['GW990267','GW990268','GW990269','GW990270','GW990271','GW990272','GW990273','GW990274','GW990275','GW990276','GW990277','GW990278','GW990279','GW990280','GW990281','GW990282','GW990283','GW990284','GW990285','GW990286','GW990287','GW990288','GW990289','GW990290','GW990291','GW990292','GW990293','GW990294','GW990295','GW990296','GW990297','GW990298','GW990299','GW990300']
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
