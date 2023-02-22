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
fout = 'test_cluster_luncer_01'

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
#-----------------------------------------GW35xxxx----------------------------------------
#subset_names=['GW350000','GW350001','GW350002','GW350003','GW350004','GW350005','GW350006','GW350007','GW350008','GW350009','GW350010','GW350011','GW350012','GW350013','GW350014','GW350015','GW350016','GW350017','GW350018','GW350019','GW350020','GW350021','GW350022','GW350023','GW350024','GW350025','GW350026','GW350027','GW350028','GW350029']#00
#subset_names=['GW350030','GW350031','GW350032','GW350033','GW350034','GW350035','GW350036','GW350037','GW350038','GW350039','GW350040','GW350041','GW350042','GW350043','GW350044','GW350045','GW350046','GW350047','GW350048','GW350049','GW350050','GW350051','GW350052','GW350053','GW350054','GW350055','GW350056','GW350057','GW350058']#01
#subset_names=['GW350059','GW350060','GW350061','GW350062','GW350063','GW350064','GW350065','GW350066','GW350067','GW350068','GW350069','GW350070','GW350071','GW350072','GW350073','GW350074','GW350075','GW350076','GW350077','GW350078','GW350079','GW350080','GW350081','GW350082','GW350083','GW350084','GW350085','GW350086','GW350087']#02
#subset_names=['GW350088','GW350089','GW350090','GW350091','GW350092','GW350093','GW350094','GW350095','GW350096','GW350097','GW350098','GW350099','GW350100','GW350101','GW350102','GW350103','GW350104','GW350105','GW350106','GW350107','GW350108','GW350109','GW350110','GW350111','GW350112','GW350113','GW350114','GW350115','GW350116']#03
#subset_names=['GW350117','GW350118','GW350119','GW350120','GW350121','GW350122','GW350123','GW350124','GW350125','GW350126','GW350127','GW350128','GW350129','GW350130','GW350131','GW350132','GW350133','GW350134','GW350135','GW350136','GW350137','GW350138','GW350139','GW350140','GW350141','GW350142','GW350143','GW350144','GW350145']#04
#subset_names=['GW350146','GW350147','GW350148','GW350149','GW350150','GW350151','GW350152','GW350153','GW350154','GW350155','GW350156','GW350157','GW350158','GW350159','GW350160','GW350161','GW350162','GW350163','GW350164','GW350165','GW350166','GW350167','GW350168','GW350169','GW350170','GW350171','GW350172','GW350173','GW350174']#05
#subset_names=['GW350175','GW350176','GW350177','GW350178','GW350179','GW350180','GW350181','GW350182','GW350183','GW350184','GW350185','GW350186','GW350187','GW350188','GW350189','GW350190','GW350191','GW350192','GW350193','GW350194','GW350195','GW350196','GW350197','GW350198','GW350199','GW350200','GW350201','GW350202','GW350203']#06
#subset_names=['GW350204','GW350205','GW350206','GW350207','GW350208','GW350209','GW350210','GW350211','GW350212','GW350213','GW350214','GW350215','GW350216','GW350217','GW350218','GW350219','GW350220','GW350221','GW350222','GW350223','GW350224','GW350225','GW350226','GW350227','GW350228','GW350229','GW350230','GW350231','GW350232']#07
#subset_names=['GW350233','GW350234','GW350235','GW350236','GW350237','GW350238','GW350239','GW350240','GW350241','GW350242','GW350243','GW350244','GW350245','GW350246','GW350247','GW350248','GW350249','GW350250','GW350251','GW350252','GW350253','GW350254','GW350255','GW350256','GW350257','GW350258','GW350259','GW350260','GW350261']#08
#subset_names=['GW350262','GW350263','GW350264','GW350265','GW350266','GW350267','GW350268','GW350269','GW350270','GW350271','GW350272','GW350273','GW350274','GW350275','GW350276','GW350277','GW350278','GW350279','GW350280','GW350281','GW350282','GW350283','GW350284','GW350285','GW350286','GW350287','GW350288','GW350289','GW350290']#09
#subset_names=['GW350291','GW350292','GW350293','GW350294','GW350295','GW350296','GW350297','GW350298','GW350299','GW350300','GW350301','GW350302','GW350303','GW350304','GW350305','GW350306','GW350307','GW350308','GW350309','GW350310','GW350311','GW350312','GW350313','GW350314','GW350315','GW350316','GW350317','GW350318','GW350319']#10
subset_names=['GW350320','GW350321','GW350322','GW350323','GW350324','GW350325','GW350326','GW350327','GW350328','GW350329','GW350330','GW350331','GW350332','GW350333','GW350334','GW350335','GW350336','GW350337','GW350338','GW350339','GW350340','GW350341','GW350342','GW350343','GW350344','GW350345','GW350346','GW350347','GW350348']#11
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
