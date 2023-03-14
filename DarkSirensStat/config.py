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
fout = 'GW31_testpar_00'

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
subset_names=['GW310000','GW310001','GW310002','GW310003','GW310004','GW310005','GW310006','GW310007','GW310008','GW310009','GW310010','GW310011','GW310012','GW310013','GW310014','GW310015','GW310016','GW310017','GW310018','GW310019','GW310020','GW310021','GW310022','GW310023','GW310024','GW310025','GW310026','GW310027','GW310028','GW310029']#00
#subset_names=['GW310030','GW310031','GW310032','GW310033','GW310034','GW310035','GW310036','GW310037','GW310038','GW310039','GW310040','GW310041','GW310042','GW310043','GW310044','GW310045','GW310046','GW310047','GW310048','GW310049','GW310050','GW310051','GW310052','GW310053','GW310054','GW310055','GW310056','GW310057','GW310058']#01
#subset_names=['GW310059','GW310060','GW310061','GW310062','GW310063','GW310064','GW310065','GW310066','GW310067','GW310068','GW310069','GW310070','GW310071','GW310072','GW310073','GW310074','GW310075','GW310076','GW310077','GW310078','GW310079','GW310080','GW310081','GW310082','GW310083','GW310084','GW310085','GW310086','GW310087']#02
#subset_names=['GW310088','GW310089','GW310090','GW310091','GW310092','GW310093','GW310094','GW310095','GW310096','GW310097','GW310098','GW310099','GW310100','GW310101','GW310102','GW310103','GW310104','GW310105','GW310106','GW310107','GW310108','GW310109','GW310110','GW310111','GW310112','GW310113','GW310114','GW310115','GW310116']#03
#subset_names=['GW310117','GW310118','GW310119','GW310120','GW310121','GW310122','GW310123','GW310124','GW310125','GW310126','GW310127','GW310128','GW310129','GW310130','GW310131','GW310132','GW310133','GW310134','GW310135','GW310136','GW310137','GW310138','GW310139','GW310140','GW310141','GW310142','GW310143','GW310144','GW310145']#04
#subset_names=['GW310146','GW310147','GW310148','GW310149','GW310150','GW310151','GW310152','GW310153','GW310154','GW310155','GW310156','GW310157','GW310158','GW310159','GW310160','GW310161','GW310162','GW310163','GW310164','GW310165','GW310166','GW310167','GW310168','GW310169','GW310170','GW310171','GW310172','GW310173','GW310174']#05
#subset_names=['GW310175','GW310176','GW310177','GW310178','GW310179','GW310180','GW310181','GW310182','GW310183','GW310184','GW310185','GW310186','GW310187','GW310188','GW310189','GW310190','GW310191','GW310192','GW310193','GW310194','GW310195','GW310196','GW310197','GW310198','GW310199','GW310200','GW310201','GW310202','GW310203']#06
#subset_names=['GW310204','GW310205','GW310206','GW310207','GW310208','GW310209','GW310210','GW310211','GW310212','GW310213','GW310214','GW310215','GW310216','GW310217','GW310218','GW310219','GW310220','GW310221','GW310222','GW310223','GW310224','GW310225','GW310226','GW310227','GW310228','GW310229','GW310230','GW310231','GW310232']#07
#subset_names=['GW310233','GW310234','GW310235','GW310236','GW310237','GW310238','GW310239','GW310240','GW310241','GW310242','GW310243','GW310244','GW310245','GW310246','GW310247','GW310248','GW310249','GW310250','GW310251','GW310252','GW310253','GW310254','GW310255','GW310256','GW310257','GW310258','GW310259','GW310260','GW310261']#08
#subset_names=['GW310262','GW310263','GW310264','GW310265','GW310266','GW310267','GW310268','GW310269','GW310270','GW310271','GW310272','GW310273','GW310274','GW310275','GW310276','GW310277','GW310278','GW310279','GW310280','GW310281','GW310282','GW310283','GW310284','GW310285','GW310286','GW310287','GW310288','GW310289','GW310290']#09
#subset_names=['GW310291','GW310292','GW310293','GW310294','GW310295','GW310296','GW310297','GW310298','GW310299','GW310300','GW310301','GW310302','GW310303','GW310304','GW310305','GW310306','GW310307','GW310308','GW310309','GW310310','GW310311','GW310312','GW310313','GW310314','GW310315','GW310316','GW310317','GW310318','GW310319']#10
#subset_names=['GW310320','GW310321','GW310322','GW310323','GW310324','GW310325','GW310326','GW310327','GW310328','GW310329','GW310330','GW310331','GW310332','GW310333','GW310334','GW310335','GW310336','GW310337','GW310338','GW310339','GW310340','GW310341']#11
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
