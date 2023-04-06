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
fout = 'GW33_Enzo_sig10samesig_11'
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
#--------------------------------------GW33xxxx-------------------------------------------
#subset_names=['GW330000','GW330001','GW330002','GW330003','GW330004','GW330005','GW330006','GW330007','GW330008','GW330009','GW330010','GW330011','GW330012','GW330013','GW330014','GW330015','GW330016','GW330017','GW330018','GW330019','GW330020','GW330021','GW330022','GW330023','GW330024','GW330025','GW330026','GW330027','GW330028','GW330029']#00
#subset_names=['GW330030','GW330031','GW330032','GW330033','GW330034','GW330035','GW330036','GW330037','GW330038','GW330039','GW330040','GW330041','GW330042','GW330043','GW330044','GW330045','GW330046','GW330047','GW330048','GW330049','GW330050','GW330051','GW330052','GW330053','GW330054','GW330055','GW330056','GW330057','GW330058']#01
#subset_names=['GW330059','GW330060','GW330061','GW330062','GW330063','GW330064','GW330065','GW330066','GW330067','GW330068','GW330069','GW330070','GW330071','GW330072','GW330073','GW330074','GW330075','GW330076','GW330077','GW330078','GW330079','GW330080','GW330081','GW330082','GW330083','GW330084','GW330085','GW330086','GW330087']#02
#subset_names=['GW330088','GW330089','GW330090','GW330091','GW330092','GW330093','GW330094','GW330095','GW330096','GW330097','GW330098','GW330099','GW330100','GW330101','GW330102','GW330103','GW330104','GW330105','GW330106','GW330107','GW330108','GW330109','GW330110','GW330111','GW330112','GW330113','GW330114','GW330115','GW330116']#03
#subset_names=['GW330117','GW330118','GW330119','GW330120','GW330121','GW330122','GW330123','GW330124','GW330125','GW330126','GW330127','GW330128','GW330129','GW330130','GW330131','GW330132','GW330133','GW330134','GW330135','GW330136','GW330137','GW330138','GW330139','GW330140','GW330141','GW330142','GW330143','GW330144','GW330145']#04
#subset_names=['GW330146','GW330147','GW330148','GW330149','GW330150','GW330151','GW330152','GW330153','GW330154','GW330155','GW330156','GW330157','GW330158','GW330159','GW330160','GW330161','GW330162','GW330163','GW330164','GW330165','GW330166','GW330167','GW330168','GW330169','GW330170','GW330171','GW330172','GW330173','GW330174']#05
#subset_names=['GW330175','GW330176','GW330177','GW330178','GW330179','GW330180','GW330181','GW330182','GW330183','GW330184','GW330185','GW330186','GW330187','GW330188','GW330189','GW330190','GW330191','GW330192','GW330193','GW330194','GW330195','GW330196','GW330197','GW330198','GW330199','GW330200','GW330201','GW330202','GW330203']#06
#subset_names=['GW330204','GW330205','GW330206','GW330207','GW330208','GW330209','GW330210','GW330211','GW330212','GW330213','GW330214','GW330215','GW330216','GW330217','GW330218','GW330219','GW330220','GW330221','GW330222','GW330223','GW330224','GW330225','GW330226','GW330227','GW330228','GW330229','GW330230','GW330231','GW330232']#07
#subset_names=['GW330233','GW330234','GW330235','GW330236','GW330237','GW330238','GW330239','GW330240','GW330241','GW330242','GW330243','GW330244','GW330245','GW330246','GW330247','GW330248','GW330249','GW330250','GW330251','GW330252','GW330253','GW330254','GW330255','GW330256','GW330257','GW330258','GW330259','GW330260','GW330261']#08
#subset_names=['GW330262','GW330263','GW330264','GW330265','GW330266','GW330267','GW330268','GW330269','GW330270','GW330271','GW330272','GW330273','GW330274','GW330275','GW330276','GW330277','GW330278','GW330279','GW330280','GW330281','GW330282','GW330283','GW330284','GW330285','GW330286','GW330287','GW330288','GW330289','GW330290']#09
#subset_names=['GW330291','GW330292','GW330293','GW330294','GW330295','GW330296','GW330297','GW330298','GW330299','GW330300','GW330301','GW330302','GW330303','GW330304','GW330305','GW330306','GW330307','GW330308','GW330309','GW330310','GW330311','GW330312','GW330313','GW330314','GW330315','GW330316','GW330317','GW330318','GW330319']#10
subset_names=['GW330320','GW330321','GW330322','GW330323','GW330324','GW330325','GW330326','GW330327','GW330328','GW330329','GW330330','GW330331','GW330332','GW330333','GW330334','GW330335','GW330336','GW330337','GW330338','GW330339','GW330340','GW330341']#11
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
