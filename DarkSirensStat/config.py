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
fout = 'run28_confronto_comp100_03'

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
#----------------------------23xxxx runs------------------------------------
#subset_names=['GW230000','GW230001','GW230002','GW230003','GW230004','GW230005','GW230006','GW230007','GW230008','GW230009','GW230010','GW230011','GW230012','GW230013','GW230014','GW230015'] #Those will not be separated

#subset_names = ['GW230016','GW230017','GW230018','GW230019','GW230020','GW230021','GW230022','GW230023','GW230024','GW230025','GW230026','GW230027','GW230028','GW230029','GW230030','GW230031','GW230032','GW230033','GW230034','GW230035','GW230036','GW230037','GW230038','GW230039','GW230040','GW230041','GW230042','GW230043','GW230044','GW230045']

#subset_names = ['GW230046','GW230047','GW230048','GW230049','GW230050','GW230051','GW230052','GW230053','GW230054','GW230055','GW230056','GW230057','GW230058','GW230059','GW230060','GW230061','GW230062','GW230063','GW230064','GW230065','GW230066','GW230067','GW230068','GW230069','GW230070','GW230071','GW230072','GW230073','GW230074','GW230075']

#subset_names = ['GW230076','GW230077','GW230078','GW230079','GW230080','GW230081','GW230082','GW230083','GW230084','GW230085','GW230086','GW230087','GW230088','GW230089','GW230090','GW230091','GW230092','GW230093','GW230094','GW230095','GW230096','GW230097','GW230098','GW230099','GW230100','GW230101','GW230102','GW230103','GW230104','GW230105']

#subset_names = ['GW230106','GW230107','GW230108','GW230109','GW230110','GW230111']

#----------------------------24xxxx runs------------------------------------
#subset_names=['GW240000','GW240001','GW240002','GW240003','GW240004','GW240005','GW240006','GW240007','GW240008','GW240009','GW240010','GW240011','GW240012','GW240013','GW240014','GW240015'] #Those will not be separated

#subset_names = ['GW240016','GW240017','GW240018','GW240019','GW240020','GW240021','GW240022','GW240023','GW240024','GW240025','GW240026','GW240027','GW240028','GW240029','GW240030','GW240031','GW240032','GW240033','GW240034','GW240035','GW240036','GW240037','GW240038','GW240039','GW240040','GW240041','GW240042','GW240043','GW240044','GW240045']

#subset_names = ['GW240046','GW240047','GW240048','GW240049','GW240050','GW240051','GW240052','GW240053','GW240054','GW240055','GW240056','GW240057','GW240058','GW240059','GW240060','GW240061','GW240062','GW240063','GW240064','GW240065','GW240066','GW240067','GW240068','GW240069','GW240070','GW240071','GW240072','GW240073','GW240074','GW240075']

#subset_names = ['GW240076','GW240077','GW240078','GW240079','GW240080','GW240081','GW240082','GW240083','GW240084','GW240085','GW240086','GW240087','GW240088','GW240089','GW240090','GW240091','GW240092','GW240093','GW240094','GW240095','GW240096','GW240097','GW240098','GW240099','GW240100','GW240101','GW240102','GW240103','GW240104','GW240105']

#subset_names = ['GW240106','GW240107','GW240108','GW240109','GW240110','GW240111']

#---------------------------------------------------------------------------
#----------------------------25xxxx runs------------------------------------
#subset_names=['GW250000','GW250001','GW250002','GW250003','GW250004','GW250005','GW250006','GW250007','GW250008','GW250009','GW250010','GW250011','GW250012','GW250013','GW250014','GW250015'] #Those will not be separated

#subset_names = ['GW250016','GW250017','GW250018','GW250019','GW250020','GW250021','GW250022','GW250023','GW250024','GW250025','GW250026','GW250027','GW250028','GW250029','GW250030','GW250031','GW250032','GW250033','GW250034','GW250035','GW250036','GW250037','GW250038','GW250039','GW250040','GW250041','GW250042','GW250043','GW250044','GW250045']

#subset_names = ['GW250046','GW250047','GW250048','GW250049','GW250050','GW250051','GW250052','GW250053','GW250054','GW250055','GW250056','GW250057','GW250058','GW250059','GW250060','GW250061','GW250062','GW250063','GW250064','GW250065','GW250066','GW250067','GW250068','GW250069','GW250070','GW250071','GW250072','GW250073','GW250074','GW250075']

#subset_names = ['GW250076','GW250077','GW250078','GW250079','GW250080','GW250081','GW250082','GW250083','GW250084','GW250085','GW250086','GW250087','GW250088','GW250089','GW250090','GW250091','GW250092','GW250093','GW250094','GW250095','GW250096','GW250097','GW250098','GW250099','GW250100','GW250101','GW250102','GW250103','GW250104','GW250105']

#subset_names = ['GW250106','GW250107','GW250108','GW250109','GW250110','GW250111']

#---------------------------------------------------------------------------
#----------------------------26xxxx runs------------------------------------
#subset_names=['GW260000','GW260001','GW260002','GW260003','GW260004','GW260005','GW260006','GW260007','GW260008','GW260009','GW260010','GW260011','GW260012','GW260013','GW260014','GW260015'] #Those will not be separated

#subset_names = ['GW260016','GW260017','GW260018','GW260019','GW260020','GW260021','GW260022','GW260023','GW260024','GW260025','GW260026','GW260027','GW260028','GW260029','GW260030','GW260031','GW260032','GW260033','GW260034','GW260035','GW260036','GW260037','GW260038','GW260039','GW260040','GW260041','GW260042','GW260043','GW260044','GW260045']

#subset_names = ['GW260046','GW260047','GW260048','GW260049','GW260050','GW260051','GW260052','GW260053','GW260054','GW260055','GW260056','GW260057','GW260058','GW260059','GW260060','GW260061','GW260062','GW260063','GW260064','GW260065','GW260066','GW260067','GW260068','GW260069','GW260070','GW260071','GW260072','GW260073','GW260074','GW260075']

#subset_names = ['GW260076','GW260077','GW260078','GW260079','GW260080','GW260081','GW260082','GW260083','GW260084','GW260085','GW260086','GW260087','GW260088','GW260089','GW260090','GW260091','GW260092','GW260093','GW260094','GW260095','GW260096','GW260097','GW260098','GW260099','GW260100','GW260101','GW260102','GW260103','GW260104','GW260105']

#subset_names = ['GW260106','GW260107','GW260108','GW260109','GW260110','GW260111']

#---------------------------------------------------------------------------
#----------------------------27xxxx runs------------------------------------
#subset_names=['GW270000','GW270001','GW270002','GW270003','GW270004','GW270005','GW270006','GW270007','GW270008','GW270009','GW270010','GW270011','GW270012','GW270013','GW270014','GW270015'] #Those will not be separated

#subset_names = ['GW270016','GW270017','GW270018','GW270019','GW270020','GW270021','GW270022','GW270023','GW270024','GW270025','GW270026','GW270027','GW270028','GW270029','GW270030','GW270031','GW270032','GW270033','GW270034','GW270035','GW270036','GW270037','GW270038','GW270039','GW270040','GW270041','GW270042','GW270043','GW270044']
#--------------------------------------------------------------------------------
#----------------------------28xxxx runs------------------------------------
#subset_names= ['GW280000','GW280001','GW280002','GW280003','GW280004','GW280005','GW280006','GW280007','GW280008','GW280009','GW280010','GW280011','GW280012','GW280013','GW280014','GW280015','GW280016','GW280017','GW280018','GW280019','GW280020','GW280021','GW280022','GW280023','GW280024','GW280025','GW280026','GW280027','GW280028','GW280029']

#subset_names= ['GW280030','GW280031','GW280032','GW280033','GW280034','GW280035','GW280036','GW280037','GW280038','GW280039','GW280040','GW280041','GW280042','GW280043','GW280044','GW280045','GW280046','GW280047','GW280048','GW280049','GW280050','GW280051','GW280052','GW280053','GW280054','GW280055','GW280056','GW280057','GW280058']

#subset_names= ['GW280059','GW280060','GW280061','GW280062','GW280063','GW280064','GW280065','GW280066','GW280067','GW280068','GW280069','GW280070','GW280071','GW280072','GW280073','GW280074','GW280075','GW280076','GW280077','GW280078','GW280079','GW280080','GW280081','GW280082','GW280083','GW280084','GW280085','GW280086','GW280087']

subset_names= ['GW280088','GW280089','GW280090','GW280091','GW280092','GW280093','GW280094','GW280095','GW280096','GW280097','GW280098','GW280099','GW280100','GW280101','GW280102','GW280103','GW280104','GW280105','GW280106','GW280107','GW280108','GW280109','GW280110','GW280111','GW280112','GW280113','GW280114','GW280115','GW280116']

#subset_names= ['GW280117','GW280118','GW280119','GW280120','GW280121','GW280122','GW280123','GW280124','GW280125','GW280126','GW280127','GW280128','GW280129','GW280130','GW280131','GW280132','GW280133','GW280134','GW280135','GW280136','GW280137','GW280138','GW280139','GW280140','GW280141','GW280142','GW280143','GW280144','GW280145']

#subset_names= ['GW280146','GW280147','GW280148','GW280149','GW280150','GW280151','GW280152','GW280153','GW280154','GW280155','GW280156','GW280157','GW280158','GW280159','GW280160','GW280161','GW280162','GW280163','GW280164','GW280165','GW280166','GW280167','GW280168','GW280169','GW280170','GW280171','GW280172','GW280173','GW280174']

#subset_names= ['GW280175','GW280176','GW280177','GW280178','GW280179','GW280180','GW280181','GW280182','GW280183','GW280184','GW280185','GW280186','GW280187','GW280188','GW280189','GW280190','GW280191','GW280192','GW280193','GW280194','GW280195','GW280196','GW280197','GW280198','GW280199','GW280200','GW280201','GW280202','GW280203']

#subset_names= ['GW280204','GW280205','GW280206','GW280207','GW280208','GW280209','GW280210','GW280211','GW280212','GW280213','GW280214','GW280215','GW280216','GW280217','GW280218','GW280219','GW280220','GW280221','GW280222','GW280223','GW280224','GW280225','GW280226','GW280227','GW280228','GW280229','GW280230','GW280231','GW280232']

#subset_names= ['GW280233','GW280234','GW280235','GW280236','GW280237','GW280238','GW280239','GW280240','GW280241','GW280242','GW280243','GW280244','GW280245','GW280246','GW280247','GW280248','GW280249','GW280250','GW280251','GW280252','GW280253','GW280254','GW280255','GW280256','GW280257','GW280258','GW280259','GW280260','GW280261']

#subset_names= ['GW280262','GW280263','GW280264','GW280265','GW280266','GW280267','GW280268','GW280269','GW280270','GW280271','GW280272','GW280273','GW280274','GW280275','GW280276','GW280277','GW280278','GW280279','GW280280','GW280281','GW280282','GW280283','GW280284','GW280285','GW280286','GW280287','GW280288','GW280289']
#--------------------------------------------------------------------------------
#subset_names = ['GW222000','GW222001','GW222002','GW222003','GW222004','GW222005','GW222006','GW222007','GW222008','GW222009','GW222010','GW222011','GW222012','GW222013','GW222014','GW222015']
#subset_names = ['GW222016','GW222017','GW222018','GW222019','GW222020','GW222021','GW222022','GW222023','GW222024','GW222025','GW222026','GW222027']
#subset_names = ['GW222028','GW222029','GW222030','GW222031','GW222032','GW222033','GW222034','GW222035','GW222036','GW222037','GW222038','GW222039','GW222040','GW222041']
#subset_names = ['GW222042','GW222043','GW222044','GW222045','GW222046','GW222047','GW222048','GW222049','GW222050','GW222051','GW222052','GW222053','GW222054','GW222055']
#subset_names = ['GW222056','GW222057','GW222058','GW222059','GW222060','GW222061','GW222062','GW222063','GW222064','GW222065','GW222066','GW222067','GW222068','GW222069']
#subset_names = ['GW222070','GW222071']

#subset_names = ['GW221900','GW221901','GW221902','GW221903','GW221904','GW221905','GW221906','GW221907','GW221908','GW221909','GW221910','GW221911','GW221912','GW221913']
#subset_names=['GW221914','GW221915','GW221916','GW221917','GW221918','GW221919','GW221920','GW221921','GW221922','GW221923','GW221924','GW221925','GW221926','GW221927']
#subset_names=['GW221928','GW221929','GW221930','GW221931','GW221932','GW221933','GW221934','GW221935','GW221936','GW221937','GW221938','GW221939','GW221940','GW221941']

#subset_names = ['GW221800']

#subset_names = ['GW221800','GW221801','GW221802','GW221803','GW221804','GW221805','GW221806','GW221807','GW221808','GW221809']

#subset_names = ['GW221700','GW221701','GW221703','GW221704','GW221705','GW221706','GW221707','GW221709','GW221710','GW221711']
#subset_names = ['GW221702','GW221708']

#subset_names = ['GW221600','GW221611','GW221602','GW221613','GW221604','GW221615','GW221606','GW221617','GW221608','GW221619']

#subset_names = ['GW221600']

#subset_names = ['GW221500','GW221511','GW221502','GW221513','GW221504','GW221515','GW221506','GW221517','GW221508','GW221519','GW221520','GW221531','GW221522','GW221533','GW221524','GW221535','GW221526','GW221537','GW221528','GW221539']

#subset_names = ['GW221500','GW221511','GW221502','GW221513','GW221504','GW221515','GW221506','GW221517','GW221508','GW221519','GW221520','GW221531','GW221522','GW221533','GW221524']

#subset_names = ['GW221500','GW221521','GW221502','GW221523','GW221504','GW221525','GW221506','GW221527','GW221508','GW221529']

#subset_names = ['GW221510','GW221531','GW221512','GW221533','GW221514','GW221535','GW221516','GW221537','GW221518','GW221539']


#subset_names = ['GW221529','GW221500']

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
