import pandas as pd 
import numpy as np

Parent_Catalogue=pd.read_csv('Uniform_paper.txt', header=None)

colnames=['Ngal','Comoving Distance','Luminosity Distance','z','phi','theta']
Parent_Catalogue.columns=colnames
print(Parent_Catalogue.columns)
print(Parent_Catalogue.shape)
print(Parent_Catalogue.head(10))