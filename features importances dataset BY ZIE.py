# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:04:06 2019

@author: HP User
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

dataset=pd.read_csv('C:/Users/HP User/Desktop/datacat.csv')
dataset.columns.values
#*******************diviser les données*****************

X=dataset.loc[:,dataset.columns!='ROAS']
print X.columns.values
print X.shape

Y=dataset.loc[:,dataset.columns=='ROAS']
print Y.shape


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)



np.isnan(dataset.any()) #and gets False
np.isfinite(dataset.all()) #and gets True
#dataset[np.isfinite(dataset) == True] = 0


## Importez le modèle de forêt aléatoire. 
from sklearn.ensemble import RandomForestClassifier 
## Cette ligne instancie le modèle. 
rf = RandomForestClassifier () 
## Ajustez le modèle sur vos données d’entraînement. 
rf.fit (X_train, Y_train) 
## Et notez-le sur vos données de test. 
rf.score (X_train,Y_train)
feature_importances = pd.DataFrame(rf.feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance', ascending=False)
print feature_importances