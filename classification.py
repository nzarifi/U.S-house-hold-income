# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:09:26 2019

@author: Niloofar-Z
"""
import matplotlib.pyplot as plt
from matplotlib import patheffects
from numpy import array
import os
import subprocess
from time import time
from operator import itemgetter
from scipy.stats import randint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from random import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\Niloofar-Z\Desktop\MetroCo\datamining\us-household-income')


df = pd.read_csv("cleaned-income.csv")        
df.columns.values
#here I drop zero value in Mean
df.drop(df.loc[df['Mean']==0].index, inplace=True)
X=df.drop(columns=['target_mean','target_median','Mean'],axis=1)
y=df[['target_mean']]

##############################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

#import learning model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#train the model
model.fit(X_train,y_train) #training


#predict testcases
y_pred = model.predict(X_test)
#'LogisticRegression' object has no attribute 'feature_importance_'
y_pred_probs = model.predict_proba(X_test)

#performance measures on the test set
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score 

accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
from sklearn.metrics import recall_score, precision_score, classification_report,accuracy_score
print(classification_report(y_test,y_pred))



df.target_mean.value_counts()
####################################

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=0)

seed = 7

kfold = model_selection.KFold(n_splits=5, random_state=seed)
model = LogisticRegression()
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
print("LogisticRegression, Cross_val_score=",results.mean() )

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print('confusion_matrix\n',
       confusion_matrix(y_test,y_pred))

##X.values[train_index]  to convert pandas to array
scores = []
max_score = 0
from sklearn.model_selection import KFold
kf = KFold(n_splits=4,random_state=0,shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = y.values[train_index], y.values[test_index]
    current_model = LogisticRegression()
    #train the model
    current_model.fit(X_train,y_train)
    #see performance score
    current_score = model.score(X_test,y_test)
    scores.append(current_score)
    if max_score < current_score:
        max_score = current_score
        best_model = current_model

print ('all scores: ', scores)
##########################################
"""
len(X.values[0]) I do not iterate the loop for 1196 times!
for this example is meaningless since I kept dummies for states name
and Type and ....
"""
##here I only show RFE works and result is not valid 
#(32526, 1196)
ss=list(df.columns.values)
dd=ss[1187:1195] #ugly solution 
#better solution is filtering Type to create list
#...=[x for x in ss if not x.startswith('Type')]

X=df[['Median','Mean','Stdev','sum_w','Primary_place','Type_CDP',
      'Type_City', 'Type_Community','Type_County','Type_Municipality',
      'Type_Town','Type_Track','Type_Urban']].values
y=df[['target_mean']].values
#now I have 13 features to run RFE    
from sklearn.feature_selection import RFE
accuracies = []
feature_set = []
max_accuracy_so_far = 0
for i in range(1,len(X[0])+1):
    selector = RFE(LogisticRegression(), i,verbose=1)
    selector = selector.fit(X, y)
    current_accuracy = selector.score(X,y)
    accuracies.append(current_accuracy)
    feature_set.append(selector.support_)
    if max_accuracy_so_far < current_accuracy:
        max_accuracy_so_far = current_accuracy
        selected_features = selector.support_
    print('End of iteration no. {}'.format(i))    
""" meaningless ressult based on RFE
chosen features:
'Median','Mean','Stdev','sum_w', 'Type_Track'   
"""