# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:05:18 2020

@author: 1804499
"""

from sklearn.model_selection import train_test_split
import numpy as np
import lightgbm as lgb
import time
from memory_profiler import profile
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
#from sklearn.decomposition import PCA

def data():
   
    alldata = np.loadtxt("UNSW_2018_IoT_Botnet_Datasets.csv", delimiter = ",")
    j = len(alldata[0])
    data = alldata[:, 1:j]
    #data = PCA(0.99).fit_transform(data)
    benlabel = alldata[:, 0]
    bendata = (data - data.min()) / (data.max() - data.min())
    bendata, benmir, benlabel, benslabel = train_test_split(bendata, benlabel, test_size = 0.3)
    return bendata, benmir, benlabel, benslabel

    
#training complexity profiling
    
start_train_time = time.time()

precision = 10

fp = open('BoT-IoT_GBDT_Default_Parameter_Mem_Train.log', 'w+')

@profile(precision=precision, stream=fp)

# train the classifier

def train_gbc(train, trainlabel):
    #iotbinclf = lgb.LGBMClassifier(boosting_type = 'gbdt',num_leaves =10,
                        #n_estimators =2, learning_rate = 0.0001, reg_alpha=0.0001, min_data_in_leaf =1000, max_depth =2)
    iotbinclf = lgb.LGBMClassifier()
    trainclf = iotbinclf.fit(train, trainlabel)
    return trainclf

#load data

train, test, trainlabel, testl = data()

clf = train_gbc(train, trainlabel)

end_train_time = time.time()

print(end_train_time - start_train_time)

#testing complexity profile

start_test_time = time.time()

precision = 10

fp = open('BoT-IoT_GBDT_Default_Parameter_Mem_Test.log', 'w+')

@profile(precision=precision, stream=fp)

def predicted_gbc(clf, test):
    predictions = clf.predict(test)
    return predictions

# measure performance

y_pred = predicted_gbc(clf, test)

print('The accuracy of prediction is:', accuracy_score(testl, y_pred))
print('The roc_auc_score of prediction is:', roc_auc_score(testl, y_pred))
print('The null acccuracy is:', max(testl.mean(), 1 - testl.mean()))

end_test_time = time.time()

print(end_test_time -start_test_time ) 


