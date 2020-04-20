# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:17:33 2020

@author: 1804499
"""

from sklearn.model_selection import train_test_split
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, ParameterGrid 
import csv
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


#clf = lgb.LGBMClassifier()

gridParams = {
         'num_leaves': [10, 20, 30, 40],
         'n_estimators': [2, 4, 6, 8],
         'boosting_type': ['rf', 'gbdt'],
         'bagging_freq': [2, 4, 6, 8],
         'bagging_fraction': [0.2, 0.4, 0.6, 0.8],
         'learning_rate': [0.0001, 0.001, 0.01, 0.1],
         'objective': ['binary'],
         'reg_alpha': [0.0001, 0.001, 0.01, 0.1],
         'min_data_in_leaf': [1000, 100, 10, 0],
         'max_depth':[2, 4, 6, 8]}




train, test, trainlabel, testl = data()


start_train_time = time.time()

precision = 10

fp = open('Full_BoT_IoT_LGBM_OPZ_Mem_Train.log', 'w+')

@profile(precision=precision, stream=fp)

def traind(train, trainlabel, param):
    lgb_train =lgb.Dataset(train, trainlabel)
    gbm = lgb.train(param, lgb_train)
    return gbm


start_test_time = time.time()

precision = 10

fp = open('Full_BoT_IoT_LGBM_OPZ_Mem_Test.log', 'w+')

@profile(precision=precision, stream=fp)

def predict_clf(gbm, test):
    ypred = gbm.predict(test, num_iteration=gbm.best_iteration)
    acc = roc_auc_score(testl, ypred)
    return acc
    
    
with open('Full_BoT_IoT_LGBM_OPTZ_Results.csv', 'w', newline ='') as f:
    writer = csv.writer(f)
    for param in ParameterGrid(gridParams):
        optz = []
        mdl = traind(train, trainlabel, param)
        end_train_time = time.time()
        train_time = end_train_time - start_train_time
        acc = predict_clf(mdl, test)
        end_test_time = time.time()
        test_time = end_test_time - start_test_time
        optz.append(param)
        optz.append(train_time)
        optz.append(test_time)
        optz.append(acc)
        writer.writerow(optz)






