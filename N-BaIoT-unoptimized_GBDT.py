# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:46:44 2020

@author: 1804499
"""

from sklearn.model_selection import train_test_split
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, ParameterGrid
import time
from memory_profiler import profile
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
import os
import psutil

def data():
    benign = np.loadtxt("benign_train.csv", delimiter = ",")
    benscan = np.loadtxt("ben_mir_gaf.csv", delimiter = ",")
    alldata = np.concatenate((benign, benscan))
    j = len(benscan[0])
    data = alldata[:, 1:j]
    benlabel = alldata[:, 0]
    bendata = (data - data.min()) / (data.max() - data.min())
    bendata, benmir, benlabel, benslabel = train_test_split(bendata, benlabel, test_size = 0.3)
    return bendata, benmir, benlabel, benslabel
    
param = { 'task': 'train',
         'num_leaves': 31,
         #'n_estimators': 2,
         'boosting_type': 'gbdt',
         'bagging_freq': 2,
         'bagging_fraction': 0.1,
         'feature_fraction': 0.1,
         'learning_rate': 0.1,
         'objective': 'binary',
         'reg_alpha': 0.0,
         'verbose': 0
         #'min_data_in_leaf': 0,
         #'max_depth':-1
         }



train, test, trainlabel, testl = data()

def traind(train, trainlabel, param):
    lgb_train =lgb.Dataset(train, trainlabel)
    gbm = lgb.train(param, lgb_train)
    return gbm

mdl = traind(train, trainlabel, param)




def predict_clf(gbm, test):
    ypred = gbm.predict(test)
    acc = roc_auc_score(testl, ypred)
    return acc

def eval():
    mi = psutil.Process(os.getpid())
    mit = mi.memory_info()
    sti = mit.rss
    start_test_time = time.time()
    acc = predict_clf(mdl, test)
    end_test_time = time.time()
    algtime = end_test_time - start_test_time
    mp = psutil.Process(os.getpid())
    mpt = mp.memory_info()
    stm = mpt.rss - sti
    return acc, algtime, stm


ac, algt, stm = eval()   
