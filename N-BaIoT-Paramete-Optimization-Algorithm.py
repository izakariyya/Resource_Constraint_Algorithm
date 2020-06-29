# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:05:18 2020

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
    
gridParams = { 'task': ['train'],
         'num_leaves': [5, 10, 15, 30, 50],
         'boosting_type': ['gbdt'],
         'bagging_freq': [3,5,7,9,10],
         'bagging_fraction': [0.2, 0.4, 0.6, 0.8, 0.9],
         'feature_fraction': [0.2, 0.4, 0.6, 0.8, 0.9],
         'learning_rate': [0.00001, 0.0001, 0.001, 0.01, 0.1],
         'objective': ['binary'],
         'reg_alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
         'verbose': [0]         
         }

p1 = { 'task': 'train',
         'num_leaves': 2,
         'boosting_type': 'gbdt',
         'bagging_freq': 2,
         'bagging_fraction': 0.1,
         'feature_fraction': 0.1,
         'learning_rate': 0.000001,
         'objective': 'binary',
         'reg_alpha': 0.000001,
         'verbose': 0         
         }

train, test, trainlabel, testl = data()

def traind(train, trainlabel, param):
    lgb_train =lgb.Dataset(train, trainlabel)
    gbm = lgb.train(param, lgb_train)
    return gbm

def predict_clf(gbm, test):
    ypred = gbm.predict(test)
    acc = roc_auc_score(testl, ypred)
    return acc

def optimz():
    md = traind(train, trainlabel, p1)
    mp1 = psutil.Process(os.getpid())
    mp1 = mp1.memory_info()
    sm1 = mp1.rss
    stt = time.time()
    ap1 = predict_clf(md, test)
    et1 = time.time()
    et1 = et1 - stt
    for param in ParameterGrid(gridParams):
        mdl = traind(train, trainlabel, param)
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
        while ((ap1 < acc) or (ap1 > acc)):
            if ((sm1 < stm) and (et1 < algtime)):
                print('optimized memory is done!')
            break
    return sm1, et1, ap1
          
efmem, eftime,  efacc = optimz()   
