# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 21:48:23 2020

@author: Shashidhar
"""
import numpy as np

def normalizeData(x):
    return x/np.max(x, axis=1, keepdims=True)

def test_train_split(X, Y, ratio):
    X_train = X[:, :int(X.shape[1]*(1-ratio))]
    X_test = X[:, int(X.shape[1]*(1-ratio)):]
    Y_train = Y[:,:int(Y.shape[1]*(1-ratio))]
    Y_test = Y[:,int(Y.shape[1]*(1-ratio)):]
    return X_train, X_test, Y_train, Y_test