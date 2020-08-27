# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:20:05 2020

@author: Shashidhar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocess_utils import normalizeData, test_train_split
from lr_model_utils import model

dataset = pd.read_csv('Admission_Predict_Ver1.1.csv')
X = dataset.iloc[:, 1:8].values.T
Y = dataset.iloc[:, 8].values.T
Y = Y.reshape((1,Y.shape[0]))

X_Norm = normalizeData(X)

X_train, X_test, Y_train, Y_test = test_train_split(X_Norm, Y, 0.2)

d = model(X_train, Y_train, X_test, Y_test, num_iterations = 400000, learning_rate = 0.25, print_cost = True)


costs = np.squeeze(d['costs'])
test_costs = np.squeeze(d['test_costs'])
plt.plot(costs, label='training cost')
plt.plot(test_costs, label='testing cost')
plt.ylabel('cost')
plt.xlabel('iterations (per Hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
