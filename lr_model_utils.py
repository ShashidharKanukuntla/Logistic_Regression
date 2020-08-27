# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 21:50:35 2020

@author: Shashidhar
"""
import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def getPrediction(w, b, X):
    
    return sigmoid(np.dot(w.T,X)+b)

def getCost(A, Y, m):
    return np.squeeze(-(1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))

def initialize_weights(dim):
    
    w = np.zeros((dim,1))
    b = 0
    
    return w, b

def getGrads(X, Y, A, w, m):
    
    dw = (1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*np.sum(A-Y)
    
    grads = {"dw": dw,
             "db": db}
    
    return grads

def optimize(w, b, X, Y, X_test, Y_test, num_iterations, learning_rate, print_cost = False):
    
    m = X.shape[1]
    costs = []
    test_costs = []
    
    for i in range(num_iterations):
        
        A = getPrediction(w, b, X)  
        cost = getCost(A, Y, m)
        A_test = getPrediction(w, b, X_test) 
        test_cost = getCost(A_test, Y_test, X_test.shape[1])
        grads = getGrads(X, Y, A, w, m)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w-learning_rate*dw
        b = b-learning_rate*db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
            test_costs.append(test_cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs, test_costs

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.2, print_cost = False):
    
    w, b = initialize_weights(X_train.shape[0])

    parameters, grads, costs, test_costs = optimize(w, b, X_train, Y_train,X_test, Y_test, num_iterations, learning_rate, print_cost)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = getPrediction(w, b, X_test)
    Y_prediction_train = getPrediction(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "test_costs": test_costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d