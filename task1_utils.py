import pandas as pd
import numpy as np

def generateDataVectors(data):
    t = pd.get_dummies(data["Class"]).astype(int).to_numpy()
    t = t.transpose()

    x = data[data.columns[:-1]].to_numpy()
    ones = np.ones((x.shape[0],1))
    x = np.hstack((x, ones))
    x = x.transpose()
    return x, t

def forward_pass(x,W):
    # Linear layer
    z = W @ x

    # sigmoid function
    return 1/(1 + np.exp(-z))

def loss(pred,label):
    squared_norms = np.sum((pred-label)**2,axis=0,keepdims=True) # (g_k-t_k)^T (g_k-t_k)
    return 1/2 * np.sum(squared_norms)

def getConfused(pred, t):
    conf_matrix = np.zeros((3,3))
    for i in range(pred.shape[1]):
        conf_matrix[np.argmax(t[:,i]),np.argmax(pred[:,i])] += 1
    return conf_matrix

def getErrorRate(conf_matrix):
    total_sum = np.sum(conf_matrix)
    diag_sum = np.trace(conf_matrix)
    return (total_sum-diag_sum)/total_sum