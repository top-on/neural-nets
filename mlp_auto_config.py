# -*- coding: utf-8 -*-
"""
Exploring SCIPY's capabilities to automatically tune a neural net

Created on Mon Oct 31 16:10:56 2016

@author: Thorben Jensen
"""

#%% working folder

import os

os.chdir('Documents\\repositories\\neural_nets')
os.getcwd()

#%% import data

import pandas as pd

# training data
train = pd.read_csv("occupancy_data/datatraining.txt")
x_train = train.ix[:,1:6]
y_train = train.ix[:, 6]

# testing data
test = pd.read_csv("occupancy_data/datatest.txt")
x_test = test.ix[:,1:6]
y_test = test.ix[:, 6]

# testing data2
test2 = pd.read_csv("occupancy_data/datatest2.txt")                                      
x_test2 = test2.ix[:,1:6]
y_test2 = test2.ix[:, 6]


#%% Visualize data

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 4)

x_train.plot()
plt.show()

y_train.plot()
plt.show()


#%% regularize the data

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()

# transform training data
scaler.fit(x_train)
x_train_transform = pd.DataFrame( scaler.fit_transform(x_train ))

# transform testing data
x_test_transform = pd.DataFrame( scaler.transform(x_test ))
x_test2_transform = pd.DataFrame( scaler.transform(x_test2 ))


#%% train MLP with GridSearchCV

from sklearn.neural_network import MLPClassifier
from sklearn.grid_search import GridSearchCV

hidden_layer_sizes = [
    (4, 4, 4, 2),
    (5, 5, 5, 2),
    (4, 4, 4, 2)
]

model = MLPClassifier(solver='lbfgs', 
                      alpha=1e-5, 
                      random_state=1)

grid = GridSearchCV(estimator  = model,
                    param_grid = dict(hidden_layer_sizes=hidden_layer_sizes),
                    cv = 5, 
                    verbose = 1,
                    n_jobs = 2)

grid.fit(x_train_transform, y_train)

print(grid)
print(grid.best_score_)
print(grid.best_estimator_.hidden_layer_sizes)


#%% tune the MLP's hyperparameters with scipy.optimize

import numpy as np
from sklearn.model_selection import cross_val_score

# define training function with score output
def train_mlp(net_width):
    net_width = int(np.round( net_width ))
    
    model = MLPClassifier(solver='lbfgs', 
                      alpha=1e-5, 
                      random_state=1,
                      hidden_layer_sizes = (net_width, net_width, net_width, 2))
    scores = cross_val_score(model, x_train_transform, y_train, 
                             n_jobs=2, verbose=1, cv=2)
    return 1 - np.mean(scores) # inverse score for minimization

    
#%% 

train_mlp(5)


#%% Stepwise exploration configuration space with 'GridSearchCV'

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', 
                      alpha=1e-5, 
                      random_state=1)

# TODO: create network configurations from for-loop
depth = 2
width = 4

def adjacent_hidden_layer_sizes( best_depth, best_width ):
    hidden_layer_sizes = []
    for depth in range(best_depth-1,best_depth+2):
        for width in range(best_width-1,best_width+2):
            hidden_layer_sizes.append( tuple([width  ] * depth + [2]) )
    return hidden_layer_sizes

param_dist = { "hidden_layer_sizes": adjacent_hidden_layer_sizes( depth, width) }

# run randomized search
n_iter_search = 3
grid = GridSearchCV(clf, 
                             param_grid=param_dist,
                             cv=5,
                             verbose=1)
grid.fit(x_train_transform, y_train)

print(grid)
print(grid.best_score_)
print(grid.best_estimator_.hidden_layer_sizes)


    
#%%  predictions (positives)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

y_predict  = pd.DataFrame( grid.predict( x_test_transform  ))
y_predict2 = pd.DataFrame( grid.predict( x_test2_transform ))

print("Test set 1:")
print(confusion_matrix(y_test,  y_predict))
print("precision: ", precision_score(y_test, y_predict))
print("recall: ", recall_score(y_test, y_predict))
print()

print("Test set 2:")
print(confusion_matrix(y_test2, y_predict2))
print("precision: ", precision_score(y_test2, y_predict2))
print("recall: ", recall_score(y_test2, y_predict2))    
    