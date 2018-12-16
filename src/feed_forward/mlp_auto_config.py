# -*- coding: utf-8 -*-
"""
Reusable solution to automatically tune a neural net

Created on Mon Oct 31 16:10:56 2016

@author: Thorben Jensen
"""

#%% imports

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  

#%% working folder


#os.chdir('Documents\\repositories\\neural_nets')
#os.chdir('it/neural_nets/src')
os.getcwd()

#%% import data


# training data
train = pd.read_csv("../data/datatraining.txt")
x_train = train.ix[:,1:6]
y_train = train.ix[:, 6]

# testing data
test = pd.read_csv("../data/datatest.txt")
x_test = test.ix[:,1:6]
y_test = test.ix[:, 6]

# testing data2
test2 = pd.read_csv("../data/datatest2.txt")                                      
x_test2 = test2.ix[:,1:6]
y_test2 = test2.ix[:, 6]


#%% Visualize data

plt.rcParams['figure.figsize'] = (8, 3)

x_train.plot()
plt.show()

y_train.plot()
plt.show()


#%% regularize the data

scaler = StandardScaler()

# transform training data
scaler.fit(x_train)
x_train_transform = pd.DataFrame( scaler.fit_transform(x_train ))

# transform testing data
x_test_transform = pd.DataFrame( scaler.transform(x_test ))
x_test2_transform = pd.DataFrame( scaler.transform(x_test2 ))


#%% grow net dimensions with loop over 'GridSearchCV'


def adjacent_hidden_layer_sizes( best_depth, best_width ):
    hidden_layer_sizes = []
    for depth in range( max( 1, best_depth - 2 ), best_depth + 3 ):
        for width in range( max( 1, best_width - 2 ),best_width + 3 ):
            hidden_layer_sizes.append( tuple( [ width ] * depth ) )
    return hidden_layer_sizes


def optimize_MLPClassifier_layout(x, y, depth, width):
    
    # print referent parameters
    print("Variying net layout around:")
    print("depth: " + str(depth) )
    print("width: " + str(width) )
    
    # get adjacent hyperparameters
    params = {"hidden_layer_sizes": adjacent_hidden_layer_sizes(depth, width)}
    
    # define general model
    model = MLPClassifier(solver='lbfgs', # 'adam' for larger data sets
                          alpha=1e-5, 
                          random_state=1,
                          learning_rate='adaptive')
    
    # test model instancces with GridSearch (5-fold cross-validation)
    grid = GridSearchCV(model, 
                        param_grid=params,
                        cv=5,
                        iid=False,
                        n_jobs=2)

    # create and get best results
    grid.fit(x_train_transform, y_train)
    best_depth = len(grid.best_estimator_.hidden_layer_sizes)
    best_width = grid.best_estimator_.hidden_layer_sizes[0]
    
    # IF no better layout found: continue the search via recursion
    if (depth == best_depth) & (width == best_width):
        return grid, best_depth, best_width
    else:
        return optimize_MLPClassifier_layout( x, y, best_depth, best_width )
    
    
# initial width and depth
grid, best_depth, best_width = optimize_MLPClassifier_layout(x_train_transform, 
                                                             y_train, 
                                                             1, 
                                                             1)

print("Best result:")
print("best_depth: " + str(best_depth))
print("best_width: " + str(best_width))

    
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
    