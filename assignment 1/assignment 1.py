# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:53:37 2019

@author: aaron
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


original_data = pd.read_csv('hw2_data.txt', sep=None, header=0,engine='python')
print(original_data.shape)
data = original_data._get_numeric_data()
data.reindex(np.arange(data.shape[0]))

############################   Statistics
data_mean = data.mean()
data_var = data.var()
kurtosis = data.kurtosis()
skew = data.skew()
print("Mean values of the numerical feature are:")
print(data_mean)
print("Variances of the numerical features are:")
print(data_var)
print("Kurtosis of the numerical features are:")
print(kurtosis)
print("Skewness of the numerical features are:")
print(skew)
print(data.shape)
###########################



####### KNN where K=5#############################################
y = data.iloc[:,0:1] 
X = data.iloc[:,1:14] 

y = y.to_numpy().reshape((-1,1))
X = X.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.3,
                                                    random_state = 1,
                                                    stratify = None)

KNN_classifier = KNeighborsClassifier(n_neighbors = 5,
                                      weights = 'uniform',
                                      algorithm = 'kd_tree',
                                      leaf_size = 30,
                                      p = 2,
                                      metric = 'minkowski',
                                      metric_params = None,
                                      n_jobs = 1)
####Holdout
KNN_classifier.fit(X_train, y_train)

KNN_y_pred = KNN_classifier.predict(X_test)

KNN_y_pred_hit_rate = np.mean(y_test == KNN_y_pred)
KNN_y_pred_rmse = ((((y_test - KNN_y_pred)**2).mean())**0.5)
KNN_y_pred__mae = np.absolute(y_test - KNN_y_pred).mean()

print("Holdout prediction hit rate:", KNN_y_pred_hit_rate)
print("Holdout prediction RMSE:", KNN_y_pred_rmse)
print("Holdout prediction MAE:", KNN_y_pred__mae)

####

#### Repeated Holdout
rng = np.random.RandomState(seed = 12345)
seeds = np.arange(10*15);
rng.shuffle(seeds);
seeds = seeds[:50]

accuracies_hit = []
accuracies_rmse = []
accuracies_mae = []

for i in seeds:
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size = 0.3,
                                                        random_state = i,
                                                        stratify = None)
    KNN_classifier.fit(X_train, y_train)
    KNN_y_pred_i__hit = KNN_classifier.score(X_test, y_test)
    KNN_y_pred2 = KNN_classifier.predict(X_test)
    KNN_y_pred_i__rmse = (((y_test - KNN_y_pred2)**2).mean())**0.5
    KNN_y_pred_i__mae = np.absolute(y_test - KNN_y_pred2).mean()
    
    accuracies_hit.append(KNN_y_pred_i__hit)
    accuracies_rmse.append(KNN_y_pred_i__rmse)
    accuracies_mae.append(KNN_y_pred_i__mae)

accuracies_hit = np.asarray(accuracies_hit)
accuracies_rmse = np.asarray(accuracies_rmse)
accuracies_mae = np.asarray(accuracies_mae)

print("Repeated holdout average hit rate:", accuracies_hit.mean())
print("Repeated holdout average RMSE:", accuracies_rmse.mean())
print("Repeated holdout average MAE:", accuracies_mae.mean())

KNN_classifier.fit(X,y)
print ("Resubstitution optimistic prediction accuracy:", KNN_classifier.score(X,y))    


#############################  Hyperparameter selection (Question 3)

##### 5-fold CV
params = range(1,21)
cv_acc_rmse, cv_acc_mae, cv_std_rmse, cv_std_mae, cv_stderr_rmse, cv_stderr_mae = [], [], [], [], [], []
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.3,
                                                    random_state = 1,
                                                    stratify = None)
#cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1)
cv = KFold(n_splits = 5, shuffle = True, random_state = 1)
for c in params:
    KNN_classifier = KNeighborsClassifier(n_neighbors = c,
                                          weights = 'uniform',
                                          algorithm = 'kd_tree',
                                          leaf_size = 30,
                                          p = 2,
                                          metric = 'minkowski',
                                          metric_params = None,
                                          n_jobs = 1)
    all_acc_rmse, all_acc_mae = [], []
    for train_index, valid_index in cv.split(X_train, y_train):
        pred = KNN_classifier.fit(X_train[train_index], y_train[train_index])\
               .predict(X_train[valid_index])
        acc_rmse = (((y_train[valid_index] - pred)**2).mean())**0.5
        acc_mae =(np.absolute(y_train[valid_index] - pred)).mean()
        
        all_acc_rmse.append(acc_rmse)
        all_acc_mae.append(acc_mae)
        
    all_acc_rmse = np.array(all_acc_rmse)
    all_acc_mae = np.array(all_acc_mae)
    
    y_pred_cv5_rmse_mean = all_acc_rmse.mean()
    y_pred_cv5_rmse_std = all_acc_rmse.std()
    y_pred_cv5_rmse_stderr = y_pred_cv5_rmse_std / np.sqrt(5)
    y_pred_cv5_mae_mean = all_acc_mae.mean()
    y_pred_cv5_mae_std = all_acc_mae.std()
    y_pred_cv5_mae_stderr = y_pred_cv5_mae_std / np.sqrt(5)  
    
    cv_acc_rmse.append(y_pred_cv5_rmse_stderr)
    cv_stderr_rmse.append(y_pred_cv5_rmse_stderr)
    cv_acc_mae.append(y_pred_cv5_mae_stderr)
    cv_stderr_mae.append(y_pred_cv5_mae_stderr)
    
best_k_rmse = np.argmin(cv_acc_rmse)
best_k_mae = np.argmin(cv_acc_mae)

print('The best K chosen according to the minimum RMSE is:', best_k_rmse+1)
print('The best K chosen according to the minimum MAE is:', best_k_mae+1)

KNN_classifier_rmse = KNeighborsClassifier(n_neighbors = params[best_k_rmse],
                                     weights = 'uniform',
                                     algorithm = 'kd_tree',
                                     leaf_size = 30,
                                     p = 2,
                                     metric = 'minkowski',
                                     metric_params = None,
                                     n_jobs = 1)
KNN_classifier_rmse.fit(X_train, y_train)
KNN_rmse_pred_y = KNN_classifier_rmse.fit(X_train, y_train).predict(X_train)
KNN_rmse_pred_rmse = (((KNN_rmse_pred_y-y_train)**2).mean())**0.5
KNN_rmse_pred_mae = np.absolute(KNN_rmse_pred_y-y_train).mean()

print('5-fold prediction RMSE of the best K chose by RMSE is :', KNN_rmse_pred_rmse)
print('5-fold prediction MAE of the best K chose by RMSE is :', KNN_rmse_pred_mae)

KNN_classifier_mae = KNeighborsClassifier(n_neighbors = params[best_k_mae],
                                     weights = 'uniform',
                                     algorithm = 'kd_tree',
                                     leaf_size = 30,
                                     p = 2,
                                     metric = 'minkowski',
                                     metric_params = None,
                                     n_jobs = 1)
KNN_classifier_mae.fit(X_train, y_train)
KNN_mae_pred_y = KNN_classifier_mae.fit(X_train, y_train).predict(X_train)
KNN_mae_pred_rmse = (((KNN_mae_pred_y-y_train)**2).mean())**0.5
KNN_mae_pred_mae = np.absolute(KNN_mae_pred_y-y_train).mean()

print('5-fold prediction RMSE of the best K chose by RMSE is :', KNN_mae_pred_rmse)
print('5-fold prediction MAE of the best K chose by RMSE is :', KNN_mae_pred_mae)
#####################

###### Repeated 5-fold CV
params = range(1,21)
rng = np.random.RandomState(seed = 12345);seeds = np.arange(10**5)
rng.shuffle(seeds); seeds = seeds[:5]
params_by_seed_rmse, params_by_seed_mae = [], []
for seed in seeds:
    #cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    cv = KFold(n_splits = 5, shuffle = True, random_state = seed)
    acc_by_param_rmse, acc_by_param_mae = [], []
    
    for c in params:
        
        KNN_classifier = KNeighborsClassifier(n_neighbors = params[c-1],
                                     weights = 'uniform',
                                     algorithm = 'kd_tree',
                                     leaf_size = 30,
                                     p = 2,
                                     metric = 'minkowski',
                                     metric_params = None,
                                     n_jobs = 1)
        all_acc_rmse = []
        all_acc_mae = []
        
        for train_index, valid_index in cv.split(X_train, y_train):
            
            pred = KNN_classifier.fit(X_train[train_index], y_train[train_index])\
                   .predict(X_train[valid_index])
            acc_rmse =(((pred - y_train[valid_index])**2).mean())**0.5
            acc_mae = (np.absolute(pred - y_train[valid_index])).mean()
            all_acc_rmse.append(acc_rmse)
            all_acc_mae.append(acc_mae)
        
        all_acc_rmse = np.array(all_acc_rmse)
        all_acc_mae = np.array(all_acc_mae)
        
        acc_by_param_rmse.append(all_acc_rmse.mean())
        acc_by_param_mae.append(all_acc_mae.mean())

    params_by_seed_rmse.append(acc_by_param_rmse) 
    params_by_seed_mae.append(acc_by_param_mae)

best_k_rmse = np.argmin(np.mean(params_by_seed_rmse,0))    
best_k_mae = np.argmin(np.mean(params_by_seed_mae,0))    

print('The best K chosen according to the minimum RMSE is:', best_k_rmse+1)
print('The best K chosen according to the minimum MAE is:', best_k_mae+1)

KNN_classifier_rmse = KNeighborsClassifier(n_neighbors = params[best_k_rmse],
                                     weights = 'uniform',
                                     algorithm = 'kd_tree',
                                     leaf_size = 30,
                                     p = 2,
                                     metric = 'minkowski',
                                     metric_params = None,
                                     n_jobs = 1)
KNN_classifier_rmse.fit(X_train, y_train)
KNN_rmse_pred_y = KNN_classifier_rmse.fit(X_train, y_train).predict(X_train)
KNN_rmse_pred_rmse = (((KNN_rmse_pred_y-y_train)**2).mean())**0.5
KNN_rmse_pred_mae = np.absolute(KNN_rmse_pred_y-y_train).mean()

print('Repeated 5-fold prediction RMSE of the best K chose by RMSE is :', KNN_rmse_pred_rmse)
print('Repeated 5-fold prediction MAE of the best K chose by RMSE is :', KNN_rmse_pred_mae)

KNN_classifier_mae = KNeighborsClassifier(n_neighbors = params[best_k_mae],
                                     weights = 'uniform',
                                     algorithm = 'kd_tree',
                                     leaf_size = 30,
                                     p = 2,
                                     metric = 'minkowski',
                                     metric_params = None,
                                     n_jobs = 1)
KNN_classifier_mae.fit(X_train, y_train)
KNN_mae_pred_y = KNN_classifier_mae.fit(X_train, y_train).predict(X_train)
KNN_mae_pred_rmse = (((KNN_mae_pred_y-y_train)**2).mean())**0.5
KNN_mae_pred_mae = np.absolute(KNN_mae_pred_y-y_train).mean()

print('Repeated 5-fold prediction RMSE of the best K chose by RMSE is :', KNN_mae_pred_rmse)
print('Repeated5-fold prediction MAE of the best K chose by RMSE is :', KNN_mae_pred_mae)
############

##############################################################        
        
        
    
    
    
    
        
        





















