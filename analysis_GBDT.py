# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 17:57:31 2022

@author: z.rao
"""


import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import  mean_squared_error
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
import numpy as np

#%% read the data
all_data = pd.read_excel('all_data_Ms.xlsx').drop('Unnamed: 0',axis=1)
all_data = all_data.iloc[:,0:25]

x_all = all_data.iloc[:,0:23]
y_all = all_data['Ms']
# y_all = all_data['Hc']
# x_all =x_all[y_all.isnull()==False]
# y_all =y_all[y_all.isnull()==False]

def train_model(num_leaves,
                min_child_samples,
            learning_rate,
            n_estimators, 
            max_bin,
            colsample_bytree, 
            subsample, 
            max_depth, 
            reg_alpha,
            reg_lambda,
            min_split_gain,
            min_child_weight
            ):
    params = {
        "num_leaves": int(round(num_leaves)),
        'min_child_samples':int(round(min_child_samples)),
        'learning_rate': learning_rate,
        'n_estimators': int(round(n_estimators)),
        'max_bin': int(round(max_bin)),
        'colsample_bytree': max(min(colsample_bytree, 1), 0),
        'subsample': max(min(subsample, 1), 0),
        'max_depth': int(round(max_depth)),
        'reg_alpha':  max(reg_alpha, 0),
        'reg_lambda': max(reg_lambda, 0),
        'min_split_gain': min_split_gain,
        'min_child_weight': min_child_weight,
        'verbose': -1
                  }
    model = LGBMRegressor(metric='l2',**params)
# model = LGBMRegressor()
    model.fit(training_data, training_y)
    y_pred = model.predict(validation_data)
    error = -np.mean(np.abs(validation_y - y_pred))       # print(error)     
    return error

bounds = {'num_leaves': (5, 60),#50
          'min_child_samples':(1, 50),
          'learning_rate': (0.001, 1),
          'n_estimators': (5, 200),#100
            'max_bin': (5, 100),#10
          'colsample_bytree': (0.5, 1),
          'subsample': (0.1, 2),
          'max_depth': (1, 60),#10
          'reg_alpha': (0.01, 1), #5
          'reg_lambda': (0.01, 1),#5
          'min_split_gain': (0.001, 0.1),
          'min_child_weight': (0.0001, 30)}

table = pd.DataFrame(columns=['target', 'colsample_bytree', 'learning_rate', 'max_bin',
                      'max_depth','min_child_samples','min_child_weight','min_split_gain',
                      'n_estimators','num_leaves','reg_alpha','reg_lambda','subsample'])


#%%train the GBDT model
    
kf = KFold(n_splits = 5) 
fold_var = 1 #start value of fivefold cross-validation
for train_index, val_index in kf.split(y_all):
    plt.close('all')
    print("fold_var=", fold_var)
    print(train_index)
    print(val_index)
    training_data = x_all.iloc[train_index,:]
    validation_data = x_all.iloc[val_index,:]
    training_y=y_all.iloc[train_index]
    validation_y=y_all.iloc[val_index]
    batch_size=32 
    optimizer = BayesianOptimization(f=train_model,pbounds=bounds,random_state=1)
    optimizer.maximize(init_points = 100, n_iter=100)
    print(optimizer.max)
    num_leaves = optimizer.max['params']['num_leaves']
    min_child_samples = optimizer.max['params']['min_child_samples']
    learning_rate = optimizer.max['params']['learning_rate']
    n_estimators = optimizer.max['params']['n_estimators']
    max_bin = optimizer.max['params']['max_bin']
    colsample_bytree = optimizer.max['params']['colsample_bytree']
    subsample = optimizer.max['params']['subsample']
    max_depth= optimizer.max['params']['max_depth']
    reg_alpha = optimizer.max['params']['reg_alpha']
    reg_lambda = optimizer.max['params']['reg_lambda']
    min_split_gain = optimizer.max['params']['min_split_gain']
    min_child_weight = optimizer.max['params']['min_child_weight']
    params = {
    "num_leaves": int(round(num_leaves)),
    'min_child_samples':int(round(min_child_samples)),
    'learning_rate': learning_rate,
    'n_estimators': int(round(n_estimators)),
    'max_bin': int(round(max_bin)),
    'colsample_bytree': max(min(colsample_bytree, 1), 0),
    'subsample': max(min(subsample, 1), 0),
    'max_depth': int(round(max_depth)),
    'reg_alpha':  max(reg_alpha, 0),
    'reg_lambda': max(reg_lambda, 0),
    'min_split_gain': min_split_gain,
    'min_child_weight': min_child_weight,
    'verbose': -1
              }
    model = LGBMRegressor(metric='l2',**params)
    # model = LGBMRegressor()
    model.fit(training_data,  training_y, eval_set=[(validation_data, validation_y), (training_data, training_y)], 
          eval_metric=None,)
    lgb.plot_metric(model)
    plt.ylabel('loss')
    plt.xlabel ('epoch')
    plt.legend(['train','test'])
    plt.figtext(0.4, 0.75, 'Loss=%.4f' % model.evals_result_['valid_0']['l2'][-1], fontdict={'size': 15, 'color':  'black'})
    plt.savefig('Figures/GBDT_training_history_{}.png'.format(fold_var), format='png', dpi=300)
    # lgb.record_evaluation(eval_result)
    prediction = model.predict(x_all)
    plt.figure()
    sns.regplot(x=prediction, y=y_all, color='g') 
    plt.xlabel('predicted')
    plt.ylabel('experimental')
    fig_name = 'Figures/GBDT_predict_experiments_cv_Ms_{}.png'.format(fold_var)
    plt.savefig(fig_name, format='png', dpi=300)
    fold_var+=1
    table=table.append(pd.DataFrame({'target':[optimizer.max['target']],'colsample_bytree':[optimizer.max['params']['colsample_bytree']],
                                 'colsample_bytree':[optimizer.max['params']['colsample_bytree']],
                                 'learning_rate':[optimizer.max['params']['learning_rate']],
                                 'max_bin':[optimizer.max['params']['max_bin']],
                                 'max_depth':[optimizer.max['params']['max_depth']],
                                 'min_child_samples':[optimizer.max['params']['min_child_samples']],
                                 'min_child_weight':[optimizer.max['params']['min_child_weight']],
                                 'min_split_gain':[optimizer.max['params']['min_split_gain']],
                                 'n_estimators':[optimizer.max['params']['n_estimators']],
                                 'num_leaves':[optimizer.max['params']['num_leaves']],
                                 'reg_alpha':[optimizer.max['params']['reg_alpha']],
                                 'reg_lambda':[optimizer.max['params']['reg_lambda']],
                                 'subsample':[optimizer.max['params']['subsample']]}),
                                 ignore_index=True)
table.to_csv('Figures/GBDT_predict_Ms_BO_results.csv')



