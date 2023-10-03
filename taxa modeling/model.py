import pandas as pd
import numpy as np
import taxamodelinghelper as helper

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

from adspy_shared_utilities import plot_feature_importances
from matplotlib import pyplot as plt

import joblib
from datetime import datetime

# supress future warnings
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

#set up random seed
rng = 42

def gridsearch_pipelines(genus_prefix, trade_type, X_train, y_train, 
                         X_test, y_test, clf, clf_prefix, params):
    
    # default pipeline
    start = datetime.now()
    pipe = helper.gridsearch_pipeline(X_train, y_train, clf, params) 
    end = datetime.now()
    y_pred = pipe.predict(X_test)
    print(f'recall score on holdout: {recall_score(y_test, y_pred)}')  
    print(f'model training time: {end - start}\n')
    joblib.dump(pipe, f'{genus_prefix}_{trade_type}_{clf_prefix}_pipe.joblib')
    
        
    # imbalanced pipeline with SMOTE upsampling
    start = datetime.now()
    pipe_SMOTE = helper.gridsearch_pipeline(X_train, y_train, clf, params, imbalance_pipe=True) 
    end = datetime.now()
    y_pred = pipe_SMOTE.predict(X_test)
    print(f'recall score on holdout: {recall_score(y_test, y_pred)}')
    print(f'SMOTE model training time: {end - start}')
    joblib.dump(pipe_SMOTE, f'{genus_prefix}_{trade_type}_{clf_prefix}_pipe_SMOTE.joblib')
    

    return pipe, pipe_SMOTE
    
def run_models(df, imp=True, genus_prefix='all'):
    
    prefix = genus_prefix
    
    if imp:
        trade_type = 'I' 
    else:
        trade_type = 'E' 
    
    print(f'>>>>>> Begin model training for {genus_prefix} {trade_type} <<<<<<<<')
    
    # set up feature columns
    feature_cols = ['species_code', 'wildlf_desc', 'wildlf_cat',
                    'ctry_org', 'ctry_ie','purp', 'src', 'trans_mode', 'pt_cd', 
                     'value', 'ship_date_mm']

    print(f'number of records in df: {len(df)}')
    
    # filter by import vs. export
    df = helper.df_filtering(df, i_e = trade_type, f_cols = feature_cols)
    print(f'number of records in df after trade type {trade_type} filtering: {len(df)}')
    n_refused = len(df[df['act']=='R'])
    print(f'number of refused records in df: {n_refused}, {round(n_refused/len(df),2)}%')
    
    # check if there's enough refused samples to train
    if n_refused < 5:
        print('!! not enough refused samples to train !!')
    else:
        X_train, X_test, y_train, y_test = helper.data_split(df)

        print(f'\n-------- training LogRegression classifer ----------')
        clf = LogisticRegression(random_state=rng)
        lr_params={'clf__class_weight':[None,'balanced' 
                                         #{0:100, 1:1}, 
                                         #{0:50, 1:1}
                                        ],
                    'clf__solver': ['lbfgs', 'liblinear'],# 'sag', 'saga'],
                    'clf__C': [0.0000001, 0.000001, 0.00001, 
                               0.0001, 0.001, 0.1, #0.5, 
                               1]
                }
        
        gridsearch_pipelines(prefix, trade_type, 
                             X_train, y_train, X_test, y_test, 
                             clf, 'lr', lr_params)

        print(f'\n-------- training DecisionTree classifer ----------')
        clf = DecisionTreeClassifier(random_state=rng)
        dt_params={'clf__class_weight':[None,'balanced'], 
                                         #{0:100, 1:1}, 
                                         #{0:50, 1:1}],
                    'clf__max_depth': [2,3,4,5,6,7,8,9,10, None]
                  }
        
        gridsearch_pipelines(prefix, trade_type, 
                             X_train, y_train, X_test, y_test, 
                             clf, 'dt', dt_params)
   

        print(f'\n-------- training RandomForest classifer ----------')
        clf = RandomForestClassifier(random_state=rng)
        rf_params={'clf__class_weight':['balanced', None],
                    'clf__n_estimators':[4,6,8,10, 20, 30, 50, 100],
                    'clf__max_depth': [None, 3, 4, 5,7,8,9, 10]
                  }
        
        gridsearch_pipelines(prefix, trade_type, 
                             X_train, y_train, X_test, y_test,
                             clf, 'rf', rf_params)

        
        print(f'\n-------- training GradientBoostingClassifier ----------')
        clf = GradientBoostingClassifier(random_state=rng)
        gbc_params={'clf__learning_rate':[0.01, 0.1, 1, 10],
                    'clf__loss':['log_loss'],#, 'exponential'],
                    #'clf__criterion': ['friedman_mse', 'squared_error'],
                    'clf__n_estimators':[4,10, 50, 100],
                    'clf__max_depth': [None, 4, 6, 8, 10],
                    }

        gridsearch_pipelines(prefix, trade_type, 
                                     X_train, y_train, X_test, y_test,
                                     clf, 'gbc', gbc_params)
        
        print(f'\n-------- training XGBoostingClassifier ----------')
        clf = XGBClassifier(random_state=rng, use_label_encoder=False)
        xgb_params={'clf__eta':[0.001, 0.1, 0.3],
                    'clf__gamma': [0, 0.1, 1, 10],
                    'clf__n_estimators':[4, 10, 50, 100],
                    'clf__max_depth': [4, 6, 8, 10],
                    'clf__eval_metric':['logloss','auc']#,'error']
                    }

        # xgb_params={'clf__eta':[0.001,0.01, 0.1, 0.2, 0.3],
        #             'clf__gamma': [0, 0.01, 0.1, 0.5, 1],
        #             'clf__n_estimators':[4, 8, 10, 30, 50, 100],
        #             'clf__max_depth': [4, 6, 8, 10],
        #             'clf__eval_metric':['logloss','auc','error']
        #             }

        gridsearch_pipelines(prefix, trade_type, 
                                     X_train, y_train, X_test, y_test,
                                     clf, 'xgb', xgb_params)
        
        print('\n')

