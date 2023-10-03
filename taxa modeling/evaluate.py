import pandas as pd
import numpy as np
import taxamodelinghelper as helper

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, average_precision_score, precision_recall_fscore_support

from adspy_shared_utilities import plot_feature_importances
from matplotlib import pyplot as plt

import joblib
from datetime import datetime

# supress future warnings
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

#set up random seed
rng = 42


def subgroup_eval(df, trade_type, pipe, subgroup_name):
    feature_cols = ['species_code', 'wildlf_desc', 'wildlf_cat',
                        'ctry_org', 'ctry_ie','purp', 'src', 'trans_mode', 'pt_cd', 
                         'value', 'ship_date_mm']


    # filter by import vs. export
    df = helper.df_filtering(df, i_e = trade_type, f_cols = feature_cols)
    X_train, X_test, y_train, y_test = helper.data_split(df)
    
    y_predicted = pipe.predict(X_test)
    print(f'{subgroup_name} recall score on {trade_type} holdout: {recall_score(y_test, y_predicted)}')
    
    confusion = confusion_matrix(y_test, y_predicted, labels = [1,0])
    cm_display = ConfusionMatrixDisplay(confusion, display_labels=[1,0])
    cm_display.plot()
    
    
def subgroup_eval_all(df, trade_type, pipe,subgroup_name):
    feature_cols = ['species_code', 'wildlf_desc', 'wildlf_cat',
                        'ctry_org', 'ctry_ie','purp', 'src', 'trans_mode', 'pt_cd', 
                         'value', 'ship_date_mm']


    # filter by import vs. export
    df = helper.df_filtering(df, i_e = trade_type, f_cols = feature_cols)
    X_train, X_test, y_train, y_test = helper.data_split(df)
    
    y_predicted = pipe.predict(X_test)
    print(f'{subgroup_name} recall score on {trade_type} holdout: {recall_score(y_test, y_predicted)}')
    
    y_scores = pipe.predict_proba(X_test)[:, 1]
    
    scores = []
    # adding additional metrics
    prec, recall, fscore, _ = precision_recall_fscore_support(y_test, y_predicted, average='binary')
    
    scores.extend([recall, fscore, prec])
    scores.append(roc_auc_score(y_test, y_scores, average='weighted'))
    scores.append(average_precision_score(y_test, y_scores))
    
    return scores
    


def plot_tree_importance(genus_prefix, trade_type, pipe, clf_name, clf_prefix):
    plt.figure(figsize=(5,4), dpi=80)
    
    feature_cols = ['species_code', 'wildlf_desc', 'wildlf_cat',
                        'ctry_org', 'ctry_ie','purp', 'src', 'trans_mode', 'pt_cd', 
                         'value', 'ship_date_mm']
    
    if trade_type == 'I':
        trade = 'Import'
    else:
        trade = 'Export'
    
    plot_feature_importances(pipe.best_estimator_.named_steps['clf'], feature_cols)
    plt.title(f'{genus_prefix} {trade_type} {clf_name} Feature Importance')
    plt.xlim(0, 1)
    plt.show()
    # if outputs:
    #     #print('Feature importances: {}'.format(pipe.best_estimator_.named_steps['clf'].feature_importances_[::-1]))
    #     #plt.savefig(f'{genus_prefix}_{clf_prefix}_feature_importance.png')
    #     plt.show()

def plot_LogReg_feature_importance(pipe):
    feature_cols = ['species_code', 'wildlf_desc', 'wildlf_cat',
                            'ctry_org', 'ctry_ie','purp', 'src', 'trans_mode', 'pt_cd', 
                             'value', 'ship_date_mm']

    imp_df = pd.DataFrame(feature_cols, columns=['feature'])
    imp_df['importance'] = pipe.best_estimator_.named_steps['clf'].coef_[0]
    #imp_df = imp_df.sort_values(by=['importance'])
    imp_df.plot.barh(x='feature', y='importance')