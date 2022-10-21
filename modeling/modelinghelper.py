# imports
import pandas as pd
import numpy as np
from datetime import datetime
from category_encoders.target_encoder import TargetEncoder

# sklearn
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


# environment variables
rng = 42


def df_filtering(df, i_e = 'I', f_cols = []):

    filtered_df = df[df.i_e == i_e]
    filtered_df = filtered_df[f_cols+['act']]

    return filtered_df

def data_split(df):
    X, y = df.iloc[:,:-1], df.iloc[:,-1:]
    y = np.where(y['act']=='R',1,0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng, stratify=y)

    return X_train, X_test, y_train, y_test

def upsampler(df_x, df_y): # https://www.analyticsvidhya.com/blog/2021/06/5-techniques-to-handle-imbalanced-data-for-a-classification-problem/
    # combine results
    df_x['act'] = df_y
    # filter into majority / minority
    df_majority = df_x.loc[df_x['act']==0]
    df_minority = df_x.loc[df_x['act']==1]
    # upsample to the size of the majority
    df_minority_upsampled = resample(df_minority, replace=True, n_samples= len(df_majority), random_state=rng)
    # combine and split back into x and y dfs
    df_upsampled = pd.concat([df_minority_upsampled, df_majority])
    new_y = df_upsampled['act']
    new_x = df_upsampled.drop(columns=['act'])
    return new_x, new_y


def gridsearch_pipeline(X_train, y_train, classifer, grid_param): 
    # set
    clf = classifer
    params = grid_param
    categorical_var = ['species_code', 'wildlf_desc', 'wildlf_cat',
                       'ctry_org','ctry_ie','purp', 'src', 
                       'trans_mode', 'pt_cd','ship_date_mm']
    numerical_var = ['value']

    
    ct_target = make_column_transformer((TargetEncoder(), categorical_var),
                                        remainder='passthrough')

    pipe = Pipeline([('targetEncoding', ct_target), 
                     ('standardScaler', StandardScaler()), 
                     ('clf', clf)
                    ], verbose=False)

    grid_pipe = GridSearchCV(pipe,
                             param_grid=params,
                             scoring='recall',
                             cv=5,
                             verbose=3)

    grid_pipe.fit(X_train, y_train)
    print('Grid best parameter (max. recall): ', grid_pipe.best_params_)
    print('Grid best score (recall): ', grid_pipe.best_score_)
    
    return grid_pipe