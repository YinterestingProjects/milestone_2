import pandas as pd
import datetime
import numpy as np

def process_columns(df):
    # combining specific and generic name columns...
    df['specific_generic_name'] = df['specific_name'].fillna('') + ' ' + df['generic_name'].fillna('')
    df['specific_generic_name'] = df['specific_generic_name'].str.strip()
    df[['control_number', 'generic_name']].to_csv('../data/control_generic_key')
    
    # dropping extra columns
    drop_cols = ['Unnamed: 0', 'subspecies', 'specific_name', 'generic_name']
    df = df.drop(columns=drop_cols)
    print(f'Dropped cols: {str(drop_cols)}')
    
    return df
    
def process_duplicated_rows(df, method='drop'):
    duplicated_df = df[df.duplicated(keep='first')]
    print(f'Number of duplicated rows {duplicated_df.shape[0]:,} out of {df.shape[0]:,} total rows')
    if method == 'drop':
        df = df.drop_duplicates(ignore_index=True)
        print(f'Rows remaining {df.shape[0]:,}')
    else:
        pass
    
    return df


def process_dates(df):    
    df['ship_date'] = df['ship_date'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d'))
    df['ship_date_yyyy'] = (pd.to_datetime(df['ship_date']).dt.year).map(str)
    df['ship_date_mm'] = (pd.to_datetime(df['ship_date']).dt.month).map(str)
    
    mm_dict = {'1':'Jan','2':'Feb','3':'Mar','4':'Apr','5':'May','6':'Jun',
               '7':'Jul','8':'Aug','9':'Sep','10':'Oct','11':'Nov','12':'Dec'}
    
    for k, v in mm_dict.items():
        if df['ship_date_mm'].str.contains(k).any():
            index_nums = df[df['ship_date_mm']==k].index
            for index_num in index_nums:
                df.at[index_num,'ship_date_mm']=v
                
    yyyy_dict = {'2016':'yr_2016','2017':'yr_2017','2018':'yr_2018','2019':'yr_2019','2020':'yr_2020'}
    
    for k, v in yyyy_dict.items():
        if df['ship_date_yyyy'].str.contains(k).any():
            index_nums = df[df['ship_date_yyyy']==k].index
            for index_num in index_nums:
                df.at[index_num,'ship_date_yyyy']=v
    
    return df

def process_units(df):
    # clean up weird units
    unit_dict = {'N0':'NO','N':'NO','EA':'NO','MU':'NO','10':'NO','O':'NO','PC':'NO','ID':'NO','MO':'NO','PR':'NO',
                 '1':'NO','2':'NO','GA':'GM','22':'GM','24':'GM','GL':'KG'}
    for k, v in unit_dict.items():
        if df['unit'].str.contains(k).any():
            index_nums = df[df['unit']==k].index
            for index_num in index_nums:
                df.at[index_num,'unit']=v            
            
    columns = ['unit_NO','unit_KG','unit_LT','unit_MT','unit_M2']
    for column in columns:
        df[column] = False
    
    # unify units 
    unify_dict = {'NO':'NO','KG':'KG','GM':'KG','LB':'KG','MG':'KG','ML':'LT','LT':'LT','M3':'LT','CM3':'LT',
                  'MT':'MT','CM':'MT','M2':'M2','C2':'M2'}
    for k, v in unify_dict.items():
        if df['unit'].str.contains(k).any():
            index_nums = df[df['unit']==k].index
            for index_num in index_nums:
                df.at[index_num,'unit_'+v]=True
    
    # generate a column of np.NaN
    df['qty_new'] = np.NaN
    
    # recalculate units 
    ### NO = Number of Specimens
    df['qty_new'] = np.where((df['unit']=='NO'), df['qty'], np.NaN)
    df_no = df[df['unit']=='NO']
    ### KG, GM, LB, MG = Weight 
    df['qty_new'] = np.where((df['unit']=='KG'), df['qty'], np.NaN)
    df_kg = df[df['unit']=='KG']
    df['qty_new'] = np.where((df['unit']=='GM'), df['qty']/1000, np.NaN)
    df_gm = df[df['unit']=='GM']
    df['qty_new'] = np.where((df['unit']=='LB'), df['qty']/2.20462, np.NaN)
    df_lb = df[df['unit']=='LB']
    df['qty_new'] = np.where((df['unit']=='MG'), df['qty']/1000000, np.NaN)
    df_mg = df[df['unit']=='MG']
    ### ML, LT, M3, C3 = Volume
    df['qty_new'] = np.where((df['unit']=='LT'), df['qty'], np.NaN)
    df_lt = df[df['unit']=='LT']
    df['qty_new'] = np.where((df['unit']=='ML'), df['qty']/1000, np.NaN)
    df_ml = df[df['unit']=='ML']
    df['qty_new'] = np.where((df['unit']=='C3'), df['qty']/1000, np.NaN)
    df_c3 = df[df['unit']=='C3']
    df['qty_new'] = np.where((df['unit']=='M3'), df['qty']/0.001, np.NaN)
    df_m3 = df[df['unit']=='M3']
    ### M2, C2 = Area
    df['qty_new'] = np.where((df['unit']=='C2'), df['qty']/10000, np.NaN)
    df_c2 = df[df['unit']=='C2']
    df['qty_new'] = np.where((df['unit']=='M2'), df['qty'], np.NaN)
    df_m2 = df[df['unit']=='M2']
    ### MT, CM = Length
    df['qty_new'] = np.where((df['unit']=='CM'), df['qty']/100, np.NaN)
    df_cm = df[df['unit']=='CM']
    df['qty_new'] = np.where((df['unit']=='MT'), df['qty'], np.NaN)
    df_mt = df[df['unit']=='MT']
    
    # combine the datasets
    merged_df = pd.concat([df_no, df_kg, df_gm, df_lb, df_mg, df_lt, df_ml, df_c3, df_m3, df_c2, df_m2, df_cm, df_mt])
    return merged_df

def add_gdp_values(df, imf, ctry):
    df['ctry_org'] = df['ctry_org'].str.upper()
    df['ctry_ie'] = df['ctry_ie'].str.upper()
    
    df_ctry = df.merge(ctry, how='left', left_on='ctry_ie', right_on='code')
    df_ctry = df_ctry.rename(columns={'value_y':'ctry_ie_name'})
    
    ctry_dict = {'G5':'United Kingdom','G2':'United Kingdom','G4':'United Kingdom','G1':'United Kingdom',
                 'G6':'United Kingdom','G3':'United Kingdom','E1':'Germany','GB':'United Kingdom'}
    
    for k, v in ctry_dict.items():
        if df_ctry['ctry_ie'].str.contains(k).any():
            index_nums = df_ctry[df_ctry['ctry_ie']==k].index
            for index_num in index_nums:
                df_ctry.at[index_num,'ctry_ie_name']=v
    
    df_ctry_imf = df_ctry.merge(imf, how='left', left_on='ctry_ie_name', right_on='Country')
    df_ctry_imf = df_ctry_imf.rename(columns={'2020':'2020GDP'}).drop(columns=['field','code','Country'])
    
    return df_ctry_imf

def cleanup_nulls(df):
    for name in df.columns:
        df.loc[df[name].isnull(), name] = "NaN_"+name
    return df