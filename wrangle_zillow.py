import pandas as pd
import numpy as np
import scipy as sp 
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
import acquire
import prepare
import summarize

############# Clustering Exercises ###########

def wrangle_zillow_cluster():
    '''
    Get zillow data and prepare data, return df ready for split and scale with all nulls managed
    '''
    # get data with acquire file
    df = acquire.get_zillow_cluster_data()
    
    # drop known duplicate columns and those with proportion of null values above threshold with prepare file
    df = prepare.data_prep(df, cols_to_remove=['id', 'id.1', 'pid', 'tdate'], prop_required_column=.6, prop_required_row=.75)
    # drop additional columns with nulls that are duplicate or have too many remaining nulls to use
    cols_to_remove2 = ['heatingorsystemtypeid', 'buildingqualitytypeid', 'finishedsquarefeet12',  
                    'lotsizesquarefeet', 'propertyzoningdesc', 'regionidcity', 'structuretaxvaluedollarcnt', 
                    'censustractandblock', 'heatingorsystemdesc', 'calculatedbathnbr']
    df = prepare.remove_columns(df, cols_to_remove2)
    # fill null values in unitcount with 1 for single unit
    df.unitcnt = df.unitcnt.fillna(value=1)
    # drop remaining 190 (out of 71,600) with null vaules
    df = df.dropna()
    
    # print summary info
    df = summarize.df_summary(df)
    
    # remove outliers above 50th percentile of upperbound and drop
    df = prepare.add_upper_outlier_columns(df, k=1.5)
    zup_drop_index = df[df.taxamount_up_outliers > 5365].index
    df.drop(zup_drop_index, inplace=True)
    # remove outliers above 75th percentile of lowerbound and drop
    df = prepare.add_lower_outlier_columns(df, k=1.5)
    zlow_drop_index = df[df.taxamount_low_outliers > 9695].index
    df.drop(zlow_drop_index, inplace=True)
    #(new shape = 51,735, 90)
    
    # drop rows not needed for explore or modeling
    cols_to_remove3 = [col for col in df if col.endswith('_outliers')]
    df = prepare.remove_columns(df, cols_to_remove3)
    cols_to_remove4 = ['null_count', 'pct_null']
    df = prepare.remove_columns(df, cols_to_remove4)
    # (new shape = 51,735, 22)

    # df is now ready to split and scale
    return df