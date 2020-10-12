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
    
    # for now return dataframe ready to split and scale (shape = 71,431, 22)
    return df