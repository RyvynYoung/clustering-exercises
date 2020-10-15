import pandas as pd
import numpy as np
import scipy as sp 
import os
import sklearn.preprocessing
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
import acquire
import prepare

def add_scaled_columns(train, validate, test, scaler, columns_to_scale):
    """This function scales the iris data"""
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler.fit(train[columns_to_scale])

    train_scaled = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index)], axis=1)
    validate_scaled = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index)], axis=1)
    test_scaled = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index)], axis=1)
    
    return train_scaled, validate_scaled, test_scaled

def scale_iris(train, validate, test):
    """This function provides the inputs and runs the add_scaled_columns function"""
    train_scaled, validate_scaled, test_scaled = add_scaled_columns(
    train,
    validate,
    test,
    scaler=sklearn.preprocessing.MinMaxScaler(),
    columns_to_scale=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    )
    # drop rows not needed for modeling
    cols_to_remove = ['species', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    train_scaled = train_scaled.drop(columns=cols_to_remove)
    validate_scaled = validate_scaled.drop(columns=cols_to_remove)
    test_scaled = test_scaled.drop(columns=cols_to_remove)
    return train_scaled, validate_scaled, test_scaled


def wrangle_iris_data():
    """
    This function takes acquired iris data, completes the prep
    and splits the data into train, validate, and test datasets
    """
    df = acquire.get_iris_data()
    train, test, validate = prepare.prep_iris_data(df)
    #train_and_validate, test = train_test_split(df, test_size=.15, random_state=123)
    #train, validate = train_test_split(train_and_validate, test_size=.15, random_state=123)
    # return train, test, validate
    train_scaled, validate_scaled, test_scaled = scale_iris(train, validate, test)
    return train, validate, test, train_scaled, validate_scaled, test_scaled


####### NOTE: to call wrangle_iris_data 
##### train, validate, test, train_scaled, validate_scaled, test_scaled = wrangle_iris_data()