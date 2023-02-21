import pandas as pd
import sys
import os

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import joblib

from src.dataloader import DataReader
import config as conf


def DataPrep() -> None:
    
    os.environ['AWS_ACCESS_KEY_ID'] = conf.conn["data.aws.aws_access_key_id"]
    os.environ['AWS_SECRET_ACCESS_KEY'] = conf.conn["data.aws.aws_secret_access_key"]

    print('dataprep started')

    # read data --------------------------------------------------------------------------------------------------------
    
    #pairs = DataReader(conf, conf.data['pairs'], conf.params['main_path'])
    df = DataReader(conf, conf.data['training_data'], conf.params['main_path'])

    # data prep --------------------------------------------------------------------------------------------------------
    # Divide training and testing columns
    
    X = df.drop(conf.data["target_column"],axis=1)
    y = df[conf.data["target_column"]]
    print("Shape of X: ", X.shape)
    print("Shape of y: ", y.shape)
    

    # Split data into training and test set
    ##------------------------------------------------------------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(conf.data["test_size"]), random_state=42)
   
    print("X_train Shape: ",X_train.shape)
    print("X_test Shape: ",X_test.shape)
    print("y_train Shape: ",y_train.shape)
    print("y_test Shape: ",y_test.shape)

    # Scaling data
    ##------------------------------------------------------------------------------------------------------------------
    
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # create and fit model
    ##------------------------------------------------------------------------------------------------------------------
    
    lgbm = LGBMClassifier()
    
    lgbm.fit(X_train_scaled,y_train)
    

    # Model Parameters
    ##------------------------------------------------------------------------------------------------------------------
    
    pd.DataFrame(lgbm.get_params()).to_csv(conf.params['model_params_path'],index=False)
    
    
    # Saving Model
    ##------------------------------------------------------------------------------------------------------------------
    
    #joblib.dump(X_test_scaled,conf.params['main_path'] +'X_test.pkl')
    joblib.dump(y_test,conf.params['main_path'] +'y_test.pkl')
    
    if conf.params['save_the_model']:
        joblib.dump(lgbm,conf.params['main_path'] +"/model_saving/"+'LightGBM_model.pkl')
        print("Model saved")
        
    else:
        pass

    print('dataprep completed')




