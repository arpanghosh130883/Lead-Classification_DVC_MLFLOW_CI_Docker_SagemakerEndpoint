import pandas as pd
import argparse
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
# print(xgboost.sklearn.__version__)
import mlflow
import joblib
# import shutil
# from tqdm import tqdm
import logging
from sklearn.metrics import confusion_matrix
from src.utils.common import read_yaml, create_directories
# import random


STAGE = "STAGE_02_MODEL_TRAINING" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):

    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    ## Read train, val and test data 

    artifacts = config['artifacts']
    train_val_test_data_directory = os.path.join(artifacts['artifacts_directory'], artifacts['train_val_test_data_directory'])

    train_data_path = os.path.join(train_val_test_data_directory, artifacts["train_data_file"])
    train_df = pd.read_csv(train_data_path, sep=",")

    train_y = train_df['IsClientConverted']
    train_x = train_df.drop(['IsClientConverted'],axis=1)

    ## Get the model parameters
    subsample = params['BestModelParameters']['subsample']
    silent = params['BestModelParameters']['silent']
    n_estimators = params['BestModelParameters']['n_estimators']
    max_depth = params['BestModelParameters']['max_depth']
    learning_rate = params['BestModelParameters']['learning_rate']
    gamma = params['BestModelParameters']['gamma']
    colsample_bytree = params['BestModelParameters']['colsample_bytree']
    colsample_bylevel = params['BestModelParameters']['colsample_bylevel']

    mlflow.log_params(params["BestModelParameters"])

    # mlflow.set_tracking_uri("sqlite:///mlflow_db.db")

    """
    mlflow.set_tracking_uri("sqlite:///mlflow_db.db")
    model_params = {"subsample":subsample,"silent":silent, "n_estimators":n_estimators,"max_depth":max_depth, 
                                   "learning_rate":learning_rate,"gamma":gamma,"colsample_bytree":colsample_bytree, 
                                   "colsample_bylevel":colsample_bylevel}
    # with mlflow.start_run():
    model = XGBClassifier(**model_params)
    model.fit(train_x, train_y)
    # mlflow.log_params(model_params)
    mlflow.sklearn.log_model(model, "xgbclassifier")
    
    run1 = mlflow.active_run()
    if run1:
        print("Active run_id: {}".format(run1.info.run_id))
        # mlflow.end_run()
    else:
        print("No active runs")

    """

    
     

    ## Model creation
    model_params = {"subsample":subsample,"silent":silent, "n_estimators":n_estimators,"max_depth":max_depth, \
                                   "learning_rate":learning_rate,"gamma":gamma,"colsample_bytree":colsample_bytree, 
                                   "colsample_bylevel":colsample_bylevel}

    model = XGBClassifier(**model_params)

    


    ## Fit the model on train set

    model.fit(train_x, train_y)

    ## Log model parameters
    mlflow.log_params(model_params)

    ## Save the model
    model_path =  os.path.join(artifacts['artifacts_directory'], artifacts['model_directory'])
    create_directories([model_path])
    model_file = os.path.join(model_path, artifacts['model_name'])
    joblib.dump(model, model_file)

    ## Mlflow logging of model
    mlflow.sklearn.log_model(model, "xgbclassifier")

    run = mlflow.active_run()
    model_uri = "runs:/{}/xgbclassifier-model".format(run.info.run_id)
    mv = mlflow.register_model(model_uri, "xgbclassifiermodel")
    print("Name: {}".format(mv.name))
    print("Version: {}".format(mv.version))

    
    # model_uri = "runs:/{}/xgbclassifier-model".format(run.info.run_id)
    # mv = mlflow.register_model(model_uri, "xgbclassifiermodelv2")
    # print("Name: {}".format(mv.name))
    # print("Version: {}".format(mv.version))
    

    
   

"""  
# mlflow.set_tracking_uri("sqlite:////tmp/mlruns.db")
# params = {"n_estimators": 3, "random_state": 42}

# # Log MLflow entities
# with mlflow.start_run() as run:
#    rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])
#    mlflow.log_params(params)
#    mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")

# model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
# mv = mlflow.register_model(model_uri, "RandomForestRegressionModel")
# print("Name: {}".format(mv.name))
# print("Version: {}".format(mv.version))
"""
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e