import pandas as pd
import argparse
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import mlflow
import joblib
import logging
from src.utils.common import read_yaml, create_directories
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


STAGE = "STAGE_02_MODEL_TRAINING"  ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)


def main(config_path, params_path):

    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    ## Read train, val and test data

    artifacts = config["artifacts"]
    train_val_test_data_directory = os.path.join(
        artifacts["artifacts_directory"], artifacts["train_val_test_data_directory"]
    )

    # train_data_path = os.path.join(train_val_test_data_directory, artifacts["train_data_file"])
    # train_df = pd.read_csv(train_data_path, sep=",")

    x_train_data_path = os.path.join(
        train_val_test_data_directory, artifacts["x_train_data_file"]
    )
    x_train = pd.read_csv(x_train_data_path, sep=",")

    y_train_data_path = os.path.join(
        train_val_test_data_directory, artifacts["y_train_data_file"]
    )
    y_train = pd.read_csv(y_train_data_path, sep=",")

    """
    train_y = train_df['IsClientConverted']
    train_x = train_df.drop(['IsClientConverted'],axis=1)

    ## Apply smote 
    oversample = SMOTE()
    train_x_balanced, train_y_balanced = oversample.fit_resample(train_x, train_y)
    """

    ## Scaling the
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train), index=x_train.index, columns=x_train.columns
    )

    ## Save the scaler
    scaler_path = os.path.join(
        artifacts["artifacts_directory"], artifacts["scaler_directory"]
    )
    create_directories([scaler_path])
    scaler_file = os.path.join(scaler_path, artifacts["scaler_name"])
    joblib.dump(scaler, scaler_file)

    artifacts = config["artifacts"]
    scaled_file = os.path.join(
        artifacts["artifacts_directory"], artifacts["scaled_train_dir"]
    )
    create_directories([scaled_file])
    scaled_train_path = os.path.join(scaled_file, artifacts["xtrain_scaled_file"])
    x_train_scaled.to_csv(scaled_train_path, header=False, index=False)

    ## Get the model parameters
    subsample = params["BestModelParameters"]["subsample"]
    silent = params["BestModelParameters"]["silent"]
    n_estimators = params["BestModelParameters"]["n_estimators"]
    max_depth = params["BestModelParameters"]["max_depth"]
    learning_rate = params["BestModelParameters"]["learning_rate"]
    gamma = params["BestModelParameters"]["gamma"]
    colsample_bytree = params["BestModelParameters"]["colsample_bytree"]
    colsample_bylevel = params["BestModelParameters"]["colsample_bylevel"]

    mlflow.log_params(params["BestModelParameters"])

    ## Model creation
    model_params = {
        "subsample": subsample,
        "silent": silent,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "colsample_bytree": colsample_bytree,
        "colsample_bylevel": colsample_bylevel,
    }

    model = XGBClassifier(**model_params)

    ## Fit the model on train set

    model.fit(x_train_scaled, y_train)

    ## Log model parameters
    mlflow.log_params(model_params)

    ## Save the model
    model_path = os.path.join(
        artifacts["artifacts_directory"], artifacts["model_directory"]
    )
    create_directories([model_path])
    model_file = os.path.join(model_path, artifacts["model_name"])
    joblib.dump(model, model_file)

    ## Mlflow logging of model
    mlflow.sklearn.log_model(model, "xgbclassifier")

    run = mlflow.active_run()
    model_uri = "runs:/{}/xgbclassifier-model".format(run.info.run_id)
    mv = mlflow.register_model(model_uri, "xgbclassifiermodel")
    print("Name: {}".format(mv.name))
    print("Version: {}".format(mv.version))


if __name__ == "__main__":
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
