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

# import shutil
# from tqdm import tqdm
import logging
from sklearn.metrics import confusion_matrix
from src.utils.common import read_yaml, create_directories

# import random
from sklearn_evaluation import SQLiteTracker


STAGE = "STAGE_03_MODEL_EVALUATION"  ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)


def evaluate(actual, predicted):
    (tn, fp), (fn, tp) = metrics.confusion_matrix(actual, predicted)
    (tnr, fpr), (fnr, tpr) = metrics.confusion_matrix(
        actual, predicted, normalize="true"
    )
    precision_score = metrics.precision_score(actual, predicted)
    recall_score = metrics.recall_score(actual, predicted)
    return tnr, fpr, fnr, tpr, precision_score, recall_score


def main(config_path, params_path):

    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    ## Read train, val and test data
    artifacts = config["artifacts"]
    train_val_test_data_directory = os.path.join(
        artifacts["artifacts_directory"], artifacts["train_val_test_data_directory"]
    )

    x_val_data_path = os.path.join(
        train_val_test_data_directory, artifacts["x_val_data_file"]
    )
    x_val = pd.read_csv(x_val_data_path, sep=",")

    y_val_data_path = os.path.join(
        train_val_test_data_directory, artifacts["y_val_data_file"]
    )
    y_val = pd.read_csv(y_val_data_path, sep=",")

    x_test_data_path = os.path.join(
        train_val_test_data_directory, artifacts["x_test_data_file"]
    )
    x_test = pd.read_csv(x_test_data_path, sep=",")

    y_test_data_path = os.path.join(
        train_val_test_data_directory, artifacts["y_test_data_file"]
    )
    y_test = pd.read_csv(y_test_data_path, sep=",")

    threshold = params["BestModelParameters"]["threshold"]

    scaler_path = os.path.join(
        artifacts["artifacts_directory"], artifacts["scaler_directory"]
    )
    scaler_file = os.path.join(scaler_path, artifacts["scaler_name"])
    scaler = joblib.load(scaler_file)

    model_path = os.path.join(
        artifacts["artifacts_directory"], artifacts["model_directory"]
    )
    model_file = os.path.join(model_path, artifacts["model_name"])
    model = joblib.load(model_file)

    # prediction_val  = model.predict(val_x)
    #Y_val_predicted_proba = model.predict_proba(scaler.transform(x_val))
    #prediction_val = (Y_val_predicted_proba[:, 1] >= threshold).astype("int")
    #(
        tnr_val,
        fpr_val,
        fnr_val,
        tpr_val,
        precision_score_val,
        recall_score_val,
    #) = evaluate(y_val, prediction_val)

    mlflow.log_metric("Val_TNR", tnr_val)
    mlflow.log_metric("Val_FPR", fpr_val)
    mlflow.log_metric("Val_FNR", fnr_val)
    mlflow.log_metric("Val_TPR", tpr_val)
    mlflow.log_metric("Val_precision_score", precision_score_val)
    mlflow.log_metric("Val_recall_score", recall_score_val)

    Y_test_predicted_proba = model.predict_proba(scaler.transform(x_test))
    prediction_test = (Y_test_predicted_proba[:, 1] >= threshold).astype("int")
    (
        tnr_test,
        fpr_test,
        fnr_test,
        tpr_test,
        precision_score_test,
        recall_score_test,
    ) = evaluate(y_test, prediction_test)

    mlflow.log_metric("Test_TNR", tnr_test)
    mlflow.log_metric("Test_FPR", fpr_test)
    mlflow.log_metric("Test_FNR", fnr_test)
    mlflow.log_metric("Test_TPR", tpr_test)
    mlflow.log_metric("Test_precision_score", precision_score_test)
    mlflow.log_metric("Test_recall_score", recall_score_test)

    print("Test_TPR", tpr_test)
    print("Test_precision_score", precision_score_test)


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
