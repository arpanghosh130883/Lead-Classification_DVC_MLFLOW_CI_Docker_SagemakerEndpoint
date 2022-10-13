import pytest
import pandas as pd
import argparse
import os
import mlflow
import logging
from src.utils.common import read_yaml
import dvc.api
import joblib
from xgboost.sklearn import XGBClassifier
import xgboost
import sklearn
from sklearn.metrics import confusion_matrix


class CustomError(Exception):
    pass


## read config files

config_path = "configs/config.yaml"
params_path = "params.yaml"

config = read_yaml(config_path)
params = read_yaml(params_path)

artifacts = config["artifacts"]
output_file = os.path.join(
    artifacts["artifacts_directory"], artifacts["processed_data_dir"]
)
train_val_test_data_directory = os.path.join(
    artifacts["artifacts_directory"], artifacts["train_val_test_data_directory"]
)


source_data = config["source_data"]

path = os.path.join(source_data["path"])
repo = os.path.join(source_data["repo"])
version = os.path.join(source_data["version"])

data_url = dvc.api.get_url(path=path, repo=repo, rev=version)

input_df = pd.read_csv(data_url, sep=",")


def test_data_url():

    try:
        assert os.path.exists(data_url)

    except:
        raise CustomError("Data url does not exist")


def test_processed_file_is_saved():

    prepared_data_path = os.path.join(output_file, artifacts["processed_data_file"])
    try:
        assert os.path.exists(prepared_data_path) == True
        assert os.path.getsize(prepared_data_path) > 0

    except:
        raise CustomError("Processed data file is not saved")


def test_input_features_are_saved():

    input_features_file = os.path.join(
        artifacts["artifacts_directory"], artifacts["input_features_dir"]
    )
    input_features_path = os.path.join(
        input_features_file, artifacts["input_features_file"]
    )

    try:
        assert os.path.exists(input_features_path) == True
        assert os.path.getsize(input_features_path) > 0

    except:
        raise CustomError("Input features are not saved")


def test_overseas_countries_feature():

    try:
        assert "overseas_countries" in input_df.columns

    except:
        raise CustomError("overseas_countries column does not exist")


def test_acc_status_feature():

    try:
        assert "acc_status" in input_df.columns

    except:
        raise CustomError("acc_status column does not exist")


def test_num_of_input_features():

    x_train_data_path = os.path.join(
        train_val_test_data_directory, artifacts["x_train_data_file"]
    )
    x_train = pd.read_csv(x_train_data_path, sep=",")
    features = x_train.columns

    try:
        assert len(features) == 26

    except:
        raise CustomError("Number of input columns is not 26")


def test_input_feature_names():

    try:

        prepared_data_path = os.path.join(output_file, artifacts["processed_data_file"])
        df_req = pd.read_csv(prepared_data_path, sep=",")

        assert df_req.columns.all() in [
            "Org",
            "OwnershipType",
            "OperatingStatus",
            "SalesTurnoverGBP",
            "ProfitOrLossAmount",
            "ImportTrue",
            "ExportTrue",
            "LineOfBusinessDescription",
            "EmpCount",
            "MinorityOwnedIndicator",
            "FamilyTreeHierarchyLevel",
            "GlobalUltimateFamilyTreeLinkageCount",
            "FamilyTreeMemberRoleText",
            "ForeignIndicator",
            "TotalAssetsAmount",
            "PrimaryTownName",
            "NetWorth",
            "IsClientConverted",
            "SalesTurnoverAvailable",
            "ProfitOrLossAmountAvailable",
            "EmpCountAvailable",
            "FamilyTreeHierarchyLevelAvailable",
            "GlobalUltimateFamilyTreeLinkageCountAvailable",
            "TotalAssetsAmountAvailable",
            "NetworthAvailable",
            "OverseasCountriesKnown",
            "StandaloneOrgTrue",
        ]

    except:
        raise CustomError("Input feature names do not match")


def test_shape_xtrain_ytrain():

    x_train_data_path = os.path.join(
        train_val_test_data_directory, artifacts["x_train_data_file"]
    )
    x_train = pd.read_csv(x_train_data_path, sep=",")

    y_train_data_path = os.path.join(
        train_val_test_data_directory, artifacts["y_train_data_file"]
    )
    y_train = pd.read_csv(y_train_data_path, sep=",")
    try:
        assert x_train.shape[0] == y_train.shape[0]
    except:
        raise CustomError("Shape of x_train is not equal to y_train")


def test_null_values():

    prepared_data_path = os.path.join(output_file, artifacts["processed_data_file"])
    df_req = pd.read_csv(prepared_data_path, sep=",")

    try:
        assert df_req.isnull().sum().all() == 0
    except:
        raise CustomError("There are null values in the processed data set")


def test_val_test_split_ratio():

    test_split = params["ModelTraining"]["test_split"]
    val_split = params["ModelTraining"]["val_split"]

    try:
        assert isinstance(test_split, float)
        assert isinstance(val_split, float)

        assert test_split >= 0.0
        assert test_split <= 1.0

        assert val_split >= 0.0
        assert val_split <= 1.0

    except:
        raise CustomError(
            "Test split or Validation split is not in the range of 0 to 1"
        )


def test_model_params():

    model_params = params["BestModelParameters"]

    try:

        assert model_params["learning_rate"] == 0.300000012
        assert model_params["silent"] == False
        assert model_params["n_estimators"] == 100
        assert model_params["max_depth"] == 6
        assert model_params["gamma"] == 0.0
        assert model_params["colsample_bytree"] == 1.0
        assert model_params["colsample_bylevel"] == 1.0
    except:
        raise CustomError("Loaded ML parameters are not correct")


def test_xtrain_data_present():

    x_train_data_path = os.path.join(
        train_val_test_data_directory, artifacts["x_train_data_file"]
    )

    try:
        assert os.path.exists(x_train_data_path) == True

    except:
        raise CustomError("x_train file does not exist")


def test_ytrain_data_present():

    y_train_data_path = os.path.join(
        train_val_test_data_directory, artifacts["y_train_data_file"]
    )

    try:
        assert os.path.exists(y_train_data_path) == True

    except:
        raise CustomError("y_train file does not exist")


def test_xtrain_data_size():

    x_train_data_path = os.path.join(
        train_val_test_data_directory, artifacts["x_train_data_file"]
    )

    try:
        assert os.path.getsize(x_train_data_path) != 0

    except:
        raise CustomError("x_train file size is zero")


def test_ytrain_data_size():

    y_train_data_path = os.path.join(
        train_val_test_data_directory, artifacts["y_train_data_file"]
    )

    try:
        assert os.path.getsize(y_train_data_path) != 0

    except:
        raise CustomError("y_train file size is zero")


def test_values_numeric():

    prepared_data_path = os.path.join(output_file, artifacts["processed_data_file"])
    df_req = pd.read_csv(prepared_data_path, sep=",")

    df_num = df_req.select_dtypes(include=["int64", "float64", "bool"])
    df_cat = df_req.select_dtypes(include=["object"])
    cat_features = list(df_cat.columns)

    try:
        assert df_num.shape[1] == 27  ## 27 is including target variable
        assert df_cat.shape[1] == 0
    except:
        raise CustomError(
            "There are {num} non-numeric features in input data: {list}".format(
                num=len(cat_features), list=cat_features
            )
        )


def test_cat_after_encoding():

    prepared_data_path = os.path.join(output_file, artifacts["processed_data_file"])
    df_req = pd.read_csv(prepared_data_path, sep=",")

    cat_features = [
        "OwnershipType",
        "OperatingStatus",
        "Org",
        "LineOfBusinessDescription",
        "FamilyTreeMemberRoleText",
        "PrimaryTownName",
    ]

    try:
        assert (df_req[cat_features].values < 0).any() == False
    except:
        raise CustomError("Encoding of categorical variables has negative values")


def test_scaler_exists():

    scaler_path = os.path.join(
        artifacts["artifacts_directory"], artifacts["scaler_directory"]
    )
    scaler_file = os.path.join(scaler_path, artifacts["scaler_name"])

    try:
        assert os.path.exists(scaler_file) == True
        assert os.path.getsize(scaler_file) > 0
    except:
        raise CustomError("Scaler file does not exist")


def test_input_features_are_saved():

    scaler_path = os.path.join(
        artifacts["artifacts_directory"], artifacts["scaler_directory"]
    )
    scaler_file = os.path.join(scaler_path, artifacts["scaler_name"])

    try:
        assert os.path.exists(scaler_file) == True
    except:
        raise CustomError("Scaler file does not exist")


def test_model():

    # Check that the model file can be loaded properly
    model_path = os.path.join(
        artifacts["artifacts_directory"], artifacts["model_directory"]
    )
    model_file = os.path.join(model_path, artifacts["model_name"])
    model = joblib.load(model_file)

    try:
        assert isinstance(model, xgboost.sklearn.XGBClassifier)
    except:
        raise CustomError("Loaded ML model is not XGBClassifier")


def test_model_accuracy():

    model_path = os.path.join(
        artifacts["artifacts_directory"], artifacts["model_directory"]
    )
    model_file = os.path.join(model_path, artifacts["model_name"])
    model = joblib.load(model_file)

    scaler_path = os.path.join(
        artifacts["artifacts_directory"], artifacts["scaler_directory"]
    )
    scaler_file = os.path.join(scaler_path, artifacts["scaler_name"])
    scaler = joblib.load(scaler_file)

    x_test_data_path = os.path.join(
        train_val_test_data_directory, artifacts["x_test_data_file"]
    )
    x_test = pd.read_csv(x_test_data_path, sep=",")

    y_test_data_path = os.path.join(
        train_val_test_data_directory, artifacts["y_test_data_file"]
    )
    y_test = pd.read_csv(y_test_data_path, sep=",")

    prediction_test = model.predict(scaler.transform(x_test))

    tpr = confusion_matrix(y_test, prediction_test, normalize="true")[3]

    try:
        assert tpr >= 0.90
    except:
        raise CustomError("Test TPR dropped below 90%")


if __name__ == "__main__":
    pytest.main([__file__, "-k", "test_", "-v", "-s"])
