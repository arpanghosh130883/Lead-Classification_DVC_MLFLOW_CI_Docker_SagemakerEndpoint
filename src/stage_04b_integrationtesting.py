import pytest
import os
from src.utils.common import read_yaml
from src.utils.testing import *
from src.utils.customexception import CustomError
import joblib
import xgboost
import pandas as pd
from ensure import ensure
import numpy as np
from sklearn.metrics import confusion_matrix

# class CustomError(Exception):
#     pass


config_path = "configs/config.yaml"
params_path = "params.yaml"

config = read_yaml(config_path)
params = read_yaml(params_path)

artifacts = config["artifacts"]
source_data = config["source_data"]

input_data_file = os.path.join(source_data["data_directory"], source_data["data_file"])

x_train_data_file = define_path(
    artifacts["artifacts_directory"],
    artifacts["train_val_test_data_directory"],
    artifacts["x_train_data_file"],
)
y_train_data_file = define_path(
    artifacts["artifacts_directory"],
    artifacts["train_val_test_data_directory"],
    artifacts["y_train_data_file"],
)
x_val_data_file = define_path(
    artifacts["artifacts_directory"],
    artifacts["train_val_test_data_directory"],
    artifacts["x_val_data_file"],
)
y_val_data_file = define_path(
    artifacts["artifacts_directory"],
    artifacts["train_val_test_data_directory"],
    artifacts["y_val_data_file"],
)
x_test_data_file = define_path(
    artifacts["artifacts_directory"],
    artifacts["train_val_test_data_directory"],
    artifacts["x_test_data_file"],
)
y_test_data_file = define_path(
    artifacts["artifacts_directory"],
    artifacts["train_val_test_data_directory"],
    artifacts["y_test_data_file"],
)
input_features_file = define_path(
    artifacts["artifacts_directory"],
    artifacts["input_features_dir"],
    artifacts["input_features_file"],
)
processed_data_file = define_path(
    artifacts["artifacts_directory"],
    artifacts["processed_data_dir"],
    artifacts["processed_data_file"],
)
scaled_train_data_file = define_path(
    artifacts["artifacts_directory"],
    artifacts["scaled_train_dir"],
    artifacts["xtrain_scaled_file"],
)

model_file = define_path(
    artifacts["artifacts_directory"],
    artifacts["model_directory"],
    artifacts["model_name"],
)
scaler_file = define_path(
    artifacts["artifacts_directory"],
    artifacts["scaler_directory"],
    artifacts["scaler_name"],
)


model_params = params["BestModelParameters"]


model = load_file(model_file)
scaler = load_file(scaler_file)
input_df = read_csv(input_data_file)
x_train_df = read_csv(x_train_data_file)
y_train_df = read_csv(y_train_data_file)
x_val_df = read_csv(x_val_data_file)
y_val_df = read_csv(y_val_data_file)
x_test_df = read_csv(x_test_data_file)
y_test_df = read_csv(y_test_data_file)
final_df = read_csv(processed_data_file)
df_num = final_df.select_dtypes(include=["int64", "float64"])
df_cat = final_df.select_dtypes(include=["object"])


initial_required_columns = ["overseas_countries", "acc_status"]
final_columns = [
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

columns = [
    (input_df.columns, initial_required_columns),
    (final_df.columns, final_columns),
]


values = [
    (x_train_df.shape[0], y_train_df.shape[0]),
    (x_train_df.shape[1], 26),
    (final_df.isnull().sum().all(), 0),
    (df_num.shape[1], 27),
    (df_cat.shape[1], 0),
    (model_params["learning_rate"], 0.300000012),
    (model_params["silent"], False),
    (model_params["n_estimators"], 100),
    (model_params["max_depth"], 6),
    (model_params["gamma"], 0.0),
    (model_params["colsample_bytree"], 1.0),
    (model_params["colsample_bylevel"], 1.0),
]

model_type = [(model, xgboost.sklearn.XGBClassifier)]

test_split = params["ModelTraining"]["test_split"]
val_split = params["ModelTraining"]["val_split"]
split_ratios = [test_split, val_split]


@pytest.mark.parametrize("model, type", model_type)
def test_model_type(model, type):
    try:
        assert isinstance(model, type)
    except:
        raise CustomError("Loaded model is not XGBoost")


@pytest.mark.parametrize("all_columns, required_columns", columns)
def test_if_columns_exist(all_columns, required_columns):
    try:
        ensure(all_columns).contains_all_of(required_columns)
    except:
        raise CustomError("Required columns not present")


@pytest.mark.parametrize("value1, value2", values)
def test_xtrain_ytrain_shape(value1, value2):
    try:
        ensure(value1).equals(value2)
    except:
        raise CustomError("Values do not match")


@pytest.mark.parametrize("split_ratio", split_ratios)
def test_split_ratios(split_ratio):
    try:
        ensure(split_ratio).is_a_positive(float)
        ensure(split_ratio).is_less_than_or_equal_to(1)
    except:
        raise CustomError("Split ratios do not have an acceptable value")


def test_categorical_features_have_nonnegative_values():

    cat_features = [
        "OwnershipType",
        "OperatingStatus",
        "Org",
        "LineOfBusinessDescription",
        "FamilyTreeMemberRoleText",
        "PrimaryTownName",
    ]
    try:
        ensure(final_df[cat_features].values.all()).is_nonnegative()
    except:
        raise CustomError("Categorical features have negative values")


def test_model_accuracy():
    prediction_test = model.predict(scaler.transform(x_test_df))
    (tnr, fpr), (fnr, tpr) = confusion_matrix(
        y_test_df, prediction_test, normalize="true"
    )

    try:
        ensure(tpr).is_greater_than_or_equal_to(0.85)
    except:
        raise CustomError("Test TPR dropped below 85%")
