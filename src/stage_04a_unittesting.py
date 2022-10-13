import pytest
import os
from src.utils.common import read_yaml
from src.utils.testing import *
from src.utils.customexception import CustomError
from ensure import ensure

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


file_paths = [
    model_file,
    scaler_file,
    x_train_data_file,
    y_train_data_file,
    x_val_data_file,
    y_val_data_file,
    x_test_data_file,
    y_test_data_file,
    scaled_train_data_file,
    input_features_file,
    processed_data_file,
]


file_extensions = [
    (model_file, "pkl"),
    (scaler_file, "pkl"),
    (x_train_data_file, "csv"),
    (y_train_data_file, "csv"),
    (x_val_data_file, "csv"),
    (y_val_data_file, "csv"),
    (x_test_data_file, "csv"),
    (y_test_data_file, "csv"),
    (scaled_train_data_file, "csv"),
    (input_features_file, "csv"),
    (processed_data_file, "csv"),
]


@pytest.mark.parametrize("path", file_paths)
def test_if_files_exist(path):
    try:
        ensure(os.path.exists(path)).is_(True)
        ensure(os.path.getsize(path)).is_greater_than(0)
    except:
        raise CustomError("Specified file does not exist")


@pytest.mark.parametrize("file, extension", file_extensions)
def test_file_format(file, extension):
    try:
        assert file.split(".")[1] == extension
    except:
        raise CustomError("Specified file format not supported")
