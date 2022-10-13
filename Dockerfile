FROM python:3.7

COPY . /mlflow_dvc_trial

# Specify work directory
WORKDIR /mlflow_dvc_trial

RUN python3 -m pip install -r requirements.txt

RUN export MLFLOW_TRACKING_URI=sqlite:///new_db.db 

EXPOSE 5000

CMD mlflow run ./ --no-conda && mlflow server --backend-store-uri sqlite:///new_db.db --default-artifact-root sqlite:///new_db.db --host 0.0.0.0 --port 5000

