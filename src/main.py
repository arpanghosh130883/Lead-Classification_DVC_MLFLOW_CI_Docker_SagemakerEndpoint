import mlflow


def main():

    mlflow.set_tracking_uri("sqlite:///new_db.db")

    mlflow.run(".", "stage_01", use_conda=False, experiment_name='leads')
    print("Stage 01 completed")
    mlflow.run(".", "stage_02", use_conda=False, experiment_name='leads')
    print("Stage 02 completed")
    mlflow.run(".", "stage_03", use_conda=False, experiment_name='leads')
    print("Stage 03 completed")
    mlflow.run(".", "stage_04a", use_conda=False, experiment_name='leads')
    print("Stage 04a completed")
    mlflow.run(".", "stage_04b", use_conda=False, experiment_name='leads')
    print("Stage 04b completed")
    mlflow.run(".", "stage_05", use_conda=False, experiment_name='leads')
    print("Stage 05 completed")
    mlflow.run(".", "stage_06", use_conda=False, experiment_name='leads')
    print("Stage 06 completed")
        

if __name__ == "__main__":
    main()