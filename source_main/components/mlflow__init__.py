import mlflow
import dagshub


def init_mlflow():
    """
    Initializes MLflow with DagsHub tracking & registry.
    Safe for multiple runs.
    """

    # 1️⃣ Initialize DagsHub (sets tracking + registry URI internally)
    dagshub.init(
        repo_owner="yashwanth0098",
        repo_name="Bank_Term_Deposit_Prediction_Azure_Aws",
        mlflow=True,
    )

    # 2️⃣ Set experiment directly (auto-create if not exists)
    EXPERIMENT_NAME = "Bank_Term_Deposit_Model_Training_and_Evaluate"
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 3️⃣ (Optional but recommended) Explicit registry URI
    mlflow.set_registry_uri(mlflow.get_tracking_uri())
