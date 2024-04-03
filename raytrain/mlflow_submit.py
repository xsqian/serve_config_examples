#submit a train with mlflow job
from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("http://127.0.0.1:8265")

kick_off_xgboost_benchmark = (
    # Clone ray. If ray is already present, don't clone again.
    "rm serve_config_examples -rf;"
    # "git clone https://github.com/ray-project/ray || true;"
    "git clone https://github.com/xsqian/serve_config_examples.git || False;"


    # Run the benchmark.
    " python serve_config_examples/raytrain/train_func_with_mlflow.py"
)


submission_id = client.submit_job(
    entrypoint=kick_off_xgboost_benchmark,
    runtime_env={
        "pip": ["requests==2.26.0"],
    }
)

print("Use the following command to follow this Job's logs:")
print(f"ray job logs '{submission_id}' --follow --address http://127.0.0.1:8265")