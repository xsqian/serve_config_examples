from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("http://127.0.0.1:8265")

kick_off_xgboost_benchmark = (
    # Clone ray. If ray is already present, don't clone again.
    "rm serve_config_examples -rf;"
    # "git clone https://github.com/ray-project/ray || true;"
    "git clone https://github.com/xsqian/serve_config_examples.git || False;"

    # Run the benchmark.
    " python serve_config_examples/raytrain/train_func.py"
)


submission_id = client.submit_job(
    entrypoint=kick_off_xgboost_benchmark,
)

print("Use the following command to follow this Job's logs:")
print(f"ray job logs '{submission_id}' --follow")
