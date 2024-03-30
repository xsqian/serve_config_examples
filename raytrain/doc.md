## Steps for ray train on k8s

### write your training code with raytrain

### set up a k8s cluster (already exists in Iguazio)

### Deploy the KubeRay operator (already exists in Iguazio)

### Deploy a Ray cluster
```
kubectl apply -f https://raw.githubusercontent.com/ray-project/ray/releases/2.0.0/doc/source/cluster/kubernetes/configs/xgboost-benchmark.yaml
```

### Run the workload
To observe the startup progress of the Ray head pod, run the following command.
```
watch -n 1 kubectl get pod
```

### Connect to the cluster

```
kubectl port-forward --address 0.0.0.0 service/raycluster-xgboost-benchmark-head-svc 8265:8265
```

### Submit the workload 

```python
from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("http://127.0.0.1:8265")

kick_off_xgboost_benchmark = (
    # Clone ray. If ray is already present, don't clone again.
    "git clone https://github.com/ray-project/ray || true; "
    # Run the benchmark.
    "python ray/release/train_tests/xgboost_lightgbm/train_batch_inference_benchmark.py"
    " xgboost --size=100G --disable-check"
)


submission_id = client.submit_job(
    entrypoint=kick_off_xgboost_benchmark,
)

print("Use the following command to follow this Job's logs:")
print(f"ray job logs '{submission_id}' --follow")
```

```
# Download the above script.
curl https://raw.githubusercontent.com/ray-project/ray/releases/2.0.0/doc/source/cluster/doc_code/xgboost_submit.py -o xgboost_submit.py
# Run the script.
python xgboost_submit.py
```

```
ray job logs 'raysubmit_PunLcB8KZbePxABJ' --follow --address http://127.0.0.1:8265     
```

```
watch -n 1 kubectl exec -it raycluster-xgboost-mini-head-dthg2 -- ray status
```