https://docs.ray.io/en/latest/cluster/kubernetes/examples/ml-example.html


± kubectl apply -f xgboost-mini.yaml
# raycluster.ray.io/raycluster-xgboost-mini created

kubectl get svc
raycluster-xgboost-mini-head-svc



kubectl port-forward --address 0.0.0.0 service/raycluster-xgboost-mini-head-svc 8265:8265