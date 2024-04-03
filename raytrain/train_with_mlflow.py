import os
import mlflow

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/xsqian/dagshub-demo.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "xsqian"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "rfg8@5LZ88igXX@"

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

from ray import tune

def train_mlflow(config):
    with mlflow.start_run():
        # Training logic here
        mlflow.log_param('param', config['param'])
        mlflow.log_metric('metric', 0.88)

tune.run(train_mlflow, config={
    'param': tune.grid_search([0.1, 0.2, 0.3])
})