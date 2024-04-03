import os
import tempfile

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose
import ray.train.torch

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/xsqian/dagshub-demo.mlflow"

def train_func():
    # Model, Loss, Optimizer
    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    # [1] Prepare model.
    model = ray.train.torch.prepare_model(model)
    # model.to("cuda")  # This is done by `prepare_model`
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Data
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    data_dir = os.path.join(tempfile.gettempdir(), "data")
    train_data = FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    # [2] Prepare dataloader.
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    # Training
    for epoch in range(1):
        if ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)

        for images, labels in train_loader:
            # This is done by `prepare_data_loader`!
            # images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # [3] Report metrics and checkpoint.
        metrics = {"loss": loss.item(), "epoch": epoch}
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                model.module.state_dict(),
                os.path.join(temp_checkpoint_dir, "model.pt")
            )
            ray.train.report(
                metrics,
                checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            )
        if ray.train.get_context().get_world_rank() == 0:
            print(metrics)

# [4] Configure scaling and resource requirements.
scaling_config = ray.train.ScalingConfig(num_workers=2, use_gpu=False)


# [5] Launch distributed training job.
    
def train_mlflow(config):
    trainer = ray.train.torch.TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        # [5a] If running in a multi-node cluster, this is where you
        # should configure the run's persistent storage that is accessible
        # across all worker nodes.
        run_config=ray.train.RunConfig(
            storage_path="s3://mlrun-ce-cfn/mlflow", 
            failure_config=ray.train.FailureConfig(3)
            ),
    )
    result = trainer.fit()

# train_mlflow({})
    
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
from ray import train, tune
import mlflow
mlflow.pytorch.autolog(log_models=True) 
def tune_with_callback(mlflow_tracking_uri, finish_fast=True):
    tuner = tune.Tuner(
        train_mlflow,
        tune_config=tune.TuneConfig(num_samples=3),
        run_config=train.RunConfig(
            name="mlflow",
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=mlflow_tracking_uri,
                    experiment_name="mlflow_callback_train_func",
                    save_artifact=True,
                    # tracking_token=os.environ['MLFLOW_TRACKING_TOKEN']  
                )
            ],
        ),
        param_space={
            "width": tune.choice([10, 11]),
            "steps": 2 if finish_fast else 3,
        },
    )
    result = tuner.fit()
    return result

result = tune_with_callback(os.environ['MLFLOW_TRACKING_URI'], finish_fast=True)

# # [6] Load the trained model.
# with result.checkpoint.as_directory() as checkpoint_dir:
#     model_state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))
#     model = resnet18(num_classes=10)
#     model.conv1 = torch.nn.Conv2d(
#         1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
#     )
#     model.load_state_dict(model_state_dict)