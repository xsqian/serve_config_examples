from ray import serve

import pickle
import json
import numpy as np
import os
import tempfile
from starlette.requests import Request
from typing import Dict

from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error

MODEL_PATH = "iris_model_gradient_boosting_classifier.pkl"
LABEL_PATH = "iris_labels.json"

@serve.deployment
class BoostingModel:
    def __init__(self, model_path: str, label_path: str):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(label_path) as f:
            self.label_list = json.load(f)

    async def __call__(self, starlette_request: Request) -> Dict:
        payload = await starlette_request.json()
        print("Worker: received starlette request with data", payload)

        input_vector = [
            payload["sepal length"],
            payload["sepal width"],
            payload["petal length"],
            payload["petal width"],
        ]
        prediction = self.model.predict([input_vector])[0]
        human_name = self.label_list[prediction]
        return {"result": human_name}
    
boosting_model = BoostingModel.bind(MODEL_PATH, LABEL_PATH)