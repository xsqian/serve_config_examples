from ray import serve
from starlette.requests import Request
import tensorflow as tf

MODEL_PATH = "file.txt"
@serve.deployment
class Model:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def __call__(self):
        with open(self.model_path) as f:
            return f.read().strip()
app = Model.bind(MODEL_PATH)
