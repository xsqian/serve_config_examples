from ray import serve
from starlette.requests import Request
import tensorflow as tf

MODEL_PATH = "model.h5"
print(MODEL_PATH)
@serve.deployment
class Model:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)

    def __call__(self):

        with open("file.txt") as f:
            return f.read().strip()
app = Model.bind(MODEL_PATH)
