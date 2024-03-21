# https://docs.ray.io/en/latest/serve/tutorials/serve-ml-models.html
# pip install "tensorflow>=2.0"  requests

from ray import serve

import os
import tempfile
import numpy as np
from starlette.requests import Request
from typing import Dict
import tensorflow as tf

# # TRAINED_MODEL_PATH = os.path.join(tempfile.gettempdir(), "mnist_model.h5")

# TRAINED_MODEL_PATH = os.path.join(os.getcwd(), "model", "mnist_model.h5")

# TRAINED_MODEL_PATH = "https://mlrun-ce-cfn.s3.us-east-2.amazonaws.com/mnist_model.h5" #use a remote model registry, actuall will receive error below:
# # (ServeController pid=35245) tensorflow.python.framework.errors_impl.UnimplementedError: File system scheme 'https' not implemented (file: 'https://mlrun-ce-cfn.s3.us-east-2.amazonaws.com/mnist_model.h5')

# from urllib.request import urlretrieve

# urlretrieve('https://mlrun-ce-cfn.s3.us-east-2.amazonaws.com/mnist_model.h5', 'model.h5')
TRAINED_MODEL_PATH = "model.h5"
print(TRAINED_MODEL_PATH)

@serve.deployment
class TFMnistModel:
    def __init__(self, model_path: str):
        self.model_path = model_path 
        self.model = tf.keras.models.load_model(model_path)

    async def __call__(self, starlette_request: Request) -> Dict:
        # Step 1: transform HTTP request -> tensorflow input
        # Here we define the request schema to be a json array.
        input_array = np.array((await starlette_request.json())["array"])
        reshaped_array = input_array.reshape((1, 28, 28))

        # Step 2: tensorflow input -> tensorflow output
        prediction = self.model(reshaped_array)

        # Step 3: tensorflow output -> web output
        return {"prediction": prediction.numpy().tolist(), "file": self.model_path}
    
mnist_model = TFMnistModel.bind(TRAINED_MODEL_PATH)

