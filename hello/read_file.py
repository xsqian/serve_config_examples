from ray import serve

MODEL_PATH = "file.txt"
@serve.deployment
class Model:
    def __call__(self, model_path: str):
        self.model_path = model_path
        with open(model_path) as f:
            return f.read().strip()
app = Model.bind(MODEL_PATH)
