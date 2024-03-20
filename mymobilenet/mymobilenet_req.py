import requests

# TODO: Change this to your image path
image_path = "shiba-inu.jpeg"

url = "http://127.0.0.1:8000"
files = {"image": open(image_path, "rb")}
response = requests.post(url, files=files)
print(response.text)


# Xingsheng_Qian@[24-03-19 16:08] ~/src/serve_config_examples/mymobilnet on master!
# ± python mymobilenet_req.py
# {"prediction": ["n02115641", "dingo", 0.6235566735267639]}
# (ray) 