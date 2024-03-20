serve run read_file.app

serve build hello.read_file.app -o ray-service.hello.yaml

serve deploy -f ray-service.hello.yaml

kubectl port-forward svc/rayservice-hello-serve-svc 8000

