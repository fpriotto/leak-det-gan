# Imagem Docker base para rodar no SageMaker ou ECS
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
