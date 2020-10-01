FROM tensorflow/tensorflow:latest-gpu

COPY ./requirements-tensorflow.txt /workspace/
WORKDIR /workspace

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx

RUN pip install --upgrade pip
RUN pip install -r ./requirements-tensorflow.txt
