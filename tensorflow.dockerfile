FROM tensorflow/tensorflow:1.2.1-gpu

COPY ./requirements-tensorflow.txt /workspace/
WORKDIR /workspace

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx

RUN pip install -r ./requirements-tensorflow.txt
