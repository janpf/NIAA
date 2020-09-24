FROM tensorflow/tensorflow:latest

COPY ./requirements-pytorch.txt /workspace/
WORKDIR /workspace

RUN pip install -r ./requirements-pytorch.txt
RUN pip install tensorpack
