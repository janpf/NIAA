FROM tensorflow/tensorflow:latest

COPY ./requirements-pytorch.txt /workspace/
WORKDIR /workspace

RUN apt-get install -y libgl1-mesa-glx

RUN pip install -r ./requirements-pytorch.txt
RUN pip install tensorpack
