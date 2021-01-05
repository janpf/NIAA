FROM nvcr.io/nvidia/pytorch:20.12-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y python3-gi python3-gi-cairo libgirepository1.0-dev libmagickwand-dev libgl1-mesa-glx protobuf-compiler gir1.2-gegl-0.4

COPY ./requirements-pytorch.txt /workspace/
WORKDIR /workspace
RUN pip install -r ./requirements-pytorch.txt
