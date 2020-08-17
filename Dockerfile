FROM pytorch/pytorch:latest

RUN apt-get update
RUN apt-get install -y libgtk2.0-dev libmagickwand-dev libgl1-mesa-glx

COPY ./requirements-pytorch.txt /workspace/
WORKDIR /workspace
RUN pip install -r ./requirements-pytorch.txt
