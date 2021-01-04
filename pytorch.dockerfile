FROM pytorch/pytorch:latest

RUN apt-get update
RUN apt-get install -y libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev python3-gi python3-gi-cairo libgtk2.0-dev libmagickwand-dev libgl1-mesa-glx protobuf-compiler gegl gir1.2-gtk-3.0

COPY ./requirements-pytorch.txt /workspace/
WORKDIR /workspace
RUN pip install -r ./requirements-pytorch.txt
