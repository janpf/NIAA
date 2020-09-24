FROM mapler/caffe-py3:gpu

RUN apt-get update && apt-get dist-upgrade -y
RUN pip3 install --upgrade pip

WORKDIR /workspace
COPY ./requirements-pytorch.txt /workspace/
RUN sed -i "s/torch$/torch==1.4/" requirements-pytorch.txt
RUN sed -i "s/torchvision$/torchvision==0.5/" requirements-pytorch.txt

RUN pip3 install -r ./requirements-pytorch.txt
