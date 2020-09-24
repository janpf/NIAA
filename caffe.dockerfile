FROM bvlc/caffe:gpu

# caffe image is horribly outdated
RUN apt-get update && apt-get dist-upgrade -y
RUN pip install --upgrade pip

COPY ./requirements-pytorch.txt /workspace/
WORKDIR /workspace

RUN pip install -r ./requirements-pytorch.txt
