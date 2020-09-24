FROM bvlc/caffe:gpu

COPY ./requirements-pytorch.txt /workspace/
WORKDIR /workspace

RUN pip install -r ./requirements-pytorch.txt
