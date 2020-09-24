FROM tensorflow/tensorflow:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV CAFFE_ROOT=/opt/caffe
ENV CLONE_TAG=1.0

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    debhelper \
    devscripts \
    fakeroot \
    git \
    wget \
    libatlas-base-dev \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libprotobuf-dev \
    libsnappy-dev \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-pip \
    python-setuptools \
    python-scipy
RUN rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip


WORKDIR $CAFFE_ROOT

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git .
RUN cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done

RUN git clone --depth 1 https://github.com/NVIDIA/nccl.git
RUN cd nccl && make -j install
RUN rm -rf nccl

RUN mkdir build
RUN cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 .. && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

RUN pip install tensorpack

WORKDIR /workspace
