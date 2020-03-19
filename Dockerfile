FROM pytorch/pytorch:latest

# because ubuntu only has darktable v2.6: downloading v3
ADD https://download.opensuse.org/repositories/graphics:/darktable/xUbuntu_18.04/amd64/darktable_3.0.1-1.1_amd64.deb ./darktable.deb
RUN apt-get update
RUN apt-get install -y ./darktable.deb
# no dpkg, because dependency resolution
RUN rm darktable.deb

COPY ./requirements-pytorch.txt /workspace/
WORKDIR /workspace
RUN pip install -r ./requirements-pytorch.txt
