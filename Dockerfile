FROM pytorch/pytorch:latest

COPY ./requirements.txt /workspace/

RUN conda install --y pip
RUN pip install -r /workspace/requirements.txt

RUN python -c "import torch"
