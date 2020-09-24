FROM ufoym/deepo:latest
RUN pip install -r ./requirements-pytorch.txt
RUN pip install tensorpack
