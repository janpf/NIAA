FROM python:3.7

# because debian only has darktable v2.6: downloading v3
ADD https://download.opensuse.org/repositories/graphics:/darktable/Debian_10/amd64/darktable_3.0.1-1.1_amd64.deb ./darktable.deb
RUN apt-get update
RUN apt-get install -y ./darktable.deb
# no dpkg, because dependency resolution
RUN rm darktable.deb

COPY ./survey/requirements-flask.txt /app/survey/
WORKDIR /app
RUN pip install -r ./survey/requirements-flask.txt

COPY . /app
