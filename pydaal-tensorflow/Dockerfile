FROM intel/pydaal-contrib

LABEL maintainer="vilen.jumutcs@intel.com"
LABEL version="2017.0"
LABEL description="Converter from Tensorflow to (Py)DAAL"

WORKDIR /work

RUN conda install -y -c conda-forge tensorflow 

ADD . ./pydaal-tensorflow

VOLUME /media

RUN conda build -c intel -c conda-forge ./pydaal-tensorflow
RUN conda install -y --use-local pydaaltensorflow