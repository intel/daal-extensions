FROM intel/pydaal-contrib

LABEL maintainer="vilen.jumutcs@intel.com"
LABEL version="2017.0"
LABEL description="Converter from Caffe to (Py)DAAL"

WORKDIR /work

RUN conda install -y -c conda-forge protobuf 

ADD . ./pydaal-caffe

VOLUME /media

RUN conda build -c intel -c conda-forge ./pydaal-caffe
RUN conda install -y --use-local pydaalcaffe