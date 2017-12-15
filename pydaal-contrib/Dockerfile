FROM continuumio/miniconda

LABEL maintainer="vilen.jumutcs@intel.com"
LABEL version="2017.0"
LABEL description="(Py)DAAL contrib and extensions library bundeled with Intel® Python, DAAL and MKL"

WORKDIR /work

RUN conda install -y -c intel pydaal
RUN conda install -y -c conda-forge future
RUN conda install -y -c conda-forge jsonpickle
RUN conda install -y -c conda-forge multipledispatch
RUN conda install -y conda-build

ADD . ./pydaal-contrib

VOLUME /media

RUN conda build -c intel -c conda-forge ./pydaal-contrib
RUN conda install -y --use-local pydaalcontrib