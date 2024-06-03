FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda create -n ccf_reg python=3.8

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "ccf_reg", "/bin/bash", "-c"]

RUN pip install -U --no-cache-dir \
    antspyx==0.4.2 \
    argschema==3.0.4 \
    s3fs==2022.11.0 \
    scikit-image==0.19.3 \
    tifffile==2022.10.10 \
    bokeh==2.4.2 \
    zarr==2.13.3 \
    aind-data-schema==0.22.1 \
    xarray_multiscale==1.1.0 \
    dask[distributed]==2022.11.1 \
    matplotlib==3.7.3 \
    ome-zarr==0.8.2 \
    natsort==8.4.0 \
    aicsimageio@git+https://github.com/camilolaiton/aicsimageio.git@feature/zarrwriter-multiscales-daskjobs

RUN conda install -c conda-forge awscli
