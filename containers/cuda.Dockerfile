FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       autoconf \
       automake \
       bison \
       build-essential \
       flex \
       gengetopt \
       help2man \
       libtool \
       libtool-bin \
       perl \
       pkg-config \
       python3 \
       rsync \
       swig \
       texinfo \
       vim-common \
       xxd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
