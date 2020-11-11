FROM ubuntu:18.04

ARG PREFIX=/usr/local
ARG GPU_ARCH=";"
ARG MIOTENSILE_VER="default"

# Support multiarch
RUN dpkg --add-architecture i386

# Add rocm repository
# RUN sh -c 'echo deb [arch=amd64 trusted=yes] http://repo.radeon.com/rocm/apt/.apt_3.7/ xenial main > /etc/apt/sources.list.d/rocm.list'
RUN sh -c 'echo deb [arch=amd64 trusted=yes] http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-deb/ compute-rocm-dkms-amd-feature-targetid 3004 > /etc/apt/sources.list.d/rocm.list'

# Install dependencies
RUN apt-get update --fix-missing --allow-insecure-repositories && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    sudo \
    ca-certificates \
    apt-utils \
    build-essential \
    make \
    cmake \
    curl \
    doxygen \
    g++ \
    gdb \
    git \
    graphviz \
    hip-rocclr \
    lcov \
    libelf-dev \
    libncurses5-dev \
    libpthread-stubs0-dev \
    libnuma1 \
    llvm-6.0-dev \
    libboost-all-dev \
    miopengemm \
    pkg-config \
    python \
    python3 \
    python-yaml \
    python3-yaml \
    python-dev \
    python3-dev \
    python-pip \
    python3-pip \
    python3-distutils \
    python3-pytest \
    python3-setuptools \
    python3-venv \
    software-properties-common \
    wget \
    pkg-config \
    rocm-dev \
    rocm-device-libs \
    rocm-opencl \
    rocm-opencl-dev \
    zlib1g-dev \
    libomp-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
 
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install cget
RUN pip3 install wheel && pip3 install pyyaml && pip3 install cget
RUN pip install https://github.com/pfultz2/rbuild/archive/master.tar.gz

# Add symlink to /opt/rocm
RUN [ -d /opt/rocm ] || ln -sd $(realpath /opt/rocm-*) /opt/rocm

# Install rocm-cmake
RUN cget -p $PREFIX install RadeonOpenCompute/rocm-cmake@master

# Manually download and install MessagePack DEB files
RUN curl -O http://ftp.us.debian.org/debian/pool/main/m/msgpack-c/libmsgpack-dev_3.0.1-3_amd64.deb \
         -O http://ftp.us.debian.org/debian/pool/main/m/msgpack-c/libmsgpackc2_3.0.1-3_amd64.deb
RUN dpkg -i libmsgpack-dev_3.0.1-3_amd64.deb libmsgpackc2_3.0.1-3_amd64.deb

RUN pip3 install setuptools --upgrade && \
    pip3 install wheel && \
    pip3 install tox pyyaml msgpack

# Install dependencies
RUN cget -p $PREFIX install pfultz2/rocm-recipes
ADD requirements.txt /requirements.txt
RUN CXXFLAGS='-isystem $PREFIX/include' cget -p $PREFIX install -f /requirements.txt

# Set sudo privileges
RUN useradd --create-home -G video --shell /bin/bash jenkins && \
    echo '%video   ALL=(ALL) NOPASSWD:ALL' | tee /etc/sudoers.d/sudo-nopasswd && \
    chmod 400 /etc/sudoers.d/sudo-nopasswd && \
    usermod -aG video root && \
    echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf && \
    echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf

