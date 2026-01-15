# Stage 1: RMG setup & ARC setup
# The parent image is the base image that the Dockerfile builds upon.
# The RMG installation instructions suggest Anaconda for installation by source, however, we use micromamba for the Docker image due to its smaller size and less overhead.
# Installation of ARC will also be done in this stage.
FROM --platform=linux/amd64 mambaorg/micromamba:2.2-ubuntu24.04 AS builder

# Set ARGS
ARG RMG_PY_BRANCH=main
ARG RMG_DATABASE_BRANCH=main
ARG ARC_BRANCH=main

# Set Global ENV
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
ENV MAMBA_DOCKERFILE_ACTIVATE=1

# Switch to root to install dependencies
USER root

RUN apt-get update && apt-get install -y \
    git gcc g++ make wget libxrender1 ca-certificates sudo nano make && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /home/mambauser/Code && \
    chown -R mambauser:mambauser /home/mambauser && \
    chown -R mambauser:mambauser /opt/conda && \
    chown -R mambauser:mambauser /home/mambauser

# Change to unprivileged user
USER mambauser
ENV MAMBA_USER=mambauser

# Set JuliaUp PATH and install Julia 1.10 as req. by RMG
ENV PATH="/home/mambauser/.juliaup/bin:$PATH"
RUN wget -qO- https://install.julialang.org | sh -s -- --yes --default-channel 1.10

# Switch directory to Code and RMG clone
WORKDIR /home/mambauser/Code
RUN git clone --branch ${RMG_PY_BRANCH} https://github.com/ReactionMechanismGenerator/RMG-Py.git && \
    git clone --branch ${RMG_DATABASE_BRANCH} https://github.com/ReactionMechanismGenerator/RMG-database.git && \
    git clone --branch ${ARC_BRANCH} https://github.com/ReactionMechanismGenerator/ARC.git

# Create RMG-Py environment
RUN micromamba create -y -n rmg_env -f /home/mambauser/Code/RMG-Py/environment.yml && \
    micromamba run -n rmg_env micromamba install -y -c conda-forge pyjuliacall conda && \
    micromamba clean --all --yes && \
    micromamba run -n rmg_env make -C /home/mambauser/Code/RMG-Py -j"$(nproc)" && \
    micromamba run -n rmg_env bash -c "\
      cd /home/mambauser/Code/RMG-Py && \
      source install_rms.sh \
    "

WORKDIR /home/mambauser/Code/ARC
RUN micromamba create -y -n arc_env -f environment.yml --channel-priority flexible && \
    micromamba install -y -n arc_env -c conda-forge pytest && \
    micromamba clean --all -f -y && \
    micromamba run -n arc_env bash -euxo pipefail -c \
      "make compile && bash ./devtools/install_pyrdl.sh" && \
    micromamba clean --all --yes

# Stage 2: Final image
# The final image is based on the same micromamba image, but we copy over the installed RMG and ARC from the builder stage.
# This keeps the final image size smaller and avoids unnecessary layers.
FROM --platform=linux/amd64 mambaorg/micromamba:2.2-ubuntu24.04

ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:/home/mambauser/.juliaup/bin:/home/mambauser/Code/RMG-Py:/home/mambauser/Code/ARC:$PATH
ENV PYTHONPATH="/home/mambauser/Code/RMG-Py:/home/mambauser/Code/ARC"
ENV RMG_PY_DIR="/home/mambauser/Code/RMG-Py"
ENV ARC_DIR="/home/mambauser/Code/ARC"
ENV MAMBA_DOCKERFILE_ACTIVATE=1

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        libxrender1 \
        ca-certificates \
        nano \
        make \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
USER mambauser

COPY --from=builder --chown=mambauser:mambauser /opt/conda /opt/conda
COPY --from=builder --chown=mambauser:mambauser /home/mambauser/.juliaup /home/mambauser/.juliaup
COPY --from=builder --chown=mambauser:mambauser /home/mambauser/Code /home/mambauser/Code

# Need to copy the tests separately as they are not in the ARC git by default
COPY --chown=mambauser:mambauser dockerfiles/docker_tests /home/mambauser/Code/ARC/dockerfiles/docker_tests

# --- Entry wrapper ----------------------------------------------------------
COPY --chmod=755  dockerfiles/entrywrapper.sh  /usr/local/bin/entrywrapper.sh
COPY --chmod=644  dockerfiles/aliases.sh       /etc/profile.d/aliases.sh
COPY --chmod=755  dockerfiles/aliases_print.sh /usr/local/bin/aliases
RUN touch /home/mambauser/.bashrc && \
    grep -qxF 'source /etc/profile.d/aliases.sh' /home/mambauser/.bashrc || \
    echo 'source /etc/profile.d/aliases.sh' >> /home/mambauser/.bashrc

USER root
ENTRYPOINT ["/usr/local/bin/entrywrapper.sh"]
