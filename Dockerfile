# syntax=docker/dockerfile:1.4

# Stage 1: Base build with micromamba
FROM --platform=linux/amd64 mambaorg/micromamba:2.2-ubuntu24.04 AS builder

# --- Build‑time arguments (override in CI if needed) ------------------------
ARG RMG_PY_BRANCH=main
ARG RMG_DATABASE_BRANCH=main
ARG ARC_BRANCH=molecule

# --- Global env vars --------------------------------------------------------
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
ENV MAMBA_DOCKERFILE_ACTIVATE=1

# --- System packages --------------------------------------------------------
USER root
RUN apt-get update && apt-get install -y \
    git gcc g++ make wget libxrender1 ca-certificates sudo nano make && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Back to unprivileged user ---------------------------------------------
USER mambauser
ENV MAMBA_USER=mambauser

# --- Julia via juliaup ------------------------------------------------------
RUN wget -qO- https://install.julialang.org | sh -s -- --yes --default-channel 1.10
ENV PATH="/home/mambauser/.juliaup/bin:$PATH"

# --- Clone sources ----------------------------------------------------------
WORKDIR /home/mambauser/Code
RUN git clone --branch ${RMG_PY_BRANCH} https://github.com/ReactionMechanismGenerator/RMG-Py.git && \
    git clone --branch ${RMG_DATABASE_BRANCH} https://github.com/ReactionMechanismGenerator/RMG-database.git && \
    git clone --branch ${ARC_BRANCH} https://github.com/ReactionMechanismGenerator/ARC.git

# --- Create RMG conda environment ------------------------------------------
RUN micromamba create -y -n rmg_env -f /home/mambauser/Code/RMG-Py/environment.yml && \
    # Install modern juliacall (install_rms.sh still checks for it!)
    micromamba run -n rmg_env micromamba install -y -c conda-forge pyjuliacall && \
    micromamba clean --all --yes

# --- Build RMG‑Py (parallel) ------------------------------------------------
WORKDIR /home/mambauser/Code/RMG-Py
RUN micromamba run -n rmg_env make -j$(nproc)

# --- RMS / Julia integration -----------------------------------------------
RUN micromamba run -n rmg_env bash -c "\
    export current_env=rmg_env && \
    source install_rms.sh \
 "
# --- Quick sanity test (optional, can comment out to speed CI) --------------
RUN micromamba run -n rmg_env python rmg.py examples/rmg/minimal/input.py

# --- ARC environment --------------------------------------------------------
WORKDIR /home/mambauser/Code/ARC
# Copy ARC environment file from the local directory - Temporary solution
COPY ./environment.yml /home/mambauser/Code/ARC/environment.yml
RUN micromamba create -y -n arc_env \
      -c rmg -c conda-forge -c cantera \
      --channel-priority flexible \
      -f environment.yml \
 && micromamba run -n arc_env bash -euxo pipefail -c \
      "make compile && bash ./devtools/install_pyrdl.sh" \
 && micromamba clean --all --yes

# --- Strip cruft ------------------------------------------------------------
RUN rm -rf ~/.cache ~/.npm ~/.yarn /tmp/* && \
    find /opt/conda -name '*.a' -delete && \
    find /opt/conda -name '*.pyc' -delete && \
    find /opt/conda -name '__pycache__' -type d -exec rm -rf {} +

# ---------------------------------------------------------------------------
# Stage 2: Slim runtime image
# ---------------------------------------------------------------------------
FROM --platform=linux/amd64 mambaorg/micromamba:2.2.0-ubuntu24.04

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

# --- Copy artefacts from builder -------------------------------------------
COPY --from=builder --chown=user:mambauser /opt/conda /opt/conda
COPY --from=builder --chown=user:mambauser /home/mambauser/.juliaup /home/mambauser/.juliaup
COPY --from=builder --chown=user:mambauser /home/mambauser/Code /home/mambauser/Code

# --- Runtime env vars -------------------------------------------------------
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:/home/mambauser/.juliaup/bin:/home/mambauser/Code/RMG-Py:$PATH
ENV PYTHONPATH="/home/mambauser/Code/RMG-Py:/home/mambauser/Code/ARC"
ENV RMG_PY_DIR="/home/mambauser/Code/RMG-Py"
ENV ARC_DIR="/home/mambauser/Code/ARC"
ENV MAMBA_DOCKERFILE_ACTIVATE=1

# --- Entry wrapper ----------------------------------------------------------
COPY --chmod=755  dockerfiles/entrywrapper.sh  /usr/local/bin/entrywrapper.sh
COPY --chmod=644  dockerfiles/aliases.sh       /etc/profile.d/aliases.sh
COPY --chmod=755  dockerfiles/aliases_print.sh /usr/local/bin/aliases

ENTRYPOINT ["/usr/local/bin/entrywrapper.sh"]
