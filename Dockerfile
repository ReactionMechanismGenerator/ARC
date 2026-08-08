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
    git gcc g++ make wget libxrender1 ca-certificates nano && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /home/mambauser/Code && \
    chown -R mambauser:mambauser /home/mambauser && \
    chown -R mambauser:mambauser /opt/conda && \
    chown -R mambauser:mambauser /home/mambauser

# Change to unprivileged user
USER mambauser
ENV MAMBA_USER=mambauser

# This image ships no .condarc, so the `conda` that gets installed into rmg_env below starts with
# an empty channel list. RMG's install_rms.sh runs `conda install 'conda-forge::pyjuliacall<0.9.35'`,
# which then dies with NoChannelsConfiguredError (non-fatally) and silently drops that pin.
RUN printf 'channels:\n  - conda-forge\n' > /home/mambauser/.condarc

# Set JuliaUp PATH and install Julia 1.10 as req. by RMG (mirrors RMG-Py's own Dockerfile)
ENV PATH="/home/mambauser/.juliaup/bin:$PATH"
RUN wget -qO- https://install.julialang.org | sh -s -- --yes --default-channel 1.10 && \
    juliaup add 1.10 && \
    juliaup default 1.10 && \
    juliaup list && \
    rm -rf /home/mambauser/.juliaup/downloads /home/mambauser/.juliaup/tmp

# Switch directory to Code and RMG clone
WORKDIR /home/mambauser/Code
RUN git clone --depth 1 --branch ${RMG_PY_BRANCH} https://github.com/ReactionMechanismGenerator/RMG-Py.git && \
    git clone --depth 1 --branch ${RMG_DATABASE_BRANCH} https://github.com/ReactionMechanismGenerator/RMG-database.git && \
    git clone --depth 1 --branch ${ARC_BRANCH} https://github.com/ReactionMechanismGenerator/ARC.git

# Record the resolved commit SHAs of the three clones. A pulled image is otherwise
# only identifiable by branch name; this file states exactly which revisions are
# baked in (and stays truthful even when the clone layer comes from the build cache).
RUN { \
      echo "ARC_BRANCH=${ARC_BRANCH}"; \
      echo "ARC_COMMIT=$(git -C /home/mambauser/Code/ARC rev-parse HEAD)"; \
      echo "RMG_PY_BRANCH=${RMG_PY_BRANCH}"; \
      echo "RMG_PY_COMMIT=$(git -C /home/mambauser/Code/RMG-Py rev-parse HEAD)"; \
      echo "RMG_DATABASE_BRANCH=${RMG_DATABASE_BRANCH}"; \
      echo "RMG_DATABASE_COMMIT=$(git -C /home/mambauser/Code/RMG-database rev-parse HEAD)"; \
    } > /home/mambauser/Code/image_versions.env

# Create RMG-Py environment (split into separate layers for GHA cache)
# Pin python=3.11 to drastically reduce solver search space
# pyjuliacall is pinned to <0.9.35 to match RMG's install_rms.sh: from 0.9.35 juliacall derives
# Julia's BINDIR from dirname(exe) instead of asking Julia for Sys.BINDIR, which breaks when the
# exe is the juliaup launcher shim (it looks for <shim dir>/../lib/julia/sys.so, which never exists).
RUN micromamba create -y -v -n rmg_env python=3.11 -f /home/mambauser/Code/RMG-Py/environment.yml && \
    micromamba run -n rmg_env micromamba install -y -v -c conda-forge "pyjuliacall<0.9.35" conda && \
    micromamba clean --all --yes

RUN micromamba run -n rmg_env make -C /home/mambauser/Code/RMG-Py -j"$(nproc)"

# Julia/PythonCall wiring, as done in RMG-Py's Dockerfile. install_rms.sh exports most of these
# itself, but only from the point it starts running; RMS_INSTALLER/RMS_BRANCH are inputs it reads
# rather than sets, and JULIA_CPU_TARGET it never touches at all.
# JULIA_CPU_TARGET yields portable multi-architecture pkgimages: this image is built on
# a GitHub Actions runner but runs on arbitrary user hardware.
# Syntax: ';' separates targets, ',' separates features/flags WITHIN one target. The value was
# comma-separated, so Julia read it as a single x86-64 target plus a list of bogus features, logged
# "'+haswell' is not a recognized feature for this target (ignoring feature)" once per name, and
# baked one generic pkgimage instead of eleven. The target list is otherwise RMG-Py's own, kept
# deliberately identical to it - the same typo is at RMG-Py/Dockerfile:78 and is being fixed
# upstream. No 'clone_all': LLVM's heuristic decides what to specialize, which is the Julia default
# and keeps build time and image size down. This image is not run on our HPC nodes, so it carries no
# targets chosen for them. znver3 is in any case the newest AMD target Julia 1.10 (LLVM 15) accepts
# - znver4/znver5 are an "Invalid CPU name" hard error - and Zen 4/Zen 5 load znver3 as a superset.
ENV JULIA_CPU_TARGET="x86-64;haswell;skylake;broadwell;znver1;znver2;znver3;cascadelake;icelake-client;cooperlake;generic"
ENV JULIA_CONDAPKG_BACKEND=Null
ENV JULIA_PYTHONCALL_EXE=/opt/conda/envs/rmg_env/bin/python
ENV PYTHON_JULIAPKG_PROJECT=/opt/conda/envs/rmg_env/julia_env
ENV RMS_INSTALLER=continuous
ENV RMS_BRANCH=for_rmg

RUN micromamba run -n rmg_env bash -c "\
      cd /home/mambauser/Code/RMG-Py && \
      source install_rms.sh \
    "

WORKDIR /home/mambauser/Code/ARC
RUN micromamba create -y -v -n arc_env python=3.14 -c conda-forge -c danagroup -f environment.yml && \
    micromamba install -y -v -n arc_env -c conda-forge pytest && \
    micromamba clean --all -f -y

RUN micromamba run -n arc_env bash -euxo pipefail -c \
      "make compile" && \
    micromamba clean --all --yes

# Stage 2: Final image
# The final image is based on the same micromamba image, but we copy over the installed RMG and ARC from the builder stage.
# This keeps the final image size smaller and avoids unnecessary layers.
FROM --platform=linux/amd64 mambaorg/micromamba:2.2-ubuntu24.04

# --- OCI image metadata -----------------------------------------------------
# ARGs do not cross stages, so ARC_BRANCH is re-declared here.
# Docker cannot set a LABEL from the output of a RUN, so the *resolved* commit
# SHAs live in /home/mambauser/Code/image_versions.env inside the image (printed
# by the `aliases` helper). Pass --build-arg ARC_COMMIT=<sha> to also surface the
# ARC revision as a label; CI does this.
ARG ARC_BRANCH=main
ARG ARC_COMMIT=unknown
LABEL org.opencontainers.image.title="ARC" \
      org.opencontainers.image.description="Automated Rate Calculator (ARC), with RMG-Py, RMG-database and Arkane preinstalled" \
      org.opencontainers.image.source="https://github.com/ReactionMechanismGenerator/ARC" \
      org.opencontainers.image.documentation="https://reactionmechanismgenerator.github.io/ARC/" \
      org.opencontainers.image.vendor="Reaction Mechanism Generator" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${ARC_BRANCH}" \
      org.opencontainers.image.revision="${ARC_COMMIT}"

ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:/home/mambauser/.juliaup/bin:/home/mambauser/Code/RMG-Py:/home/mambauser/Code/ARC:$PATH
ENV PYTHONPATH="/home/mambauser/Code/RMG-Py:/home/mambauser/Code/ARC"
ENV RMG_PY_DIR="/home/mambauser/Code/RMG-Py"
ENV ARC_DIR="/home/mambauser/Code/ARC"
ENV MAMBA_DOCKERFILE_ACTIVATE=1

# Same Julia/PythonCall wiring as the builder, so RMS loads at runtime without re-resolving.
# install_rms.sh also persisted four of these onto rmg_env via `conda env config vars set`, and
# those env-level vars take PRECEDENCE over the image ENV below: `micromamba run -n rmg_env` warns
# "Overwriting variables: JULIA_CONDAPKG_BACKEND,JULIA_PYTHONCALL_EXE,PYTHON_JULIAPKG_EXE,
# PYTHON_JULIAPKG_PROJECT" and substitutes the recorded values. The two agree, so this is a no-op
# there; the image ENV is what covers callers that bypass `micromamba run` (e.g. invoking
# /opt/conda/envs/rmg_env/bin/python by absolute path). JULIA_CPU_TARGET is not among the recorded
# vars, so the image ENV is its only source. Must stay byte-identical to the builder stage's value
# above (see there for the ';' vs ',' syntax), or a runtime recompile would target a different set.
ENV JULIA_CPU_TARGET="x86-64;haswell;skylake;broadwell;znver1;znver2;znver3;cascadelake;icelake-client;cooperlake;generic"
ENV JULIA_CONDAPKG_BACKEND=Null
ENV JULIA_PYTHONCALL_EXE=/opt/conda/envs/rmg_env/bin/python
ENV PYTHON_JULIAPKG_PROJECT=/opt/conda/envs/rmg_env/julia_env
ENV PYTHON_JULIAPKG_EXE=/home/mambauser/.juliaup/bin/julia
# juliaup resolves its channel config from $HOME/.julia/juliaup/juliaup.json, and `docker exec` (or
# any --entrypoint override) lands on root with HOME=/root rather than going through
# entrywrapper.sh's `runuser -u mambauser`. juliaup then finds no config, falls back to the
# `release` channel, and downloads a newer Julia over the network - silently bypassing the pinned
# 1.10 and every pkgimage baked above, or hard-failing when the host is offline. Pinning the depot
# explicitly makes resolution HOME-independent. Note this is JULIAUP_DEPOT_PATH, not
# JULIA_DEPOT_PATH; juliaup does not read the latter for channel lookup.
ENV JULIAUP_DEPOT_PATH=/home/mambauser/.julia

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
# Carry the builder's channel config over, so `conda install` inside the running container does not
# hit the same NoChannelsConfiguredError that silently defeated install_rms.sh's pyjuliacall pin.
COPY --from=builder --chown=mambauser:mambauser /home/mambauser/.condarc /home/mambauser/.condarc
COPY --from=builder --chown=mambauser:mambauser /home/mambauser/.juliaup /home/mambauser/.juliaup
# ~/.juliaup only holds the juliaup launcher and its config; the Julia toolchain itself and the
# Julia depot (RMS and its dependencies, plus their precompilation cache) live in ~/.julia.
COPY --from=builder --chown=mambauser:mambauser /home/mambauser/.julia /home/mambauser/.julia
COPY --from=builder --chown=mambauser:mambauser /home/mambauser/Code /home/mambauser/Code

# Need to copy the tests separately as they are not in the ARC git by default
COPY --chown=mambauser:mambauser dockerfiles/docker_tests /home/mambauser/Code/ARC/dockerfiles/docker_tests

# --- Entry wrapper ----------------------------------------------------------
COPY --chmod=755  dockerfiles/entrywrapper.sh  /usr/local/bin/entrywrapper.sh
COPY --chmod=644  dockerfiles/aliases.sh       /etc/profile.d/aliases.sh
COPY --chmod=755  dockerfiles/job_helpers.sh   /usr/local/bin/arc_job_helpers.sh
COPY --chmod=755  dockerfiles/aliases_print.sh /usr/local/bin/aliases
RUN touch /home/mambauser/.bashrc && \
    grep -qxF 'source /etc/profile.d/aliases.sh' /home/mambauser/.bashrc || \
    echo 'source /etc/profile.d/aliases.sh' >> /home/mambauser/.bashrc

# Land interactive shells in arc_env instead of base (base has no python at all).
# The micromamba base image's ~/.bashrc hook activates ${ENV_NAME:-base}; this is
# the micromamba equivalent of RMG's `sed -i 's/conda activate base/.../' ~/.bashrc`.
# It must come after /opt/conda is copied in, otherwise the RUN steps above would
# try to activate an environment that does not exist yet. Non-interactive shells
# never source ~/.bashrc, and entrywrapper.sh names its environment explicitly with
# `micromamba run -n ...`, so the entrypoint path is unaffected. `rmge` still switches
# to rmg_env, `arce` back to arc_env.
ENV ENV_NAME=arc_env
ENV ARC_IMAGE_VERSIONS=/home/mambauser/Code/image_versions.env

USER root
ENTRYPOINT ["/usr/local/bin/entrywrapper.sh"]
