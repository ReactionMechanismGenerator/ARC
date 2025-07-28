# Stage 1: RMG setup
# RMG Dockerfile
# The parent image is the base image that the Dockerfile builds upon.
# The RMG installation instructions suggest Anaconda for installation by source, however, we use micromamba for the Docker image due to its smaller size and less overhead.
# https://hub.docker.com/layers/mambaorg/micromamba/1.4.3-jammy/images/sha256-0c7c97be938c5522dcb9e1737bfa4499c53f6cf9e32e53897607a57ba8b148d5?context=explore
# We are using the sha256 hash to ensure that the image is not updated without our knowledge. It considered best practice to use the sha256 hash
FROM --platform=linux/amd64 mambaorg/micromamba@sha256:20fb02f2d1160265f7fabaf1601707a902ae65c6dc9e053d305441182450c368 AS rmg-stage

# Set the user as root
USER root

# Create a login user named rmguser
# Create a login user named rmguser
ARG NEW_MAMBA_USER=rmguser
ARG NEW_MAMBA_USER_ID=1000
ARG NEW_MAMBA_USER_GID=1000
RUN usermod "--login=${NEW_MAMBA_USER}" "--home=/home/${NEW_MAMBA_USER}" \
        --move-home "-u ${NEW_MAMBA_USER_ID}" "${MAMBA_USER}" && \
    groupmod "--new-name=${NEW_MAMBA_USER}" \
             "-g ${NEW_MAMBA_USER_GID}" "${MAMBA_USER}" && \
    echo "${NEW_MAMBA_USER}" > "/etc/arg_mamba_user" && \
    :

# Set the environment variables
ARG MAMBA_ROOT_PREFIX=/opt/conda
ENV MAMBA_USER=$NEW_MAMBA_USER
ENV BASE=$MAMBA_ROOT_PREFIX

# Install system dependencies
#
# List of deps and why they are needed:
#  - make, gcc, g++ for building RMG
#  - git for downloading RMG respoitories
#  - wget for downloading conda install script
#  - libxrender1 required by RDKit
# Clean up the apt cache to reduce the size of the image
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    make \
    libgomp1\
    libxrender1 \
    sudo \
    nano \
    && apt-get clean \
    && apt-get autoclean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && echo "${NEW_MAMBA_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# Change user to the non-root user
USER $MAMBA_USER

# Make directory for RMG-Py and RMG-database
RUN mkdir -p /home/rmguser/Code

# Change working directory to Code
WORKDIR /home/rmguser/Code

# ------------------------------------------------------------------ clone & checkout
WORKDIR /home/rmguser/Code
RUN git clone --filter=blob:none --no-checkout https://github.com/ReactionMechanismGenerator/RMG-Py.git RMG-Py \
 && git -C RMG-Py checkout --detach 55464c54d1fa61b531e865682df598d33718597d  \
 && git clone --filter=blob:none --depth 1 https://github.com/ReactionMechanismGenerator/RMG-database.git RMG-database




ENV PATH=/opt/conda/envs/rmg_env/bin:/home/rmguser/Code/RMG-Py:$PATH \
    PYTHONPATH=/home/rmguser/Code/RMG-Py


ENV JULIA_DEPOT_PATH="/home/rmguser/julia-ver/packages"
ENV JULIA_HISTORY=/home/rmguser/repl_history.jl
# Since 1.9.0 Julia, the CPU target is set to "native" by default. This is not ideal for a Docker image, so we set it to a list of common CPU targets
# This avoids the need to compile the Julia packages for the specific CPU architecture of the host machine
ENV JULIA_CPU_TARGET="x86-64,haswell,skylake,broadwell,znver1,znver2,znver3,cascadelake,icelake-client,cooperlake,generic,native"
# Install RMS
# The extra arguments are required to install PyCall and RMS in this Dockerfile. Will not work without them.
# Final command is to compile the RMS during Docker build - This will reduce the time it takes to run RMS for the first time
# Julia + PyCall + RMS in rmg_env
ENV CONDA_JL_CONDA_EXE=/opt/conda/envs/rmg_env/bin/conda
# ------------------------------------------------------------------ env create
WORKDIR /home/rmguser/Code/RMG-Py
RUN micromamba create -y -n rmg_env -f environment.yml \
    && touch /opt/conda/envs/rmg_env/condarc-julia.yml \
    && micromamba install -y -n rmg_env -c conda-forge conda \
    && micromamba clean -a -y \
    && echo "export PYTHONPATH=/home/rmguser/Code/RMG-Py" >> ~/.bashrc \
    && echo "export PATH=/home/rmguser/Code/RMG-Py:$PATH" >> ~/.bashrc \
    &&  micromamba run -n rmg_env make -j"$(nproc)" \
    && micromamba run -n rmg_env bash -lc "\
   julia -e 'ENV[\"CONDA_JL_CONDA_EXE\"]=\"${CONDA_JL_CONDA_EXE}\"; using Pkg; \
             Pkg.add(PackageSpec(name=\"PyCall\", rev=\"master\")); Pkg.build(\"PyCall\"); \
             Pkg.add(PackageSpec(name=\"ReactionMechanismSimulator\", url=\"https://github.com/ReactionMechanismGenerator/ReactionMechanismSimulator.jl\", rev=\"8dbb07eedebaacf9b9f77081623c572d054b9a6f\")); \
             using ReactionMechanismSimulator'; \
   python -c 'import julia, diffeqpy; julia.install(); diffeqpy.install()'; \
   python-jl -c 'from pyrms import rms' \
 " 

RUN micromamba run -n rmg_env python-jl /home/rmguser/Code/RMG-Py/rmg.py /home/rmguser/Code/RMG-Py/examples/rmg/minimal/input.py \
# delete the results, preserve input.py
&& mv /home/rmguser/Code/RMG-Py/examples/rmg/minimal/input.py /home/rmguser/Code/RMG-Py/examples/input.py \
&& rm -rf /home/rmguser/Code/RMG-Py/examples/rmg/minimal/* \
&& mv /home/rmguser/Code/RMG-Py/examples/input.py /home/rmguser/Code/RMG-Py/examples/rmg/minimal/input.py

# Installing ARC
# Change directory to Code
WORKDIR /home/rmguser/Code

# Clone main branch ARC repository from GitHub and set as working directory
RUN git clone -b main https://github.com/ReactionMechanismGenerator/ARC.git
WORKDIR /home/rmguser/Code/ARC

# Set environment variables for the Docker run and container
ENV PYTHONPATH="${PYTHONPATH}:/home/rmguser/Code/ARC"
ENV PYTHONPATH="${PYTHONPATH}:/home/rmguser/Code/AutoTST"
ENV PYTHONPATH="${PYTHONPATH}:/home/rmguser/Code/TS-GCN"
ENV PATH=/home/rmguser/Code/ARC:$PATH

# Install ARC Environment
COPY --chown=rmguser:rmguser ./environment.yml /home/rmguser/Code/ARC/environment.yml
RUN micromamba create -y -n arc_env -f environment.yml --channel-priority flexible && \
    micromamba clean --all -f -y && \
bash -euxo pipefail <<'EOF'
PYDIR=$(echo /opt/conda/envs/arc_env/lib/python*/site-packages)

# cache
rm -rf /home/rmguser/.cache/pip /home/rmguser/.cache/yarn

# strip compiled cruft
find "$PYDIR" -type f \( -name "*.a" -o -name "*.py[co]" \) -delete
find "$PYDIR" -type d -name "__pycache__" -prune -exec rm -rf {} +

# drop test directories in one traversal
find "$PYDIR" -type d \( -path "*/scipy/tests" -o -path "*/numpy/tests" -o -path "*/pandas/tests" \) \
     -prune -exec rm -rf {} +

# remove cython sources
find "$PYDIR" -name "*.pyx" -delete
rm -f "$PYDIR/uvloop/loop.c"
EOF

FROM --platform=linux/amd64 mambaorg/micromamba@sha256:20fb02f2d1160265f7fabaf1601707a902ae65c6dc9e053d305441182450c368 AS arc-stage
USER root
ARG NEW_MAMBA_USER=rmguser
ARG NEW_MAMBA_USER_ID=1000
ARG NEW_MAMBA_USER_GID=1000
RUN usermod "--login=${NEW_MAMBA_USER}" "--home=/home/${NEW_MAMBA_USER}" \
        --move-home "-u ${NEW_MAMBA_USER_ID}" "${MAMBA_USER}" && \
    groupmod "--new-name=${NEW_MAMBA_USER}" \
             "-g ${NEW_MAMBA_USER_GID}" "${MAMBA_USER}" && \
    echo "${NEW_MAMBA_USER}" > "/etc/arg_mamba_user" && \
    :

# Set the environment variables
ARG MAMBA_ROOT_PREFIX=/opt/conda
ENV MAMBA_USER=$NEW_MAMBA_USER
ENV BASE=$MAMBA_ROOT_PREFIX

# Install system dependencies
#
# List of deps and why they are needed:
#  - make, gcc, g++ for building RMG
#  - git for downloading RMG respoitories
#  - wget for downloading conda install script
#  - libxrender1 required by RDKit
# Clean up the apt cache to reduce the size of the image
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    make \
    libgomp1\
    libxrender1 \
    sudo \
    nano \
    && apt-get clean \
    && apt-get autoclean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && echo "${NEW_MAMBA_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && printf '\n# print cheat-sheet once per interactive Bash\nif [[ $- == *i* && $SHLVL -eq 1 ]]; then\n  aliases\nfi\n' \
        >> /etc/bash.bashrc
# Change user to the non-root user
USER $MAMBA_USER

# Make directory for RMG-Py and RMG-database
RUN mkdir -p /home/rmguser/Code

COPY --from=rmg-stage --chown=rmguser:rmguser /opt/conda /opt/conda
COPY --from=rmg-stage --chown=rmguser:rmguser /home/rmguser/Code/RMG-Py /home/rmguser/Code/RMG-Py
COPY --from=rmg-stage --chown=rmguser:rmguser /home/rmguser/Code/RMG-database /home/rmguser/Code/RMG-database
COPY --from=rmg-stage --chown=rmguser:rmguser /home/rmguser/Code/ARC /home/rmguser/Code/ARC


ENV PYTHONPATH="${PYTHONPATH}:/home/rmguser/Code/ARC"
ENV PYTHONPATH="${PYTHONPATH}:/home/rmguser/Code/AutoTST"
ENV PYTHONPATH="${PYTHONPATH}:/home/rmguser/Code/TS-GCN"
ENV PATH=/home/rmguser/Code/ARC:/home/rmguser/Code/RMG-Py:/home/rmguser/Code/RMG-database:$PATH
ENV PYTHONPATH=/home/rmguser/Code/ARC:/home/rmguser/Code/RMG-Py:/home/rmguser/Code/RMG-database:$PYTHONPATH

ENV JULIA_DEPOT_PATH="/home/rmguser/julia-ver/packages"
ENV JULIA_HISTORY=/home/rmguser/repl_history.jl
# Since 1.9.0 Julia, the CPU target is set to "native" by default. This is not ideal for a Docker image, so we set it to a list of common CPU targets
# This avoids the need to compile the Julia packages for the specific CPU architecture of the host machine
ENV JULIA_CPU_TARGET="x86-64,haswell,skylake,broadwell,znver1,znver2,znver3,cascadelake,icelake-client,cooperlake,generic"
# Install RMS
# The extra arguments are required to install PyCall and RMS in this Dockerfile. Will not work without them.
# Final command is to compile the RMS during Docker build - This will reduce the time it takes to run RMS for the first time
# Julia + PyCall + RMS in rmg_env
ENV CONDA_JL_CONDA_EXE=/opt/conda/envs/rmg_env/bin/conda

WORKDIR /home/rmguser/

RUN mkdir -p /home/rmguser/.arc && \
    cp /home/rmguser/Code/ARC/arc/settings/settings.py /home/rmguser/.arc/settings.py && \
    cp /home/rmguser/Code/ARC/arc/settings/submit.py /home/rmguser/.arc/submit.py

# Copy alias_print.sh and entrywrapper.sh to the container
# Copy your login‐wide aliases into /etc/profile.d
COPY --chown=rmguser:rmguser dockerfiles/aliases.sh /etc/profile.d/99-rmg-aliases.sh

# Copy the cheat‐sheet and entrypoint
COPY --chown=rmguser:rmguser dockerfiles/aliases_print.sh /usr/local/bin/aliases
COPY --chown=rmguser:rmguser dockerfiles/entrywrapper.sh  /home/rmguser/entrywrapper.sh

# Fix permissions & make the scripts executable & ensure rms is run once
RUN chmod 644 /etc/profile.d/99-rmg-aliases.sh \
 && chmod +x /home/rmguser/entrywrapper.sh \
 && chmod +x /usr/local/bin/aliases \
    && micromamba run -n rmg_env python-jl /home/rmguser/Code/RMG-Py/rmg.py /home/rmguser/Code/RMG-Py/examples/rmg/minimal/input.py \
    # delete the results, preserve input.py
    && mv /home/rmguser/Code/RMG-Py/examples/rmg/minimal/input.py /home/rmguser/Code/RMG-Py/examples/input.py \
    && rm -rf /home/rmguser/Code/RMG-Py/examples/rmg/minimal/* \
    && mv /home/rmguser/Code/RMG-Py/examples/input.py /home/rmguser/Code/RMG-Py/examples/rmg/minimal/input.py

# Use a login shell so /etc/profile and /etc/profile.d/* are sourced automatically
ENTRYPOINT ["bash","-l","/home/rmguser/entrywrapper.sh"]

# Activate the ARC environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV ENV_NAME=arc_env
