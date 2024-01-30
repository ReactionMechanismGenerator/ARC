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
    && rm -rf /var/lib/apt/lists/*\
    && echo "${NEW_MAMBA_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# Change user to the non-root user
USER $MAMBA_USER

# Make directory for RMG-Py and RMG-database
RUN mkdir -p /home/rmguser/Code

# Change working directory to Code
WORKDIR /home/rmguser/Code

# Clone the RMG base and database repositories. The pulled branches are only the main branches.
RUN git clone -b main https://github.com/ReactionMechanismGenerator/RMG-Py.git \ 
    && git clone -b main https://github.com/ReactionMechanismGenerator/RMG-database.git

# cd into RMG-Py
WORKDIR /home/rmguser/Code/RMG-Py

# Install RMG-Py and then clean up the micromamba cache
RUN micromamba create -y -f environment.yml && \
    micromamba install -n rmg_env -c conda-forge conda && \
    micromamba clean --all -f -y

# Activate the RMG environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV ENV_NAME=rmg_env

# Set environment variables
# These need to be set in the Dockerfile so that they are available to the build process
ENV PATH /opt/conda/envs/rmg_env/bin:$PATH
ENV PYTHONPATH /home/rmguser/Code/RMG-Py:$PYTHONPATH
ENV PATH /home/rmguser/Code/RMG-Py:$PATH

# Build RMG
RUN make \
    && echo "export PYTHONPATH=/home/rmguser/Code/RMG-Py" >> ~/.bashrc \
    && echo "export PATH=/home/rmguser/Code/RMG-Py:$PATH" >> ~/.bashrc

# Install RMS
# The extra arguments are required to install PyCall and RMS in this Dockerfile. Will not work without them.
# Final command is to compile the RMS during Docker build - This will reduce the time it takes to run RMS for the first time
RUN touch /opt/conda/envs/rmg_env/condarc-julia.yml
RUN CONDA_JL_CONDA_EXE=/bin/micromamba julia -e 'ENV["CONDA_JL_CONDA_EXE"]="/opt/conda/envs/rmg_env/bin/conda";using Pkg;Pkg.add(PackageSpec(name="PyCall", rev="master")); Pkg.build("PyCall"); Pkg.add(PackageSpec(name="ReactionMechanismSimulator", rev="main"))' \
    && python -c "import julia; julia.install(); import diffeqpy; diffeqpy.install()"

RUN python-jl /home/rmguser/Code/RMG-Py/rmg.py /home/rmguser/Code/RMG-Py/examples/rmg/minimal/input.py \
    # delete the results, preserve input.pyז
    && mv /home/rmguser/Code/RMG-Py/examples/rmg/minimal/input.py /home/rmguser/Code/RMG-Py/examples/input.py \
    && rm -rf /home/rmguser/Code/RMG-Py/examples/rmg/minimal/* \
    && mv /home/rmguser/Code/RMG-Py/examples/input.py /home/rmguser/Code/RMG-Py/examples/rmg/minimal/input.py

# Add alias to bashrc - rmge to activate the environment
# These commands are not necessary for the Docker image to run, but they are useful for the user
RUN echo "alias rmge='micromamba activate rmg_env'" >> ~/.bashrc \
    && echo "alias arce='micromamba activate arc_env'" >> ~/.bashrc \
    && echo "alias rmg='python-jl /home/rmguser/Code/RMG-Py/rmg.py input.py'" >> ~/.bashrc \
    && echo "alias deact='micromamba deactivate'" >> ~/.bashrc \
    && echo "export rmgpy_path='/home/rmguser/Code/RMG-Py/'" >> ~/.bashrc \
    && echo "export rmgdb_path='/home/rmguser/Code/RMG-database/'" >> ~/.bashrc \
    && echo "alias rmgcode='cd \$rmgpy_path'" >> ~/.bashrc \
    && echo "alias rmgdb='cd \$rmgdb_path'" >> ~/.bashrc \
    && echo "alias arcode='cd /home/rmguser/Code/ARC'" >> ~/.bashrc \
    && echo "alias conda='micromamba'" >> ~/.bashrc \
    && echo "alias mamba='micromamba'" >> ~/.bashrc

FROM rmg-stage AS arc-stage

# Log in as rmguser
USER rmguser

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
ENV PATH /home/rmguser/Code/ARC:$PATH

# Install ARC Environment
RUN micromamba create -y -f environment.yml && \
    micromamba clean --all -f -y && \
    rm -rf /home/rmguser/.cache/yarn \
    rm -rf /home/rmguser/.cache/pip &&\
    rm -rf /home/rmguser/.cache/pip && \
    find -name '*.a' -delete && \
    find -name '*.pyc' -delete && \
    find -name '__pycache__' -type d -exec rm -rf '{}' '+' && \
    find /opt/conda/envs/arc_env/lib/python3.7/site-packages/scipy -name 'tests' -type d -exec rm -rf '{}' '+' && \
    find /opt/conda/envs/arc_env/lib/python3.7/site-packages/numpy -name 'tests' -type d -exec rm -rf '{}' '+' && \
    find /opt/conda/envs/arc_env/lib/python3.7/site-packages/pandas -name 'tests' -type d -exec rm -rf '{}' '+' && \
    find /opt/conda/envs/arc_env/lib/python3.7/site-packages -name '*.pyx' -delete && \
    rm -rf /opt/conda/envs/arc_env/lib/python3.7/site-packages/uvloop/loop.c &&\
    make clean

WORKDIR /home/rmguser/

RUN mkdir -p /home/rmguser/.arc && \
    cp /home/rmguser/Code/ARC/arc/settings/settings.py /home/rmguser/.arc/settings.py && \
    cp /home/rmguser/Code/ARC/arc/settings/submit.py /home/rmguser/.arc/submit.py

# Copy alias_print.sh and entrywrapper.sh to the container
COPY --chown=rmguser:rmguser ./dockerfiles/alias_print.sh /home/rmguser/alias_print.sh
COPY --chown=rmguser:rmguser ./dockerfiles/entrywrapper.sh /home/rmguser/entrywrapper.sh

# Make the scripts executable
RUN chmod +x /home/rmguser/alias_print.sh \
    && chmod +x /home/rmguser/entrywrapper.sh

# Set the wrapper script as the entrypoint
ENTRYPOINT ["/home/rmguser/entrywrapper.sh"]

# Activate the ARC environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV ENV_NAME=arc_env
