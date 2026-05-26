.. _installation:

Installation
============

ARC is normally installed from source into the Conda environment defined by the
repository. Install ARC on every machine that will execute ARC itself. Electronic
structure software can be local to that machine or available on remote servers
that ARC reaches over SSH.

ARC currently targets Python 3.14 through ``environment.yml`` and
``pyproject.toml``. Linux and macOS are the supported practical targets. Windows
is not expected to be smooth unless used through a Linux-like environment.

Quick Install
-------------

Clone ARC and create the environment:

.. code-block:: bash

   git clone https://github.com/ReactionMechanismGenerator/ARC.git
   cd ARC
   conda env create -f environment.yml
   conda activate arc_env

Compile ARC's Cython extensions:

.. code-block:: bash

   make compile

Run the unit tests when you want to verify the installation:

.. code-block:: bash

   make test

If you only need the Docker image, see :ref:`docker`.

Core Requirements
-----------------

The Conda environment installs ARC's Python dependencies, including RDKit,
OpenBabel bindings, Sphinx, pytest, and the packages needed by ARC's molecule
and scheduler layers.

Install system build tools before creating the environment if they are missing:

.. code-block:: bash

   sudo apt install git gcc g++ make

On macOS, install the equivalent compiler toolchain through Xcode command-line
tools or your normal package manager.

Optional External Tools
-----------------------

ARC can use several external projects and programs. Install only the ones you
need for your workflows and the electronic structure software you can actually
run.

ARC does not install commercial or third-party ESS binaries such as Gaussian,
ORCA, Q-Chem, Molpro, TeraChem, Psi4, or CFour. Those programs must already be
licensed where required, installed on the routed machine, and loadable from the
submit environment used by ARC.

Useful Make targets include:

.. code-block:: bash

   make install-rmg       # RMG-Py, RMG-database, and Arkane support
   make install-xtb       # xTB support
   make install-kinbot    # KinBot TS search support
   make install-gcn       # TS-GCN support
   make install-gcn-cpu   # CPU TS-GCN support
   make install-autotst   # AutoTST support
   make install-sella     # Sella support
   make install-torchani  # TorchANI support
   make install-ob        # OpenBabel support
   make install-all       # install the full external stack

The RMG installer accepts flags:

.. code-block:: bash

   make install-rmg
   make install-rmg RMG_ARGS=--source
   make install-rmg RMG_ARGS="--source --rms --ssh"

Run this for the full list:

.. code-block:: bash

   devtools/install_rmg.sh --help

Personal Configuration
----------------------

Do not edit repository defaults for routine use. Create a personal ``~/.arc``
directory and add short override files containing only the settings you need to
customize:

.. code-block:: bash

   mkdir -p ~/.arc
   touch ~/.arc/settings.py

Copy the full templates only when you want a starting point for site-specific
configuration:

.. code-block:: bash

   mkdir -p ~/.arc
   cp arc/settings/settings.py ~/.arc/settings.py
   cp arc/settings/submit.py ~/.arc/submit.py
   cp arc/settings/inputs.py ~/.arc/inputs.py

Most users should keep a short ``~/.arc/settings.py`` containing only changed
values, such as ``servers``, ``global_ess_settings``, ``levels_ess``, and job
defaults. Missing values fall back to ARC's repository defaults.

Remote Servers and SSH
----------------------

Use SSH keys for remote servers. A remote server entry needs:

* ``cluster_soft`` - one of the cluster software names configured in ARC;
* ``address`` - SSH hostname;
* ``un`` - username;
* ``key`` - path to the private SSH key on the machine running ARC.

Example:

.. code-block:: python

   servers = {
       'my_slurm': {
           'cluster_soft': 'Slurm',
           'address': 'login.cluster.edu',
           'un': 'my_user',
           'key': '/home/my_user/.ssh/id_rsa',
           'cpus': 32,
           'memory': 128,
       },
   }

Local and HPC Execution
-----------------------

If ARC runs on the same machine or login node that submits the quantum chemistry
jobs, define a server named ``local``. The name ``local`` is reserved.
In this context, ``local`` means "no SSH hop"; ARC still submits ESS jobs through
the configured scheduler unless your site-specific submit templates implement a
different execution path.

.. code-block:: python

   servers = {
       'local': {
           'cluster_soft': 'Slurm',
           'un': 'my_user',
           'cpus': 32,
           'memory': 128,
       },
   }

ARC supports scheduler definitions for OGE/SGE, Slurm, PBS, and HTCondor. If
your site uses different command paths, queue names, or submit script headers,
customize ``check_status_command``, ``submit_command``, ``delete_command``,
``submit_filenames``, and the templates in ``~/.arc/submit.py``.

Map Software to Servers
-----------------------

Tell ARC where electronic structure software is available using
``global_ess_settings``:

.. code-block:: python

   global_ess_settings = {
       'gaussian': ['local', 'my_slurm'],
       'orca': 'local',
       'qchem': 'my_slurm',
       'xtb': 'local',
   }

Per-project ``ess_settings`` in a YAML input or Python script overrides the
global mapping. If neither is provided, ARC can scan configured servers for
known software, but explicit mappings are more reproducible.

SSH and HPC Preflight
---------------------

Before running ARC on remote or scheduled resources, verify the pieces ARC will
depend on:

* passwordless SSH works from the ARC machine to each remote ``address``;
* the remote host key is already accepted in ``known_hosts``;
* scheduler commands such as ``sbatch``, ``qsub``, ``qstat``, ``squeue``, or
  ``condor_submit`` work in a normal shell;
* each routed ESS executable can be loaded by the submit script environment;
* scratch and project paths have enough space and write permissions;
* any required module loads, license variables, or environment setup commands
  are present in ``~/.arc/submit.py`` or the relevant site startup scripts.

Verify the Install
------------------

After activating ``arc_env``:

.. code-block:: bash

   make compile
   make test

For a smaller functional check, run a minimal YAML project from :ref:`examples`.
If you use remote servers, also verify SSH access and queue submission with your
cluster's native tools before asking ARC to submit jobs.

.. include:: links.txt
