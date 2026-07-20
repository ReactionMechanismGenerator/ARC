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
   make install-rmg RMG_ARGS="--source --ssh"

Run this for the full list:

.. code-block:: bash

   devtools/install_rmg.sh --help

UMA Engine (Optional)
---------------------

ARC can use `UMA <https://github.com/facebookresearch/fairchem>`_, Meta FAIR's
``fairchem-core`` foundation machine-learned interatomic potential, as a fast local
engine for geometry optimization, frequencies, single points, hindered-rotor scans,
IRCs, and transition-state searches. Use ``method='uma'`` in a level of theory to
select it (it resolves to the latest UMA model implemented in ARC).

UMA runs in its own ``uma_env`` conda environment and is **not** installed by
``make install-all`` or in CI, because the model is gated behind a Meta license and a
HuggingFace token and is heavy to download. To set it up, run:

.. code-block:: bash

   make install-uma          # or: bash devtools/install_uma.sh

This creates ``uma_env`` (``fairchem-core`` + ``sella`` + ``ase``), verifies the
required imports, and walks you through the one-time HuggingFace login for the gated
model. Before running it, accept the model license at
https://huggingface.co/facebook/UMA (in a browser logged into HuggingFace) and create
a token with read access to gated repositories. To also run the UMA model-dependent
unit tests after installing, use ``bash devtools/install_uma.sh --test``.

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

Updating ARC
------------

ARC is updated frequently. Update regularly to enjoy new features and bug fixes.

.. note::

   If you change ARC's parameters within the repository rather than copies thereof
   (see the Personal Configuration section above), it is highly recommended to
   back up the files you manually changed before updating ARC. These are usually
   ``arc/settings/settings.py`` and ``arc/settings/submit.py``.

You can update ARC to a specific version, or to the most recent developer version.
To get the most recent developer version, do the following (change ``~/Path/to/ARC/``
accordingly):

.. code-block:: bash

   cd ~/Path/to/ARC/
   git stash
   git fetch origin
   git pull origin main
   git stash pop

The above will update your ``main`` branch of ARC.

To update to a specific version (e.g., version 1.1.0), do the following:

.. code-block:: bash

   cd ~/Path/to/ARC/
   git stash
   git fetch origin
   git checkout tags/1.1.0 -b v1.1.0
   git stash pop

The above creates a ``v1.1.0`` branch that replicates the stable 1.1.0 version.

This process might cause merge conflicts if the updated version changes a file you
changed locally. Although we try to avoid causing merge conflicts for ARC's users as
much as we can, it can still sometimes happen. You'll identify a merge conflict if git
prints a message similar to this:

.. code-block:: bash

   $ git merge BRANCH-NAME
   > Auto-merging settings.py
   > CONFLICT (content): Merge conflict in styleguide.md
   > Automatic merge failed; fix conflicts and then commit the result

Detailed steps to resolve a git merge conflict can be found `online`__.

__ mergeConflict_

Open the files that have merge conflicts and look for the following markings:

.. code-block:: text

   <<<<<<< HEAD
   this is some content introduced by updating ARC
   =======
   totally different content the user added, adding different changes
   to the same lines that were also updated remotely
   >>>>>>> new_branch_to_merge_later

Resolving a merge conflict consists of three stages:

- Determine which version of the code you'd like to keep (usually you should manually
  append your own changes to the more updated ARC code). Make the changes and get rid
  of the unneeded ``<<<<<<< HEAD``, ``=======``, and
  ``>>>>>>> new_branch_to_merge_later`` markings. Repeat for all conflicts.
- Stage the changes by typing ``git add .``.
- If you don't plan to commit your changes, unstage them by typing
  ``git reset --soft origin/main``.

.. include:: links.txt
