.. _installation:

Installation instructions
=========================

ARC can be installed on a server, as well as on your local desktop / laptop, submitting jobs to your server(s).
The instructions below make this differentiation when relevant (the only difference is that ARC should be "aware"
of software installed on the same machine, where the communication isn't done via SSH).

Note:
    ARC was only tested on Linux (Ubuntu_ 18.04.1 and 20.04 LTS) and Mac machines.
    We don't expect it to work smoothly on Windows machines.

Note:
    These installation instructions assume you already have access to a server with a cluster scheduling software
    (ARC currently supports SGE, Slurm, PBS, and HTCondor) and with properly installed electronic structure software
    (ARC currently supports Gaussian, QChem, Molpro, TeraChem, Orca, and Psi4). We further assume that you have
    some experience working with the server, e.g., writing appropriate submit scripts (example scripts are given,
    but modifications are usually required).

.. _path:

Clone and setup path
^^^^^^^^^^^^^^^^^^^^

- Download and install the `Anaconda Python Platform`__ for Python 3.7 or higher if you haven't already.
- Get git and appropriate compilers if you don't have them already by typing ``sudo apt install git gcc g++ make``
  in a terminal.
- Clone ARC's repository to by typing the following command in the desired folder (e.g., under `~/Code/`)::

    git clone https://github.com/ReactionMechanismGenerator/ARC.git

- Add ARC to your local path in ``.bashrc`` (make sure to change ``~/Path/to/ARC/`` accordingly)::

    export PYTHONPATH=$PYTHONPATH:~/Path/to/ARC/

__ anaconda_


.. _arce:

Install dependencies
^^^^^^^^^^^^^^^^^^^^

- Create the Anaconda environment for ARC (after changing the directory to the
  installation folder by, e.g., ``cd ~/Code/ARC/``)::

    conda env create -f environment.yml

  Activate the ARC environment every time before you run ARC::

     conda activate arc_env

- Install the latest **DEVELOPER** version of RMG (which has Arkane).
  It is recommended to follow RMG's `Developer installation by source using Anaconda
  <http://reactionmechanismgenerator.github.io/RMG-Py/users/rmg/installation/index.html
  #for-developers-installation-by-source-using-anaconda-environment>`_ instructions.
  Make sure to add RMG-Py to your PATH and PYTHONPATH variables as explained in RMG's documentation.
- Type ``make install-all`` under the ARC repository folder to install the following 3rd party repositories:
  `AutoTST <https://github.com/ReactionMechanismGenerator/AutoTST>`_
  (`West et al. <https://doi.org/10.1021/acs.jpca.7b07361>`_),
  `KinBot <https://github.com/zadorlab/KinBot>`_
  (`Van de Vijver et al. <https://doi.org/10.1016/j.cpc.2019.106947>`_),
  and `TS-GCN <https://github.com/ReactionMechanismGenerator/TS-GCN.git>`_
  (`Pattanaik et al. <https://chemrxiv.org/articles/Genereting_Transition_States_of_Isomerization_Reactions_with_Deep_Learning/12302084>`_).
- Test ARC by typing ``make test`` under the ARC folder after activating the anaconda `arc_env` environment.


Create a ``.arc`` folder (optional but recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users are encouraged to create a ``.arc`` folder under their ``HOME`` folder on the machine running ARC.
Copy (and modify as appropriate, see below) the following python files
from the ARC repository into the newly created folder:
``<base_folder>/ARC/arc/settings/settings.py`` --> ``HOME/.arc/settings.py``
``<base_folder>/ARC/arc/settings/input.py`` --> ``HOME/.arc/input.py``
``<base_folder>/ARC/arc/settings/submit.py`` --> ``HOME/.arc/submit.py``

By doing this, ARC will use the respective settings and definitions from the files under ``.arc``
to override its defaults. Users many (carefully) modify the definitions in the local files
as appropriate. Note that you may choose to copy only some of these files, in which case the
definitions from any non-copied files will be taken from ARC's defaults (e.g., most users will
not need to modify ``input.py``).  Note also that definitions within these files may be partial
(i.e., you may keep only those parameters you may wish to change within each file), and that any
missing parameter will be assigned its default value from ARC's defaults. It is recommended to
only keep parameters you modified in ``settings.py``, so that other parameters will be updated
as you update the ARC repository in the future. This recommendation helps avoid merge conflicts
when updating ARC, and allows a single ARC instance on a server to be used by different users
with different preferences.

Principally, ARC would also work fine if users directly change the respective files within ARC's repository
instead of making copies. However, modifying the files in ARC directly may cause merging conflicts when
updating ARC. The down side is that users are responsible to keep their copies up to date with ARC's format
if major changes are made. Such changes will be listed under the Release Notes and will result in an increase
of the MINOR version number (i.e., ,major.MINOR.patch, e.g., `1.1.5` --> `1.2.0`).


Generating RSA SSH keys and defining servers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first two directives are only required if you'd like ARC to access remote servers
(ARC could also run "locally" when it's installed on a server).

- Generate RSA SSH keys for your favorite server(s) on which relevant electronic structure software
  (ESS, e.g., Gaussian) are installed. `Instructions for generating RSA keys could be found here
  <https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys--2>`_.
- Copy the RSA SSH key path(s) on your local machine to ``settings.py`` in the ``servers``
  dictionary under keys.
- Update the ``servers`` dictionary in your copy of ARC's `settings.py
  <https://github.com/ReactionMechanismGenerator/ARC/blob/main/arc/settings/settings.py>`_.

  * **A local server must be named with the reserved keyword ``local``**.
    ``cluster_soft`` and username (``un``) are mandatory.
  * A remote server has no limitations for naming. ``cluster_soft``, ``address``, username (``un``), and ``key``
    (the path to the local RSA SSH key) are mandatory.
  * Optional parameters for both local and remote servers are ``cpus``, ``memory``, and ``path``.
    The first two parameters stand for the maximum amount of cpu cores and memory in GB available on a node.
    If a job crashes due to cpu or memory issues, ARC will automatically re-run the job with different cpu
    and memory allocations within the specified limitations. By default, ``cpus`` is 8 and ``memory`` is 14 GB.
    The optional ``path`` key is used if it is necessary to direct ARC to create project files in a specific
    location on the server. E.g., if the value ``/storage/group_name/`` is given for ``path``, ARC will
    create project files under ``/storage/group_name/$USER/runs/ARC_Projects/project_name/``.
  * Although ARC currently does not allocate computing resources dynamically based on system size or ESS,
    the user can manually control memory specifications for each project.
    See :ref:`Advanced Features <advanced>` for details.
  * A default ESS job in ARC has 14 GB of memory, 8 cpu cores, and 120 hours of maximum execution time.
    The default settings can be changed by providing different values to the ``job_total_memory_gb``,
    ``job_cpu_cores``, and ``job_time_limit_hrs`` keys in the ``default_job_settings`` dictionary under ``settings.py``.
  * ARC will alter job memory, cpu, and time settings when troubleshoot jobs crashed due to resource allocation issues.
    The ``job_max_server_node_memory_allocation`` key stands for the maximum percentage of total node memory ARC will
    use when troubleshoot a job. The default value is 80%.

- Update the submit scripts in your copy of ARC's `submit.py
  <https://github.com/ReactionMechanismGenerator/ARC/blob/main/arc/settings/submit.py>`_
  according to your servers' definitions.
  * See the given template examples, and follow the structure of nested dictionaries (by server name, then by ESS name).
  * Preserve the variables in curly braces (e.g., ``{memory}``), so that ARC is able to auto-fill them.


Associating software with servers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ARC keeps track of software location on servers using a Python dictionary associating the different software (keys)
with the servers they are installed on (values). The server name must be consistent with the respective definition
in the ``servers`` dictionary mentioned above. Typically, you would update the ``global_ess_settings`` dictionary in
your copy of ARC's
`settings.py <https://github.com/ReactionMechanismGenerator/ARC/blob/main/arc/settings/settings.py>`_
to reflect your software and servers, for example::

  global_ess_settings = {
      'gaussian': ['server1', 'server2'],
      'molpro': 'server2',
      'qchem': 'local',
  }

Note that the above example reflects a situation where QChem in installed on the same machine as ARC, while Gaussian
and Molpro are installed on different servers ARC has access to. You can of course make any combination as you'd like.
The servers can be listed as a simple string for a single server, or as a list for multiple servers, where relevant.

These global settings are used by default unless ARC is given an ``ess_settings`` dictionary through an input file
or the API, thus allowing more flexibility when running several instances of ARC simultaneously (e.g., if Gaussian is
installed on two servers, where one has more memory in its nodes, the user can request ARC to use that specific server
for the more memory-intensive jobs). More about the ``ess_settings`` dictionary can be found in the
:ref:`Advanced Features <advanced>` section of the documentation.

If neither ``global_ess_settings`` (in settings.py) nor ``ess_settings`` (via an input file or the API) are specified,
ARC will use its "radar" feature to "scans" the servers it has access to, and assign relevant ESS it is familiar with
to the respective server. In order for this feature to function properly, make sure your .bashrc file on the remote
server\s does not have an interactive shell check. If it does, disable it.

It is recommended, though, to use the ``global_ess_settings`` and/or ``ess_settings`` dictionaries rather than allowing
the "radar" to do its thing blindly. The "radar" feature, however, is very useful for diagnostics
(see Tests_ below).

You can check what the "radar" detects using the ARC ESS diagnostics notebook.


Cluster software definitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ARC supports Slurm and Oracle/Sun Grid Engine (OGE / SGE). If you're using other `cluster software`__, or if your
server's definitions are different that ARC's, you should also modify the following variables in your copy of ARC's
`settings.py <https://github.com/ReactionMechanismGenerator/ARC/blob/main/arc/settings/settings.py>`_:

- ``check_status_command``
- ``submit_command``
- ``delete_command``
- ``list_available_nodes_command``
- ``submit_filename``
- ``t_max_format``

__ cluster_

You will find the values for ``check_status_command``, ``submit_command``, ``delete_command``, and
``list_available_nodes_command`` by typing on the respective server the `which` command, e.g.::

  which sbatch

If you have different servers with the same cluster software that have different cluster software definitions, just name
them differently, e.g., `Slurm1` and `Slurm2`, and make sure to pair them accordingly under the ``servers`` dictionary.


.. _Tests:

Tests
^^^^^

- If you'd like to make sure ARC has access to your servers and recognises your ESS, use the "radar" tool, available
  as an iPython notebook (see :ref:`Standalone tools <tools>`).
- Run the minimal example (see :ref:`Examples <examples>`), and a couple more examples, if you'd like, using both input files
  and the API (via iPython notebooks or any other method).
- Run ARC's unit tests. Note that for all tests to pass, ARC expects to find the unmodified settings in settings.py.
  If you made changes to any settings instead of creating respective files under a local ``.arc.`` folder,
  it is recommended to first stash your changes (``git stash``). To run the tests, type::

    make test

  After the tests complete, you may unstash your changes, if relevant (``git stash pop``).


.. _aliases:

Optional: Add ARC aliases to your .bashrc (for convenience)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below are optional aliases to make ARC (even) more convenient (make sure to change `~/Path/to/ARC/` accordingly).
Add these to your ``.bashrc`` file (edit it by typing, e.g., ``nano ~/.bashrc``)::

  export arc_path=$HOME'/Path/to/ARC/'
  alias arce='source activate arc_env'
  alias arc='python $arc_path/ARC.py input.yml'
  alias arcrestart='python $arc_path/ARC.py restart.yml'
  alias arcode='cd $arc_path'
  alias j='cd $arc_path/ipython/ && jupyter notebook'



Updating ARC
^^^^^^^^^^^^

ARC is being updated frequently. Make sure to update ARC and enjoy new features and bug fixes.

Note:
    If you change ARC's parameters within the repository rather than copies thereof as explained above,
    it is highly recommended to backup the files you manually changed before updating ARC.
    These are usually `ARC/arc/settings/settings.py` and `ARC/arc/settings/submit.py`.

You can update ARC to a specific version, or to the most recent developer version.
To get the most recent developer version, do the following
(and make sure to change `~/Path/to/ARC/` accordingly)::

    cd ~/Path/to/ARC/
    git stash
    git fetch origin
    git pull origin main
    git stash pop

The above will update your `main` branch of ARC.

To update to a specific version (e.g., version 1.1.0), do the following
(and make sure to change `~/Path/to/ARC/` accordingly)::

    cd ~/Path/to/ARC/
    git stash
    git fetch origin
    git checkout tags/1.1.0 -b v1.1.0
    git stash pop

The above will create a `v1.1.0` branch which replicates the stable 1.1.0 version.

**Note:** This process might cause merge conflicts if the updated version (either the developer version
or a stable version) changes a file you changed locally. Although we try to avoid causing merge conflicts
for ARC's users as much as we can, it could still sometimes happen.
You'll identify a merge conflict if git prints a message similar to this::

    $ git merge BRANCH-NAME
    > Auto-merging settings.py
    > CONFLICT (content): Merge conflict in styleguide.md
    > Automatic merge failed; fix conflicts and then commit the result

Detailed steps to resolve a git merge conflict can be found `online`__.

__ mergeConflict_

Principally, you should open the files that have merge conflicts, and look for the following markings::

    <<<<<<< HEAD
    this is some content introduced by updating ARC
    =======
    totally different content the user added, adding different changes
    to the same lines that were also updated remotely
    >>>>>>> new_branch_to_merge_later

Resolving a merge conflict consists of three stages:

- determine which version of the code you'd like to keep
  (usually you should manually append your oun changes to the more
  updated ARC code). Make the changes and get rid of the unneeded ``<<<<<<< HEAD``,
  ``=======``, and ``>>>>>>> new_branch_to_merge_later`` markings. Repeat for all conflicts.
- Stage the changed by typing: ``git add .``
- If you don't plan to commit your changes, unstage them by typing: ``git reset --soft origin/main``


.. include:: links.txt
