.. _installation:

Installation instructions
=========================

Note: ARC was only tested on Linux (Ubuntu_ 18.04.1 LTS) and Mac machines. We don't expect it to work on Windows.

ARC can be installed on a server, as well as on your local desktop / laptop, submitting jobs to the server/s.
The instructions below make this differentiation when relevant (the only difference is that ARC should be aware
of software installed on the same machine, where the communication isn't done via SSH).


.. _path:

Clone and setup path
^^^^^^^^^^^^^^^^^^^^

- Download and install the `Anaconda Python Platform`__ for Python 3.7 or higher if you haven't already.
- Get git if you don't have it already by typing sudo apt-get install git in a terminal.
- Clone ARC's repository to by typing the following command in the desired folder (e.g., under `~/home/Code/`)::

    git clone https://github.com/ReactionMechanismGenerator/ARC.git

- Add ARC to your local path in .bashrc (make sure to change `~/Path/to/ARC/` accordingly)::

    export PYTHONPATH=$PYTHONPATH:~/Path/to/ARC/

__ anaconda_


.. _arce:

Install dependencies
^^^^^^^^^^^^^^^^^^^^

- Install the latest DEVELOPER version of RMG (which has Arkane).
  It is recommended to follow RMG's `Developer installation by source using Anaconda
  <http://reactionmechanismgenerator.github.io/RMG-Py/users/rmg/installation/index.html
  #for-developers-installation-by-source-using-anaconda-environment>`_ instructions.
  Make sure to add RMG-Py to your PATH and PYTHONPATH variables as explained in RMG's documentation.
- If you'd like to use `AutoTST <https://github.com/ReactionMechanismGenerator/AutoTST>`_ in ARC (optional),
  clone it in a separate folder and add it to your PYTHONPATH just as well.
- Create the Anaconda environment for ARC::

    conda env create -f environment.yml

  Activate the ARC environment every time before you run ARC::

     source activate arc_env


Generate RSA SSH keys and define servers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first two directives are only required if you'd like ARC to access remote servers
(ARC could also run "locally" on a server).

- Generate RSA SSH keys for your favorite server/s on which relevant electronic structure software
  (ESS, e.g., Gaussian etc.) are installed. `Instructions for generating RSA keys could be found here
  <https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys--2>`_.
- Copy the RSA SSH key path/s on your local machine to ARC/arc/settings.py in the servers
  dictionary under keys.
- Update the `servers` dictionary in `ARC/arc/settings.py
  <https://github.com/ReactionMechanismGenerator/ARC/blob/master/arc/settings.py>`_.

  * A local server must be named with the reserved keyword ``local``. ``cluster_soft`` and username (``un``) are
    mandatory.
  * A remote server has no limitations for naming. ``cluster_soft``, ``address``, username (``un``), and ``key``
    (the path to the local RSA SSH key) are mandatory.
  * Optional parameters for both local and remote servers are ``cpus`` and ``memory``. The number of CPUs ARC will use
    when spawning ESS jobs defaults to 8 unless otherwise specified under ``cpus``.
    Likewise, the default memory is 16 GB. ARC will not use more than 90% of the node memory specified under ``memory``.

- Update the submit scripts in `ARC/arc/job/submit.py
  <https://github.com/ReactionMechanismGenerator/ARC/blob/master/arc/job/submit.py>`_
  according to your servers' definitions.
  * See the given template examples, and follow the structure of nested dictionaries (by server name, then by ESS name).
  * Preserve the variables in curly braces (e.g., ``{mem_per_cpu}``), so that ARC is able to auto-complete them.


Associating software with servers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ARC keeps track of software location on servers using a Python dictionary associating the different software (keys)
with the servers they are installed on (values). The server name must be consistent with the respective definition
in the ``servers`` dictionary mentioned above. Typically, you would update the ``global_ess_settings`` dictionary in
`ARC/arc/settings.py <https://github.com/ReactionMechanismGenerator/ARC/blob/master/arc/settings.py>`_ to reflect
your software and servers, for example::

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
server's definitions are different that ARC's, you should also modify the following variables in
`ARC/arc/settings.py <https://github.com/ReactionMechanismGenerator/ARC/blob/master/arc/settings.py>`_:

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
  Therefore it is recommended to first stash your changes. If you'd like to test ARC with changes you made to the code,
  first commit these changes to a git branch.::

    git stash
    make test

  After the tests complete, don't forget to::

    git stash pop


.. _aliases:

Add ARC aliases to your .bashrc (for convenience)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below are optional aliases to make ARC (even) more convenient (make sure to change `~/Path/to/ARC/` accordingly)::

  export arc_path=$HOME'~/Path/to/ARC/'
  alias arce='source activate arc_env'
  alias arc='source activate arc_env && python $arc_path/ARC.py input.yml'
  alias arcrestart='source activate arc_env && python $arc_path/ARC.py restart.yml'
  alias arcode='cd $arc_path'
  alias j='cd $arc_path/ipython/ && arce && jupyter notebook'



Updating ARC
^^^^^^^^^^^^

ARC is being updated frequently. Make sure to update ARC and enjoy new features and bug fixes.

**Note:** It is highly recommended to backup files you manually changed in ARC before updating the version,
these are usually `ARC/arc/settings.py` and `ARC/arc/job/submit.py`.

You can update ARC to a specific version, or to the most recent developer version.
To get the most recent developer version, do the following
(and make sure to change `~/Path/to/ARC/` accordingly)::

    cd ~/Path/to/ARC/
    git stash
    git fetch origin
    git pull origin master
    git stash pop

The above will update your `master` branch of ARC.

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
- If you don't plan to commit your changes, unstage them by typing: ``git reset --soft origin/master``


.. include:: links.txt
