.. _docker:

Docker image
============

The published Docker image includes ARC and RMG along with convenience entrypoints.
Bind-mount your working directory and pass an input file path that exists inside
the container. For best write access on bind mounts, pass your host UID/GID as
``PUID``/``PGID`` (the entrypoint remaps the ``mambauser`` account).

Run ARC non-interactively::

    docker run --rm \
        -v "$PWD:/work" -w /work \
        -e PUID=$(id -u) -e PGID=$(id -g) \
        laxzal/arc:latest arc my_case/input.yml

Run RMG non-interactively::

    docker run --rm \
        -v "$PWD:/work" -w /work \
        -e PUID=$(id -u) -e PGID=$(id -g) \
        laxzal/arc:latest rmg my_case/input.py

Manual RMG invocation::

    docker run --rm \
        -v "$PWD:/work" -w /work \
        -e PUID=$(id -u) -e PGID=$(id -g) \
        laxzal/arc:latest \
        micromamba run -n rmg_env python /home/mambauser/Code/RMG-Py/rmg.py my_case/input.py

Manual ARC invocation::

    docker run --rm \
        -v "$PWD:/work" -w /work \
        -e PUID=$(id -u) -e PGID=$(id -g) \
        laxzal/arc:latest \
        micromamba run -n arc_env python /home/mambauser/Code/ARC/ARC.py my_case/input.yml

Open an interactive shell::

    docker run --rm -it \
        -v "$PWD:/work" -w /work \
        -e PUID=$(id -u) -e PGID=$(id -g) \
        laxzal/arc:latest

For job submission, the scheduler client tools must be available in the container
or accessed via SSH on a remote host.

Aliases in interactive shells
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you open an interactive shell, the image provides these aliases::

    rc              # reload ~/.bashrc
    rce, erc        # edit ~/.bashrc with nano

    mamba, conda    # micromamba
    deact           # micromamba deactivate

    rmge, arce      # activate rmg_env / arc_env

    rmgcode         # cd to /home/mambauser/Code/RMG-Py
    dbcode          # cd to /home/mambauser/Code/RMG-database
    arcode          # cd to /home/mambauser/Code/ARC

    rmg             # run RMG with input.py, tee logs
    arkane          # run Arkane with input.py, tee logs
    arc             # run ARC with input.yml, tee logs
    arcrestart      # run ARC with restart.yml, tee logs
