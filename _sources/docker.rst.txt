.. _docker:

Docker image
============

The published Docker image includes ARC and RMG along with convenience entrypoints.
Bind-mount your working directory and pass an input file path that exists inside
the container. For best write access on bind mounts, pass your host UID/GID as
``PUID``/``PGID`` (the entrypoint remaps the ``mambauser`` account).

Platform support
^^^^^^^^^^^^^^^^

The image is built for ``linux/amd64`` only.

* **Linux (x86-64)** -- the platform the image is built and tested on.
* **macOS on Apple Silicon** -- Docker Desktop will run the image under
  ``qemu``/Rosetta emulation. This is untested; expect a substantial slowdown,
  and be aware that some numerical dependencies may misbehave under emulation.
* **Windows** -- use Docker Desktop with the WSL2 backend. Untested.

No ``linux/arm64`` image is published.

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

An interactive shell starts with the ``arc_env`` environment already activated,
and provides these shortcuts::

    rc              # reload ~/.bashrc
    rce, erc        # edit ~/.bashrc with nano

    mamba, conda    # micromamba
    deact           # micromamba deactivate

    rmge, arce      # activate rmg_env / arc_env

    rmgcode         # cd to /home/mambauser/Code/RMG-Py
    dbcode          # cd to /home/mambauser/Code/RMG-database
    arcode          # cd to /home/mambauser/Code/ARC

    rmg [args] <input.py>     # run RMG in rmg_env    (default file: input.py)
    arkane [args] <input.py>  # run Arkane in rmg_env (default file: input.py)
    arc [args] <input.yml>    # run ARC in arc_env    (default file: input.yml)
    arcrestart [args] [file]  # run ARC in arc_env    (default file: restart.yml)

The four job runners forward their arguments, so ``arc my_case/input.yml`` runs
``my_case/input.yml``; with no arguments the default file shown above is used.
They dispatch through the same entrypoint as ``docker run IMAGE arc <file>``, so
interactive and non-interactive invocations behave identically. Both streams are
echoed to the terminal and appended to ``./stdout.log`` and ``./stderr.log``.

Run ``aliases`` at any time to reprint this cheat-sheet; it also shows the
branch and resolved commit SHA of ARC, RMG-Py and RMG-database baked into the
image (also available in ``/home/mambauser/Code/image_versions.env`` and, for
ARC, in the ``org.opencontainers.image.revision`` label).
