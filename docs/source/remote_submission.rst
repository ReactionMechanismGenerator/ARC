.. _remote_submission:

Remote Job Submission over SSH
==============================

ARC submits electronic structure jobs to a remote cluster by opening an SSH session
with paramiko. This page describes how to authenticate that session, how host keys
are verified, and how to do both from inside the Docker image.

If ARC and the electronic structure software run on the same machine, you do not
need any of this - define a server named ``local`` instead, as described in
:ref:`running`.

Server Settings
---------------

Remote servers are declared in the ``servers`` dictionary of your personal
``~/.arc/settings.py``:

.. code-block:: python

   servers = {
       'cluster_a': {
           'cluster_soft': 'Slurm',
           'address': 'login.cluster.edu',
           'un': 'my_user',
       },
   }

The keys that affect the SSH connection itself are:

* ``address`` - the hostname ARC connects to;
* ``un`` - the username on the remote machine;
* ``key`` - **optional**. The path, on the machine running ARC, of the SSH
  **private key** to authenticate with;
* ``strict_host_key_checking`` - optional, ``False`` by default. See
  `Host Key Verification`_.

Authentication
--------------

ARC hands ``key`` to paramiko as the identity to authenticate with. There are two
supported ways to authenticate, and the choice is made simply by whether you set
``key``.

Using an ssh-agent (preferred)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Omit ``key`` entirely. paramiko then looks for a running ssh-agent, and after that
for the default key paths ``~/.ssh/id_rsa``, ``~/.ssh/id_ecdsa`` and
``~/.ssh/id_ed25519``. There is no ``id_dsa`` in that list; paramiko 4 dropped DSA.

.. code-block:: python

   servers = {
       'cluster_a': {
           'cluster_soft': 'Slurm',
           'address': 'login.cluster.edu',
           'un': 'my_user',
       },
   }

Add the key to your agent once per login session:

.. code-block:: bash

   ssh-add ~/.ssh/id_ed25519

This is the preferred route. Passphrase-protected keys keep working, because the
agent holds the decrypted key and ARC never has to prompt for the passphrase. In a
container it also means the key material itself never has to enter the container.

Using a key file
^^^^^^^^^^^^^^^^

Set ``key`` to the path of the private key:

.. code-block:: python

   servers = {
       'cluster_a': {
           'cluster_soft': 'Slurm',
           'address': 'login.cluster.edu',
           'un': 'my_user',
           'key': '/home/my_user/.ssh/id_ed25519',
       },
   }

The path must exist and be readable **on the machine running ARC**, and it must
name the private key, not the public ``.pub`` half and not ``known_hosts``.
paramiko raises if it cannot read the file, and ARC then retries the connection for
up to 24 hours, so a wrong path shows up as a run that appears to hang rather than
as an immediate error.

``~/.ssh/config`` is not read
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ARC connects with paramiko, and paramiko only applies ``~/.ssh/config`` when the
application explicitly builds a ``paramiko.SSHConfig``. ARC does not, so **no**
directive in that file has any effect: not ``IdentityFile``, not ``User``, not
``HostName``, not ``Port``, and not ``ProxyJump`` or ``ProxyCommand``.

The practical consequences are:

* every connection detail must be spelled out in the server entry - ``address``,
  ``un``, and ``key`` if you use one - even when your ``ssh`` command line works
  without them;
* **bastion and jump hosts are not supported**. A cluster that can only be reached
  through a jump host cannot be driven by ARC directly. Run ARC on a machine that
  has direct access to the login node instead, or establish the tunnel outside ARC
  and point ``address`` at the local end of it.

This is a property of ARC on any machine, not something introduced by running it
in a container.

Host Key Verification
---------------------

ARC loads the system host keys from paramiko's default location, which is
``~/.ssh/known_hosts`` on the machine running ARC, and nothing else. In particular
``/etc/ssh/ssh_known_hosts`` is **not** read, even though the OpenSSH command line
client reads it; paramiko's ``load_system_host_keys()`` consults the user file only,
and silently ignores it if it cannot be read. The path is not configurable from
``settings.py``.

What happens when a host is *not* in ``known_hosts`` depends on the per-server
``strict_host_key_checking`` flag:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Value
     - Behavior for an unknown host
   * - ``False`` (default)
     - The connection proceeds and a warning is logged. Convenient, but a
       machine-in-the-middle is indistinguishable from a first-ever connection,
       so the warning is the only signal you get.
   * - ``True``
     - The connection is refused. The host must be present in ``known_hosts``
       before ARC can reach it. The refusal is immediate -- it raises its own
       exception type, which ARC does not retry -- so it is reported as an error
       naming the host rather than as a silent wait. Seed ``known_hosts`` first.

.. code-block:: python

   servers = {
       'cluster_a': {
           'cluster_soft': 'Slurm',
           'address': 'login.cluster.edu',
           'un': 'my_user',
           'strict_host_key_checking': True,
       },
   }

Seed ``known_hosts`` before enabling the flag:

.. code-block:: bash

   ssh-keyscan -H login.cluster.edu >> ~/.ssh/known_hosts

Verify the fingerprints against a trusted source before trusting the result -
``ssh-keyscan`` trusts whatever answers on the network.

Why warning is the default
^^^^^^^^^^^^^^^^^^^^^^^^^^

Refusing an unknown host is the safer policy in the abstract, and it is a poor
default for ARC specifically, because ARC is a scheduler that runs unattended for
days rather than a client that makes one connection.

A refused host key does not fail the run once and stop. Every submission, status
poll and download for that server fails while the ARC driver stays alive, so the run
continues, submits nothing, and looks like a stall. Each individual refusal is
reported at once rather than retried, but nothing takes the driver down, so whoever
is running it typically finds out hours later, and recovery means stopping ARC,
running ``ssh-keyscan``, and starting again.
Weighed against a first-ever connection to a login node reached over a network the
user already trusts enough to submit jobs to, that failure mode costs more than the
risk it removes.

What makes the default defensible is that the risk is *reported*, not hidden. ARC
checks ``known_hosts`` at startup (see `Startup Checks`_), so an unseeded host is
named before any calculation is submitted, and
``'strict_host_key_checking': True`` remains available per server for anyone who
wants the connection refused instead.

Running in the Docker Image
---------------------------

Everything above applies unchanged inside the container; what changes is that
``~/.arc``, the SSH material, and possibly the agent socket have to be bind-mounted
in. See :ref:`docker` for the image's general usage.

Mounting your ARC settings
^^^^^^^^^^^^^^^^^^^^^^^^^^

The container user is ``mambauser`` with ``HOME=/home/mambauser``, so your personal
settings must be mounted at ``/home/mambauser/.arc``:

.. code-block:: bash

   -v "$HOME/.arc:/home/mambauser/.arc:ro"

ARC reads ``settings.py``, ``submit.py`` and ``inputs.py`` from that directory. All
three matter for remote submission: ``submit.py`` holds the cluster's PBS/Slurm
submit script templates, and without it ARC will submit jobs with the repository's
generic templates.

Two properties of that overlay are worth knowing:

* It is a **replacement of top-level names**, not a deep merge. A ``settings.py``
  that defines only ``servers`` replaces the whole ``servers`` dictionary and
  leaves every other setting at its repository default - which is what you
  usually want - but a ``settings.py`` that defines, say, ``levels_ess`` replaces
  that whole dictionary too.
* Whenever a local ``settings.py`` exists, ARC forces ``global_ess_settings`` to
  ``None`` unless that file defines a truthy value of its own. This is deliberate:
  the repository defaults are dummies. If you route software to servers through
  ``global_ess_settings``, define it in your own ``settings.py``.

Mount the directory read-only, and set ``PYTHONDONTWRITEBYTECODE=1`` so Python does
not try to write ``__pycache__`` into a read-only mount:

.. code-block:: bash

   -e PYTHONDONTWRITEBYTECODE=1

With agent forwarding (preferred)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forward the host's ssh-agent socket and omit ``key`` from the server entry. No key
material enters the container:

.. code-block:: bash

   docker run --rm \
       -v "$PWD:/work" -w /work \
       -e PUID=$(id -u) -e PGID=$(id -g) \
       -e PYTHONDONTWRITEBYTECODE=1 \
       -v "$HOME/.arc:/home/mambauser/.arc:ro" \
       -v "$HOME/.ssh/known_hosts:/home/mambauser/.ssh/known_hosts:ro" \
       -v "$SSH_AUTH_SOCK:/ssh-agent" -e SSH_AUTH_SOCK=/ssh-agent \
       laxzal/arc:latest arc my_case/input.yml

On macOS, Docker Desktop exposes the host agent at a fixed path, so use
``-v /run/host-services/ssh-auth.sock:/ssh-agent`` instead of ``$SSH_AUTH_SOCK``.

``PUID``/``PGID`` remap the ``mambauser`` account to your host UID/GID. This is not
optional for agent forwarding, it is the mechanism: an agent socket is mode ``0600``
and owned by you, so only a container user carrying your UID can open it.

The entrypoint deliberately does **not** relax the socket's permissions to work
around a missing remap. A bind mount shares the inode with the host, so doing that
would make your real, live agent socket readable and writable by every other user on
the machine, for as long as the agent runs - and because the entrypoint hands off
with ``exec``, nothing would ever restore it. If you genuinely need that behaviour,
set ``ARC_WIDEN_AGENT_SOCKET=1``; the entrypoint will then relax the mode, say
exactly what it changed, and remind you to run ``chmod 600 "$SSH_AUTH_SOCK"`` on the
host afterwards. Passing ``PUID``/``PGID`` is almost always the better answer.

A note on stale sockets: if ``$SSH_AUTH_SOCK`` points at an agent that has since
exited, Docker creates the missing bind-mount source rather than failing, leaving a
root-owned directory at that path on your host. The entrypoint reports it as "not a
socket" and continues without agent forwarding. The same applies to ``$HOME/.arc``
if that directory does not exist.

With a mounted key (fallback)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For headless runs where no agent is available, mount the key read-only and point
``key`` at the path **inside** the container:

.. code-block:: python

   servers = {
       'cluster_a': {
           'cluster_soft': 'Slurm',
           'address': 'login.cluster.edu',
           'un': 'my_user',
           'key': '/home/mambauser/.ssh/id_ed25519',
       },
   }

.. code-block:: bash

   docker run --rm \
       -v "$PWD:/work" -w /work \
       -e PUID=$(id -u) -e PGID=$(id -g) \
       -e PYTHONDONTWRITEBYTECODE=1 \
       -v "$HOME/.arc:/home/mambauser/.arc:ro" \
       -v "$HOME/.ssh/id_ed25519:/home/mambauser/.ssh/id_ed25519:ro" \
       -v "$HOME/.ssh/known_hosts:/home/mambauser/.ssh/known_hosts:ro" \
       laxzal/arc:latest arc my_case/input.yml

A read-only mount is fine: ARC uses paramiko, which unlike the OpenSSH command line
client does not refuse key files with permissive modes. The key must be
passphrase-free, since there is nothing to prompt for it in a batch run.

To mount the whole directory instead of individual files, use
``-v "$HOME/.ssh:/home/mambauser/.ssh:ro"``. The entrypoint detects that
``/home/mambauser/.ssh`` is a bind mount and leaves its ownership and modes alone,
so nothing leaks back to your host files.

Docker Compose
^^^^^^^^^^^^^^

``docker-compose.yml`` in the repository root wires all of this up already:

.. code-block:: bash

   ARC_WORKDIR=$PWD ARC_INPUT=my_case/input.yml \
   PUID=$(id -u) PGID=$(id -g) \
   docker compose run --rm arc

It mounts ``$HOME/.arc`` read-only, forwards ``$SSH_AUTH_SOCK`` to ``/ssh-agent``,
mounts ``$ARC_KNOWN_HOSTS`` read-only at ``/home/mambauser/.ssh/known_hosts``, and
sets ``PYTHONDONTWRITEBYTECODE=1``. The key-file mount is present but commented out;
uncomment it if you are not using an agent.

Point ``ARC_KNOWN_HOSTS`` at your host keys to share them with the container:

.. code-block:: bash

   ARC_WORKDIR=$PWD ARC_INPUT=my_case/input.yml \
   ARC_KNOWN_HOSTS=$HOME/.ssh/known_hosts \
   PUID=$(id -u) PGID=$(id -g) \
   docker compose run --rm arc

The variable defaults to ``/dev/null``, which reads as an empty set of host keys,
rather than to ``$HOME/.ssh/known_hosts`` directly. Docker creates any bind-mount
source that does not exist, so naming that path unconditionally would leave a
root-owned *directory* at ``$HOME/.ssh/known_hosts`` on a host that has never
written the file - which then breaks ``ssh`` on the host itself.

known_hosts in a fresh container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A fresh container starts with an empty ``/home/mambauser/.ssh``, so *every* host is
an unknown host. With the default ``strict_host_key_checking: False`` that only
produces a warning per connection, but with ``strict_host_key_checking: True`` it
refuses every connection until the file is seeded. paramiko reads it from its
default location, so the mount target has to be exactly
``/home/mambauser/.ssh/known_hosts`` - mounting it anywhere else has no effect.

Either mount the host's file, as in the examples above, or seed one inside the
container. The image ships ``openssh-client`` for exactly this:

.. code-block:: bash

   docker run --rm -it \
       -v "$HOME/.ssh:/home/mambauser/.ssh" \
       -e PUID=$(id -u) -e PGID=$(id -g) \
       laxzal/arc:latest \
       ssh-keyscan login.cluster.edu

Append the verified output to the ``known_hosts`` file you then mount read-only.

Network reachability
^^^^^^^^^^^^^^^^^^^^

A container on Docker's default bridge network reaches an ordinary cluster login
node without any extra flags: the host acts as a NAT router, DNS resolves, and
paramiko speaks SSH itself rather than shelling out to ``ssh``, so nothing but
outbound TCP on port 22 is required. ``--network host`` is not needed in the normal
case.

Two situations do differ from running on the host:

* **The cluster is reachable only over a VPN on the host.** Split-tunnel routing
  and firewall rules frequently exclude the ``docker0`` bridge, so the container's
  traffic never enters the tunnel. ``--network host`` is the escape hatch: the
  container then shares the host's network namespace and its VPN routes.
* **Internal hostnames resolved by a VPN-pushed resolver.** The container uses
  Docker's resolver, not the host's, so a name that only the VPN's DNS server knows
  will not resolve. Use the IP address in ``address``, or pass
  ``--dns <resolver-ip>``.

To tell the two apart, check reachability from inside the container before blaming
ARC:

.. code-block:: bash

   docker run --rm laxzal/arc:latest \
       bash -lc 'getent hosts login.cluster.edu && ssh-keyscan -T 5 login.cluster.edu'

A resolved address followed by a host key means the network path is fine and the
problem is authentication.

Startup Checks
--------------

Every ARC run, in a container or not, reports the configured servers that have no
host key in ``~/.ssh/known_hosts`` before it submits anything. The message names the
server, its address, and the ``ssh-keyscan`` command that fixes it. It is a warning,
never fatal, and the check is entirely local: the ``known_hosts`` file is read, and
no name is resolved and no connection opened.

Three kinds of server entry are skipped, because reporting them would be noise
rather than information: the ``local`` server, entries with no ``address``, and
entries still carrying the repository's placeholder ``*.host.edu`` address or
``<username>`` user name.

Inside the container the entrypoint checks more, and earlier - before ARC itself
starts - because the failure modes there are otherwise invisible until much later in
the run. Most notably, ARC ignores a ``~/.arc/settings.py`` that fails to import
and silently continues with the repository's dummy servers
(``server1.host.edu``, ``<username>``) - in a container, a mis-typed mount path
produces exactly that.

The entrypoint therefore:

* refuses to start ARC, with exit code 78, if ``/home/mambauser/.arc/settings.py``
  exists but cannot be imported, printing the import traceback;
* warns if no ``settings.py`` is mounted at all;
* warns about a server whose ``key`` does not exist or is not readable inside the
  container;
* warns about a server with no ``key`` when the container has neither a forwarded
  agent nor a default key under ``/home/mambauser/.ssh``;
* warns about a server with ``strict_host_key_checking`` when no ``known_hosts``
  file is present.

Only the unimportable ``settings.py`` is fatal. The rest are warnings, so a server
that is configured but not used in this particular run cannot abort it. Set
``ARC_SKIP_PREFLIGHT=1`` to skip the checks entirely.

Exit codes from the entrypoint follow ``sysexits.h``: 64 for a usage error, 66 for
a missing input file, and 78 for a configuration error.

Troubleshooting
---------------

**ARC connects to** ``server1.host.edu`` **, or reports unknown server names.**
Your ``~/.arc/settings.py`` was not picked up. Outside a container, check that the
file is at ``$HOME/.arc/settings.py`` and imports cleanly with
``python -c "import sys; sys.path.insert(0, '$HOME/.arc'); import settings"``.
Inside a container, check the mount target is ``/home/mambauser/.arc``.

**The run appears to hang while connecting.** ARC retries a failed connection every
60 seconds, up to 1440 times, so a full 24 hours -- but only a transport-level one,
such as a login node that is down or a ``key`` pointing at a missing file. The
reason is reported on every attempt, but only every tenth attempt goes through the
logger, at info level; the others are written straight to standard output, so look
there as well as in the log if the run seems stuck.

Failures that retrying cannot resolve are not retried at all: a rejected identity
(a wrong username, or a key the server does not accept), a host key that contradicts
``known_hosts``, and a rejected host key under ``strict_host_key_checking`` raise a
``ServerError`` naming the cause on the first attempt.

**Permission denied, or the forwarded agent is not usable.** Pass
``-e PUID=$(id -u) -e PGID=$(id -g)``. Without it the container user's UID does not
match the owner of your mounts or of the agent socket, and the entrypoint will say
so.

**"cannot remap mambauser to uid/gid N".** The requested ID belongs to an account
the container cannot safely displace - the superuser, or a system account reserved
by the distribution. Ordinary, idle accounts are shared automatically, so this only
appears for IDs below 1000. Drop the flag named in the message and mount ownership
will fall back to the image's own ``mambauser`` IDs.

**Host key refused.** The server sets ``strict_host_key_checking: True`` and the
host is not in ``known_hosts``. Seed it with ``ssh-keyscan``, verify the
fingerprint, and mount the file at ``/home/mambauser/.ssh/known_hosts``.

.. include:: links.txt
