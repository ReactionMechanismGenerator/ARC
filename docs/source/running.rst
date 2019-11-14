.. _running:

Running ARC
===========

Using an input file
^^^^^^^^^^^^^^^^^^^

To run ARC, make sure to first activate the ARC environment
(see the :ref:`installation instructions <arce>`).
Then simply type::

    python <path_to_the_ARC_folder>/ARC.py input.yml

replacing ``<path_to_the_ARC_folder>`` in the above command with your actual local path to ARC.
You could of course name the input file whatever you like.
However, if you're using the :ref:`recommended aliases <aliases>`, then simply typing::

    arc

in any folder with a valid ``input.yml`` file will execute ARC using that file.

ARC automatically creates restart files in the same format as its input files.
If ARC crashes (e.g., due to a bug which was later fixed, or connectivity issues), typing::

    arcrestart

in a folder containing an ARC restart.yml file
(assuming you're using the :ref:`recommended aliases <aliases>`)
will cause ARC to execute, considering all previously spawned jobs specified in the restart file.
In restart mode, ARC is aware of all past submitted jobs and collects their
output files or waits for them to terminate if they are still running.

ARC's adopts the `YAML`__ format for its input/restart files.
In fact, a restart file is nothing but a very detailed
input file, and ARC treats them both identically.
Other than the file name, the difference is that the restart file
was automatically generated.

__ yaml_

A (very) simple input file might look like this::

    project: example1

    species:
      - label: ethanol
        smiles: CCO

All the parameters of `arc.main.ARC`__ class are allowed input file keywords.
Specifying species and reactions lists define :ref:`ARCSpecies <species>` and :ref:`ARCReaction <reaction>`
object. See ARC's API for a complete and updated list of keywords along with their allowed types.

__ api/main.html


Additional input file examples are available in the examples folder.
Another convenient way to see a valid and detailed input file is to run an ARC job
and peak at the automatically generated ``restart.yml`` file.

A sample reaction input file with a user-supplied TS geometry guess is::

    project: example2

    species:
      - label: N2H4
        smiles: NN

      - label: NH
        smiles: '[NH]'

      - label: N2H3
        smiles: N[NH]

      - label: NH2
        smiles: '[NH2]'

    reactions:
      - label: N2H4 + NH <=> N2H3 + NH2
        ts_xyz_guess:
        - |
          N      -0.4465194713     0.6830090994    -0.0932618217
          H      -0.4573825998     1.1483344874     0.8104886823
          H       0.6773598975     0.3820642106    -0.2197000290
          N      -1.2239012380    -0.4695695875    -0.0069891203
          H      -1.8039356973    -0.5112019151     0.8166872835
          H      -1.7837217777    -0.5685801608    -0.8405154279
          N       1.9039017235    -0.1568337145    -0.0766247796
          H       1.7333130781    -0.8468572038     0.6711695415



Using the API
^^^^^^^^^^^^^

To run ARC, make sure to first activate the ARC environment
(see the :ref:`installation instructions <arce>`).

ARC's API can be reached from any python platform, if ARC was added to the PYTHONPATH
(see the :ref:`installation instructions <path>`).

Running ARC using `Jupyter notebooks`__ (comes pre-installed with Anaconda)
has the benefit of displaying "live" and interactive
3D geometries for the species of interest.

__ jupyter_

Example iPython notebooks are available in the ``ipython/Demo`` folder.
Various :ref:`standalone tools <tools>` in an iPython format are also available,
demonstrating different utilizations of the API.
Users are of course directed to read :ref:`ARC's API <api>`.

.. include:: links.txt
