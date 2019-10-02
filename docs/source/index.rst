ARC - Automated Rate Calculator v\ |release|
============================================

.. image:: arc.png
   :align: center

**ARC - Automated Rate Calculator** is a software for automating
electronic structure calculations relevant for chemical kinetic modeling.
ARC has many advanced options, yet at its core it is simple: it accepts 2D
graph representations of chemical species (e.g., SMILES_ or `adjacency lists`__),
and  automatically executes, tracks, and processes relevant electronic structure
jobs on user-defined servers. The principal outputs of ARC are thermodynamic properties
(H, S, Cp) and high-pressure limit kinetic rate coefficients for the defined species
and reactions.
(Note that automating transition states is still in progress).

.. _adj: http://reactionmechanismgenerator.github.io/RMG-Py/reference/molecule/adjlist.html
__ adj_

ARC is written in Python 3.7, and was made open-source under the :ref:`MIT licence <licence>`.

We use ARC to facilitate our research and have made it available
as a benefit to the community in the hopes that others may find it useful as well.
Its code is hosted on GitHub_, which is where comments, issues, and community
contributions are welcomed.

The following pages describe how to install and execute ARC,
show some advanced features and what to expect from its output.


Documentation Contents
======================

.. toctree::
   :maxdepth: 2

   installation
   running
   examples
   advanced
   output
   api/index
   tools
   release
   credits
   contribute
   cite
   licence


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. include:: links.txt
