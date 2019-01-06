ARC - Automatic Rate Calculator

|codacy| |codecov|

|arc|


Version 0.1

This program automates quantum chemical calculations. Currently Gaussian, Molpro, and QChem are supported.
The current version should be run locally; it communicates with servers (defined in settings.py) using an RSA key to spawn the calculations.

Currently ARC can calculate Thermodynamic properties relying on Arkane (see Arkane's user guide: http://reactionmechanismgenerator.github.io/RMG-Py/users/arkane/index.html).
We plan to elaborate ARC to automatically identify and compute transition states and eventually calculate kinetic rates and pressure-dependent networks using Arkane.

Installation instructions are in the Wiki pages of this project.


.. |arc| image:: https://github.com/ReactionMechanismGenerator/ARC/blob/master/logo/ARC-logo.jpg
    :target: https://github.com/ReactionMechanismGenerator/ARC
    :alt: arc logo
    :width: 250px
    :align: middle


.. |codacy| image:: https://api.codacy.com/project/badge/Grade/932aa16ac3f747d9b236bcd29e5dc9a9
   :alt: Codacy Badge
   :target: https://app.codacy.com/app/ReactionMechanismGenerator/ARC?utm_source=github.com&utm_medium=referral&utm_content=ReactionMechanismGenerator/ARC&utm_campaign=Badge_Grade_Dashboard
   

.. |codecov| image:: https://codecov.io/gh/ReactionMechanismGenerator/ARC/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/ReactionMechanismGenerator/ARC

