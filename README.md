# ARC
ARC - Automatic Rate Calculator

Version 0.1

This program automates quantum chemical calculations. Currently Gaussian, Molpro, and QChem are supported.
The current version should be run locally; it communicates with servers (defined in settings.py) using an RSA key to spawn the calculations.

Currently ARC can calculate Thermodynamic properties relying on Arkane (see Arkane's user guide: http://reactionmechanismgenerator.github.io/RMG-Py/users/arkane/index.html).
We plan to elaborate ARC to automatically identify and compute transition states and eventually calculate kinetic rates and pressure-dependent networks using Arkane.

Installation instructions are in the Wiki pages of this project.
