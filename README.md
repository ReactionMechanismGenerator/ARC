# ARC
ARC - Automatic Rate Calculator

Version 0.1

This program automates quantum chemical calculations. Currently Gaussian, Molpro, and QChem are supported.
The current version should be run locally; it communicates with servers (defined in settings.py) using an RSA key to spawn the calculations.

Currently ARC can calculate Thermodynamic properties relying on Arkane (see Arkane's user guide: http://reactionmechanismgenerator.github.io/RMG-Py/users/arkane/index.html).
We plan to elaborate ARC to automatically identify and compute transition states and eventually calculate kinetic rates and pressure-dependent networks using Arkane.

To install ARC, follow these steps:

1. Clone this repository to your local machine.
2. Add ARC to your local path in .bashrc: export PYTHONPATH=$PYTHONPATH:~/Path/to/ARC/
3. Install the latest version of RMG (which has Arkane) and activate the RMG environment (usually called rmg_env)
4. Install the following libraries onto the RMG environment:
   - `paramico` (https://anaconda.org/anaconda/paramiko)
   - `cclib` (https://anaconda.org/omnia/cclib)
   - `py3dmol` (https://anaconda.org/RMG/py3dmol)
5. Generate RSA SSH keys to your favorite server/s. Instructions could be found here: https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys--2
6. Copy the RSA SSH key path/s to ARC/arc/settings.py in the servers dictionary under "keys".
7. Make sure that the server/s address/es and your username under "un" in that dictionary are all correct
8. Update the arc_path in ARC/arc/settings.py
9. Have all relevant ESS software (currently in this version g03, QChem, molpro2012, molpro2015 are supported) defined in your .bashrc on the server
10. Update all additional dictionaries in ARC/arc/settings.py (e.g., software_server, submit_command, delete_command...) accordingly.
11. Update the submit scripts in ARC/arc/job/submit.py to align with your servers definitions.
12. Run the ARCDemo.ipynb to test. This demo also shows different methods to define species for thermo calculations.
