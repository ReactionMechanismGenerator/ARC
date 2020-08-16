## External submit scripts example

Sometimes it is desired to use submit scripts stored outside of the ARC repository
(e.g., when ARC is installed for a group of users on a server, and each user would like to customize
the scripts used by ARC).
It is possible to do so by passing a YAML file containing the desired scripts into
the ``external_submit_scripts`` argument.
Note that the structure of the YAML file must be identical to that in ARC's submit scripts file
(e.g., nested dictionaries). 
This folder includes an example for how such YAML should be set up.
