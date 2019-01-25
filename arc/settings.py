#!/usr/bin/env python
# encoding: utf-8

import os

##################################################################

# If ARC is run locally and communication with servers is desired,
# complete the following server dictionary.
# Instructions for RSA key generation can be found here:
# https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys--2
# The `precedence` key is optional, and will cause ARC to use the respective server
# for the specified ESS even if it finds it first on a different server.
# If this aut-ESS determination method doesn't work for you, you could also
# just pass an `ess_settings` dictionary to ARC() with the desired software/server as keys/values.
# servers = {
#     'pharos': {
#         'cluster_soft': 'OGE',  # Oracle Grid Engine (Sun Grin Engine)
#         'adddress': 'pharos.mit.edu',
#         'un': '<username>',
#         'key': '/home/<username>/.ssh/known_hosts',
#     },
#     'rmg': {
#         'cluster_soft': 'Slurm',  # Simple Linux Utility for Resource Management
#         'adddress': 'rmg.mit.edu',
#         'un': '<username>',
#         'key': '/home/<username>/.ssh/id_rsa',
#     }
# }
servers = {
    'server1': {
        'cluster_soft': 'OGE',  # Oracle Grid Engine
        'adddress': 'server1.host.edu',
        'un': '<username>',
        'key': 'path_to_rsa_key',
        'precedence': 'molpro',
    },
    'server2': {
        'cluster_soft': 'Slurm',  # Simple Linux Utility for Resource Management
        'adddress': 'server2.host.edu',
        'un': '<username>',
        'key': 'path_to_rsa_key',
    }
}

check_status_command = {'OGE': 'export SGE_ROOT=/opt/sge; /opt/sge/bin/lx24-amd64/qstat',
                        'Slurm': '/usr/bin/squeue'}

submit_command = {'OGE': 'export SGE_ROOT=/opt/sge; /opt/sge/bin/lx24-amd64/qsub',
                  'Slurm': '/usr/bin/sbatch'}

delete_command = {'OGE': 'export SGE_ROOT=/opt/sge; /opt/sge/bin/lx24-amd64/qdel',
                  'Slurm': '/usr/bin/scancel'}

list_available_nodes_command = {'OGE': 'export SGE_ROOT=/opt/sge; /opt/sge/bin/lx24-amd64/qstat -f | grep "/8 " | grep "long" | grep -v "8/8"| grep -v "aAu"',
                                'Slurm': 'sinfo'}

submit_filename = {'OGE': 'submit.sh',
                   'Slurm': 'submit.sl'}

input_filename = {'gaussian': 'input.gjf',
                   'qchem': 'input.in',
                   'molpro': 'input.in',
}

output_filename = {'gaussian': 'input.log',
                   'qchem': 'output.out',
                   'molpro': 'input.out',
}

default_levels_of_theory = {'conformer': 'b97-d3/6-311+g(d,p)',
                            'opt': 'wb97x-d3/6-311+g(d,p)',
                            'freq': 'wb97x-d3/6-311+g(d,p)',
                            'sp': 'ccsd(t)-f12/cc-pvtz-f12',  # This should be a level for which BAC is available
                            'scan': 'b3lyp/6-311+g(d,p)',
                            'irc': 'b3lyp/6-31+g(d)',
                            'gsm': 'b3lyp/6-31+g(d)',
                            'scan_for_composite': 'B3LYP/CBSB7',  # This is the frequency level of the CBS-QB3 method
                            'freq_for_composite': 'B3LYP/CBSB7',  # This is the frequency level of the CBS-QB3 method
}

arc_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # absolute path to the ARC folder

# A scan with better resolution (lower number here) takes more time to compute,
# but the automatically-derived rotor symmetry number is more likely to be correct.
rotor_scan_resolution = 10.0  # degrees. Default: 10.0

# rotor validation parameters
inconsistency_az = 5   # maximum allowed inconsistency (kJ/mol) between initial and final rotor scan points. Default: 5
inconsistency_ab = 10  # maximum allowed inconsistency (kJ/mol) between consecutive points in the scan. Default: 10
maximum_barrier = 40  # maximum allowed barrier (kJ/mol) for a hindered rotor. Default: 40 (~10 kcal/mol)
