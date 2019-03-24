#!/usr/bin/env python
# encoding: utf-8

import os
import string

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
#         'address': 'pharos.mit.edu',
#         'un': '<username>',
#         'key': '/home/<username>/.ssh/known_hosts',
#     },
#     'rmg': {
#         'cluster_soft': 'Slurm',  # Simple Linux Utility for Resource Management
#         'address': 'rmg.mit.edu',
#         'un': '<username>',
#         'key': '/home/<username>/.ssh/id_rsa',
#     }
# }
servers = {
    'server1': {
        'cluster_soft': 'OGE',  # Oracle Grid Engine
        'address': 'server1.host.edu',
        'un': '<username>',
        'key': 'path_to_rsa_key',
        'precedence': 'molpro',
    },
    'server2': {
        'cluster_soft': 'Slurm',  # Simple Linux Utility for Resource Management
        'address': 'server2.host.edu',
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

t_max_format = {'OGE': 'hours',
                'Slurm': 'days'}

input_filename = {'gaussian': 'input.gjf',
                   'qchem': 'input.in',
                   'molpro': 'input.in',
}

output_filename = {'gaussian': 'input.log',
                   'qchem': 'output.out',
                   'molpro': 'input.out',
}

default_levels_of_theory = {'conformer': 'b97-d3/6-311+g(d,p)',
                            'ts_guesses': 'b3lyp/6-31+g(d,p)',  # used for IRC as well
                            'opt': 'wb97xd/6-311++g(d,p)',
                            'freq': 'wb97xd/6-311++g(d,p)',  # should be the same level as opt
                            'sp': 'ccsd(t)-f12/cc-pvtz-f12',  # This should be a level for which BAC is available
                            # 'sp': 'b3lyp/6-311+g(3df,2p)',
                            'orbitals': 'b3lyp/6-311++g(3df,3pd)',  # save orbitals for visualization
                            'scan': 'b3lyp/6-311+g(d,p)',
                            'scan_for_composite': 'B3LYP/CBSB7',  # This is the frequency level of the CBS-QB3 method
                            'freq_for_composite': 'B3LYP/CBSB7',  # This is the frequency level of the CBS-QB3 method
}

# default_ts_methods = ['QST2', 'DEGSM', 'NEB', 'Kinbot', 'AutoTST']
default_ts_methods = ['AutoTST']

arc_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # absolute path to the ARC folder

valid_chars = "-_()[]=., %s%s" % (string.ascii_letters, string.digits)

# A scan with better resolution (lower number here) takes more time to compute,
# but the automatically-derived rotor symmetry number is more likely to be correct.
rotor_scan_resolution = 8.0  # degrees. Default: 8.0

# rotor validation parameters
inconsistency_az = 5    # maximum allowed inconsistency (kJ/mol) between initial and final rotor scan points. Default: 5
inconsistency_ab = 0.5  # maximum allowed inconsistency between consecutive points in the scan given as a fraction
#  of the maximum scan energy. Default: 50%
maximum_barrier = 40    # a rotor threshold (kJ/mol) above which the rotor is not considered. Default: 40 (~10 kcal/mol)
minimum_barrier = 0.5   # a rotor threshold (kJ/mol) below which it is considered a FreeRotor. Default: 0.5 kJ/mol
