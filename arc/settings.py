#!/usr/bin/env python
# encoding: utf-8

"""
ARC's settings
"""

import os
import string

##################################################################

# If ARC communication with remote servers is desired, complete the following server dictionary.
# Instructions for RSA key generation can be found here:
# https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys--2
# If ARC is being executed on a server, and ESS are available on that server, define a server named 'local',
# for which only the cluster software and user name are required.
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
#     },
#    'local': {
#        'cluster_soft': 'OGE',
#        'un': '<username>',
#    },
# }
servers = {
    'server1': {
        'cluster_soft': 'OGE',
        'address': 'server1.host.edu',
        'un': '<username>',
        'key': 'path_to_rsa_key',
    },
    'server2': {
        'cluster_soft': 'Slurm',
        'address': 'server2.host.edu',
        'un': '<username>',
        'key': 'path_to_rsa_key',
        'cpus': 48,  # number of cpu's per node, optional (default: 8)
        'memory': 128,  # amount of memory per node in GB, optional (default: 16)
    },
    'local': {
        'cluster_soft': 'OGE',
        'un': '<username>',
    },
}

# List here servers you'd like to associate with specific ESS.
# An ordered list of servers indicates priority
# Keeping this dictionary empty will cause ARC to scan for software on the servers defined above
global_ess_settings = {
    'gaussian': ['local', 'server2'],
    'molpro': 'server2',
    'qchem': 'server1',
    'onedmin': 'server1',
}

# List here job types to execute by default
default_job_types = {'conformers': True,      # defaults to True if not specified
                     'opt': True,             # defaults to True if not specified
                     'fine_grid': True,       # defaults to True if not specified
                     'freq': True,            # defaults to True if not specified
                     'sp': True,              # defaults to True if not specified
                     '1d_rotors': True,       # defaults to True if not specified
                     'orbitals': False,       # defaults to False if not specified
                     'lennard_jones': False,  # defaults to False if not specified
                     }

# List here (complete or partial) phrases of methods or basis sets you'd like to associate to specific ESS
# Avoid ascribing the same phrase to more than one software, this may cause undeterministic assignment of software
# Format is levels_ess = {ess: ['phrase1', 'phrase2'], ess2: ['phrase3', 'phrase3']}
levels_ess = {
    'gaussian': ['b3lyp', 'm062x'],
    'molpro': ['ccsd', 'cisd', 'vpz'],
    'qchem': ['m06-2x', 'def2']
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
                  'onedmin': 'input.in',
                  }

output_filename = {'gaussian': 'input.log',
                   'qchem': 'output.out',
                   'molpro': 'input.out',
                   'onedmin': 'output.out',
                   'gromacs': 'output.yml'
                   }

default_levels_of_theory = {'conformer': 'b97d3/6-31+g(d,p)',
                            'ts_guesses': 'b97d3/6-31+g(d,p)',  # used for IRC as well
                            'opt': 'wb97xd/6-311++g(d,p)',
                            'freq': 'wb97xd/6-311++g(d,p)',  # should be the same level as opt (to calc freq at min E)
                            'scan': 'b97d3/6-31+g(d,p)',  # should be the same level as freq (to project out rotors)
                            'sp': 'ccsd(t)-f12/cc-pvtz-f12',  # This should be a level for which BAC is available
                            # 'sp': 'b3lyp/6-311+g(3df,2p)',
                            'orbitals': 'wb97x-d3/6-311++g(d,p)',  # save orbitals for visualization
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
maximum_barrier = 40    # a rotor threshold (kJ/mol) above which the rotor is not considered. Default: 40 (~10 kcal/mol)
minimum_barrier = 1.0   # a rotor threshold (kJ/mol) below which it is considered a FreeRotor. Default: 1.0 kJ/mol
inconsistency_az = 5    # maximum allowed inconsistency (kJ/mol) between initial and final rotor scan points. Default: 5
inconsistency_ab = 0.3  # maximum allowed inconsistency between consecutive points in the scan given as a fraction
#  of the maximum scan energy. Default: 30%
