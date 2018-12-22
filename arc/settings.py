#!/usr/bin/env python
# encoding: utf-8


##################################################################

# Modifications to this file aren't tracked by git

arc_path = '/home/alongd/Code/ARC/'  # on local machine

servers = {
    'server1': {
        'cluster_soft': 'OGE',  # Oracle Grid Engine
        'adddress': 'server1.host.edu',
        'un': 'username',
        'key': 'rsa key path',
    },
    'server2': {
        'cluster_soft': 'Slurm',  # Simple Linux Utility for Resource Management
        'adddress': 'server2.host.edu',
        'un': 'username',
        'key': 'rsa key path',
    }
}

# If ARC is run locally and communication with servers is desired,
# complete the following server dictionary.
# Note that the order matters, and that if two serers have the same ESS installed,
# ARC will used the copy on the latest server it finds it on.
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


check_status_command = {'OGE': 'export SGE_ROOT=/opt/sge; /opt/sge/bin/lx24-amd64/qstat',
                        'Slurm': '/usr/bin/squeue'}

submit_command = {'OGE': 'export SGE_ROOT=/opt/sge; /opt/sge/bin/lx24-amd64/qsub',
                  'Slurm': '/usr/bin/sbatch'}

delete_command = {'OGE': 'export SGE_ROOT=/opt/sge; /opt/sge/bin/lx24-amd64/qdel',
                  'Slurm': '/usr/bin/scancel'}

list_available_nodes_command = {'OGE': 'export SGE_ROOT=/opt/sge; /opt/sge/bin/lx24-amd64/qstat -f | grep "/8 " | grep "long" | grep -v "8/8"| grep -v "aAu"',
                                'Slurm': 'list? '}

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
                            'freq_for_composite': 'B3LYP/CBSB7',  # This is the frequency level of the CBS-QB3 method
}

# A scan with better resolution (lower number here) takes more time to compute,
# but the automatically-derived rotor symmetry number is more likely to be correct.
rotor_scan_resolution = 5.0  # degrees. Default: 5.0

# rotor validation parameters
inconsistency_az = 5   # maximum allowed inconsistency (kJ/mol) between initial and final rotor scan points. Default: 5
inconsistency_ab = 10  # maximum allowed inconsistency (kJ/mol) between consecutive points in the scan. Default: 10
maximum_barrier = 100  # maximum allowed barrier (kJ/mol) for a hindered rotor. Default: 100
