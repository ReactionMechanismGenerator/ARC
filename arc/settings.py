#!/usr/bin/env python
# encoding: utf-8


##################################################################

# Modifications to this file aren't tracked by git

servers = {
    'server1': {
        'adddress': 'server1.host.edu',
        'un': 'username',
        'key': 'rsa key path',
    },
    'server2': {
        'adddress': 'server2.host.edu',
        'un': 'username',
        'key': 'rsa key path',
    }
}

software_server = {'gaussian03': 'server1',
                   'qchem': 'server1',
                   'molpro_2015': 'server2',
                   'molpro_2012': 'server1',
}

check_status_command = {'server1': 'export SGE_ROOT=/opt/sge; /opt/sge/bin/lx24-amd64/qstat -u ',
                        'server2': 'squeue -u '}

submit_command = {'server1': 'export SGE_ROOT=/opt/sge; /opt/sge/bin/lx24-amd64/qsub',
                  'server2': 'sbatch'}

delete_command = {'server1': 'export SGE_ROOT=/opt/sge; /opt/sge/bin/lx24-amd64/qdel ',
                  'server2': 'scancel '}

submit_filename = {'server1': 'submit.sh',
                   'server2': 'submit.sl'}

output_filename = {'gaussian03': 'input.log',
                   'qchem': 'output.out',
                   'molpro_2015': 'input.out',
                   'molpro_2012': 'input.out',
}

arc_path = 'path/to/ARC/'  # on local machine
rotor_scan_resolution = 30.0  # degrees. default: 10

# rotor validation parameters
inconsistency_az = 6   # maximum allowed inconsistency (kJ/mol) between initial and final rotor scan points. Default: 6
inconsistency_ab = 25  # maximum allowed inconsistency (kJ/mol) between consecutive points in the scan. Default: 25
maximum_barrier = 100  # maximum allowed barrier (kJ/mol) for a hindered rotor
