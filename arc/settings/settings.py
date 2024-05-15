"""
ARC's settings

You may keep a short version of this file in a local ".arc" folder under your home folder.
Any definitions made to the local file will take precedence over this file.
"""

import os
import string
import sys

# Users should update the following server dictionary.
# Instructions for RSA key generation can be found here:
# https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys--2
# If ARC is being executed on a server, and ESS are available on that server, define a server named 'local',
# for which only the cluster software and username are required.
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
#        'path': '/storage/group_name/',  # an absolute path on the server, under which ARC runs will be executed (e.g., '/storage/group_name/$USER/runs/ARC_Projects/project/')
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
        'cpus': 24,  # number of cpu's per node, optional (default: 8)
        'memory': 256,  # amount of memory per node in GB, optional (default: 16)
    },
    'server3': {
        'cluster_soft': 'PBS',
        'address': 'server3.host.edu',
        'un': '<username>',
        'key': 'path_to_rsa_key',
    },
    'local': {
        'cluster_soft': 'HTCondor',
        'un': '<username>',
        'cpus': 48,
        'queues': {'':''}, #{'queue_name':'HH:MM:SS'}
        'excluded_queues': ['queue_name1', 'queue_name2'],
    },
}

# List here servers you'd like to associate with specific ESS.
# An ordered list of servers indicates priority
# Keeping this dictionary empty will cause ARC to scan for software on the servers defined above
global_ess_settings = {
    'cfour': 'local',
    'gaussian': ['local', 'server2'],
    'gcn': 'local',
    'molpro': ['local', 'server2'],
    'onedmin': 'server1',
    'orca': 'local',
    'qchem': 'server1',
    'terachem': 'server1',
    'xtb': 'local',
    'xtb_gsm': 'local',
    'torchani': 'local',
    'openbabel': 'local',
}

# Electronic structure software ARC may access (use lowercase):
supported_ess = ['cfour', 'gaussian', 'molpro', 'orca', 'qchem', 'terachem', 'onedmin', 'xtb', 'torchani', 'openbabel']

# TS methods to try when appropriate for a reaction (other than user guesses which are always allowed):
ts_adapters = ['heuristics', 'AutoTST', 'GCN', 'xtb_gsm']

# List here job types to execute by default
default_job_types = {'conformers': True,      # defaults to True if not specified
                     'opt': True,             # defaults to True if not specified
                     'fine_grid': True,       # defaults to True if not specified
                     'freq': True,            # defaults to True if not specified
                     'sp': True,              # defaults to True if not specified
                     'rotors': True,          # defaults to True if not specified
                     'irc': True,             # defaults to True if not specified
                     'orbitals': False,       # defaults to False if not specified
                     'lennard_jones': False,  # defaults to False if not specified
                     'bde': False,            # defaults to False if not specified
                     }

# List here (complete or partial) phrases of methods or basis sets you'd like to associate to specific ESS
# Avoid ascribing the same phrase to more than one software, this may cause undeterministic assignment of software
# Format is levels_ess = {ess: ['phrase1', 'phrase2'], ess2: ['phrase3', 'phrase3']}
levels_ess = {
    'cfour': ['casscf'],
    'gaussian': ['apfd', 'b3lyp', 'm062x'],
    'molpro': ['ccsd', 'cisd', 'vpz'],
    'qchem': ['m06-2x'],
    'orca': ['dlpno'],
    'terachem': ['pbe'],
    'xtb': ['xtb', 'gfn'],
    'torchani': ['torchani'],
    'openbabel': ['mmff94s', 'mmff94', 'gaff', 'uff', 'ghemical'],
}

check_status_command = {'OGE': 'export SGE_ROOT=/opt/sge; /opt/sge/bin/lx24-amd64/qstat -u $USER',
                        'Slurm': '/usr/bin/squeue -u $USER',
                        'PBS': '/usr/local/bin/qstat -u $USER',
                        'HTCondor': """condor_q -cons 'Member(Jobstatus,{1,2})' -af:j '{"0","P","R","X","C","H",">","S"}[JobStatus]' RequestCpus RequestMemory JobName  '(Time() - EnteredCurrentStatus)'""",
                        }

submit_command = {'OGE': 'export SGE_ROOT=/opt/sge; /opt/sge/bin/lx24-amd64/qsub',
                  'Slurm': '/usr/bin/sbatch',
                  'PBS': '/usr/local/bin/qsub',
                  'HTCondor': 'condor_submit',
                  }

delete_command = {'OGE': 'export SGE_ROOT=/opt/sge; /opt/sge/bin/lx24-amd64/qdel',
                  'Slurm': '/usr/bin/scancel',
                  'PBS': '/usr/local/bin/qdel',
                  'HTCondor': 'condor_rm',
                  }

list_available_nodes_command = {
    'OGE': 'export SGE_ROOT=/opt/sge; /opt/sge/bin/lx24-amd64/qstat -f | grep "/8 " | grep "long" | grep -v "8/8"| grep -v "aAu"',
    'Slurm': 'sinfo -o "%n %t %O %E"',
    'PBS': 'pbsnodes',
    }

submit_filenames = {'OGE': 'submit.sh',
                    'Slurm': 'submit.sl',
                    'PBS': 'submit.sh',
                    'HTCondor': 'submit.sub',
                    }

t_max_format = {'OGE': 'hours',
                'Slurm': 'days',
                'PBS': 'hours',
                'HTCondor': 'hours',
                }

input_filenames = {'cfour': 'ZMAT',
                   'gaussian': 'input.gjf',
                   'molpro': 'input.in',
                   'onedmin': 'input.in',
                   'orca': 'input.in',
                   'qchem': 'input.in',
                   'terachem': 'input.in',
                   'xtb': 'input.sh',
                   }

output_filenames = {'cfour': 'output.out',
                    'gaussian': 'input.log',
                    'gcn': 'output.yml',
                    'molpro': 'input.out',
                    'onedmin': 'output.out',
                    'orca': 'input.log',
                    'qchem': 'output.out',
                    'terachem': 'output.out',
                    'torchani': 'output.yml',
                    'xtb': 'output.out',
                    'openbabel':'output.yml',
                    }

default_levels_of_theory = {'conformer': 'wb97xd/def2svp',  # it's recommended to choose a method with dispersion
                            'conf_generation': {'method': 'UFF',
                                                'solvation_method': 'SMD',
                                                'solvent': 'water'}, # good default for Gaussian
                            'ts_guesses': 'wb97xd/def2svp',
                            'opt': 'wb97xd/def2tzvp',  # good default for Gaussian
                            # 'opt': 'wb97m-v/def2tzvp',  # good default for QChem
                            'freq': 'wb97xd/def2tzvp',  # should be the same level as opt (to calc freq at min E)
                            'scan': 'wb97xd/def2tzvp',  # should be the same level as freq (to project out rotors)
                            'sp': 'ccsd(t)-f12/cc-pvtz-f12',  # This should be a level for which BAC is available
                            # 'sp': 'b3lyp/6-311+g(3df,2p)',
                            'irc': 'wb97xd/def2tzvp',  # should be the same level as opt
                            'orbitals': 'wb97x-d3/def2tzvp',  # save orbitals for visualization
                            'scan_for_composite': 'B3LYP/CBSB7',  # This is the frequency level of CBS-QB3
                            'freq_for_composite': 'B3LYP/CBSB7',  # This is the frequency level of CBS-QB3
                            'irc_for_composite': 'B3LYP/CBSB7',  # This is the frequency level of CBS-QB3
                            'orbitals_for_composite': 'B3LYP/CBSB7',  # This is the frequency level of CBS-QB3
                            }

# Software specific default settings
# Orca
# ARC accepts all the Orca options listed in the dictionary below. For specifying additional Orca options, please see
# documentation and Orca manual.
orca_default_options_dict = {
    'opt': {'keyword': {'opt_convergence': 'NormalOpt',
                        'fine_opt_convergence': 'TightOpt'}},
    'freq': {'keyword': {'use_num_freq': False}},
    'global': {'keyword': {'scf_convergence': 'TightSCF',
                           'dlpno_threshold': 'normalPNO'}},
}
tani_default_options_dict = {"model" : "ani2x", # available: 'ANI1ccx', 'ANI1x', 'ANI2x'
                             "device" : "cpu",  # available: 'cpu', 'cuda'
                             "engine" : "bfgs", # available:
                                                # 'BFGS': Broyden–Fletcher–Goldfarb–Shanno. This algorithm chooses each step from
                                                # the current atomic forces and an approximation of the Hessian matrix.
                                                # The Hessian is established from an initial guess which is gradually
                                                # improved as more forces are evaluated. Implemented in ASE.
                                                # 'SciPyFminBFGS': A Quasi-Newton method (Broydon-Fletcher-Goldfarb-Shanno).
                                                # An ASE interface to SciPy.
                                                # 'SciPyFminCG': A non-linear (Polak-Ribiere) conjugate gradient algorithm.
                                                # An ASE interface to SciPy.
                             "fmax" : 0.001,    # Make sure it is an int or a float.
                             "steps" : None}    # Make sure it is an int.
                             
ob_default_settings = {"FF" : "MMFF94s",
                       "opt_gradient_settings" : {"steps" : 2000,
                                                  "econv" : 1e-6}
                       }

# xTB-GSM
xtb_gsm_settings = {'sm_type': 'GSM',
                    'restart': 0,
                    'max_opt_iters': 160,
                    'step_opt_iters': 30,
                    'conv_tol': 0.0005,
                    'add_node_tol': 0.1,
                    'scaling': 1.0,
                    'ssm_dqmax': 0.8,
                    'growth_direction': 0,
                    'int_thresh': 2.0,
                    'min_spacing': 5.0,
                    'bond_fragments': 1,
                    'initial_opt': 0,
                    'final_opt': 150,
                    'product_limit': 100.0,
                    'ts_final_type': 1,
                    'nnodes': 15,
                    }

valid_chars = "-_[]=.,%s%s" % (string.ascii_letters, string.digits)

# A scan with better resolution (lower number here) takes more time to compute,
# but the automatically-derived rotor symmetry number is more likely to be correct.
rotor_scan_resolution = 8.0  # degrees. Default: 8.0

# rotor validation parameters
maximum_barrier = 40    # a rotor threshold (kJ/mol) above which the mode will be considered as vibrational if
                        # there's only one well. Default: 40 (~10 kcal/mol)
minimum_barrier = 1.0   # a rotor threshold (kJ/mol) below which it is considered a FreeRotor. Default: 1.0 kJ/mol
inconsistency_az = 5    # maximum allowed inconsistency (kJ/mol) between initial and final rotor scan points. Default: 5
inconsistency_ab = 0.3  # maximum allowed inconsistency between consecutive points in the scan given as a fraction
                        # of the maximum scan energy. Default: 30%
max_rotor_trsh = 4      # maximum number of times to troubleshoot the same rotor scan

# Thresholds for identifying significant changes in bond distance, bond angle,
# or torsion angle during a rotor scan. For a TS, only 'bond' and 'torsion' are considered.
preserve_params_in_scan = {
    'bond': 0.1,  # Default: 10% of the original bond length
    'angle': 10,  # Default: 10 degrees
    'dihedral': 30,  # Default: 30 degrees
}

# Coefficients to be used in a y = A * x ** b fit
# to determine the number of workers in a pipe job to execute in parallel vs. the number of processes.
# This is y = 2.0 * x ** 0.30 by default, corresponding the following output:
# 10 -> 4, 100 -> 8, 1000 -> 16, 1e4 -> 32, 1e5 -> 63.
# 'cap' is the maximal number of workers to use per pipe.
# If the number of processes is equal or lesser than 'max_one', only a single worker will be used.
# If the number of processes is greater than 'max_one' but equal or lesser than 'max_two',
# only two workers will be used.
workers_coeff = {'A': 2.0, 'b': 0.25, 'cap': 100, 'max_one': 3, 'max_two': 9}

# Default job memory, cpu, time settings
default_job_settings = {
    'job_total_memory_gb': 14,
    'job_cpu_cores': 8,
    'job_time_limit_hrs': 120,
    'job_max_server_node_memory_allocation': 0.95,  # e.g., at most 95% node memory will be used per job **if needed**
}

# Criteria for identification of imaginary frequencies for transition states.
# An imaginary frequency is valid if it is between the following range (in cm-1):
LOWEST_MAJOR_TS_FREQ, HIGHEST_MAJOR_TS_FREQ = 75.0, 10000.0

# default environment names for sister repos
TS_GCN_PYTHON, TANI_PYTHON, AUTOTST_PYTHON, ARC_PYTHON, XTB, OB_PYTHON = None, None, None, None, None, None
home = os.getenv("HOME") or os.path.expanduser("~")

tani_pypath_1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))),
                            'tani_env', 'bin', 'python')
tani_pypath_2 = os.path.join(home, 'mambaforge', 'envs', 'tani_env', 'bin', 'python')
tani_pypath_3 = os.path.join(home, 'anaconda3', 'envs', 'tani_env', 'bin', 'python')
tani_pypath_4 = os.path.join(home, 'miniconda3', 'envs', 'tani_env', 'bin', 'python')
tani_pypath_5 = os.path.join(home, '.conda', 'envs', 'tani_env', 'bin', 'python')
tani_pypath_6 = os.path.join('/Local/ce_dana', 'anaconda3', 'envs', 'tani_env', 'bin', 'python')
for tani_pypath in [tani_pypath_1, tani_pypath_2, tani_pypath_3, tani_pypath_4, tani_pypath_5, tani_pypath_6]:
    if os.path.isfile(tani_pypath):
        TANI_PYTHON = tani_pypath

ob_pypath_1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))),
                            'ob_env', 'bin', 'python')
ob_pypath_2 = os.path.join(home, 'mambaforge', 'envs', 'ob_env', 'bin', 'python')
ob_pypath_3 = os.path.join(home, 'anaconda3', 'envs', 'ob_env', 'bin', 'python')
ob_pypath_4 = os.path.join(home, 'miniconda3', 'envs', 'ob_env', 'bin', 'python')
ob_pypath_5 = os.path.join(home, '.conda', 'envs', 'ob_env', 'bin', 'python')
ob_pypath_6 = os.path.join('/Local/ce_dana', 'anaconda3', 'envs', 'ob_env', 'bin', 'python')
for ob_pypath in [ob_pypath_1, ob_pypath_2, ob_pypath_3, ob_pypath_4, ob_pypath_5, ob_pypath_6]:
    if os.path.isfile(ob_pypath):
        OB_PYTHON = ob_pypath
        break

gcn_pypath_1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))),
                            'ts_gcn', 'bin', 'python')
gcn_pypath_2 = os.path.join(home, 'anaconda3', 'envs', 'ts_gcn', 'bin', 'python')
gcn_pypath_3 = os.path.join(home, 'miniconda3', 'envs', 'ts_gcn', 'bin', 'python')
gcn_pypath_4 = os.path.join(home, '.conda', 'envs', 'ts_gcn', 'bin', 'python')
gcn_pypath_5 = os.path.join('/Local/ce_dana', 'anaconda3', 'envs', 'ts_gcn', 'bin', 'python')
for gcn_pypath in [gcn_pypath_1, gcn_pypath_2, gcn_pypath_3, gcn_pypath_4, gcn_pypath_5]:
    if os.path.isfile(gcn_pypath):
        TS_GCN_PYTHON = gcn_pypath
        break

autotst_pypath_1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))),
                                'tst_env', 'bin', 'python')
autotst_pypath_2 = os.path.join(home, 'anaconda3', 'envs', 'tst_env', 'bin', 'python')
autotst_pypath_3 = os.path.join(home, 'miniconda3', 'envs', 'tst_env', 'bin', 'python')
autotst_pypath_4 = os.path.join(home, '.conda', 'envs', 'tst_env', 'bin', 'python')
autotst_pypath_5 = os.path.join('/Local/ce_dana', 'anaconda3', 'envs', 'tst_env', 'bin', 'python')
for autotst_pypath in [autotst_pypath_1, autotst_pypath_2, autotst_pypath_3, autotst_pypath_4, autotst_pypath_5]:
    if os.path.isfile(autotst_pypath):
        AUTOTST_PYTHON = autotst_pypath
        break

paths = list()
paths.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))),
                            'xtb_env', 'bin', 'xtb'))
paths.append(os.path.join(home, 'anaconda3', 'envs', 'xtb_env', 'bin', 'xtb'))
paths.append(os.path.join(home, 'miniconda3', 'envs', 'xtb_env', 'bin', 'xtb'))
paths.append(os.path.join(home, '.conda', 'envs', 'xtb_env', 'bin', 'xtb'))
paths.append(os.path.join('/Local/ce_dana', 'anaconda3', 'envs', 'xtb_env', 'bin', 'xtb'))
for xtb_path in paths:
    if os.path.isfile(xtb_path):
        XTB = xtb_path
        break

arc_pypath_1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))),
                            'arc_env', 'bin', 'python')
arc_pypath_2 = os.path.join(home, 'anaconda3', 'envs', 'arc_env', 'bin', 'python')
arc_pypath_3 = os.path.join(home, 'miniconda3', 'envs', 'arc_env', 'bin', 'python')
arc_pypath_4 = os.path.join(home, '.conda', 'envs', 'arc_env', 'bin', 'python')
arc_pypath_5 = os.path.join('/Local/ce_dana', 'anaconda3', 'envs', 'arc_env', 'bin', 'python')
for arc_pypath in [arc_pypath_1, arc_pypath_2, arc_pypath_3, arc_pypath_4, arc_pypath_5]:
    if os.path.isfile(arc_pypath):
        ARC_PYTHON = arc_pypath
        break
