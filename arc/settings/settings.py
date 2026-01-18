"""
ARC's settings

You may keep a short version of this file in a local ".arc" folder under your home folder.
Any definitions made to the local file will take precedence over this file.
"""

import glob
import os
import string
import sys
import shutil

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
        'max_simultaneous_jobs': 10,  # optional, "check_status_command" must be set to only return jobs for your user
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
        'queues': {'':''},  # {'queue_name':'HH:MM:SS'}
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
    'mockter': 'local',
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
supported_ess = ['cfour', 'gaussian', 'mockter', 'molpro', 'orca', 'qchem', 'terachem', 'onedmin', 'xtb', 'torchani', 'openbabel']

# TS methods to try when appropriate for a reaction (other than user guesses which are always allowed):
ts_adapters = ['heuristics', 'AutoTST', 'GCN', 'xtb_gsm']

# List here job types to execute by default
default_job_types = {'conf_opt': True,        # defaults to True if not specified
                     'conf_sp': False,        # defaults to False if not specified
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
    'mockter': ['mock'],
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
                   'mockter': 'input.yml',
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
                    'mockter': 'output.yml',
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

# ARC families folder path
ARC_FAMILIES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'families')

# Default environment names for sister repos
TS_GCN_PYTHON, TANI_PYTHON, AUTOTST_PYTHON, ARC_PYTHON, XTB, OB_PYTHON, RMG_PYTHON, RMG_PATH, RMG_DB_PATH = \
    None, None, None, None, None, None, None, None, None

home = os.getenv("HOME") or os.path.expanduser("~")

# Helper function to find executables in common paths
def find_executable(env_name, executable_name='python'):
    candidate_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))), env_name, 'bin', executable_name),
        os.path.join(home, 'mambaforge', 'envs', env_name, 'bin', executable_name),
        os.path.join(home, 'anaconda3', 'envs', env_name, 'bin', executable_name),
        os.path.join(home, 'miniconda3', 'envs', env_name, 'bin', executable_name),
        os.path.join(home, '.conda', 'envs', env_name, 'bin', executable_name),
        os.path.join(home, 'micromamba', 'envs', env_name, 'bin', executable_name),
        os.path.join(home, '.micromamba', 'envs', env_name, 'bin', executable_name),
        os.path.join(home, '.local', 'share', 'mamba', 'envs', env_name, 'bin', executable_name),
        os.path.join('/Local/ce_dana', 'anaconda3', 'envs', env_name, 'bin', executable_name),
    ]
    mamba_root = os.getenv('MAMBA_ROOT_PREFIX')
    if mamba_root:
        candidate_paths.append(os.path.join(mamba_root, 'envs', env_name, 'bin', executable_name))
    conda_prefix = os.getenv('CONDA_PREFIX')
    if conda_prefix:
        candidate_paths.append(os.path.join(os.path.dirname(conda_prefix), 'envs', env_name, 'bin', executable_name))
    conda_exe = os.getenv('CONDA_EXE')
    if conda_exe:
        conda_base = os.path.dirname(os.path.dirname(conda_exe))
        candidate_paths.append(os.path.join(conda_base, 'envs', env_name, 'bin', executable_name))
    conda_envs_path = os.getenv('CONDA_ENVS_PATH')
    if conda_envs_path:
        for path in conda_envs_path.split(os.pathsep):
            candidate_paths.append(os.path.join(path, env_name, 'bin', executable_name))
    for path in candidate_paths:
        if os.path.isfile(path):
            return path
    return None

TANI_PYTHON = find_executable('tani_env')
OB_PYTHON = find_executable('ob_env')
TS_GCN_PYTHON = find_executable('ts_gcn')
AUTOTST_PYTHON = find_executable('tst_env')
ARC_PYTHON = find_executable('arc_env')
RMG_ENV_NAME = 'rmg_env'
RMG_PYTHON = find_executable('rmg_env')
XTB = find_executable('xtb_env', 'xtb')

# Set RMG_DB_PATH with fallback methods
rmg_db_candidates, rmg_candidates = list(), list()


def add_rmg_db_candidates(prefix: str) -> None:
    """Add RMG-database candidates relative to a conda/mamba env prefix."""
    if not prefix:
        return
    rmg_db_candidates.extend([
        os.path.join(prefix, 'share', 'RMG-database'),
        os.path.join(prefix, 'share', 'rmg-database'),
        os.path.join(prefix, 'share', 'rmg', 'database'),
        os.path.join(prefix, 'share', 'rmg_database'),
        os.path.join(prefix, 'share', 'RMG_database'),
        os.path.join(prefix, 'share', 'rmgdatabase'),
        os.path.join(prefix, 'share', 'RMGdatabase'),
    ])
    rmg_db_candidates.extend(glob.glob(os.path.join(prefix, 'lib', 'python*', 'site-packages', 'RMG-database')))
    rmg_db_candidates.extend(glob.glob(os.path.join(prefix, 'lib', 'python*', 'site-packages', 'rmg-database')))
    rmg_db_candidates.extend(glob.glob(os.path.join(prefix, 'lib', 'python*', 'site-packages', 'rmg_database')))
    rmg_db_candidates.extend(glob.glob(os.path.join(prefix, 'lib', 'python*', 'site-packages', 'RMG_database')))
    rmg_db_candidates.extend(glob.glob(os.path.join(prefix, 'lib', 'python*', 'site-packages', 'rmgdatabase')))
    rmg_db_candidates.extend(glob.glob(os.path.join(prefix, 'lib', 'python*', 'site-packages', 'RMGdatabase')))
    for candidate in glob.glob(os.path.join(prefix, 'share', '**', 'recommended.py'), recursive=True):
        if candidate.endswith(os.path.join('input', 'kinetics', 'families', 'recommended.py')):
            rmg_db_candidates.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(candidate)))))

# Use exported RMG_PATH & RMG_DB_PATH if available
exported_rmg_path = os.getenv("RMG_PATH")
exported_rmg_db_path = os.getenv("RMG_DB_PATH") or os.getenv("RMG_DATABASE")
if exported_rmg_path:
    rmg_candidates.append(exported_rmg_path)
if exported_rmg_db_path:
    rmg_db_candidates.append(exported_rmg_db_path)

gw = os.getenv("GITHUB_WORKSPACE")  # e.g., /home/runner/work/ARC/ARC
if gw:
    rmg_candidates.append(os.path.join(gw, 'RMG-Py'))
    rmg_db_candidates.append(os.path.join(gw, 'RMG-database'))

for python_path in sys.path:
    if 'RMG-database' in python_path or 'rmgdatabase' in python_path or 'rmg_database' in python_path:
        rmg_db_candidates.append(python_path)
    if 'RMG-Py' in python_path:
        rmg_db_candidates.append(os.path.join(os.path.dirname(python_path), 'RMG-database'))

for p in sys.path:
    if 'RMG-Py' in p:
        rmg_candidates.append(p)
        rmg_db_candidates.append(os.path.join(os.path.dirname(p), 'RMG-database'))
    if 'RMG-database' in p or 'rmgdatabase' in p or 'rmg_database' in p:
        rmg_db_candidates.append(p)

add_rmg_db_candidates(os.path.dirname(os.path.dirname(sys.executable)))
if RMG_PYTHON:
    add_rmg_db_candidates(os.path.dirname(os.path.dirname(RMG_PYTHON)))
if os.getenv('MAMBA_ROOT_PREFIX'):
    add_rmg_db_candidates(os.path.join(os.getenv('MAMBA_ROOT_PREFIX'), 'envs', 'rmg_env'))
if os.getenv('CONDA_PREFIX'):
    add_rmg_db_candidates(os.path.join(os.path.dirname(os.getenv('CONDA_PREFIX')), 'envs', 'rmg_env'))
if os.getenv('CONDA_EXE'):
    conda_base = os.path.dirname(os.path.dirname(os.getenv('CONDA_EXE')))
    add_rmg_db_candidates(os.path.join(conda_base, 'envs', 'rmg_env'))
if os.getenv('CONDA_ENVS_PATH'):
    for path in os.getenv('CONDA_ENVS_PATH').split(os.pathsep):
        add_rmg_db_candidates(os.path.join(path, 'rmg_env'))

rmg_candidates.extend([
    os.path.join(home, 'Code', 'RMG-Py'),
    os.path.join(home, 'runner', 'work', 'ARC', 'ARC', 'RMG-Py')
])
rmg_db_candidates.extend([
    os.path.join(home, 'Code', 'RMG-database'),
    os.path.join(home, 'runner', 'work', 'ARC', 'ARC', 'RMG-database')
])

# Finalize RMG_PATH and RMG_DB_PATH
for path in rmg_candidates:
    if path and os.path.isdir(path):
        RMG_PATH = path
        break
for path in rmg_db_candidates:
    if path and os.path.isdir(path):
        RMG_DB_PATH = path
        break



def parse_version(folder_name):
    """
    Parses the version from the folder name and returns a tuple for comparison.
    Supports versions like: 3.0.2, v212, 2.1, 2
    """
    version_regex = re.compile(r"(?:v?(\d+)(?:\.(\d+))?(?:\.(\d+))?)", re.IGNORECASE)
    match = version_regex.search(folder_name)
    if not match:
        return (0, 0, 0)

    major = int(match.group(1)) if match.group(1) else 0
    minor = int(match.group(2)) if match.group(2) else 0
    patch = int(match.group(3)) if match.group(3) else 0

    # Example: v212 → (2, 1, 2)
    if major >= 100 and match.group(2) is None and match.group(3) is None:
        s = str(major).rjust(3, "0")
        major = int(s[0])
        minor = int(s[1])
        patch = int(s[2])

    return (major, minor, patch)


def find_highest_version_in_directory(directory, name_contains):
    """
    Finds the file with the highest version in a directory containing a specific string.
    """
    if not os.path.exists(directory):
        return None

    highest_version_path = None
    highest_version = ()

    for folder in os.listdir(directory):
        file_path = os.path.join(directory, folder)
        if name_contains.lower() in folder.lower() and os.path.isdir(file_path):
            crest_path = os.path.join(file_path, "crest")
            if os.path.isfile(crest_path) and os.access(crest_path, os.X_OK):
                version = parse_version(folder)
                if highest_version == () or version > highest_version:
                    highest_version = version
                    highest_version_path = crest_path
    return highest_version_path


def find_crest_executable():
    """
    Returns (crest_path, env_cmd):

    - crest_path: full path to 'crest'
    - env_cmd: shell snippet to activate its environment (may be "")
    """
    # Priority 1: /Local/ce_dana standalone builds
    crest_path = find_highest_version_in_directory("/Local/ce_dana", "crest")
    if crest_path and os.path.isfile(crest_path) and os.access(crest_path, os.X_OK):
        # Standalone binary: no env activation needed
        return crest_path, ""

    # Priority 2: Conda/Mamba/Micromamba envs
    home = os.path.expanduser("~")
    potential_env_paths = [
        os.path.join(home, "anaconda3", "envs", "crest_env", "bin", "crest"),
        os.path.join(home, "miniconda3", "envs", "crest_env", "bin", "crest"),
        os.path.join(home, "miniforge3", "envs", "crest_env", "bin", "crest"),
        os.path.join(home, ".conda", "envs", "crest_env", "bin", "crest"),
        os.path.join(home, "mambaforge", "envs", "crest_env", "bin", "crest"),
        os.path.join(home, "micromamba", "envs", "crest_env", "bin", "crest"),
    ]

    # Also check the current env's bin
    current_env_bin = os.path.dirname(sys.executable)
    potential_env_paths.insert(0, os.path.join(current_env_bin, "crest"))

    for crest_path in potential_env_paths:
        if os.path.isfile(crest_path) and os.access(crest_path, os.X_OK):
            # env_root = .../anaconda3 or .../miniforge3 or .../mambaforge etc.
            env_root = crest_path.split("/envs/crest_env/")[0]
            if "micromamba" in crest_path:
                env_cmd = (
                    f"source {env_root}/etc/profile.d/micromamba.sh && "
                    f"micromamba activate crest_env"
                )
            elif any(
                name in env_root
                for name in ("anaconda3", "miniconda3", "miniforge3", "mambaforge", ".conda")
            ):
                env_cmd = (
                    f"source {env_root}/etc/profile.d/conda.sh && "
                    f"conda activate crest_env"
                )
            else:
                # If for some reason it's just a random prefix with crest in bin
                env_cmd = ""
            return crest_path, env_cmd

    # Priority 3: PATH
    crest_in_path = shutil.which("crest")
    if crest_in_path:
        return crest_in_path, ""

    return None, None


CREST_PATH, CREST_ENV_PATH = find_crest_executable()
