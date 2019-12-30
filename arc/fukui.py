#!/usr/bin/env python3
# encoding: utf-8


import os
import time
import shutil

from arc.common import arc_path, read_yaml_file, save_yaml_file, get_logger
from arc.job.submit import submit_scripts
from arc.job.inputs import input_files
from arc.job.local import check_running_jobs_ids, submit_job
from arc.species.conformers import generate_conformers, determine_rotors
from arc.species.converter import xyz_to_str, str_to_xyz, molecules_from_xyz
from arc.species.species import ARCSpecies


def fukui(process_xyz=False, max_jobs=25):
    """
    Handle the Fukui conformers generation

    Args:
        process_xyz (bool): Whether to process the xyz files
        max_jobs (int): max number of running jobs on the server
    """

    # logger = get_logger()

    xyz_input_path = os.path.join('xyz')
    yml_input_path = os.path.join('yml')
    splits_path = os.path.join('splits')

    if process_xyz:
        # process the xyz's

        file_names = list()
        for (_, _, files) in os.walk(xyz_input_path):
            file_names.extend(files)
            break  # don't continue to explore subdirectories

        data = dict()
        data_index = 0
        for i in range(len(file_names)):
            with open(os.path.join(xyz_input_path, file_names[i]), 'r') as f:
                lines = f.readlines()
                number_of_atoms = int(lines[0])
                smiles = lines[1]
                xyz = str_to_xyz('\n'.join(lines[2:]))
                if len(xyz['symbols']) != number_of_atoms:
                    print(f'Wrong number of atoms for file {file_names[i]} with smiles {smiles}. '
                          f'Expected {number_of_atoms} but got {len(xyz["symbols"])}')
                label = file_names[i].split('.')[0]
                has_br = 'Br' in xyz['symbols']
                if has_br:
                    print(f'Skipping conformer {i} {file_names[i]} {smiles} since it has Br.')
                    continue
                try:
                    torsions = determine_rotors([molecules_from_xyz(xyz)[1]])[0]
                except:
                    torsions = None
                if len(torsions) > 10:
                    print(f'Skipping conformer {i} {file_names[i]} {smiles} since it has {len(torsions)} torsions.')
                    continue
                data[label] = {'smiles': smiles,
                               'original_xyz': xyz_to_str(xyz),
                               'torsions': torsions,
                               }
            if i > 98 and (i + 1) % 100 == 0:
                split_path = os.path.join(yml_input_path, str(data_index))
                if not os.path.isdir(split_path):
                    os.mkdir(split_path)
                yml_path = os.path.join(split_path, 'yml')
                if not os.path.isdir(yml_path):
                    os.mkdir(yml_path)
                yml_path = os.path.join(yml_path, f'input.yml')
                save_yaml_file(yml_path, data)
                with open(os.path.join(split_path, 'input.py'), 'w') as f:
                    f.write(input_files['fukui'])
                with open(os.path.join(split_path, 'submit.sl'), 'w') as f:
                    f.write(submit_scripts['fukui'].format(name=f'fukui_{data_index}'))

                print(f'\nSaved {len(list(data.keys()))} entries in {yml_path} (total {i + 1} entries)\n')
                data_index += 1
                data = dict()
        print('\n\nDone processing xyzs.\n\n\n')
    else:
        print('not processing xyzs')

    # generate conformers from yml
    print('Generating conformers from yml files...')
    yml_files = list()
    for (_, dirnames, filenames) in os.walk(yml_input_path):
        yml_files.extend(filenames)
        break  # don't continue to explore subdirectories
    if '.' in yml_files:
        yml_files.pop(yml_files.index('.'))
    if '..' in yml_files:
        yml_files.pop(yml_files.index('..'))

    fukui_job_ids = list()  # add job_id when submitting

    for yml_file in yml_files:
        index = yml_file.split('.')[0].split('input')[1]
        submitted = False
        split_path = os.path.join(splits_path, f'split_{index}')
        if not os.path.isdir(split_path):
            os.makedirs(split_path)
        shutil.copyfile(src=os.path.join(yml_input_path, yml_file),
                        dst=os.path.join(split_path, 'input.yml'))
        with open(os.path.join(split_path, 'input.py'), 'w') as f:
            f.write(input_files['fukui'])
        with open(os.path.join(split_path, 'submit.sl'), 'w') as f:
            f.write(submit_scripts['fukui'].format(name=f'f_{index}'))
        while not submitted:
            running_jobs = check_running_jobs_ids()
            fukui_running_jobs = [job_id for job_id in running_jobs if job_id in fukui_job_ids]
            if len(fukui_running_jobs) < max_jobs:
                print(f'submitting job f_{index}')
                job_id = submit_job(path=split_path)
                if job_id:
                    fukui_job_ids.extend(job_id)
                    submitted = True

            if not submitted:
                time.sleep(60)  # wait 60 sec before bugging the server again.

