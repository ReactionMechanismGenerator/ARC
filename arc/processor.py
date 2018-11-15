#!/usr/bin/env python
# encoding: utf-8

import os
import logging

from arkane.input import species as arkane_species
from arkane.statmech import StatMechJob
from arkane.thermo import ThermoJob

from arc.settings import arc_path
from arc.job.input import input_files
from arc import plotter

##################################################################


class Processor(object):
    """
    ARC Processor class. Post processes results in Arkane. The attributes are:

    ================ =================== ===============================================================================
    Attribute        Type                Description
    ================ =================== ===============================================================================
    `project`         ``str``            The project's name. Used for naming the directory.
    `species_dict`    ``dict``           Keys are labels, values are ARCSpecies objects
    ================ =================== ===============================================================================
    """
    def __init__(self, project, species_dict, output):
        self.project = project
        self.species_dict = species_dict
        self.output = output
        self.process()

    def process(self):
        for species in self.species_dict.itervalues():
            if self.output[species.label]['status'] == 'converged':
                linear = False  # TODO
                species.determine_symmetry()
                multiplicity = species.multiplicity
                model_chemistry = 'CCSD(T)-F12/aug-cc-pVTZ'.lower()  # TODO
                sp_path = self.output[species.label]['sp']
                opt_path = self.output[species.label]['geo']
                freq_path = self.output[species.label]['freq']
                rotors = ''
                for i in xrange(species.number_of_rotors):
                    if species.rotors_dict[i]['success']:
                        if not rotors:
                            rotors = '\n\nrotors = ['
                        if rotors[-1] == ',':
                            rotors += '\n'
                        rotor_path = species.rotors_dict[i]['scan_path']
                        pivots = str(species.rotors_dict[i]['pivots'])
                        top = str(species.rotors_dict[i]['top'])
                        rotor_symmetry = 1  # TODO
                        rotors += input_files['arkane_rotor'].format(rotor_path=rotor_path, pivots=pivots, top=top,
                                                                     symmetry=rotor_symmetry)
                if rotors:
                    rotors += ']'
                # write the Arkane species input file
                input_file_path = os.path.join(arc_path, 'Projects', self.project, species.label, 'arkane.py')
                output_dir = os.path.join(arc_path, 'Projects', self.project, 'output', species.label)
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                output_file_path = os.path.join(output_dir, species.label + '.py')
                input_file = input_files['arkane_species']
                input_file= input_file.format(linear=linear, symmetry=species.external_symmetry,
                                              multiplicity=multiplicity, optical=species.optical_isomers,
                                              model_chemistry=model_chemistry, sp_path=sp_path, opt_path=opt_path,
                                              freq_path=freq_path, rotors=rotors)
                with open(input_file_path, 'wb') as f:
                    f.write(input_file)
                spec = arkane_species(species.label, input_file_path)
                stat_mech_job = StatMechJob(spec, input_file_path)
                stat_mech_job.modelChemistry = model_chemistry
                stat_mech_job.execute(outputFile=output_file_path, plot=True)
                thermo_job = ThermoJob(spec, 'NASA')
                thermo_job.execute(outputFile=output_file_path, plot=True)
                # plotter.log_thermo(species)


