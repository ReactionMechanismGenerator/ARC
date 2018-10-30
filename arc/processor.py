#!/usr/bin/env python
# encoding: utf-8

from rmgpy.molecule import Molecule
from rmgpy import Species

from arc.molecule.conformer import ConformerSearch
from arc.molecule.rotors import Rotors
from arc.job.job import Job

##################################################################


class Processor(object):
    """
    ARC Processor class. Calls the scheduler and post processes results in Arkane. The attributes are:

    ================ =================== ===============================================================================
    Attribute        Type                Description
    ================ =================== ===============================================================================
    `species_list`    ``list``           Contains RMG ``Species`` objects
    `project`         ``str``            The project's name. Used for naming the directory.
    `level_of_theory` ``str``            FULL Level of theory, e.g. 'CBS-QB3',
                                           'CCSD(T)-F12a/aug-cc-pVTZ//B3LYP/6-311++G(3df,3pd)'...
    ================ =================== ===============================================================================
    """
    def __init__(self, species_list, project, level_of_theory, ):
        pass

