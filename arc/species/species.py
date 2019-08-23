#!/usr/bin/env python
# encoding: utf-8

"""
A module for representing stationary points (chemical species and transition states).
If the species is a transition state (TS), its ``ts_guesses`` attribute will have one or more ``TSGuess`` objects.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import numpy as np
import datetime

from arkane.common import ArkaneSpecies, symbol_by_number
from arkane.statmech import determine_qm_software, is_linear
from rmgpy.molecule.molecule import Atom, Molecule
from rmgpy.reaction import Reaction
from rmgpy.species import Species
from rmgpy.statmech import NonlinearRotor, LinearRotor
from rmgpy.molecule.resonance import generate_kekule_structure
from rmgpy.transport import TransportData
import rmgpy.molecule.element as elements
from rmgpy.exceptions import InvalidAdjacencyListError

from arc.common import get_logger, get_atom_radius, determine_symmetry
from arc.arc_exceptions import SpeciesError, RotorError, InputError, TSError, SanitizationError
from arc.settings import default_ts_methods, valid_chars, minimum_barrier
from arc.parser import parse_xyz_from_file, parse_dipole_moment, parse_polarizability, process_conformers_file
from arc.species.converter import get_xyz_string, get_xyz_matrix, rdkit_conf_from_mol, standardize_xyz_string,\
    molecules_from_xyz, rmg_mol_from_inchi, order_atoms_in_mol_list, check_isomorphism, set_rdkit_dihedrals,\
    translate_to_center_of_mass
from arc.species import conformers
from arc.ts import atst

##################################################################

logger = get_logger()


class ARCSpecies(object):
    """
    A class for representing stationary points.

    Dictionary structure::

        rotors_dict: {1: {'pivots': pivots_list,
                          'top': top_list,
                          'scan': scan_list,
                          'success': ``bool``.
                          'invalidation_reason': ``str``,
                          'times_dihedral_set': ``int``,
                          'scan_path': <path to scan output file>},
                          'max_e': ``float``,  # in kJ/mol
                          'symmetry': ``int``,
                      2: {}, ...
                     }

    Args:
        label (str, optional): The species label.
        is_ts (bool, optional): Whether or not the species represents a transition state.
        rmg_species (Species, optional): An RMG Species object to be converted to an ARCSpecies object.
        mol (Molecule, optional): An ``RMG Molecule`` object used for BAC determination.
                                  Atom order corresponds to the order in .initial_xyz
        xyz (str or list, optional): Entries are either string-format coordinates or file paths.
                                     (If there's only one entry, it could be given directly as a string, not in a list).
                                     The file paths could direct to either a .xyz file, ARC conformers
                                     (w/ or w/o energies), or an ESS log/input files.
        multiplicity (int, optional): The species' multiplicity. Can be determined from the adjlist/smiles/xyz
                                      (the algorithm assumes it's either a singlet or a doublet).
        charge (int, optional): The species' net charge. Assumed to be 0 be default.
        smiles (str, optional): A `SMILES <https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system>`_
                                representation for the species 2D graph.
        adjlist (str, optional): An `RMG adjacency list
                                 <http://reactionmechanismgenerator.github.io/RMG-Py/reference/molecule/adjlist.html>`_
                                 representation for the species 2D graph.
        inchi (str, optional): An `InChI <https://www.inchi-trust.org/>`_ representation for the species 2D graph.
        bond_corrections (dict, optional): The bond additivity corrections (BAC) to be used. Determined from the
                                           structure if not directly given.
        generate_thermo (bool, optional): Whether ot not to calculate thermodynamic properties for this species.
        species_dict (dict, optional): A dictionary to create this object from (used when restarting ARC).
        yml_path (str, optional): Path to an Arkane YAML file representing a species (for loading the object).
        ts_methods (list, optional): Methods to try for generating TS guesses. If Species is a TS and `ts_methods` is an
                                     empty list, then xyz (user guess) must be given. If `ts_methods` is None, it will
                                     be set to the default methods.
        ts_number (int, optional): An auto-generated number associating the TS ARCSpecies object with the corresponding
                                   :ref:`ARCReaction <reaction>` object.
        rxn_label (str, optional): The reaction string (relevant for TSs).
        external_symmetry (int, optional): The external symmetry of the species (not including rotor symmetries).
        optical_isomers (int, optional): Whether (=2) or not (=1) the species has chiral center/s.
        run_time (timedelta, optional): Overall species execution time.
        checkfile (str, optional): The local path to the latest checkfile by Gaussian for the species.
        number_of_radicals (int, optional): The number of radicals (inputted by the user, ARC won't attempt to determine
                                            it). Defaults to None. Important, e.g., if a Species is a bi-rad singlet,
                                            in which case the job should be unrestricted, but the multiplicity does not
                                            have the required information to make that decision (r vs. u).
        force_field (str, optional): The force field to be used for conformer screening. The default is MMFF94.
                                     Other optional force fields are MMFF94s, UFF, or GAFF (not recommended, slow).
                                     If 'fit' is specified for this parameter, some initial MMFF94 conformers will be
                                     generated, then force field parameters will be fitted for this molecule and
                                     conformers will be re-run with the fitted force field (recommended for drug-like
                                     species and species with many heteroatoms). Another option is specifying 'cheap',
                                     and the "old" RDKit embedding method will be used.
        svpfit_output_file (str, optional): The path to a Gaussian output file of an SVP Fit job if previously ran
                                            (otherwise, this job will be spawned if running Gromacs).
        bdes (list): Specifying for which bonds should bond dissociation energies be calculated.
                     Entries are bonded atom indices tuples (1-indexed). An 'all_h' string entry is also allowed,
                     triggering BDE calculations for all hydrogen atoms in the molecule.

    Attributes:
        label (str): The species' label.
        multiplicity (int): The species' multiplicity. Can be determined from the adjlist/smiles/xyz
                            (the algorithm assumes it's either a singlet or a doublet).
        charge (int): The species' net charge. Assumed to be 0 be default.
        number_of_radicals (int): The number of radicals (inputted by the user, ARC won't attempt to determine it).
                                  Defaults to None. Important, e.g., if a Species is a bi-rad singlet, in which case
                                  the job should be unrestricted, but the multiplicity does not have the required
                                  information to make that decision (r vs. u).
        e_elect (float): The total electronic energy (without ZPE) at the chosen sp level, in kJ/mol.
        e0 (float): The 0 Kelvin energy (total electronic energy plus ZPE) at the chosen sp level, in kJ/mol.
        is_ts (bool):  Whether or not the species represents a transition state.
        number_of_rotors (int): The number of potential rotors to scan.
        rotors_dict (dict): A dictionary of rotors. structure given below.
        conformers (list): A list of selected conformers XYZs.
        conformer_energies (list): A list of conformers E0 (in kJ/mol).
        cheap_conformer (str): A string format xyz of a cheap conformer (not necessarily the best/lowest one).
        most_stable_conformer (str): A string format xyz of the best/lowest conformer ARC found.
        recent_md_conformer (list): A length three list containing the string format xyz of the recent conformer
                                    generated by MD, the energy in kJ/mol, and the number of MD runs. Used to detect
                                    when the MD algorithm converges on a single structure.
        svpfit_output_file (str): The path to a Gaussian output file of an SVP Fit job if previously ran
                                  (otherwise, this job will be spawned if running Gromacs).
        initial_xyz (str): The initial geometry guess.
        final_xyz (str): The optimized species geometry.
        radius (float): The species radius in Angstrom.
        opt_level (str): Level of theory for geometry optimization. Saved for archiving.
        number_of_atoms (int): The number of atoms in the species/TS.
        mol (Molecule): An ``RMG Molecule`` object used for BAC determination.
                        Atom order corresponds to the order in .initial_xyz
        mol_list (list): A list of localized structures generated from 'mol', if possible.
        rmg_species (Species): An RMG Species object to be converted to an ARCSpecies object.
        bond_corrections (dict): The bond additivity corrections (BAC) to be used. Determined from the structure
                                 if not directly given.
        run_time (timedelta): Overall species execution time.
        t1 (float): The T1 diagnostic parameter from Molpro.
        neg_freqs_trshed (list): A list of negative frequencies this species was troubleshooted for.
        generate_thermo (bool): Whether ot not to calculate thermodynamic properties for this species.
        thermo (HeatCapacityModel): The thermodata calculated by ARC.
        rmg_thermo (HeatCapacityModel): The thermodata generated by RMG for comparison.
        long_thermo_description (str): A description for the species entry in the thermo library outputed.
        ts_methods (list): Methods to try for generating TS guesses. If Species is a TS and `ts_methods` is an empty
                           list, then xyz (user guess) must be given. If `ts_methods` is None, it will be set to the
                           default methods.
        ts_guesses (list): A list of TSGuess objects for each of the specified methods.
        successful_methods (list): Methods used to generate a TS guess that successfully generated an XYZ guess.
        unsuccessful_methods (list): Methods used to generate a TS guess that were unsuccessfully.
        chosen_ts (int): The TSGuess index corresponding to the chosen TS conformer used for optimization.
        chosen_ts_method (str): The TS method that was actually used for optimization.
        ts_conf_spawned (bool): Whether conformers were already spawned for the Species (representing a TS) based on its
                                TSGuess objects.
        ts_number (int): An auto-generated number associating the TS ARCSpecies object with the corresponding
                         :ref:`ARCReaction <reaction>` object.
        ts_report (str): A description of all methods used for guessing a TS and their ranking.
        rxn_label (str): The reaction string (relevant for TSs).
        arkane_file (str): Path to the Arkane Species file generated in Processor.
        yml_path (str): Path to an Arkane YAML file representing a species (for loading the object).
        checkfile (str): The local path to the latest checkfile by Gaussian for the species.
        external_symmetry (int): The external symmetry of the species (not including rotor symmetries).
        optical_isomers (int): Whether (=2) or not (=1) the species has chiral center/s.
        transport_data (TransportData): A placeholder for updating transport properties after Lennard-Jones
                                        calculation (using OneDMin).
        force_field (str): The force field to be used for conformer screening. The default is MMFF94.
                           Other optional force fields are MMFF94s, UFF, or GAFF (not recommended, slow).
                           If 'fit' is specified for this parameter, some initial MMFF94 conformers will be generated,
                           then force field parameters will be fitted for this molecule and conformers will be re-run
                           with the fitted force field (recommended for drug-like species and species with many
                           heteroatoms). Another option is specifying 'cheap', and the "old" RDKit embedding method
                           will be used.
        conf_is_isomorphic (bool): Whether the lowest conformer is isomorphic with the 2D graph representation
                                   of the species. `True` if it is. Defaults to `None`. If `True`, an isomorphism check
                                   will be strictly enforced for the final optimized coordinates.
        conformers_before_opt (tuple): Conformers XYZs of a species before optimization.
        bdes (list): Specifying for which bonds should bond dissociation energies be calculated.
                     Entries are bonded atom indices tuples (1-indexed). An 'all_h' string entry is also allowed,
                     triggering BDE calculations for all hydrogen atoms in the molecule.

    """
    def __init__(self, label=None, is_ts=False, rmg_species=None, mol=None, xyz=None, multiplicity=None, charge=None,
                 smiles='', adjlist='', inchi='', bond_corrections=None, generate_thermo=True, species_dict=None,
                 yml_path=None, ts_methods=None, ts_number=None, rxn_label=None, external_symmetry=None,
                 optical_isomers=None, run_time=None, checkfile=None, number_of_radicals=None, force_field='MMFF94',
                 svpfit_output_file=None, bdes=None):
        self.t1 = None
        self.ts_number = ts_number
        self.conformers = list()
        self.conformers_before_opt = None
        self.ts_guesses = list()
        self.cheap_conformer = None
        self.most_stable_conformer = None
        self.recent_md_conformer = None
        self.conformer_energies = list()
        self.initial_xyz = None
        self.thermo = None
        self.rmg_thermo = None
        self.rmg_kinetics = None
        self._number_of_atoms = None
        self._number_of_heavy_atoms = None
        self._radius = None
        self.mol = mol
        self.mol_list = None
        self.multiplicity = multiplicity
        self.number_of_radicals = number_of_radicals
        self.external_symmetry = external_symmetry
        self.optical_isomers = optical_isomers
        self.charge = charge
        self.run_time = run_time
        self.checkfile = checkfile
        self.transport_data = TransportData()
        smiles, adjlist, inchi = str(smiles), str(adjlist), str(inchi)

        if species_dict is not None:
            # Reading from a dictionary
            self.from_dict(species_dict=species_dict)
        else:
            # Not reading from a dictionary
            self.force_field = force_field
            self.is_ts = is_ts
            self.ts_conf_spawned = False
            self.e_elect = None
            self.e0 = None
            self.arkane_file = None
            self.svpfit_output_file = svpfit_output_file
            self.conf_is_isomorphic = None
            self.bdes = bdes
            if self.bdes is not None and not isinstance(self.bdes, list):
                raise SpeciesError('The .bdes argument must be a list, got {0} which is a {1}'.format(
                                    self.bdes, type(self.bdes)))
            if self.is_ts:
                if ts_methods is None:
                    self.ts_methods = default_ts_methods
                elif isinstance(ts_methods, list):
                    self.ts_methods = ts_methods
                    if not self.ts_methods:
                        self.ts_methods = ['user guess']
                else:
                    raise TSError('ts_methods must be a list, got {0} of type {1}'.format(ts_methods, type(ts_methods)))
            else:
                self.ts_methods = None
            self.rxn_label = rxn_label
            self.successful_methods = list()
            self.unsuccessful_methods = list()
            self.chosen_ts_method = None
            self.chosen_ts = None
            self.generate_thermo = generate_thermo if not self.is_ts else False
            self.long_thermo_description = ''
            self.opt_level = None
            self.ts_report = ''
            self.yml_path = yml_path
            self.final_xyz = None
            self.number_of_rotors = 0
            self.rotors_dict = dict()
            self.rmg_species = rmg_species
            self.process_xyz(xyz)
            if bond_corrections is None:
                self.bond_corrections = dict()
            else:
                self.bond_corrections = bond_corrections

            if self.yml_path is not None:
                # a YAML path was given
                self.from_yml_file(label)
                if label is not None:
                    self.label = label
            elif self.rmg_species is not None:
                # an RMG Species was given
                if not isinstance(self.rmg_species, Species):
                    raise SpeciesError('The rmg_species parameter has to be a valid RMG Species object.'
                                       ' Got: {0}'.format(type(self.rmg_species)))
                if not self.rmg_species.molecule:
                    raise SpeciesError('If an RMG Species given, it must have a non-empty molecule list')
                if not self.rmg_species.label and not label:
                    raise SpeciesError('If an RMG Species given, it must have a label or a label must be given'
                                       ' separately')
                if label:
                    self.label = label
                else:
                    self.label = self.rmg_species.label
                self.mol_list = self.rmg_species.molecule
                if self.mol is None:
                    self.mol = self.rmg_species.molecule[0]
                    if len(self.rmg_species.molecule) == 1:
                        self.mol_list = self.rmg_species.generate_resonance_structures(
                            keep_isomorphic=False, filter_structures=True)
                    self.mol_list = self.rmg_species.molecule
                    if len(self.mol_list) > 1:
                        logger.info('Using localized structure {0} of species {1} for BAC determination. To use a'
                                    ' different structure, pass the RMG:Molecule object in the `mol` parameter'.format(
                                      self.mol.toSMILES(), self.label))
                self.multiplicity = self.rmg_species.molecule[0].multiplicity
                self.charge = self.rmg_species.molecule[0].getNetCharge()

            if label is not None:
                self.label = label
            if multiplicity is not None:
                self.multiplicity = multiplicity
            if charge is not None:
                self.charge = charge
            if self.mol is None:
                if adjlist:
                    self.mol = Molecule().fromAdjacencyList(adjlist=adjlist)
                elif inchi:
                    self.mol = rmg_mol_from_inchi(inchi)
                elif smiles:
                    self.mol = Molecule(SMILES=smiles)
            if not self.is_ts:
                # Perceive molecule from xyz coordinates. This also populates mol and mol_list.
                # It overrides self.mol from adjlist or smiles so that xyz and mol have the same atom ordering.
                if self.final_xyz or self.initial_xyz or self.most_stable_conformer or self.conformers:
                    self.mol_from_xyz(get_cheap=False)
                if self.mol is None:
                    if self.generate_thermo:
                        logger.warning('No structure (SMILES, adjList, RMG:Species, or RMG:Molecule) was given for '
                                       ' species {0}, NOT using bond additivity corrections (BAC) for thermo '
                                       'computation.'.format(self.label))
                else:
                    # Generate bond list for applying bond additivity corrections
                    if not self.bond_corrections and self.mol is not None:
                        self.bond_corrections = enumerate_bonds(self.mol)
                        if self.bond_corrections:
                            self.long_thermo_description += 'Bond corrections: {0}\n'.format(self.bond_corrections)

            if self.mol is not None and self.mol_list is None:
                mol_copy = self.mol.copy(deep=True)
                self.mol_list = mol_copy.generate_resonance_structures(keep_isomorphic=False, filter_structures=True)
            elif not self.bond_corrections and self.generate_thermo:
                logger.warning('Cannot determine bond additivity corrections (BAC) for species {0} based on xyz '
                               'coordinates only. For better thermoproperties, provide bond corrections.'.format(
                                self.label))

            self.neg_freqs_trshed = list()

        if self.multiplicity is None:
            self.determine_multiplicity(smiles, adjlist, self.mol)
            logger.debug('No multiplicity specified for {0}, assuming {1}.'.format(self.label, self.multiplicity))
        if self.charge is None:
            logger.debug('No charge specified for {0}, assuming charge 0.'.format(self.label))
            self.charge = 0
        if self.multiplicity is not None and self.multiplicity < 1:
            raise SpeciesError('Multiplicity for species {0} is lower than 1. Got: {1}'.format(
                self.label, self.multiplicity))
        if not isinstance(self.multiplicity, int) and self.multiplicity is not None:
            raise SpeciesError('Multiplicity for species {0} is not an integer. Got: {1}, a {2}'.format(
                self.label, self.multiplicity, type(self.multiplicity)))
        if not isinstance(self.charge, int):
            raise SpeciesError('Charge for species {0} is not an integer (got {1}, a {2})'.format(
                self.label, self.charge, type(self.charge)))
        if not self.is_ts and self.initial_xyz is None and self.final_xyz is None and self.mol is None\
                and not self.conformers:
            raise SpeciesError('No structure (xyz, SMILES, adjList, RMG:Species, or RMG:Molecule) was given for'
                               ' species {0}'.format(self.label))
        if self.label is None:
            raise SpeciesError('A label must be specified for an ARCSpecies object.')
        # Check that `label` is valid, since it is used for folder names
        for char in self.label:
            if char not in valid_chars:
                raise SpeciesError('Species label {0} contains an invalid character: "{1}"'.format(self.label, char))

    @property
    def number_of_atoms(self):
        """The number of atoms in the species"""
        if self._number_of_atoms is None:
            if self.mol is not None:
                self._number_of_atoms = len(self.mol.atoms)
            elif not self.is_ts:
                self._number_of_atoms = len(self.get_xyz().splitlines())
        return self._number_of_atoms

    @number_of_atoms.setter
    def number_of_atoms(self, value):
        """Allow setting number of atoms, e.g. a TS might not have Molecule or xyz when initialized"""
        self._number_of_atoms = value

    @property
    def number_of_heavy_atoms(self):
        """The number of heavy (non hydrogen) atoms in the species"""
        if self._number_of_heavy_atoms is None:
            if self.mol is not None:
                self._number_of_heavy_atoms = len([atom for atom in self.mol.atoms if atom.isNonHydrogen()])
            elif self.final_xyz is not None or self.initial_xyz is not None:
                self._number_of_heavy_atoms = len([line for line in self.get_xyz().splitlines()
                                                   if line.split()[0] != 'H'])
            elif self.is_ts:
                for ts_guess in self.ts_guesses:
                    if ts_guess.xyz is not None:
                        self._number_of_heavy_atoms =\
                            len([line for line in ts_guess.xyz.splitlines() if line.split()[0] != 'H'])
        return self._number_of_heavy_atoms

    @number_of_heavy_atoms.setter
    def number_of_heavy_atoms(self, value):
        """Allow setting number of heavy atoms, e.g. a TS might not have Molecule or xyz when initialized"""
        self._number_of_heavy_atoms = value

    @property
    def radius(self):
        """
        Determine the largest distance from the coordinate system origin attributed to one of the molecule's
        atoms in 3D space. Units are Angstrom.
        """
        if self._radius is None:
            translated_xyz = translate_to_center_of_mass(self.get_xyz())
            _, symbols, x, y, z = get_xyz_matrix(translated_xyz)
            border_elements = list()  # a list of the farthest element/s
            r = 0
            for si, xi, yi, zi in zip(symbols, x, y, z):
                ri = xi ** 2 + yi ** 2 + zi ** 2
                if ri == r:
                    border_elements.append(si)
                elif ri > r:
                    r = ri
                    border_elements = [si]
            atom_r = max([get_atom_radius(si) if get_atom_radius(si) is not None else 1.50 for si in border_elements])
            self._radius = r ** 0.5 + atom_r
            logger.info('Determined a radius of {0:.2f} Angstrom for {1}'.format(self._radius, self.label))
        return self._radius

    @radius.setter
    def radius(self, value):
        """Allow setting the radius"""
        self._radius = value

    def as_dict(self):
        """A helper function for dumping this object as a dictionary in a YAML file for restarting ARC"""
        species_dict = dict()
        species_dict['force_field'] = self.force_field
        species_dict['is_ts'] = self.is_ts
        if self.e_elect is not None:
            species_dict['e_elect'] = self.e_elect
        if self.e0 is not None:
            species_dict['e0'] = self.e0
        species_dict['arkane_file'] = self.arkane_file
        if self.yml_path is not None:
            species_dict['yml_path'] = self.yml_path
        if self.is_ts:
            species_dict['ts_methods'] = self.ts_methods
            species_dict['ts_guesses'] = [tsg.as_dict() for tsg in self.ts_guesses]
            species_dict['ts_conf_spawned'] = self.ts_conf_spawned
            species_dict['ts_number'] = self.ts_number
            species_dict['ts_report'] = self.ts_report
            species_dict['rxn_label'] = self.rxn_label
            species_dict['successful_methods'] = self.successful_methods
            species_dict['unsuccessful_methods'] = self.unsuccessful_methods
            species_dict['chosen_ts_method'] = self.chosen_ts_method
            species_dict['chosen_ts'] = self.chosen_ts
        if self.run_time is not None:
            species_dict['run_time'] = self.run_time.total_seconds()
        species_dict['t1'] = self.t1
        species_dict['label'] = self.label
        species_dict['long_thermo_description'] = self.long_thermo_description
        species_dict['multiplicity'] = self.multiplicity
        if self.number_of_radicals is not None:
            species_dict['number_of_radicals'] = self.number_of_radicals
        species_dict['charge'] = self.charge
        species_dict['generate_thermo'] = self.generate_thermo
        if self.opt_level is not None:
            species_dict['opt_level'] = self.opt_level
        if self.final_xyz is not None:
            species_dict['final_xyz'] = self.final_xyz
        species_dict['number_of_rotors'] = self.number_of_rotors
        species_dict['rotors_dict'] = self.rotors_dict
        species_dict['external_symmetry'] = self.external_symmetry
        species_dict['optical_isomers'] = self.optical_isomers
        species_dict['neg_freqs_trshed'] = self.neg_freqs_trshed
        if self.conf_is_isomorphic is not None:
            species_dict['conf_is_isomorphic'] = self.conf_is_isomorphic
        if self.bond_corrections is not None:
            species_dict['bond_corrections'] = self.bond_corrections
        if self.mol is not None:
            species_dict['mol'] = self.mol.toAdjacencyList()
        if self.initial_xyz is not None:
            species_dict['initial_xyz'] = self.initial_xyz
        if self.checkfile is not None:
            species_dict['checkfile'] = self.checkfile
        if self.most_stable_conformer is not None:
            species_dict['most_stable_conformer'] = self.most_stable_conformer
        if self.cheap_conformer is not None:
            species_dict['cheap_conformer'] = self.cheap_conformer
        if self.recent_md_conformer is not None:
            species_dict['recent_md_conformer'] = self.recent_md_conformer
        if self.svpfit_output_file is not None:
            species_dict['svpfit_output_file'] = self.svpfit_output_file
        if self._radius is not None:
            species_dict['radius'] = self._radius
        if self.conformers:
            species_dict['conformers'] = self.conformers
            species_dict['conformer_energies'] = self.conformer_energies
        if self.conformers_before_opt is not None:
            species_dict['conformers_before_opt'] = self.conformers_before_opt
        if self.bdes is not None:
            species_dict['bdes'] = self.bdes
        return species_dict

    def from_dict(self, species_dict):
        """
        A helper function for loading this object from a dictionary in a YAML file for restarting ARC
        """
        try:
            self.label = species_dict['label']
        except KeyError:
            raise InputError('All species must have a label')
        self.run_time = datetime.timedelta(seconds=species_dict['run_time']) if 'run_time' in species_dict else None
        self.t1 = species_dict['t1'] if 't1' in species_dict else None
        self.e_elect = species_dict['e_elect'] if 'e_elect' in species_dict else None
        self.e0 = species_dict['e0'] if 'e0' in species_dict else None
        self.arkane_file = species_dict['arkane_file'] if 'arkane_file' in species_dict else None
        self.yml_path = species_dict['yml_path'] if 'yml_path' in species_dict else None
        self.rxn_label = species_dict['rxn_label'] if 'rxn_label' in species_dict else None
        self._radius = species_dict['radius'] if 'radius' in species_dict else None
        self.most_stable_conformer = species_dict['most_stable_conformer'] if 'most_stable_conformer'\
                                                                              in species_dict else None
        self.cheap_conformer = species_dict['cheap_conformer'] if 'cheap_conformer' in species_dict else None
        self.recent_md_conformer = species_dict['recent_md_conformer']\
            if 'recent_md_conformer' in species_dict else None
        self.force_field = species_dict['force_field'] if 'force_field' in species_dict else 'MMFF94'
        self.svpfit_output_file = species_dict['svpfit_output_file'] if 'svpfit_output_file' in species_dict else None
        self.long_thermo_description = species_dict['long_thermo_description']\
            if 'long_thermo_description' in species_dict else ''
        self.initial_xyz = standardize_xyz_string(species_dict['initial_xyz']) if 'initial_xyz' in species_dict\
            else None
        self.final_xyz = standardize_xyz_string(species_dict['final_xyz']) if 'final_xyz' in species_dict else None
        self.conf_is_isomorphic = species_dict['conf_is_isomorphic'] if 'conf_is_isomorphic' in species_dict else None
        self.is_ts = species_dict['is_ts'] if 'is_ts' in species_dict else False
        if self.is_ts:
            self.ts_conf_spawned = species_dict['ts_conf_spawned'] if 'ts_conf_spawned' in species_dict else False
            self.ts_number = species_dict['ts_number'] if 'ts_number' in species_dict else None
            self.ts_report = species_dict['ts_report'] if 'ts_report' in species_dict else ''
            ts_methods = species_dict['ts_methods'] if 'ts_methods' in species_dict else None
            if ts_methods is None:
                self.ts_methods = default_ts_methods
            elif isinstance(ts_methods, list):
                self.ts_methods = ts_methods
                if not self.ts_methods:
                    self.ts_methods = ['user guess']
            else:
                raise TSError('ts_methods must be a list, got {0} of type {1}'.format(ts_methods, type(ts_methods)))
            self.ts_guesses = [TSGuess(ts_dict=tsg) for tsg in species_dict['ts_guesses']]\
                if 'ts_guesses' in species_dict else list()
            self.successful_methods = species_dict['successful_methods']\
                if 'successful_methods' in species_dict else list()
            self.unsuccessful_methods = species_dict['unsuccessful_methods']\
                if 'unsuccessful_methods' in species_dict else list()
            self.chosen_ts_method = species_dict['chosen_ts_method'] if 'chosen_ts_method' in species_dict else None
            self.chosen_ts = species_dict['chosen_ts'] if 'chosen_ts' in species_dict else None
            self.checkfile = species_dict['checkfile'] if 'checkfile' in species_dict else None
        else:
            self.ts_methods = None
        if 'xyz' in species_dict and self.initial_xyz is None and self.final_xyz is None:
            self.process_xyz(species_dict['xyz'])
        for char in self.label:
            if char not in valid_chars:
                raise SpeciesError('Species label {0} contains an invalid character: "{1}"'.format(self.label, char))
        self.multiplicity = species_dict['multiplicity'] if 'multiplicity' in species_dict else None
        self.charge = species_dict['charge'] if 'charge' in species_dict else 0
        if 'charge' not in species_dict:
            logger.debug('No charge specified for {0}, assuming charge 0.'.format(self.label))
        if self.is_ts:
            self.generate_thermo = False
        else:
            self.generate_thermo = species_dict['generate_thermo'] if 'generate_thermo' in species_dict else True
        self.number_of_radicals = species_dict['number_of_radicals'] if 'number_of_radicals' in species_dict else None
        self.opt_level = species_dict['opt_level'] if 'opt_level' in species_dict else None
        self.number_of_rotors = species_dict['number_of_rotors'] if 'number_of_rotors' in species_dict else 0
        self.rotors_dict = species_dict['rotors_dict'] if 'rotors_dict' in species_dict else dict()
        self.external_symmetry = species_dict['external_symmetry'] if 'external_symmetry' in species_dict else None
        self.optical_isomers = species_dict['optical_isomers'] if 'optical_isomers' in species_dict else None
        self.neg_freqs_trshed = species_dict['neg_freqs_trshed'] if 'neg_freqs_trshed' in species_dict else list()
        self.bond_corrections = species_dict['bond_corrections'] if 'bond_corrections' in species_dict else dict()
        try:
            self.mol = Molecule().fromAdjacencyList(str(species_dict['mol'])) if 'mol' in species_dict else None
        except (ValueError, InvalidAdjacencyListError) as e:
            logger.error('Could not read RMG adjacency list {0}. Got:\n{1}'.format(species_dict['mol'] if 'mol'
                                                                                   in species_dict else None,
                                                                                   e.message))
            self.mol = None
        smiles = species_dict['smiles'] if 'smiles' in species_dict else None
        inchi = species_dict['inchi'] if 'inchi' in species_dict else None
        adjlist = species_dict['adjlist'] if 'adjlist' in species_dict else None
        if self.mol is None:
            if adjlist is not None:
                self.mol = Molecule().fromAdjacencyList(adjlist=adjlist)
            elif inchi is not None:
                self.mol = rmg_mol_from_inchi(inchi)
            elif smiles is not None:
                self.mol = Molecule(SMILES=smiles)
        if self.mol is None and not self.is_ts:
            self.mol_from_xyz()
        if self.mol is not None:
            if 'bond_corrections' not in species_dict:
                self.bond_corrections = enumerate_bonds(self.mol)
                if self.bond_corrections:
                    self.long_thermo_description += 'Bond corrections: {0}\n'.format(self.bond_corrections)
            if self.multiplicity is None:
                self.multiplicity = self.mol.multiplicity
            if self.charge is None:
                self.charge = self.mol.getNetCharge()
            if self.mol_list is None:
                if not self.charge:
                    self.mol_list = self.mol.generate_resonance_structures(keep_isomorphic=False,
                                                                           filter_structures=True)
        if 'conformers' in species_dict:
            self.conformers = species_dict['conformers']
            self.conformer_energies = species_dict['conformer_energies'] if 'conformer_energies' in species_dict\
                else [None] * len(self.conformers)
        if self.mol is None and self.initial_xyz is None and self.final_xyz is None and not self.conformers\
                and not any([tsg.xyz for tsg in self.ts_guesses]):
            # TS species are allowed to be loaded w/o a structure
            raise SpeciesError('Must have either mol or xyz for species {0}'.format(self.label))
        self.bdes = species_dict['bdes'] if 'bdes' in species_dict else None
        if self.bdes is not None and not isinstance(self.bdes, list):
            raise SpeciesError('The .bdes argument must be a list, got {0} which is a {1}'.format(
                                self.bdes, type(self.bdes)))

        self.conformers_before_opt = species_dict['conformers_before_opt'] \
            if 'conformers_before_opt' in species_dict else None

    def from_yml_file(self, label=None):
        """
        Load important species attributes such as label and final_xyz from the Arkane YAML file
        Actual QM data parsing is done later when processing thermo and kinetics
        """
        rmg_spc = Species()
        arkane_spc = ArkaneSpecies(species=rmg_spc)
        # The data from the YAML file is loaded into the `species` argument of the `load_yaml` method in Arkane
        arkane_spc.load_yaml(path=self.yml_path, label=label, pdep=False)
        self.label = label if label is not None else arkane_spc.label
        self.final_xyz = get_xyz_string(coords=arkane_spc.conformer.coordinates.value,
                                        numbers=arkane_spc.conformer.number.value)
        if arkane_spc.adjacency_list is not None:
            try:
                self.mol = Molecule().fromAdjacencyList(adjlist=arkane_spc.adjacency_list)
            except ValueError:
                print('Could not read adjlist:\n{0}'.format(arkane_spc.adjacency_list))  # should *not* be logging
                raise
        elif arkane_spc.inchi is not None:
            self.mol = Molecule().fromInChI(inchistr=arkane_spc.inchi)
        elif arkane_spc.smiles is not None:
            self.mol = Molecule().fromSMILES(arkane_spc.smiles)
        if self.mol is not None:
            self.mol_list = self.mol.generate_resonance_structures(keep_isomorphic=False, filter_structures=True)
            self.multiplicity = self.mol.multiplicity
            self.charge = self.mol.getNetCharge()
        if self.multiplicity is None:
            self.multiplicity = arkane_spc.conformer.spinMultiplicity
        if self.optical_isomers is None:
            self.optical_isomers = arkane_spc.conformer.opticalIsomers
        if self.external_symmetry is None:
            external_symmetry_mode = None
            for mode in arkane_spc.conformer.modes:
                if isinstance(mode, (NonlinearRotor, LinearRotor)):
                    external_symmetry_mode = mode
                    break
            if external_symmetry_mode is not None:
                self.external_symmetry = external_symmetry_mode.symmetry
        if self.initial_xyz is not None:
            self.mol_from_xyz()
        if self.e_elect is None:  # TODO: this is actually the E0, not e_elect! be consistent!
            self.e_elect = arkane_spc.conformer.E0.value_si * 0.001  # convert to kJ/mol

    def generate_conformers(self, confs_to_dft=5, plot_path=None):
        """
        Generate conformers

        Args:
            confs_to_dft (int, optional): The number of conformers to store in the .conformers attribute of the species
                                          that will later be DFT'ed at the conformers_level.
            plot_path (str, optional): A folder path in which the plot will be saved.
                                                If None, the plot will not be shown (nor saved).
        """
        if not self.is_ts:
            if not self.charge:
                mol_list = self.mol_list
            else:
                mol_list = [self.mol]
            lowest_confs = conformers.generate_conformers(mol_list=mol_list, label=self.label,
                                                          charge=self.charge, multiplicity=self.multiplicity,
                                                          force_field=self.force_field, print_logs=False,
                                                          num_confs_to_return=confs_to_dft, return_all_conformers=False,
                                                          plot_path=plot_path)
            self.conformers.extend([conf['xyz'] for conf in lowest_confs])
            self.conformer_energies.extend([None] * len(lowest_confs))
            lowest_conf = conformers.get_lowest_confs(label=self.label, confs=lowest_confs, n=1)[0]
            logger.info('Most stable force field conformer for {0}:'.format(self.label))
            logger.info(lowest_conf['xyz'])

    def get_cheap_conformer(self):
        """
        Cheaply (limiting the number of possible conformers) get a reasonable conformer,
        this could very well not be the best (lowest) one.

        Returns:
            str: A reasonable conformer's xyz coordinates in string format.
        """
        num_confs = min(500, max(50, len(self.mol.atoms) * 3))
        rd_mol, rd_index_map = conformers.embed_rdkit(label=self.label, mol=self.mol, num_confs=num_confs)
        xyzs, energies = conformers.rdkit_force_field(label=self.label, rd_mol=rd_mol, rd_index_map=rd_index_map,
                                                      mol=self.mol, force_field='MMFF94', return_xyz_strings=True)
        if energies:
            min_energy = min(energies)
            min_energy_index = energies.index(min_energy)
            self.cheap_conformer = xyzs[min_energy_index]
        elif xyzs:
            self.cheap_conformer = xyzs[0]
        else:
            logger.warning('Could not generate a cheap conformer for {0}'.format(self.label))
            self.cheap_conformer = None

    def get_xyz(self, get_cheap=True):
        """
        Get the highest quality xyz the species has. If it doesn't have any 3D information, cheaply generate it.

        Args:
            get_cheap (bool, optional): Whether to get a cheap FF conformer if no xyz is found. True to get.

        Return:
             str: A string-format xyz coordinates.
        """
        conf = self.conformers[0] if self.conformers else None
        xyz = self.final_xyz or self.initial_xyz or self.most_stable_conformer or conf or self.cheap_conformer
        if xyz is None:
            if self.is_ts:
                for ts_guess in self.ts_guesses:
                    if ts_guess.xyz is not None:
                        xyz = ts_guess.xyz
                        return xyz
                return None
            elif get_cheap:
                self.get_cheap_conformer()
                xyz = self.cheap_conformer
        return xyz

    def determine_rotors(self):
        """
        Determine possible unique rotors in the species to be treated as hindered rotors,
        taking into account all localized structures.
        The resulting rotors are saved in {'pivots': [1, 3], 'top': [3, 7], 'scan': [2, 1, 3, 7]} format
        in self.species_dict[species.label]['rotors_dict']. Also updates 'number_of_rotors'.
        """
        if not self.is_ts:
            if not self.charge:
                mol_list = self.mol_list
            else:
                mol_list = [self.mol]
            for mol in mol_list:
                rotors = conformers.find_internal_rotors(mol)
                for new_rotor in rotors:
                    for existing_rotor in self.rotors_dict.values():
                        if existing_rotor['pivots'] == new_rotor['pivots']:
                            break
                    else:
                        self.rotors_dict[self.number_of_rotors] = new_rotor
                        self.number_of_rotors += 1
            if self.number_of_rotors == 1:
                logger.info('\nFound one possible rotor for {0}'.format(self.label))
            elif self.number_of_rotors > 1:
                logger.info('\nFound {0} possible rotors for {1}'.format(self.number_of_rotors, self.label))
            if self.number_of_rotors > 0:
                logger.info('Pivot list(s) for {0}: {1}\n'.format(
                    self.label, [self.rotors_dict[i]['pivots'] for i in range(self.number_of_rotors)]))

    def set_dihedral(self, pivots, scan, deg_increment):
        """
        Generated an RDKit molecule object from the given self.final_xyz.
        Increments the current dihedral angle between atoms i, j, k, l in the `scan` list by 'deg_increment` in degrees.
        All bonded atoms are moved accordingly. The result is saved in self.initial_xyz.
        """
        if deg_increment == 0:
            logger.warning('set_dihedral was called with zero increment for {label} with pivots {pivots}'.format(
                label=self.label, pivots=pivots))
            for rotor in self.rotors_dict.values():  # penalize this rotor to avoid inf. looping
                if rotor['pivots'] == pivots:
                    rotor['times_dihedral_set'] += 1
                    break
        else:
            for rotor in self.rotors_dict.values():
                if rotor['pivots'] == pivots and rotor['times_dihedral_set'] <= 10:
                    rotor['times_dihedral_set'] += 1
                    break
            else:
                logger.info('\n\n')
                for i, rotor in self.rotors_dict.items():
                    logger.error('Rotor {i} with pivots {pivots} was set {times} times'.format(
                        i=i, pivots=rotor['pivots'], times=rotor['times_dihedral_set']))
                raise RotorError('Rotors were set beyond the maximal number of times without converging')
            coordinates, atoms, _, _, _ = get_xyz_matrix(self.final_xyz)
            mol = molecules_from_xyz(self.final_xyz, multiplicity=self.multiplicity, charge=self.charge)[1]
            conf, rd_mol, indx_map = rdkit_conf_from_mol(mol, coordinates)
            rd_scan = [indx_map[i - 1] for i in scan]  # convert the atom indices in `scan` to RDKit indices
            new_xyz = set_rdkit_dihedrals(conf, rd_mol, indx_map, rd_scan, deg_increment=deg_increment)
            self.initial_xyz = get_xyz_string(coords=new_xyz, symbols=atoms)

    def determine_symmetry(self):
        """
        Determine external symmetry and chirality (optical isomers) of the species
        """
        if self.optical_isomers is None and self.external_symmetry is None:
            xyz = self.get_xyz()
            coords, symbols, _, _, _ = get_xyz_matrix(xyz)
            symmetry, optical_isomers = determine_symmetry(coords, symbols)
            self.optical_isomers = self.optical_isomers if self.optical_isomers is not None else optical_isomers
            if self.optical_isomers != optical_isomers:
                logger.warning("User input of optical isomers for {0} and ARC's calculation differ: {1} and {2},"
                               " respectively. Using the user input of {1}"
                               .format(self.label, self.optical_isomers, optical_isomers))
            self.external_symmetry = self.external_symmetry if self.external_symmetry is not None else symmetry
            if self.external_symmetry != symmetry:
                logger.warning("User input of external symmetry for {0} and ARC's calculation differ: {1} and {2},"
                               " respectively. Using the user input of {1}"
                               .format(self.label, self.external_symmetry, symmetry))

    def determine_multiplicity(self, smiles, adjlist, mol):
        """
        Determine the spin multiplicity of the species
        """
        if mol is not None and mol.multiplicity >= 1:
            self.multiplicity = mol.multiplicity
        elif adjlist:
            mol = Molecule().fromAdjacencyList(str(adjlist))
            self.multiplicity = mol.multiplicity
        elif self.mol is not None and self.mol.multiplicity >= 1:
            self.multiplicity = self.mol.multiplicity
        elif smiles:
            mol = Molecule(SMILES=str(smiles))
            self.multiplicity = mol.multiplicity
        else:
            xyz = self.get_xyz()
            if xyz is None and len(self.conformers):
                xyz = self.conformers[0]
            if xyz:
                atoms = get_xyz_matrix(xyz)[1]
                electrons = 0
                for atom in atoms:
                    for number, symbol in symbol_by_number.items():
                        if symbol == atom:
                            electrons += number
                            break
                    else:
                        raise SpeciesError('Could not identify atom symbol {0}'.format(atom))
                if electrons % 2 == 1:
                    self.multiplicity = 2
                    logger.warning('\nMultiplicity not specified for {0}, assuming a value of 2'.format(self.label))
                else:
                    self.multiplicity = 1
                    logger.warning('\nMultiplicity not specified for {0}, assuming a value of 1'.format(self.label))
        if self.multiplicity is None:
            raise SpeciesError('Could not determine multiplicity for species {0}'.format(self.label))

    def make_ts_report(self):
        """A helper function to write content into the .ts_report attribute"""
        self.ts_report = ''
        if self.chosen_ts_method is not None:
            self.ts_report += 'TS method summary for {0} in {1}\n'.format(self.label, self.rxn_label)
            self.ts_report += 'Methods that successfully generated a TS guess:\n'
            if self.successful_methods:
                for successful_method in self.successful_methods:
                    self.ts_report += successful_method + ','
            if self.unsuccessful_methods:
                self.ts_report += '\nMethods that were unsuccessfully in generating a TS guess:\n'
                for unsuccessful_method in self.unsuccessful_methods:
                    self.ts_report += unsuccessful_method + ','
            self.ts_report += '\nThe method that generated the best TS guess and its output used for the' \
                              ' optimization: {0}'.format(self.chosen_ts_method)

    def mol_from_xyz(self, xyz=None, get_cheap=False):
        """
        Make sure atom order in self.mol corresponds to xyz
        Important for TS discovery and for identifying rotor indices

        This works by generating a molecule from the xyz and using the
        2D structure to confirm that the perceived molecule is correct.
        Resonance structures are generated and saved to ``self.mol_list``.
        """
        if xyz is None:
            xyz = self.get_xyz(get_cheap=get_cheap)

        if self.mol is not None:
            # self.mol should have come from another source, e.g., SMILES or yml
            original_mol = self.mol.copy(deep=True)
            self.mol = molecules_from_xyz(xyz, multiplicity=self.multiplicity, charge=self.charge)[1]

            if self.mol is not None and not check_isomorphism(original_mol, self.mol):
                if original_mol.multiplicity in [1, 2]:
                    raise SpeciesError('XYZ and the 2D graph representation for {0} are not isomorphic.\n'
                                       'Got xyz:\n{1}\n\nwhich corresponds to {2}\n{3}\n\nand: {4}\n{5}'.format(
                                        self.label, xyz, self.mol.toSMILES(), self.mol.toAdjacencyList(),
                                        original_mol.toSMILES(), original_mol.toAdjacencyList()))
                else:
                    logger.warning('XYZ and the 2D graph representation for {0} are not isomorphic.\n'
                                   'Got xyz:\n{1}\n\nwhich corresponds to {2}\n{3}\n\nand: {4}\n{5}'.format(
                                    self.label, xyz, self.mol.toSMILES(), self.mol.toAdjacencyList(),
                                    original_mol.toSMILES(), original_mol.toAdjacencyList()))
            elif self.mol is None:
                # molecules_from_xyz() returned None for b_mol
                self.mol = original_mol  # todo: Atom order will not be correct, need fix
        else:
            self.mol = molecules_from_xyz(xyz, multiplicity=self.multiplicity, charge=self.charge)[1]

        if self.mol_list is None and self.mol is not None:
            # Assign atom ids first, so they carry through to the resonance structures
            self.mol.assignAtomIDs()
            # The generate_resonance_structures method changes atom order
            # Make a copy so we don't disturb the original order from xyz
            self.mol_list = self.mol.copy(deep=True).generate_resonance_structures(keep_isomorphic=False,
                                                                                   filter_structures=True)
        order_atoms_in_mol_list(ref_mol=self.mol, mol_list=self.mol_list)

    def process_xyz(self, xyz_list):
        """
        Process the user's input and add either to the .conformers attribute or to .ts_guesses.

        Args:
            xyz_list (list, str): Entries are either string-format coordinates or file paths.
                                  (If there's only one entry, it could be given directly, not in a list)
                                  The file paths could direct to either a .xyz file, ARC conformers (w/ or w/o
                                  energies), or an ESS log/input files, making this method extremely flexible.
        """
        if xyz_list is not None:
            if not isinstance(xyz_list, list):
                xyz_list = [xyz_list]
            xyzs, energies = list(), list()
            for xyz in xyz_list:
                if not isinstance(xyz, (str, unicode)):
                    raise InputError('each xyz entry in xyz_list must be a string. '
                                     'Got:\n{0}\nwhich is a {1}'.format(xyz, type(xyz)))
                if os.path.isfile(xyz):
                    file_extension = os.path.splitext(xyz)[1]
                    if 'txt' in file_extension:
                        # assume this is an ARC conformer file
                        xyzs_, energies_ = process_conformers_file(conformers_path=xyz)
                        xyzs.extend(xyzs_)
                        energies.extend(energies_)
                    else:
                        # assume this is an ESS log file
                        xyzs.append(parse_xyz_from_file(xyz))  # also calls standardize_xyz_string()
                        energies.append(None)  # dummy (lists should be the same length)
                else:
                    # assume this is a string format xyz
                    xyzs.append(standardize_xyz_string(xyz))
                    energies.append(None)  # dummy (lists should be the same length)
            if not self.is_ts:
                self.conformers.extend(xyzs)
                self.conformer_energies.extend(energies)
            else:
                tsg_index = len(self.ts_guesses)
                for xyz, energy in zip(xyzs, energies):
                    # make TSGuess objects
                    # for tsg in self.ts_guesses:
                    #     if xyz == tsg.xyz:
                    #         break
                    # else:
                    self.ts_guesses.append(TSGuess(method='user guess {0}'.format(tsg_index), xyz=xyz, energy=energy))
                    # user guesses are always successful in generating a *guess*:
                    self.ts_guesses[tsg_index].success = True
                    tsg_index += 1
            if self.multiplicity is not None and self.charge is not None:
                for xyz in xyzs:
                    consistent = check_xyz(xyz=xyz, multiplicity=self.multiplicity, charge=self.charge)
                    if not consistent:
                        raise SpeciesError('Inconsistent combination of number of electrons. multiplicity and charde  '
                                           'for {0}.'.format(self.label))

    def set_transport_data(self, lj_path, opt_path, bath_gas, opt_level, freq_path='', freq_level=None):
        """
        Set the species.transport_data attribute after a Lennard-Jones calculation (via OneDMin)
        `lj_path` is the path to a oneDMin job output file
        `opt_path` is the path to an opt job output file
        `bath_gas` is the oneDMin job bath gas
        `opt_level` is the optimization level of theory
        """
        original_comment = self.transport_data.comment
        comment = 'L-J coefficients calculated by OneDMin using a DF-MP2/aug-cc-pVDZ potential energy surface ' \
                  'with {0} as the bath gas'.format(bath_gas)
        epsilon, sigma = None, None
        with open(lj_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'Epsilons[1/cm]' in line:
                # Conversion of cm^-1 to J/mol (see https://cccbdb.nist.gov/wavenumber.asp)
                epsilon = (float(line.split()[-1]) * 11.96266, str('J/mol'))
            elif 'Sigmas[angstrom]' in line:
                # Convert Angstroms to meters
                sigma = (float(line.split()[-1]) * 1e-10, str('m'))
        if self.number_of_atoms == 1:
            shape_index = 0
            comment += '; The molecule is monoatomic'
        else:
            coordinates = get_xyz_matrix(self.get_xyz())[0]
            coordinates = np.array(coordinates)
            if is_linear(coordinates):
                shape_index = 1
                comment += '; The molecule is linear'
            else:
                shape_index = 2
        if self.number_of_atoms > 1:
            dipole_moment = parse_dipole_moment(opt_path) or 0
            if dipole_moment:
                comment += '; Dipole moment was calculated at the {0} level of theory'.format(opt_level)
        else:
            dipole_moment = 0
        polar = self.transport_data.polarizability or (0, str('angstroms^3'))
        if freq_path:
            polar = (parse_polarizability(freq_path), str('angstroms^3'))
            comment += '; Polarizability was calculated at the {0} level of theory'.format(freq_level)
        comment += '; Rotational Relaxation Collision Number was not determined, default value is 2'
        if original_comment:
            comment += '; ' + original_comment
        self.transport_data = TransportData(
            shapeIndex=shape_index,
            epsilon=epsilon,
            sigma=sigma,
            dipoleMoment=(dipole_moment, str('De')),
            polarizability=polar,
            rotrelaxcollnum=2,  # rotational relaxation collision number at 298 K
            comment=str(comment)
        )

    def check_final_xyz_isomorphism(self, allow_nonisomorphic_2d=False):
        """
        Check whether the perception of self.final_xyz is isomorphic with self.mol.

        Args:
            allow_nonisomorphic_2d (bool): Whether to continue spawning jobs for the species even if this test fails.
                                           `True` to allow (default is `False`).

        Returns:
            bool: Whether the perception of self.final_xyz is isomorphic with self.mol, `True` if it is.
        """
        passed_test, return_value = False, False
        if self.mol is not None:
            try:
                b_mol = molecules_from_xyz(self.final_xyz, multiplicity=self.multiplicity, charge=self.charge)[1]
            except SanitizationError:
                b_mol = None
            if b_mol is not None:
                is_isomorphic = check_isomorphism(self.mol, b_mol)
            else:
                is_isomorphic = False
            if is_isomorphic:
                passed_test, return_value = True, True
            else:
                # isomorphism test failed
                passed_test = False
                if self.conf_is_isomorphic:
                    if allow_nonisomorphic_2d:
                        # conformer was isomorphic, we **do** allow nonisomorphism, and the optimized structure isn't
                        return_value = True
                    else:
                        # conformer was isomorphic, we don't allow nonisomorphism, but the optimized structure isn't
                        return_value = False
                else:
                    # conformer was not isomorphic, don't strictly enforce isomorphism here
                    return_value = True
            if not passed_test:
                logger.error('The optimized geometry of species {0} is not isomorphic with the 2D structure {1}'.format(
                    self.label, self.mol.toSMILES()))
                if not return_value:
                    logger.error('Not spawning additional jobs for this species!')
            else:
                logger.info('Species {0} was found to be isomorphic with the perception '
                            'of its optimized coordinates.'.format(self.label))
        else:
            logger.error('Cannot check isomorphism for species {0}'.format(self.label))
        return return_value

    def scissors(self):
        """
        Cut chemical bonds to create new species from the original one according to the .bdes attribute,
        preserving the 3D geometry other than the splitted bond.
        If one of the scission-resulting species is a hydrogen atom, it will be returned last, labeled as 'H'.
        Other species labels will be <original species label>_BDE_index1_index2_X, where "X" is either "A" or "B",
        and the indices are 1-indexed.

        Returns:
            list: The scission-resulting species.
        """
        all_h = True if 'all_h' in self.bdes else False
        if all_h:
            self.bdes.pop(self.bdes.index('all_h'))
        for entry in self.bdes:
            if len(entry) != 2:
                raise SpeciesError('Could not interpret entry {0} in {1} for BDEs calculations.'.format(
                    entry, self.bdes))
            if not isinstance(entry, (tuple, list)):
                raise SpeciesError('`indices` entries must be tuples or lists, got {0} which is a {1} in {2}'.format(
                    entry, type(entry), self.bdes))
        self.bdes = [tuple(bde) for bde in self.bdes]
        if all_h:
            for atom1 in self.mol.atoms:
                if atom1.isHydrogen():
                    for atom2, bond12 in atom1.edges.items():
                        if bond12.isSingle():
                            atom_indices = (self.mol.atoms.index(atom2) + 1, self.mol.atoms.index(atom1) + 1)
                            atom_indices_reverse = (atom_indices[1], atom_indices[0])
                            if atom_indices not in self.bdes and atom_indices_reverse not in self.bdes:
                                self.bdes.append(atom_indices)
        resulting_species = list()
        for index_tuple in self.bdes:
            new_species_list = self._scissors(indices=index_tuple)
            for new_species in new_species_list:
                if new_species.label not in [existing_species.label for existing_species in resulting_species]:
                    # mainly checks that the H species doesn't already exist
                    resulting_species.append(new_species)
        return resulting_species

    def _scissors(self, indices):
        """
        Cut a chemical bond to create two new species from the original one, preserving the 3D geometry.

        Args:
            indices (tuple): The atom indices between which to cut (1-indexed, atoms must be bonded).

        Returns:
            list: The scission-resulting species.
        """
        if any([i < 1 for i in indices]):
            raise SpeciesError('Indices must be larger than 0')
        if not all([isinstance(i, int) for i in indices]):
            raise SpeciesError('Indices must be integers')
        if self.final_xyz is None:
            raise SpeciesError('Cannot use scissors without the .final_xyz attribute of species {0}'.format(self.label))
        indices = (indices[0] - 1, indices[1] - 1)  # convert to 0-indexed atoms
        atom1 = self.mol.atoms[indices[0]]
        atom2 = self.mol.atoms[indices[1]]
        if atom1.isHydrogen():
            top1 = [self.mol.atoms.index(atom1)]
            top2 = [i for i in range(len(self.mol.atoms)) if i not in top1]
        elif atom2.isHydrogen():
            top2 = [self.mol.atoms.index(atom2)]
            top1 = [i for i in range(len(self.mol.atoms)) if i not in top2]
        else:
            # for robustness, only use the smaller top,
            # determine_top_group_indices() might get confused for (large) convolved tops.
            top1 = conformers.determine_top_group_indices(self.mol, atom1, atom2, index=0)[0]
            top2 = conformers.determine_top_group_indices(self.mol, atom2, atom1, index=0)[0]
            if len(top1) > len(top2):
                top1 = [i for i in range(len(self.mol.atoms)) if i not in top2]
            elif len(top2) > len(top1):
                top2 = [i for i in range(len(self.mol.atoms)) if i not in top1]
        split_xyz = self.final_xyz.splitlines()
        xyz1 = '\n'.join([split_xyz[i] for i in top1])
        xyz2 = '\n'.join([split_xyz[i] for i in top2])
        xyz1, xyz2 = translate_to_center_of_mass(xyz1), translate_to_center_of_mass(xyz2)

        mol_copy = self.mol.copy(deep=True)
        # We are about to change the connectivity of the atoms in the molecule,
        # which invalidates any existing vertex connectivity information; thus we reset it.
        mol_copy.resetConnectivityValues()
        atom1 = mol_copy.atoms[indices[0]]  # Note: redefining atom1 and atom2
        atom2 = mol_copy.atoms[indices[1]]
        if not mol_copy.hasBond(atom1, atom2):
            raise SpeciesError('Attempted to remove a nonexistent bond.')
        bond = mol_copy.getBond(atom1, atom2)
        mol_copy.removeBond(bond)
        mol1, mol2 = mol_copy.split()

        used_a_label = False
        if len(mol1.atoms) == 1 and mol1.atoms[0].isHydrogen():
            label1 = 'H'
        else:
            label1 = self.label + '_BDE_' + str(indices[0] + 1) + '_' + str(indices[1] + 1) + '_A'
            used_a_label = True
        if len(mol2.atoms) == 1 and mol2.atoms[0].isHydrogen():
            label2 = 'H'
        else:
            letter = 'B' if used_a_label else 'A'
            label2 = self.label + '_BDE_' + str(indices[0] + 1) + '_' + str(indices[1] + 1) + '_' + letter

        added_radical = list()
        for mol, label in zip([mol1, mol2], [label1, label2]):
            for atom in mol.atoms:
                theoretical_charge = elements.PeriodicSystem.valence_electrons[atom.symbol] \
                                     - atom.getBondOrdersForAtom() \
                                     - atom.radicalElectrons -\
                                     2 * atom.lonePairs
                if theoretical_charge == atom.charge + 1:
                    # we're missing a radical electron on this atom
                    if label not in added_radical or label == 'H':
                        atom.radicalElectrons += 1
                        added_radical.append(label)
                    else:
                        raise SpeciesError('Could not figure out which atom should gain a radical '
                                           'due to scission in {0}'.format(self.label))
        mol1.update()
        mol2.update()

        # match xyz to mol:
        if len(mol1.atoms) != len(mol2.atoms):
            # easy
            if len(mol1.atoms) != len(top1):
                xyz1, xyz2 = xyz2, xyz1
        else:
            # harder
            element_dict_mol1, element_dict_top1 = dict(), dict()
            for atom in mol1.atoms:
                if atom.element.symbol in element_dict_mol1:
                    element_dict_mol1[atom.element.symbol] += 1
                else:
                    element_dict_mol1[atom.element.symbol] = 1
            for i in top1:
                atom = mol_copy.atoms[i - 1]
                if atom.element.symbol in element_dict_top1:
                    element_dict_top1[atom.element.symbol] += 1
                else:
                    element_dict_top1[atom.element.symbol] = 1
            for element, count in element_dict_mol1.items():
                if element not in element_dict_top1 or count != element_dict_top1[element]:
                    xyz1, xyz2 = xyz2, xyz1

        spc1 = ARCSpecies(label=label1, mol=mol1, xyz=xyz1, multiplicity=mol1.multiplicity, charge=mol1.getNetCharge(),
                          generate_thermo=False)
        spc1.initial_xyz = xyz1
        spc2 = ARCSpecies(label=label2, mol=mol2, xyz=xyz2, multiplicity=mol2.multiplicity, charge=mol2.getNetCharge(),
                          generate_thermo=False)
        spc2.initial_xyz = xyz2

        return [spc1, spc2]


class TSGuess(object):
    """
    TSGuess class

    The user can define xyz directly, and the default `method` will be 'User guess'
    Alternatively, either ARC or the user can provide reactant/s and product/s geometries with a specified `method`
    `method` could be one of the following:
    - QST2 (not implemented)
    - DEGSM (not implemented)
    - NEB (not implemented)
    - Kinbot: requires the RMG family, only works for H,C,O,S (not implemented)
    - AutoTST

    Args:
        method (str, optional): The method/source used for the xyz guess.
        reactants_xyz (list, optional): A list of tuples, each containing:
                                        (reactant label, reactant geometry in string format).
        products_xyz (list, optional): A list of tuples, each containing:
                                       (product label, product geometry in string format).
        family (str, optional): The RMG family that corresponds to the reaction, if applicable.
        xyz (str, optional): The 3D coordinates guess.
        rmg_reaction (Reaction, optional): An RMG Reaction object.
        ts_dict (dict, optional): A dictionary to create this object from (used when restarting ARC).
        energy (float): Relative energy of all TS conformers in kJ/mol.

    Attributes:
        xyz (str): The 3D coordinates guess.
        opt_xyz (str): The 3D coordinates after optimization at the ts_guesses level.
        method (str): The method/source used for the xyz guess.
        reactants_xyz (list): A list of tuples, each containing:
                              (reactant label, reactant geometry in string format).
        products_xyz (list): A list of tuples, each containing:
                             (product label, product geometry in string format).
        family (str): The RMG family that corresponds to the reaction, if applicable.
        rmg_reaction (Reaction): An RMG Reaction object.
        t0 (float): Initial time of spawning the guess job.
        execution_time (str): Overall execution time for the TS guess method.
        success (bool): Whether the TS guess method succeeded in generating an XYZ guess or not.
        energy (float): Relative energy of all TS conformers in kJ/mol.
        index (int): An index corresponding to the conformer jobs spawned for each TSGuess object.
                     Assigned only if self.success is ``True``.

    """
    def __init__(self, method=None, reactants_xyz=None, products_xyz=None, family=None, xyz=None,
                 rmg_reaction=None, ts_dict=None, energy=None):

        if ts_dict is not None:
            # Reading from a dictionary
            self.from_dict(ts_dict=ts_dict)
        else:
            # Not reading from a dictionary
            self.t0 = None
            self.index = None
            self.execution_time = None
            self.xyz = None
            self.opt_xyz = None
            self.process_xyz(xyz)  # populates self.xyz
            self.success = None
            self.energy = energy
            self.method = method.lower() if method is not None else 'user guess'
            if 'user guess' in self.method:
                if self.xyz is None:
                    raise TSError('If no method is specified, an xyz guess must be given')
                self.success = True
                self.execution_time = 0
            self.reactants_xyz = reactants_xyz if reactants_xyz is not None else list()
            self.products_xyz = products_xyz if products_xyz is not None else list()
            self.rmg_reaction = rmg_reaction
            self.family = family
            # if self.family is None and self.method.lower() in ['kinbot', 'autotst']:
            #     raise TSError('No family specified for method {0}'.format(self.method))
        if not ('user guess' in self.method or 'autotst' in self.method
                or self.method in ['user guess'] + [tsm.lower() for tsm in default_ts_methods]):
            raise TSError('Unrecognized method. Should be either {0}. Got: {1}'.format(
                          ['User guess'] + default_ts_methods, self.method))

    def as_dict(self):
        """A helper function for dumping this object as a dictionary in a YAML file for restarting ARC"""
        ts_dict = dict()
        ts_dict['t0'] = self.t0
        ts_dict['method'] = self.method
        ts_dict['success'] = self.success
        ts_dict['energy'] = self.energy
        ts_dict['index'] = self.index
        ts_dict['execution_time'] = self.execution_time
        if self.xyz:
            ts_dict['xyz'] = self.xyz
        if self.reactants_xyz:
            ts_dict['reactants_xyz'] = self.reactants_xyz
        if self.products_xyz:
            ts_dict['products_xyz'] = self.products_xyz
        if self.family is not None:
            ts_dict['family'] = self.family
        if self.rmg_reaction is not None:
            rxn_string = ' <=> '.join([' + '.join([spc.molecule[0].toSMILES() for spc in self.rmg_reaction.reactants]),
                                      ' + '.join([spc.molecule[0].toSMILES() for spc in self.rmg_reaction.products])])
            ts_dict['rmg_reaction'] = rxn_string
        return ts_dict

    def from_dict(self, ts_dict):
        """
        A helper function for loading this object from a dictionary in a YAML file for restarting ARC
        """
        self.t0 = ts_dict['t0'] if 't0' in ts_dict else None
        self.index = ts_dict['index'] if 'index' in ts_dict else None
        self.xyz = ts_dict['xyz'] if 'xyz' in ts_dict else None
        self.process_xyz(self.xyz)  # re-populates self.xyz
        self.success = ts_dict['success'] if 'success' in ts_dict else None
        self.energy = ts_dict['energy'] if 'energy' in ts_dict else None
        self.execution_time = ts_dict['execution_time'] if 'execution_time' in ts_dict else None
        self.method = ts_dict['method'].lower() if 'method' in ts_dict else 'user guess'
        if 'user guess' in self.method:
            if self.xyz is None:
                raise TSError('If no method is specified, an xyz guess must be given')
            self.success = self.success if self.success is not None else True
            self.execution_time = '0'
        self.reactants_xyz = ts_dict['reactants_xyz'] if 'reactants_xyz' in ts_dict else list()
        self.products_xyz = ts_dict['products_xyz'] if 'products_xyz' in ts_dict else list()
        self.family = ts_dict['family'] if 'family' in ts_dict else None
        if self.family is None and self.method.lower() in ['kinbot', 'autotst']:
            # raise TSError('No family specified for method {0}'.format(self.method))
            logger.warning('No family specified for method {0}'.format(self.method))
        if 'rmg_reaction' not in ts_dict:
            self.rmg_reaction = None
        else:
            rxn_string = ts_dict['rmg_reaction']
            plus = ' + '
            arrow = ' <=> '
            if arrow not in rxn_string:
                raise TSError('Could not read the reaction string. Expected to find " <=> ". '
                              'Got: {0}'.format(rxn_string))
            sides = rxn_string.split(arrow)
            reac = sides[0]
            prod = sides[1]
            if plus in reac:
                reac = reac.split(plus)
            else:
                reac = [reac]
            if plus in prod:
                prod = prod.split(plus)
            else:
                prod = [prod]
            reactants = list()
            products = list()
            for reactant in reac:
                reactants.append(Species().fromSMILES(str(reactant)))
            for product in prod:
                products.append(Species().fromSMILES(str(product)))
            self.rmg_reaction = Reaction(reactants=reactants, products=products)

    def execute_ts_guess_method(self):
        """
        Execute a TS guess method
        """
        if self.method == 'user guess':
            pass
        elif self.method == 'qst2':
            self.qst2()
        elif self.method == 'degsm':
            self.degsm()
        elif self.method == 'neb':
            self.neb()
        elif self.method == 'kinbot':
            self.kinbot()
        elif self.method == 'autotst':
            self.autotst()
        else:
            raise TSError('Unrecognized method. Should be either {0}. Got: {1}'.format(
                          ['User guess'] + default_ts_methods, self.method))

    def autotst(self):
        """
        Determine a TS guess using AutoTST for the following RMG families:
        Current supported families are:
        - H_Abstraction
        Near-future supported families might be:
        - Disproportionation
        - R_Addition_MultipleBond
        - intra_H_migration
        """
        if not isinstance(self.rmg_reaction, Reaction):
            raise InputError('AutoTST requires an RMG Reaction object. Got: {0}'.format(type(self.rmg_reaction)))
        if self.family not in ['H_Abstraction']:
            logger.debug('AutoTST currently only works for H_Abstraction. Got: {0}'.format(self.family))
            self.xyz = ''
        else:
            self.xyz = atst.autotst(rmg_reaction=self.rmg_reaction, reaction_family=self.family)

    def qst2(self):
        """
        Determine a TS guess using QST2
        """
        self.success = False

    def degsm(self):
        """
        Determine a TS guess using DEGSM
        """
        self.success = False

    def neb(self):
        """
        Determine a TS guess using NEB
        """
        self.success = False

    def kinbot(self):
        """
        Determine a TS guess using Kinbot for RMG's unimolecular families
        """
        self.success = False

    def process_xyz(self, xyz):
        """
        If xyz represents a file path, parse it.
        Standardize the xyz format using converter.standardize_xyz_string
        """
        if xyz is not None:
            if os.path.isfile(xyz):
                xyz = parse_xyz_from_file(xyz)
            self.xyz = standardize_xyz_string(xyz)


def determine_occ(label, xyz, charge):
    """
    Determines the number of occupied orbitals for an MRCI calculation
    """
    electrons = 0
    for line in xyz.split('\n'):
        if line:
            atom = Atom(element=str(line.split()[0]))
            electrons += atom.number
    electrons -= charge


def nearly_equal(a, b, sig_fig=5):
    """
    A helper function to determine whether two floats are nearly equal.
    Can be replaced by math.isclose in Py3
    """
    return a == b or int(a*10**sig_fig) == int(b*10**sig_fig)


def determine_rotor_symmetry(rotor_path, label, pivots):
    """
    Determine the rotor symmetry number from a potential energy scan.
    The *worst* resolution for each peak and valley is determined.
    The first criterion for a symmetric rotor is that the highest peak and the lowest peak must be within the
    worst peak resolution (and the same is checked for valleys).
    A second criterion for a symmetric rotor is that the highest and lowest peaks must be within 10% of
    the highest peak value. This is only applied if the highest peak is above 2 kJ/mol.

    Args:
        rotor_path (str): The path to an ESS output rotor scan file.
        label (str): The species label (used for error messages).
        pivots (list): A list of two atom indices representing the torsion pivots.

    Returns:
        int: The symmetry number (int)
    Returns:
        float: The highest torsional energy barrier in kJ/mol.
    """
    log = determine_qm_software(fullpath=rotor_path)
    energies, _ = log.loadScanEnergies()

    symmetry = None
    max_e = max(energies)
    if max_e > 2000:
        tol = 0.10 * max_e  # tolerance for the second criterion
    else:
        tol = max_e
    min_e = energies[0]
    for i, e in enumerate(energies):
        # sometimes the opt level and scan levels mismatch, causing the minimum to be close to 0 degrees, but not at 0
        if e < min_e:
            min_e = e
    peaks, valleys = list(), list()  # the peaks and valleys of the scan
    worst_peak_resolution, worst_valley_resolution = 0, 0
    for i, e in enumerate(energies):
        # identify peaks and valleys, and determine worst resolutions in the scan
        ip1 = cyclic_index_i_plus_1(i, len(energies))  # i Plus 1
        im1 = cyclic_index_i_minus_1(i)                # i Minus 1
        if i == 0 and energies[im1] == e:
            # If the first and last scan points have same energy, change im1
            im1 -= 1
            logger.debug('im1: {0}, ip1: {1}, em1: {2}, e: {3}, ep1: {4}'.format(
                im1, ip1, energies[im1], e, energies[ip1]))
        if e > energies[im1] and e > energies[ip1]:
            # this is a local peak
            if any([diff > worst_peak_resolution for diff in [e - energies[im1], e - energies[ip1]]]):
                worst_peak_resolution = max(e - energies[im1], e - energies[ip1])
            peaks.append(e)
        elif e < energies[im1] and e < energies[ip1]:
            # this is a local valley
            if any([diff > worst_valley_resolution for diff in [energies[im1] - e, energies[ip1] - e]]):
                worst_valley_resolution = max(energies[im1] - e, energies[ip1] - e)
            valleys.append(e)
    # The number of peaks and valley must always be the same (what goes up must come down), if it isn't then there's
    # something seriously wrong with the scan
    if len(peaks) != len(valleys):
        logger.error('Rotor of species {0} between pivots {1} does not have the same number'
                     ' of peaks ({2}) and valleys ({3}).'.format(label, pivots, len(peaks), len(valleys)))
        return len(peaks), max_e * 0.001  # this works for CC(=O)[O]
    min_peak = min(peaks)
    max_peak = max(peaks)
    min_valley = min(valleys)
    max_valley = max(valleys)
    # Criterion 1: worst resolution
    if max_peak - min_peak > worst_peak_resolution:
        # The rotor cannot be symmetric
        symmetry = 1
        reason = 'worst peak resolution criterion'
    elif max_valley - min_valley > worst_valley_resolution:
        # The rotor cannot be symmetric
        symmetry = 1
        reason = 'worst valley resolution criterion'
    # Criterion 2: 10% * max_peak
    elif max_peak - min_peak > tol:
        # The rotor cannot be symmetric
        symmetry = 1
        reason = '10% of the maximum peak criterion'
    else:
        # We declare this rotor as symmetric and the symmetry number is the number of peaks (and valleys)
        symmetry = len(peaks)
        reason = 'number of peaks and valleys, all within the determined resolution criteria'
    if symmetry not in [1, 2, 3]:
        logger.info('Determined symmetry number {0} for rotor of species {1} between pivots {2};'
                    ' you should make sure this makes sense'.format(symmetry, label, pivots))
    else:
        logger.info('Determined a symmetry number of {0} for rotor of species {1} between pivots {2}'
                    ' based on the {3}.'.format(symmetry, label, pivots, reason))
    return symmetry, max_e * 0.001  # max_e in kJ/mol


def cyclic_index_i_plus_1(i, length):
    """A helper function for cyclic indexing rotor scans"""
    return i + 1 if i + 1 < length else 0


def cyclic_index_i_minus_1(i):
    """A helper function for cyclic indexing rotor scans"""
    return i - 1 if i - 1 > 0 else -1


def determine_rotor_type(rotor_path):
    """
    Determine whether this rotor should be treated as a HinderedRotor of a FreeRotor
    according to it's maximum peak
    """
    log = determine_qm_software(fullpath=rotor_path)
    energies, _ = log.loadScanEnergies()
    max_val = max(energies) * 0.001  # convert to kJ/mol (Arkane used SI)
    return 'FreeRotor' if max_val < minimum_barrier else 'HinderedRotor'


def enumerate_bonds(mol):
    """
    A helper function for calling Molecule.enumerate_bonds.
    First, get the Kekulized molecule (get the Kekule version with alternating single and double bonds if the molecule
    is aromatic), since we don't have implementation for aromatic bond additivity corrections.

    Args:
        mol (Molecule): The 2D graph representation of the molecule.

    Returns:
        dict: Keys are bond types (elements and bond order symbol), values are number of occurrences per bond type.
    """
    mol_list = generate_kekule_structure(mol)
    if mol_list:
        return mol_list[0].enumerate_bonds()
    else:
        return mol.enumerate_bonds()


def check_xyz(xyz, multiplicity, charge):
    """
    Checks a string-format coordinates for electronic consistency with the spin multiplicity and charge.

    Args:
        xyz (str): The species coordinates.
        multiplicity (int): The species spin multiplicity.
        charge (int): The species net charge.

    Returns:
        bool: Whether the input arguments are all in agreement. True if they are.
    """
    symbols = get_xyz_matrix(xyz)[1]
    electrons = 0
    for symbol in symbols:
        for number, element_symbol in symbol_by_number.items():
            if symbol == element_symbol:
                electrons += number
                break
    electrons -= charge
    if electrons % 2 ^ multiplicity % 2:
        return True
    return False

