"""
A module for representing stationary points (chemical species and transition states).
If the species is a transition state (TS), its ``ts_guesses`` attribute will have one or more ``TSGuess`` objects.
"""

import datetime
import numpy as np
import os
from math import isclose
from typing import Dict, List, Optional, Tuple, Union

import rmgpy.molecule.element as elements
from arkane.common import ArkaneSpecies, symbol_by_number
from arkane.statmech import is_linear
from rmgpy.exceptions import AtomTypeError, InvalidAdjacencyListError
from rmgpy.molecule.molecule import Atom, Molecule
from rmgpy.molecule.resonance import generate_kekule_structure
from rmgpy.reaction import Reaction
from rmgpy.species import Species
from rmgpy.statmech import NonlinearRotor, LinearRotor
from rmgpy.transport import TransportData

from arc.common import (almost_equal_coords,
                        convert_list_index_0_to_1,
                        determine_symmetry,
                        dfs,
                        get_logger,
                        get_single_bond_length,
                        generate_resonance_structures,
                        is_angle_linear,
                        read_yaml_file,
                        rmg_mol_from_dict_repr,
                        rmg_mol_to_dict_repr,
                        timedelta_from_str,
                        sort_atoms_in_descending_label_order,
                        )
from arc.exceptions import InputError, RotorError, SpeciesError, TSError
from arc.imports import settings
from arc.level import Level
from arc.parser import (parse_1d_scan_energies,
                        parse_dipole_moment,
                        parse_polarizability,
                        parse_xyz_from_file,
                        process_conformers_file,
                        )
from arc.species import conformers
from arc.species.converter import (check_isomorphism,
                                   check_xyz_dict,
                                   check_zmat_dict,
                                   compare_confs,
                                   get_xyz_radius,
                                   modify_coords,
                                   molecules_from_xyz,
                                   order_atoms_in_mol_list,
                                   remove_dummies,
                                   rmg_mol_from_inchi,
                                   str_to_xyz,
                                   translate_to_center_of_mass,
                                   xyz_from_data,
                                   xyz_to_str,
                                   )
from arc.species.vectors import calculate_angle, calculate_distance, calculate_dihedral_angle

logger = get_logger()

valid_chars, minimum_barrier = settings['valid_chars'], settings['minimum_barrier']


class ARCSpecies(object):
    """
    A class for representing stationary points.

    Structures (rotors_dict is initialized in conformers.find_internal_rotors; pivots/scan/top values are 1-indexed)::

            rotors_dict: {0: {'pivots': ``List[int]``,  # 1-indexed
                              'top': ``List[int]``,  # 1-indexed
                              'scan': ``List[int]``,  # 1-indexed
                              'torsion': ``List[int]``,  # 0-indexed
                              'number_of_running_jobs': ``int``,
                              'success': Optional[``bool``],  # ``None`` by default
                              'invalidation_reason': ``str``,
                              'times_dihedral_set': ``int``,
                              'scan_path': <path to scan output file>,
                              'max_e': ``float``,  # relative to the minimum energy, in kJ/mol,
                              'trsh_counter': ``int``,
                              'trsh_methods': ``List[str]``,
                              'symmetry': ``int``,
                              'dimensions': ``int``,
                              'original_dihedrals': ``list``,
                              'cont_indices': ``list``,
                              'directed_scan_type': ``str``,
                              'directed_scan': ``dict``,  # keys: tuples of dihedrals as strings,
                                                          # values: dicts of energy, xyz, is_isomorphic, trsh
                             }
                          1: {}, ...
                         }

    Args:
        label (str, optional): The species label.
        is_ts (bool, optional): Whether the species represents a transition state.
        rmg_species (Species, optional): An RMG Species object to be converted to an ARCSpecies object.
        mol (Molecule, optional): An ``RMG Molecule``. Atom order corresponds to the order in .initial_xyz
        xyz (list, str, dict, optional): Entries are either string-format coordinates, file paths, or ARC's dict format.
                                         (If there's only one entry, it could be given directly, not in a list).
                                         The file paths could direct to either a .xyz file, ARC conformers
                                         (w/ or w/o energies), or an ESS log/input files.
        multiplicity (int, optional): The species' electron spin multiplicity. Can be determined from the
                                      adjlist/smiles/xyz (If unspecified, assumed to be either a singlet or a doublet).
        charge (int, optional): The species' net charge. Assumed to be 0 if unspecified.
        smiles (str, optional): A `SMILES <https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system>`_
                                representation for the species 2D graph.
        adjlist (str, optional): An `RMG adjacency list
                                 <https://reactionmechanismgenerator.github.io/RMG-Py/reference/molecule/adjlist.html>`_
                                 representation for the species 2D graph.
        inchi (str, optional): An `InChI <https://www.inchi-trust.org/>`_ representation for the species 2D graph.
        bond_corrections (dict, optional): The bond additivity corrections (BAC) to be used. Determined from the
                                           structure if not directly given.
        compute_thermo (bool, optional): Whether to calculate thermodynamic properties for this species.
        include_in_thermo_lib (bool, optional): Whether to include in the output RMG library.
        e0_only (bool, optional): Whether to only run statmech (w/o thermo) to compute E0.
        species_dict (dict, optional): A dictionary to create this object from (used when restarting ARC).
        yml_path (str, optional): Path to an Arkane YAML file representing a species (for loading the object).
        ts_number (int, optional): An auto-generated number associating the TS ARCSpecies object with the corresponding
                                   :ref:`ARCReaction <reaction>` object.
        rxn_label (str, optional): The reaction string (relevant for TSs).
        rxn_index (int, optional): The reaction index which is the respective key to the Scheduler rxn_dict.
        external_symmetry (int, optional): The external symmetry of the species (not including rotor symmetries).
        optical_isomers (int, optional): Whether (=2) or not (=1) the species has chiral center/s.
        run_time (timedelta, optional): Overall species execution time.
        checkfile (str, optional): The local path to the latest checkfile by Gaussian for the species.
        number_of_radicals (int, optional): The number of radicals (inputted by the user, ARC won't attempt to determine
                                            it). Defaults to None. Important, e.g., if a Species is a bi-rad singlet,
                                            in which case the job should be unrestricted, but the multiplicity does not
                                            have the required information to make that decision (r vs. u).
        force_field (str, optional): The force field to be used for conformer screening. The default is MMFF94s.
                                     Other optional force fields are MMFF94, UFF, or GAFF (not recommended, slow).
                                     If 'fit' is specified for this parameter, some initial MMFF94s conformers will be
                                     generated, then force field parameters will be fitted for this molecule and
                                     conformers will be re-run with the fitted force field (recommended for drug-like
                                     species and species with many heteroatoms). Another option is specifying 'cheap',
                                     and the "old" RDKit embedding method will be used.
        bdes (list, optional): Specifying for which bonds should bond dissociation energies be calculated.
                               Entries are bonded atom indices tuples (1-indexed). An 'all_h' string entry is also
                               allowed, triggering BDE calculations for all hydrogen atoms in the molecule.
        directed_rotors (dict): Execute a directed internal rotation scan (i.e., a series of sp or constrained opt jobs)
                                for the pivots of interest. The optional primary keys are:
                                - 'brute_force_sp'
                                - 'brute_force_opt'
                                - 'cont_opt'
                                - 'ess'
                                The brute force methods will generate all the geometries in advance and submit all
                                relevant jobs simultaneously. The continuous method will wait for the previous job
                                to terminate, and use its geometry as the initial guess for the next job.
                                Another set of three keys is allowed, adding `_diagonal` to each of the above
                                keys. the secondary keys are therefore:
                                - 'brute_force_sp_diagonal'
                                - 'brute_force_opt_diagonal'
                                - 'cont_opt_diagonal'
                                Specifying '_diagonal' will increment all the respective dihedrals together,
                                resulting in a 1D scan instead of an ND scan.
                                Values are nested lists. Each value is a list where the entries are either pivot lists
                                (e.g., [1, 5]) or lists of pivot lists (e.g., [[1, 5], [6, 8]]), or a mix
                                (e.g., [[4, 8], [[6, 9], [3, 4]]]). The requested directed scan type will be executed
                                separately for each list entry in the value. A list entry that contains only two pivots
                                will result in a 1D scan, while a list entry with N pivots will consider all of them,
                                and will result in an ND scan if '_diagonal' is not specified.
                                ARC will generate geometries using the ``rotor_scan_resolution`` argument in settings.py
                                Note: An 'all' string entry is also allowed in the value list, triggering a directed
                                internal rotation scan for all torsions in the molecule. If 'all' is specified within
                                a second level list, then all the dihedrals will be considered together.
                                Currently, ARC does not automatically identify torsions to be treated as ND, and this
                                attribute must be specified by the user.
                                An additional supported key is 'ess', in which case ARC will allow the ESS to take care
                                of spawning the ND continuous constrained optimizations (not yet implemented).
        consider_all_diastereomers (bool, optional): Whether to consider all different chiralities (tetrahedral carbon
                                                     centers, nitrogen inversions, and cis/trans double bonds) when
                                                     generating conformers. ``True`` to consider all. If no 3D
                                                     coordinates are given for the species, all diastereomers will be
                                                     considered, otherwise the chirality specified by the given
                                                     coordinates will be preserved.
        preserve_param_in_scan (list, optional): Entries are length two iterables of atom indices (1-indexed)
                                                 between which distances and dihedrals of these pivots must be
                                                 preserved. Used for identification of rotors which break a TS.
        fragments (Optional[List[List[int]]]):
            Fragments represented by this species, i.e., as in a VdW well or a TS.
            Entries are atom index lists of all atoms in a fragment, each list represents a different fragment.
        occ (int, optional): The number of occupied orbitals (core + val) from a molpro CCSD sp calc.
        irc_label (str, optional): The label of an original ``ARCSpecies`` object (a TS) for which an IRC job was spawned.
                                   The present species object instance represents a geometry optimization job of the IRC
                                   result in one direction.
        project_directory (str, optional): The path to the project directory.
        multi_species: (str, optional): The multi-species set this species belongs to. Used for running a set of species
                                       simultaneously in a single ESS input file. A species marked as multi_species
                                       will only have one conformer considered (n_confs set to 1).

    Attributes:
        label (str): The species' label.
        original_label (str): The species' label prior to modifications (removing forbidden characters).
        multiplicity (int): The species' electron spin multiplicity. Can be determined from the adjlist/smiles/xyz
                            (If unspecified, assumed to be either a singlet or a doublet).
        charge (int): The species' net charge. Assumed to be 0 if unspecified.
        number_of_radicals (int): The number of radicals (inputted by the user, ARC won't attempt to determine it).
                                  Defaults to None. Important, e.g., if a Species is a bi-rad singlet, in which case
                                  the job should be unrestricted, but the multiplicity does not have the required
                                  information to make that decision (r vs. u).
        e_elect (float): The total electronic energy (without ZPE) at the chosen sp level, in kJ/mol.
        e0 (float): The 0 Kelvin energy (total electronic energy plus ZPE) at the chosen sp level, in kJ/mol.
        is_ts (bool):  Whether the species represents a transition state. `True` if it does.
        number_of_rotors (int): The number of potential rotors to scan.
        rotors_dict (dict): A dictionary of rotors. structure given below.
        conformers (list): A list of selected conformers XYZs (dict format).
        conformer_energies (list): A list of conformers E0 (in kJ/mol).
        cheap_conformer (str): A string format xyz of a cheap conformer (not necessarily the best/lowest one).
        most_stable_conformer (int): The index of the best/lowest conformer in self.conformers.
        recent_md_conformer (list): A length three list containing the coordinates of the recent conformer
                                    generated by MD, the energy in kJ/mol, and the number of MD runs. Used to detect
                                    when the MD algorithm converges on a single structure.
        initial_xyz (dict): The initial geometry guess.
        final_xyz (dict): The optimized species geometry.
        _radius (float): The species radius in Angstrom.
        opt_level (str): Level of theory for geometry optimization. Saved for archiving.
        _number_of_atoms (int): The number of atoms in the species/TS.
        mol (Molecule): An ``RMG Molecule`` object used for BAC determination.
                        Atom order corresponds to the order in .initial_xyz
        mol_list (list): A list of localized structures generated from 'mol', if possible.
        rmg_species (Species): An RMG Species object to be converted to an ARCSpecies object.
        bond_corrections (dict): The bond additivity corrections (BAC) to be used. Determined from the structure
                                 if not directly given.
        run_time (timedelta): Overall species execution time.
        t1 (float): The T1 diagnostic parameter from Molpro.
        neg_freqs_trshed (list): A list of negative frequencies this species was troubleshooted for.
        compute_thermo (bool): Whether to calculate thermodynamic properties for this species.
        include_in_thermo_lib (bool): Whether to include in the output RMG library.
        e0_only (bool): Whether to only run statmech (w/o thermo) to compute E0.
        thermo (HeatCapacityModel): The thermodata calculated by ARC.
        rmg_thermo (Dict[str, int]): The RMG thermo, 'H298' in kJ/mol and 'S298' in J/mol*K.
        long_thermo_description (str): A description for the species entry in the thermo library outputted.
        ts_guesses (list): A list of TSGuess objects for each of the specified methods.
        successful_methods (list): Methods used to generate a TS guess that successfully generated an XYZ guess.
        unsuccessful_methods (list): Methods used to generate a TS guess that were unsuccessfully.
        chosen_ts (int): The TSGuess index corresponding to the chosen TS conformer used for optimization.
        chosen_ts_list (List[int]): The TSGuess index corresponding to the TS guesses that were tried out.
        chosen_ts_method (str): The TS method that was actually used for optimization.
        ts_checks (Dict[str, bool]): Checks that a TS species went through.
        rxn_zone_atom_indices (List[int]): 0-indexed atom indices of the active reaction zone.
        ts_conf_spawned (bool): Whether conformers were already spawned for the Species (representing a TS) based on its
                                TSGuess objects.
        tsg_spawned (bool): If this species is a TS, this attribute describes whether TS guess jobs were already spawned.
        ts_guesses_exhausted (bool): Whether all TS guesses were tried out with no luck
                                     (``True`` if no convergence achieved).
        ts_number (int): An auto-generated number associating the TS ARCSpecies object with the corresponding
                         :ref:`ARCReaction <reaction>` object.
        ts_report (str): A description of all methods used for guessing a TS and their ranking.
        rxn_label (str): The reaction string (relevant for TSs).
        rxn_index (int): The reaction index which is the respective key to the Scheduler rxn_dict.
        arkane_file (str): Path to the Arkane Species file generated in processor.
        yml_path (str): Path to an Arkane YAML file representing a species (for loading the object).
        keep_mol (bool): Label to prevent the generation of a new Molecule object.
        checkfile (str): The local path to the latest checkfile by Gaussian for the species.
        external_symmetry (int): The external symmetry of the species (not including rotor symmetries).
        optical_isomers (int): Whether (=2) or not (=1) the species has chiral center/s.
        transport_data (TransportData): A placeholder for updating transport properties after Lennard-Jones
                                        calculation (using OneDMin).
        force_field (str): The force field to be used for conformer screening. The default is MMFF94s.
                           Other optional force fields are MMFF94, UFF, or GAFF (not recommended, slow).
                           If 'fit' is specified for this parameter, some initial MMFF94s conformers will be generated,
                           then force field parameters will be fitted for this molecule and conformers will be re-run
                           with the fitted force field (recommended for drug-like species and species with many
                           heteroatoms). Another option is specifying 'cheap', and the "old" RDKit embedding method
                           will be used.
        conf_is_isomorphic (bool): Whether the lowest conformer is isomorphic with the 2D graph representation
                                   of the species. ``True`` if it is. Defaults to ``None``. If ``True``, an isomorphism
                                   check will be strictly enforced for the final optimized coordinates.
        conformers_before_opt (tuple): Conformers XYZs of a species before optimization.
        bdes (list): Specifying for which bonds should bond dissociation energies be calculated.
                     Entries are bonded atom indices tuples (1-indexed). An 'all_h' string entry is also allowed,
                     triggering BDE calculations for all hydrogen atoms in the molecule.
        directed_rotors (dict): Execute a directed internal rotation scan (i.e., a series of constrained optimizations).
                                Data is in 3 levels of nested lists, converted from pivots to four-atom torsion indices.
        consider_all_diastereomers (bool, optional): Whether to consider all different chiralities (tetrahydral carbon
                                                     centers, nitrogen inversions, and cis/trans double bonds) when
                                                     generating conformers. ``True`` to consider all.
        zmat (dict): The species internal coordinates (Z Matrix).
        preserve_param_in_scan (list): Entries are length two iterables of atom indices (1-indexed) between which
                                       distances and dihedrals of these pivots must be preserved.
        fragments (Optional[List[List[int]]]):
            Fragments represented by this species, i.e., as in a VdW well or a TS.
            Entries are atom index lists of all atoms in a fragment, each list represents a different fragment.
        occ (int): The number of occupied orbitals (core + val) from a molpro CCSD sp calc.
        irc_label (str): The label of an original ``ARCSpecies`` object (a TS)  for which an IRC job was spawned.
                         The present species object instance represents a geometry optimization job of the IRC
                         result in one direction. If a species is a transition state, then this attribute contains the
                         labels of the two corresponding "IRC species", separated by a blank space.
        project_directory (str): The path to the project directory.
        multi_species: (str): The multi-species set this species belongs to. Used for running a set of species
                             simultaneously in a single ESS input file.
    """

    def __init__(self,
                 adjlist: str = '',
                 bdes: Optional[list] = None,
                 bond_corrections: Optional[dict] = None,
                 charge: Optional[int] = None,
                 checkfile: Optional[str] = None,
                 compute_thermo: Optional[bool] = None,
                 include_in_thermo_lib: Optional[bool] = True,
                 consider_all_diastereomers: bool = True,
                 directed_rotors: Optional[dict] = None,
                 e0_only: bool = False,
                 external_symmetry: Optional[int] = None,
                 fragments: Optional[List[List[int]]] = None,
                 force_field: str = 'MMFF94s',
                 inchi: str = '',
                 is_ts: bool = False,
                 irc_label: Optional[str] = None,
                 label: Optional[str] = None,
                 mol: Optional[Molecule] = None,
                 multiplicity: Optional[int] = None,
                 multi_species: Optional[str] = None,
                 number_of_radicals: Optional[int] = None,
                 occ: Optional[int] = None,
                 optical_isomers: Optional[int] = None,
                 preserve_param_in_scan: Optional[list] = None,
                 rmg_species: Optional[Species] = None,
                 run_time: Optional[datetime.timedelta] = None,
                 rxn_label: Optional[str] = None,
                 rxn_index: Optional[int] = None,
                 smiles: str = '',
                 species_dict: Optional[dict] = None,
                 ts_number: Optional[int] = None,
                 xyz: Optional[Union[list, dict, str]] = None,
                 yml_path: Optional[str] = None,
                 keep_mol: bool = False,
                 project_directory: Optional[str] = None,
                 ):
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
        self.adjlist = adjlist
        self.multiplicity = multiplicity
        self.multi_species = multi_species
        self.number_of_radicals = number_of_radicals
        self.external_symmetry = external_symmetry
        self.irc_label = irc_label
        self.occ = occ
        self.optical_isomers = optical_isomers
        self.charge = charge
        self.run_time = run_time
        self.checkfile = checkfile
        self.transport_data = TransportData()
        self.yml_path = yml_path
        self.keep_mol = keep_mol
        self.fragments = fragments
        self.original_label = None
        self.chosen_ts = None
        self.rxn_zone_atom_indices = None
        self.ts_checks = dict()
        self.project_directory = project_directory
        self.label = label

        if species_dict is not None:
            # Reading from a dictionary (it's possible that the dict contains only a 'yml_path' argument, check first)
            if 'yml_path' in species_dict:
                if 'label' in species_dict:
                    self.label = species_dict['label']
                self.yml_path = species_dict['yml_path']
            else:
                self.from_dict(species_dict=species_dict)

        if species_dict is None or self.yml_path is not None:
            # Not reading from a dictionary.
            self.force_field = force_field
            self.is_ts = is_ts
            self.ts_conf_spawned = False
            self.ts_guesses_exhausted = False
            self.e_elect = None
            self.e0 = None
            self.arkane_file = None
            self.conf_is_isomorphic = None
            self.bdes = bdes
            self.directed_rotors = directed_rotors if directed_rotors is not None else dict()
            self.consider_all_diastereomers = consider_all_diastereomers
            self.zmat = None
            self.preserve_param_in_scan = preserve_param_in_scan
            self.rxn_label = rxn_label
            self.rxn_index = rxn_index
            self.successful_methods = list()
            self.unsuccessful_methods = list()
            self.chosen_ts_method = None
            self.chosen_ts_list = list()
            self.compute_thermo = compute_thermo if compute_thermo is not None else not self.is_ts
            self.include_in_thermo_lib = include_in_thermo_lib
            self.e0_only = e0_only
            self.long_thermo_description = ''
            self.opt_level = None
            self.ts_report = ''
            self.yml_path = self.yml_path or yml_path
            self.final_xyz = None
            self.number_of_rotors = 0
            self.rotors_dict = dict()
            self.rmg_species = rmg_species
            self.tsg_spawned = False
            regen_mol = True
            if bond_corrections is None:
                self.bond_corrections = dict()
            else:
                self.bond_corrections = bond_corrections

            if self.yml_path is not None:
                # a YAML path was given
                regen_mol = self.from_yml_file(self.label)
                if regen_mol:
                    if adjlist:
                        self.mol = Molecule().from_adjacency_list(adjlist=adjlist,
                                                                  raise_atomtype_exception=False,
                                                                  raise_charge_exception=False,
                                                                  )
                    elif inchi:
                        self.mol = rmg_mol_from_inchi(inchi)
                    elif smiles:
                        self.mol = Molecule(smiles=smiles)
                self.set_mol_list()
            elif self.rmg_species is not None:
                # an RMG Species was given
                if not isinstance(self.rmg_species, Species):
                    raise SpeciesError(f'The rmg_species parameter has to be a valid RMG Species object. '
                                       f'Got: {type(self.rmg_species)}')
                if not self.rmg_species.molecule:
                    raise SpeciesError('If an RMG Species given, it must have a non-empty molecule list')
                if not self.rmg_species.label and not label:
                    raise SpeciesError('If an RMG Species given, it must have a label or a label must be given '
                                       'separately')
                self.label = self.label or self.rmg_species.label
                if self.mol is None:
                    self.mol = self.rmg_species.molecule[0]
                self.multiplicity = self.rmg_species.molecule[0].multiplicity
                self.charge = self.rmg_species.molecule[0].get_net_charge()

            self.process_xyz(xyz)
            if multiplicity is not None:
                self.multiplicity = multiplicity
            if charge is not None:
                self.charge = charge
            if self.mol is None:
                if adjlist:
                    self.mol = Molecule().from_adjacency_list(adjlist=adjlist,
                                                              raise_atomtype_exception=False,
                                                              raise_charge_exception=False,
                                                              )
                elif inchi:
                    self.mol = rmg_mol_from_inchi(inchi)
                elif smiles:
                    self.mol = Molecule(smiles=smiles)
                if self.mol is not None:
                    if self.multiplicity is None:
                        self.multiplicity = self.mol.multiplicity
                    if self.charge is None:
                        self.charge = self.mol.get_net_charge()
            if regen_mol:
                # Perceive molecule from xyz coordinates. This also populates the .mol attribute of the Species.
                # It overrides self.mol generated from adjlist or smiles so xyz and mol will have the same atom order.
                if self.final_xyz or self.initial_xyz or self.most_stable_conformer or self.conformers or self.ts_guesses:
                    self.mol_from_xyz(get_cheap=False)
            if not self.is_ts:
                # We don't care about BACs in TSs
                if self.mol is None:
                    if self.compute_thermo:
                        logger.warning(f'No structure (SMILES, adjList, RMG Species, or RMG Molecule) was given for '
                                       f'species {self.label}, NOT using bond additivity corrections (BAC) for thermo '
                                       f'computation.')
                else:
                    # Generate bond list for applying bond additivity corrections
                    if not self.bond_corrections and self.mol is not None:
                        self.bond_corrections = enumerate_bonds(self.mol)
                        if self.bond_corrections:
                            self.long_thermo_description += f'Bond corrections: {self.bond_corrections}\n'

            if not self.bond_corrections and self.compute_thermo \
                    and self.number_of_atoms is not None and self.number_of_atoms > 1:
                logger.warning(f'Cannot determine bond additivity corrections (BAC) for species {self.label} based on '
                               f'xyz coordinates only. For better thermodynamic properties, provide bond corrections.')

            self.neg_freqs_trshed = list()

        if self.charge is None:
            self.charge = 0
        if self.multiplicity is None or self.multiplicity < 1:
            self.determine_multiplicity(smiles, adjlist, self.mol)
        if not isinstance(self.multiplicity, int) and self.multiplicity is not None:
            raise SpeciesError(f'Multiplicity for species {self.label} is not an integer. '
                               f'Got {self.multiplicity} which is a {type(self.multiplicity)}.')
        if self.multiplicity is not None and self.multiplicity < 1:
            raise SpeciesError(f'Multiplicity for species {self.label} is lower than 1. Got: {self.multiplicity}')
        if not isinstance(self.charge, int):
            raise SpeciesError(f'Charge for species {self.label} is not an integer. '
                               f'Got {self.charge} which is a {type(self.charge)}.')
        if not self.is_ts and self.initial_xyz is None and self.final_xyz is None and self.mol is None \
                and not self.conformers:
            raise SpeciesError(f'No structure (xyz, SMILES, adjList, RMG Species or Molecule) '
                               f'was given for species {self.label}')
        if self.preserve_param_in_scan is not None:
            if not isinstance(self.preserve_param_in_scan, list):
                raise SpeciesError(f'preserve_param_in_scan must be a list, got {self.preserve_param_in_scan}, '
                                   f'which is a {type(self.preserve_param_in_scan)}')
            for entry in self.preserve_param_in_scan:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise SpeciesError(f'Each entry in preserve_param_in_scan must be a length 2 list, got '
                                       f'{self.preserve_param_in_scan}')
                if 0 in entry:
                    raise SpeciesError(f'preserve_param_in_scan must be 1-indexed, got:\n{self.preserve_param_in_scan}')
        self.label, self.original_label = check_label(label=self.label, is_ts=self.is_ts)
        allowed_keys = ['brute_force_sp', 'brute_force_opt', 'cont_opt', 'ess',
                        'brute_force_sp_diagonal', 'brute_force_opt_diagonal', 'cont_opt_diagonal']
        for key in self.directed_rotors.keys():
            if key not in allowed_keys:
                raise SpeciesError(f'Allowed keys for directed_rotors are {allowed_keys}. Got {key} for {self.label}')
        if self.bdes is not None:
            if not isinstance(self.bdes, list):
                raise SpeciesError(f'The .bdes argument (of {self.label}) must be a list, '
                                   f'got {self.bdes} which is a {type(self.bdes)}')
            for bde in self.bdes:
                if not bde == 'all_h' and not (isinstance(bde, (list, tuple)) and len(bde) == 2
                                               and all(b and isinstance(b, int) for b in bde)):
                    raise SpeciesError(f'Something is wrong with the .bdes attribute of {label}. '
                                       f'Expected tuples of two 1-indexed atoms, got:\n{self.bdes}')

        if self.mol is not None and self.mol_list is None:
            self.set_mol_list()
        if self.is_ts and not any(value is not None for key, value in self.ts_checks.items() if key != 'warnings'):
            self.populate_ts_checks()

    def __str__(self) -> str:
        """Return a string representation of the object"""
        str_representation = 'ARCSpecies('
        str_representation += f'label="{self.label}", '
        if self.mol is not None:
            str_representation += f'smiles="{self.mol.copy(deep=True).to_smiles()}", '
        str_representation += f'is_ts={self.is_ts}, '
        str_representation += f'multiplicity={self.multiplicity}, '
        str_representation += f'charge={self.charge})'
        return str_representation

    @property
    def number_of_atoms(self):
        """The number of atoms in the species"""
        if self._number_of_atoms is None:
            if self.mol is not None:
                self._number_of_atoms = len(self.mol.atoms)
            else:
                xyz = self.get_xyz()
                if xyz is not None:
                    self._number_of_atoms = len(xyz['symbols'])
        return self._number_of_atoms

    @number_of_atoms.setter
    def number_of_atoms(self, value):
        """Allow setting number of atoms, e.g. a TS might not have Molecule or xyz when initialized"""
        self._number_of_atoms = value

    @property
    def number_of_heavy_atoms(self) -> int:
        """The number of heavy (non hydrogen) atoms in the species"""
        if self._number_of_heavy_atoms is None:
            if self.mol is not None:
                self._number_of_heavy_atoms = len([atom for atom in self.mol.atoms if atom.is_non_hydrogen()])
            elif self.final_xyz is not None or self.initial_xyz is not None:
                self._number_of_heavy_atoms = len([symbol for symbol in self.get_xyz()['symbols'] if symbol != 'H'])
            elif self.is_ts:
                for ts_guess in self.ts_guesses:
                    if ts_guess.get_xyz() is not None:
                        self._number_of_heavy_atoms = len([symbol for symbol in ts_guess.get_xyz()['symbols'] if symbol != 'H'])
        return self._number_of_heavy_atoms

    @number_of_heavy_atoms.setter
    def number_of_heavy_atoms(self, value):
        """Allow setting number of heavy atoms, e.g. a TS might not have Molecule or xyz when initialized"""
        self._number_of_heavy_atoms = value

    @property
    def radius(self) -> float:
        """
        Determine the largest distance from the coordinate system origin attributed to one of the molecule's
        atoms in 3D space.

        Returns:
            float: The radius in Angstrom.
        """
        if self._radius is None:
            self._radius = get_xyz_radius(self.get_xyz())
            logger.info(f'Determined a radius of {self._radius:.2f} Angstrom for {self.label}')
        return self._radius

    @radius.setter
    def radius(self, value):
        """Allow setting the radius"""
        self._radius = value

    def copy(self):
        """
        Get a copy of this object instance.

        Returns:
            ARCSpecies: A copy of this object instance.
        """
        species_dict = self.as_dict(reset_atom_ids=True)
        return ARCSpecies(species_dict=species_dict)

    def as_dict(self,
                reset_atom_ids: bool = False,
                ) -> dict:
        """
        A helper function for dumping this object as a dictionary in a YAML file for restarting ARC.

        Args:
            reset_atom_ids (bool, optional): Whether to reset the atom IDs in the .mol Molecule attribute.
                                             Useful when copying the object to avoid duplicate atom IDs between
                                             different object instances.

        Returns:
            dict: The dictionary representation of the object instance.
        """
        species_dict = dict()
        if self.force_field != 'MMFF94s':
            species_dict['force_field'] = self.force_field
        if self.is_ts:
            species_dict['is_ts'] = self.is_ts
        if self.t1 is not None:
            species_dict['t1'] = self.t1
        species_dict['label'] = self.label
        if self.long_thermo_description:
            species_dict['long_thermo_description'] = self.long_thermo_description
        species_dict['multiplicity'] = self.multiplicity
        if self.multi_species is not None:
            species_dict['multi_species'] = self.multi_species
        if self.charge != 0:
            species_dict['charge'] = self.charge
        if not self.compute_thermo and not self.is_ts:
            species_dict['compute_thermo'] = self.compute_thermo
        if not self.include_in_thermo_lib:
            species_dict['include_in_thermo_lib'] = self.include_in_thermo_lib
        species_dict['number_of_rotors'] = self.number_of_rotors
        if self.external_symmetry is not None:
            species_dict['external_symmetry'] = self.external_symmetry
        if self.adjlist:
            species_dict['adjlist'] = '\n'.join(line.strip() for line in self.adjlist.splitlines() if line.strip())
        if self.irc_label is not None:
            species_dict['irc_label'] = self.irc_label
        if self.optical_isomers is not None:
            species_dict['optical_isomers'] = self.optical_isomers
        if self.neg_freqs_trshed:
            species_dict['neg_freqs_trshed'] = self.neg_freqs_trshed.tolist() \
                if isinstance(self.neg_freqs_trshed, np.ndarray) else self.neg_freqs_trshed
        if self.arkane_file is not None:
            species_dict['arkane_file'] = self.arkane_file
        if not self.consider_all_diastereomers:
            species_dict['consider_all_diastereomers'] = self.consider_all_diastereomers
        if self.is_ts:
            if len(self.ts_guesses):
                species_dict['ts_guesses'] = [tsg.as_dict() for tsg in self.ts_guesses]
            if self.ts_conf_spawned:
                species_dict['ts_conf_spawned'] = self.ts_conf_spawned
            if self.ts_guesses_exhausted:
                species_dict['ts_guesses_exhausted'] = self.ts_guesses_exhausted
            if self.ts_number is not None:
                species_dict['ts_number'] = self.ts_number
            if self.ts_report:
                species_dict['ts_report'] = self.ts_report
            if self.rxn_label is not None:
                species_dict['rxn_label'] = self.rxn_label
            if self.rxn_index is not None:
                species_dict['rxn_index'] = self.rxn_index
            if len(self.successful_methods):
                species_dict['successful_methods'] = self.successful_methods
            if len(self.unsuccessful_methods):
                species_dict['unsuccessful_methods'] = self.unsuccessful_methods
            if self.chosen_ts_method is not None:
                species_dict['chosen_ts_method'] = self.chosen_ts_method
            if self.chosen_ts is not None:
                species_dict['chosen_ts'] = self.chosen_ts
            if self.rxn_zone_atom_indices is not None:
                species_dict['rxn_zone_atom_indices'] = self.rxn_zone_atom_indices
            if len(self.chosen_ts_list):
                species_dict['chosen_ts_list'] = self.chosen_ts_list
            if self.ts_checks:
                species_dict['ts_checks'] = self.ts_checks
        if self.original_label is not None:
            species_dict['original_label'] = self.original_label
        if self.e_elect is not None:
            species_dict['e_elect'] = self.e_elect
        if self.fragments is not None:
            species_dict['fragments'] = self.fragments
        if self.e0 is not None:
            species_dict['e0'] = self.e0
        if self.e0_only is not False:
            species_dict['e0_only'] = self.e0_only
        if self.tsg_spawned is not False:
            species_dict['tsg_spawned'] = self.tsg_spawned
        if self.yml_path is not None:
            species_dict['yml_path'] = self.yml_path
        if self.run_time is not None:
            species_dict['run_time'] = self.run_time.total_seconds()
        if self.number_of_radicals is not None:
            species_dict['number_of_radicals'] = self.number_of_radicals
        if self.opt_level is not None:
            species_dict['opt_level'] = self.opt_level
        if self.directed_rotors:
            species_dict['directed_rotors'] = self.directed_rotors
        if self.conf_is_isomorphic is not None:
            species_dict['conf_is_isomorphic'] = self.conf_is_isomorphic
        if self.bond_corrections is not None:
            species_dict['bond_corrections'] = self.bond_corrections
        if self.mol is not None:
            species_dict['mol'] = rmg_mol_to_dict_repr(self.mol, reset_atom_ids=reset_atom_ids)
        if self.initial_xyz is not None:
            species_dict['initial_xyz'] = xyz_to_str(self.initial_xyz)
        if self.final_xyz is not None:
            species_dict['final_xyz'] = xyz_to_str(self.final_xyz)
        if self.zmat is not None:
            species_dict['zmat'] = self.zmat
        if self.checkfile is not None:
            species_dict['checkfile'] = self.checkfile
        if self.occ is not None:
            species_dict['occ'] = self.occ
        if self.most_stable_conformer is not None:
            species_dict['most_stable_conformer'] = self.most_stable_conformer
        if self.cheap_conformer is not None:
            species_dict['cheap_conformer'] = xyz_to_str(self.cheap_conformer)
        if self.recent_md_conformer is not None:
            species_dict['recent_md_conformer'] = xyz_to_str(self.recent_md_conformer)
        if self._radius is not None:
            species_dict['radius'] = self._radius
        if self.conformers:
            species_dict['conformers'] = [xyz_to_str(conf) for conf in self.conformers]
            species_dict['conformer_energies'] = self.conformer_energies
        if self.conformers_before_opt is not None:
            species_dict['conformers_before_opt'] = [xyz_to_str(conf) for conf in self.conformers_before_opt]
        if self.bdes is not None:
            species_dict['bdes'] = self.bdes
        if self.preserve_param_in_scan is not None:
            species_dict['preserve_param_in_scan'] = self.preserve_param_in_scan
        if self.rotors_dict is not None and len(list(self.rotors_dict.keys())):
            rotors_dict = dict()
            for index, rotor_dict in self.rotors_dict.items():
                rotors_dict[index] = dict()
                for key, val in rotor_dict.items():
                    if key == 'directed_scan':
                        rotors_dict[index][key] = dict()
                        for dihedrals, result in val.items():
                            rotors_dict[index][key][dihedrals] = dict()
                            for result_key, result_val in result.items():
                                if result_key == 'energy':
                                    rotors_dict[index][key][dihedrals][result_key] = str(result_val)
                                elif result_key == 'xyz':
                                    rotors_dict[index][key][dihedrals][result_key] = xyz_to_str(result_val) \
                                        if isinstance(result_val, dict) else result_val
                                else:
                                    rotors_dict[index][key][dihedrals][result_key] = result_val
                    else:
                        rotors_dict[index][key] = val
            species_dict['rotors_dict'] = rotors_dict
        elif self.rotors_dict is None:
            # this marks the species to skip rotor scans (it is not an empty dict)
            # this is valuable information, store it in the restart file
            species_dict['rotors_dict'] = self.rotors_dict
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
        self.original_label = species_dict['original_label'] if 'original_label' in species_dict else None
        self.t1 = species_dict['t1'] if 't1' in species_dict else None
        self.e_elect = species_dict['e_elect'] if 'e_elect' in species_dict else None
        self.e0 = species_dict['e0'] if 'e0' in species_dict else None
        self.tsg_spawned = species_dict['tsg_spawned'] if 'tsg_spawned' in species_dict else False
        self.occ = species_dict['occ'] if 'occ' in species_dict else None
        self.arkane_file = species_dict['arkane_file'] if 'arkane_file' in species_dict else None
        self.yml_path = species_dict['yml_path'] if 'yml_path' in species_dict else None
        self.rxn_label = species_dict['rxn_label'] if 'rxn_label' in species_dict else None
        self.rxn_index = species_dict['rxn_index'] if 'rxn_index' in species_dict else None
        self._radius = species_dict['radius'] if 'radius' in species_dict else None
        self.most_stable_conformer = species_dict['most_stable_conformer'] \
            if 'most_stable_conformer' in species_dict else None
        self.cheap_conformer = check_xyz_dict(species_dict['cheap_conformer']) if 'cheap_conformer' in species_dict else None
        self.recent_md_conformer = str_to_xyz(species_dict['recent_md_conformer']) \
            if 'recent_md_conformer' in species_dict else None
        self.fragments = species_dict['fragments'] if 'fragments' in species_dict else None
        self.force_field = species_dict['force_field'] if 'force_field' in species_dict else 'MMFF94s'
        self.long_thermo_description = species_dict['long_thermo_description'] \
            if 'long_thermo_description' in species_dict else ''
        self.initial_xyz = str_to_xyz(species_dict['initial_xyz']) if 'initial_xyz' in species_dict else None
        self.final_xyz = str_to_xyz(species_dict['final_xyz']) if 'final_xyz' in species_dict else None
        self.conf_is_isomorphic = species_dict['conf_is_isomorphic'] if 'conf_is_isomorphic' in species_dict else None
        self.zmat = check_zmat_dict(species_dict['zmat']) if 'zmat' in species_dict else None
        self.is_ts = species_dict['is_ts'] if 'is_ts' in species_dict else False
        self.ts_conf_spawned = species_dict['ts_conf_spawned'] if 'ts_conf_spawned' in species_dict \
            else False if self.is_ts else None
        self.adjlist = species_dict['adjlist'] if 'adjlist' in species_dict else None
        if self.is_ts:
            self.ts_number = species_dict['ts_number'] if 'ts_number' in species_dict else None
            self.ts_guesses_exhausted = species_dict['ts_guesses_exhausted'] if 'ts_guesses_exhausted' in species_dict else False
            self.ts_report = species_dict['ts_report'] if 'ts_report' in species_dict else ''
            self.ts_guesses = [TSGuess(ts_dict=tsg) for tsg in species_dict['ts_guesses']] \
                if 'ts_guesses' in species_dict else list()
            self.successful_methods = species_dict['successful_methods'] \
                if 'successful_methods' in species_dict else list()
            self.unsuccessful_methods = species_dict['unsuccessful_methods'] \
                if 'unsuccessful_methods' in species_dict else list()
            self.chosen_ts_method = species_dict['chosen_ts_method'] if 'chosen_ts_method' in species_dict else None
            self.chosen_ts = species_dict['chosen_ts'] if 'chosen_ts' in species_dict else None
            self.rxn_zone_atom_indices = species_dict['rxn_zone_atom_indices'] \
                if 'rxn_zone_atom_indices' in species_dict else None
            self.ts_checks = species_dict['ts_checks'] if 'ts_checks' in species_dict else dict()
            self.chosen_ts_list = species_dict['chosen_ts_list'] if 'chosen_ts_list' in species_dict else list()
            self.checkfile = species_dict['checkfile'] if 'checkfile' in species_dict else None
        if 'xyz' in species_dict and self.initial_xyz is None and self.final_xyz is None:
            self.process_xyz(species_dict['xyz'])
        self.multiplicity = species_dict['multiplicity'] if 'multiplicity' in species_dict else None
        self.multi_species = species_dict['multi_species'] if 'multi_species' in species_dict else None
        self.charge = species_dict['charge'] if 'charge' in species_dict else 0
        self.compute_thermo = species_dict['compute_thermo'] if 'compute_thermo' in species_dict else not self.is_ts
        self.include_in_thermo_lib = species_dict['include_in_thermo_lib'] if 'include_in_thermo_lib' in species_dict else True
        self.e0_only = species_dict['e0_only'] if 'e0_only' in species_dict else False
        self.number_of_radicals = species_dict['number_of_radicals'] if 'number_of_radicals' in species_dict else None
        self.opt_level = species_dict['opt_level'] if 'opt_level' in species_dict else None
        self.number_of_rotors = species_dict['number_of_rotors'] if 'number_of_rotors' in species_dict else 0
        self.external_symmetry = species_dict['external_symmetry'] if 'external_symmetry' in species_dict else None
        self.irc_label = species_dict['irc_label'] if 'irc_label' in species_dict else None
        self.optical_isomers = species_dict['optical_isomers'] if 'optical_isomers' in species_dict else None
        self.neg_freqs_trshed = species_dict['neg_freqs_trshed'] if 'neg_freqs_trshed' in species_dict else list()
        self.bond_corrections = species_dict['bond_corrections'] if 'bond_corrections' in species_dict else dict()
        if 'mol' in species_dict:
            if isinstance(species_dict['mol'], str):
                try:
                    self.mol = Molecule().from_adjacency_list(species_dict['mol'],
                                                              raise_atomtype_exception=False,
                                                              raise_charge_exception=False)
                except (ValueError, AtomTypeError, InvalidAdjacencyListError) as e:
                    logger.error(f"Could not read RMG adjacency list {species_dict['mol']}.\nGot:\n{e}")
            else:
                self.mol = rmg_mol_from_dict_repr(species_dict['mol'], is_ts=self.is_ts)
        else:
            self.mol = None
        smiles = species_dict['smiles'] if 'smiles' in species_dict else None
        inchi = species_dict['inchi'] if 'inchi' in species_dict else None
        adjlist = species_dict['adjlist'] if 'adjlist' in species_dict else None
        if self.mol is None:
            if adjlist is not None:
                self.mol = Molecule().from_adjacency_list(adjlist=adjlist, raise_atomtype_exception=False,
                                                          raise_charge_exception=False)
            elif inchi is not None:
                self.mol = rmg_mol_from_inchi(inchi)
            elif smiles is not None:
                if isinstance(smiles, list):
                    raise SpeciesError(f'Got a list value type for SMILES of species {self.label}:\n'
                                       f'{smiles}, type: {type(smiles)}\n'
                                       f'Did you mean to enter this as a string? Consider adding quotation marks '
                                       f'before and after the SMILES value if entering through a YAML file.')
                self.mol = Molecule(smiles=smiles)
        # Perceive molecule from xyz coordinates. This also populates the .mol attribute of the Species.
        # It overrides self.mol generated from adjlist or smiles so xyz and mol will have the same atom order.
        if self.final_xyz or self.initial_xyz or self.most_stable_conformer or self.conformers or self.ts_guesses:
            self.mol_from_xyz(get_cheap=False)
        if self.mol is not None:
            if 'bond_corrections' not in species_dict and not self.is_ts:
                self.bond_corrections = enumerate_bonds(self.mol)
                if self.bond_corrections:
                    self.long_thermo_description += f'Bond corrections: {self.bond_corrections}\n'
            if self.multiplicity is None:
                self.multiplicity = self.mol.multiplicity
            if self.charge is None:
                self.charge = self.mol.get_net_charge()
        if 'conformers' in species_dict:
            self.conformers = [str_to_xyz(conf) for conf in species_dict['conformers']]
            self.conformer_energies = species_dict['conformer_energies'] if 'conformer_energies' in species_dict \
                else [None] * len(self.conformers)
        self.conformers_before_opt = [str_to_xyz(conf) for conf in species_dict['conformers_before_opt']] \
            if 'conformers_before_opt' in species_dict else None
        if self.mol is None and self.initial_xyz is None and self.final_xyz is None and not self.conformers \
                and not self.is_ts:
            # Only TS species are allowed to be loaded w/o a structure
            raise SpeciesError(f'Must have either mol or xyz for species {self.label}')
        self.directed_rotors = species_dict['directed_rotors'] if 'directed_rotors' in species_dict else dict()
        self.consider_all_diastereomers = species_dict['consider_all_diastereomers'] \
            if 'consider_all_diastereomers' in species_dict else True
        self.bdes = species_dict['bdes'] if 'bdes' in species_dict else None
        if self.bdes is not None and not isinstance(self.bdes, list):
            raise SpeciesError(f'The .bdes argument must be a list, got {self.bdes} which is a {type(self.bdes)}')
        self.rotors_dict = dict()
        if 'rotors_dict' in species_dict and species_dict['rotors_dict'] is None:
            self.rotors_dict = None
        self.preserve_param_in_scan = species_dict['preserve_param_in_scan'] \
            if 'preserve_param_in_scan' in species_dict else None
        if 'rotors_dict' in species_dict and self.rotors_dict is not None:
            for index, rotor_dict in species_dict['rotors_dict'].items():
                self.rotors_dict[index] = dict()
                for key, val in rotor_dict.items():
                    if key == 'directed_scan':
                        self.rotors_dict[index][key] = dict()
                        for dihedrals, result in val.items():
                            self.rotors_dict[index][key][dihedrals] = dict()
                            for directed_scan_key, directed_scan_val in result.items():
                                if directed_scan_key == 'energy':
                                    self.rotors_dict[index][key][dihedrals][directed_scan_key] = \
                                        float(directed_scan_val)
                                elif directed_scan_key == 'xyz':
                                    self.rotors_dict[index][key][dihedrals][directed_scan_key] = \
                                        str_to_xyz(directed_scan_val)
                                else:
                                    self.rotors_dict[index][key][dihedrals][directed_scan_key] = directed_scan_val
                    else:
                        self.rotors_dict[index][key] = val

    def from_yml_file(self, label: str = None) -> bool:
        """
        Load important species attributes such as label and final_xyz from an Arkane YAML file.
        Actual QM data parsing is done later when processing thermo and kinetics.

        Args:
            label (str, optional): The specie label.

        Raises:
            ValueError: If the adjlist cannot be read.

        Returns:
            bool: Whether self.mol should be regenerated
        """
        regen_mol = True
        rmg_spc = Species()
        arkane_spc = ArkaneSpecies(species=rmg_spc)
        # The data from the YAML file is loaded into the `species` argument of the `load_yaml` method in Arkane
        yml_content = read_yaml_file(self.yml_path)
        arkane_spc.load_yaml(path=self.yml_path, label=label, pdep=False)
        self.label = label or self.label or arkane_spc.label
        self.final_xyz = xyz_from_data(coords=arkane_spc.conformer.coordinates.value,
                                       numbers=arkane_spc.conformer.number.value)
        if 'mol' in yml_content:
            self.mol = rmg_mol_from_dict_repr(representation=yml_content['mol'], is_ts=yml_content['is_ts'])
            if self.mol is not None:
                regen_mol = False
        if regen_mol:
            if arkane_spc.adjacency_list is not None:
                try:
                    self.mol = Molecule().from_adjacency_list(adjlist=arkane_spc.adjacency_list,
                                                              raise_atomtype_exception=False)
                except ValueError:
                    print(f'Could not read adjlist:\n{arkane_spc.adjacency_list}')  # should *not* be logging
                    raise
            elif arkane_spc.inchi is not None:
                self.mol = Molecule().from_inchi(inchistr=arkane_spc.inchi, raise_atomtype_exception=False)
            elif arkane_spc.smiles is not None:
                self.mol = Molecule().from_smiles(arkane_spc.smiles, raise_atomtype_exception=False)
        if self.mol is not None:
            self.multiplicity = self.mol.multiplicity
            self.charge = self.mol.get_net_charge()
        if self.multiplicity is None:
            self.multiplicity = arkane_spc.conformer.spin_multiplicity
        if self.optical_isomers is None:
            self.optical_isomers = arkane_spc.conformer.optical_isomers
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
        if self.e0 is None:
            self.e0 = arkane_spc.conformer.E0.value_si * 0.001  # convert to kJ/mol
        return regen_mol

    def set_mol_list(self):
        """
        Set the .mol_list attribute from self.mol by generating resonance structures, preserving atom order.
        The mol_list attribute is used for identifying rotors and generating conformers.
        """
        if self.mol is not None:
            if all([atom.id == -1 for atom in self.mol.atoms]):
                self.mol.assign_atom_ids()
            if not self.is_ts:
                mol_copy = self.mol.copy(deep=True)
                mol_copy.reactive = True
                self.mol_list = generate_resonance_structures(mol_copy)
                if self.mol_list is None:
                    self.mol_list = [self.mol]
            else:
                self.mol_list = [self.mol]
            success = order_atoms_in_mol_list(ref_mol=self.mol.copy(deep=True), mol_list=self.mol_list)
            if not success:
                self.mol_list = None

    def is_monoatomic(self) -> Optional[bool]:
        """
        Determine whether the species is monoatomic.

        Returns:
            Optional[bool]: Whether the species is monoatomic.
        """
        if self.mol is not None and len(self.mol.atoms):
            return len(self.mol.atoms) == 1
        xyz = self.get_xyz()
        if xyz is not None:
            return len(xyz['symbols']) == 1
        return None

    def is_diatomic(self) -> Optional[bool]:
        """
        Determine whether the species is diatomic.

        Returns:
            Optional[bool]: Whether the species is diatomic.
        """
        if self.mol is not None and len(self.mol.atoms):
            return len(self.mol.atoms) == 2
        xyz = self.get_xyz()
        if xyz is not None:
            return len(xyz['symbols']) == 2
        return None

    def is_isomorphic(self, other: Union['ARCSpecies', Species, Molecule]) -> Optional[bool]:
        """
        Determine whether the species is isomorphic with ``other``.

        Args:
            other (Union[ARCSpecies, Species, Molecule]): An ARCSpecies, RMG Species, or RMG Molecule object instance
                                                          to compare isomorphism with.

        Returns:
            Optional[bool]: Whether the species is isomorphic with ``other``.
        """
        if self.mol is None:
            return None
        if isinstance(other, ARCSpecies):
            if other.mol is None:
                return None
            other = other.mol
        if isinstance(other, Molecule):
            if self.mol_list is not None and len(self.mol_list):
                for mol_ in [self.mol] + self.mol_list:
                    if mol_.copy(deep=True).is_isomorphic(other.copy(deep=True)):
                        return True
                return False
            else:
                return self.mol.copy(deep=True).is_isomorphic(other.copy(deep=True))
        if isinstance(other, Species):
            for other_mol in other.molecule:
                if self.mol_list is not None and len(self.mol_list):
                    for mol_ in [self.mol] + self.mol_list:
                        if mol_.copy(deep=True).is_isomorphic(other_mol.copy(deep=True)):
                            return True
                else:
                    return self.mol.copy(deep=True).is_isomorphic(other_mol.copy(deep=True))
            return False
        raise SpeciesError(f'Can only compare isomorphism to other ARCSpecies, RMG Species, or RMG Molecule '
                           f'object instances, got {other} which is of type {type(other)}.')

    def generate_conformers(self,
                            n_confs: int = 10,
                            e_confs: float = 5,
                            plot_path: str = None,
                            ) -> None:
        """
        Generate conformers.

        Args:
            n_confs (int, optional): The max number of conformers to store in the .conformers attribute
                                     that will later be DFT'ed at the conformers_level.
            e_confs (float, optional): The energy threshold in kJ/mol above the lowest energy conformer below which all
                                       (unique) generated conformers will be stored in the .conformers attribute.
            plot_path (str, optional): A folder path in which the plot will be saved.
                                       If None, the plot will not be shown (nor saved).
        """
        if self.is_ts:
            return
        if self.mol_list is None:
            self.set_mol_list()
        if self.mol_list is not None and not self.charge:
            mol_list = self.mol_list
        else:
            mol_list = [self.mol]
        if self.consider_all_diastereomers:
            diastereomers = None
        else:
            xyz = self.get_xyz(generate=False)
            diastereomers = [xyz] if xyz is not None else None
        lowest_confs = conformers.generate_conformers(mol_list=mol_list,
                                                      label=self.label,
                                                      charge=self.charge,
                                                      multiplicity=self.multiplicity,
                                                      force_field=self.force_field,
                                                      print_logs=False,
                                                      n_confs=n_confs,
                                                      e_confs=e_confs,
                                                      return_all_conformers=False,
                                                      plot_path=plot_path,
                                                      diastereomers=diastereomers,
                                                      )
        if len(lowest_confs):
            self.conformers.extend([conf['xyz'] for conf in lowest_confs])
            self.conformer_energies.extend([conf['FF energy'] for conf in lowest_confs])
        else:
            xyz = self.get_xyz(generate=False)
            if xyz is None or not xyz:
                logger.error(f'No 3D coordinates available for species {self.label}!')

    def get_cheap_conformer(self):
        """
        Cheaply (limiting the number of possible conformers) get a reasonable conformer,
        this could very well not be the best (lowest energy) one.
        """
        if self.is_monoatomic():
            self.cheap_conformer = \
                conformers.generate_monoatomic_conformer(symbol=self.mol_list[0].atoms[0].element.symbol)['xyz']
            self.initial_xyz = self.final_xyz = self.cheap_conformer
        elif self.is_diatomic():
            self.cheap_conformer = \
                conformers.generate_diatomic_conformer(symbol_1=self.mol_list[0].atoms[0].element.symbol,
                                                       symbol_2=self.mol_list[0].atoms[1].element.symbol,
                                                       multiplicity=self.multiplicity)['xyz']
        else:
            num_confs = min(500, max(50, len(self.mol.atoms) * 3))
            rd_mol = conformers.embed_rdkit(label=self.label, mol=self.mol, num_confs=num_confs)
            xyzs, energies = conformers.rdkit_force_field(label=self.label,
                                                          rd_mol=rd_mol,
                                                          mol=self.mol,
                                                          num_confs=num_confs,
                                                          force_field='MMFF94s',
                                                          )
            if energies:
                min_energy = min(energies)
                min_energy_index = energies.index(min_energy)
                self.cheap_conformer = xyzs[min_energy_index]
            elif xyzs:
                self.cheap_conformer = xyzs[0]
            else:
                logger.warning(f'Could not generate a cheap conformer for {self.label}')
                self.cheap_conformer = None

    def get_xyz(self,
                generate: bool = True,
                return_format: str = 'dict',
                ) -> Optional[Union[dict, str]]:
        """
        Get the highest quality xyz the species has.
        If it doesn't have any 3D information, and if ``generate`` is ``True``, cheaply generate it.
        Returns ``None`` if no xyz can be retrieved nor generated.

        Args:
            generate (bool, optional): Whether to cheaply generate an FF conformer if no xyz is found.
                                       ``True`` to generate. If generate is ``False`` and the species has no xyz data,
                                       the method will return None.
            return_format (str, optional): Whether to output a 'dict' or a 'str' representation of the respective xyz.

        Return:
             Optional[Union[dict, str]]: The xyz coordinates in the requested representation.
        """
        conf = self.conformers[0] if self.conformers else None
        xyz = self.final_xyz or self.initial_xyz or self.most_stable_conformer or conf or self.cheap_conformer
        if xyz is None:
            if self.is_ts:
                for ts_guess in self.ts_guesses:
                    if ts_guess.initial_xyz is not None:
                        xyz = ts_guess.opt_xyz or ts_guess.initial_xyz
                        return xyz
                return None
            elif generate and (self.mol is not None or self.mol_list is not None):
                self.get_cheap_conformer()
                if self.cheap_conformer is not None:
                    xyz = self.cheap_conformer
                else:
                    self.generate_conformers(n_confs=1)
                    if self.conformers is not None:
                        xyz = self.conformers[0]
        if return_format == 'str':
            xyz = xyz_to_str(xyz)
        return xyz

    def determine_rotors(self, verbose: bool = False) -> None:
        """
        Determine possible unique rotors in the species to be treated as hindered rotors,
        taking into account all localized structures.
        The resulting rotors are saved in a ``{'pivots': [1, 3], 'top': [3, 7], 'scan': [2, 1, 3, 7]}`` format
        in ``self.rotors_dict``. Also updates ``self.number_of_rotors``.
        """
        if self.rotors_dict != {}:
            # This species was marked to skip rotor scans (when ``rotors_dict`` is ``None``), or a rotors dictionary already exist.
            return
        mol_list = self.mol_list or [self.mol]
        if mol_list is None or not len(mol_list) or all([mol is None for mol in mol_list]):
            if not self.is_ts:
                logger.error(f'Could not determine rotors for {self.label} without a 2D graph structure')
        else:
            for mol in mol_list:
                if mol is None:
                    continue
                rotors = conformers.find_internal_rotors(mol.copy())
                for new_rotor in rotors:
                    for existing_rotor in self.rotors_dict.values():
                        if existing_rotor['pivots'] == new_rotor['pivots']:
                            break
                    else:
                        self.rotors_dict[self.number_of_rotors] = new_rotor
                        self.number_of_rotors += 1
        if verbose:
            if self.number_of_rotors == 1:
                logger.info(f'\nFound one possible rotor for {self.label}')
            elif self.number_of_rotors > 1:
                logger.info(f'\nFound {self.number_of_rotors} possible rotors for {self.label}')
            if self.number_of_rotors > 0:
                logger.info(f'Pivot list(s) for {self.label}: '
                            f'{[self.rotors_dict[i]["pivots"] for i in range(self.number_of_rotors)]}\n')
        self.initialize_directed_rotors()

    def initialize_directed_rotors(self):
        """
        Initialize self.directed_rotors, correcting the input to nested lists and to torsions instead of pivots.
        Also modifies self.rotors_dict to remove respective 1D rotors dicts if appropriate and adds ND rotors dicts.
        Principally, from this point we don't really need self.directed_rotors, self.rotors_dict should have all
        relevant rotors info for the species, including directed and ND rotors.

        Raises:
            SpeciesError: If the pivots don't represent a dihedral in the species.
        """
        if self.directed_rotors:
            all_pivots = [rotor_dict['pivots'] for rotor_dict in self.rotors_dict.values()]
            directed_rotors, directed_rotors_scans = dict(), dict()
            for key, vals in self.directed_rotors.items():
                # Reformat as nested lists.
                directed_rotors[key] = list()
                for val1 in vals:
                    if len(val1) != 2 and val1 not in ['all', ['all']]:
                        raise SpeciesError(f'directed_scan pivots must be lists of length 2, got {val1}.')
                    if isinstance(val1, (tuple, list)) and isinstance(val1[0], int):
                        corrected_val = val1 if list(val1) in all_pivots else [val1[1], val1[0]]  # re-order if needed
                        directed_rotors[key].append([list(corrected_val)])
                    elif isinstance(val1, (tuple, list)) and isinstance(val1[0], (tuple, list)):
                        directed_rotors[key].append([list(val2 if list(val2) in all_pivots else [val2[1], val2[0]])
                                                     for val2 in val1])
                    elif val1 == 'all':
                        # 1st level all, add all pivots, they will be treated separately
                        for i in range(self.number_of_rotors):
                            if [self.rotors_dict[i]['pivots']] not in directed_rotors[key]:
                                directed_rotors[key].append([self.rotors_dict[i]['pivots']])
                    elif val1 == ['all']:
                        # 2nd level all, add all pivots, they will be treated together
                        directed_rotors[key].append(all_pivots)
                    elif len(val1) != 2:
                        raise SpeciesError(f'directed_scan pivots must be lists of length 2, got {val1}.')
                    elif isinstance(val1, (tuple, list)) and isinstance(val1[0], int):
                        corrected_val = val1 if list(val1) in all_pivots else [val1[1], val1[0]]
                        directed_rotors[key].append([list(corrected_val)])
                    elif isinstance(val1, (tuple, list)) and isinstance(val1[0], (tuple, list)):
                        directed_rotors[key].append([list(val2 if list(val2) in all_pivots else [val2[1], val2[0]])
                                                     for val2 in val1])
            for key, vals1 in directed_rotors.items():
                # check
                for vals2 in vals1:
                    for val in vals2:
                        if val not in all_pivots:
                            raise SpeciesError(f'The pivots {val} do not represent a rotor in species {self.label}. '
                                               f'Valid rotor pivots are: {all_pivots}\n\n'
                                               f'Species coordinates:\n{xyz_to_str(self.get_xyz())}')
            # Modify self.rotors_dict to remove respective 1D rotors dicts and add ND rotors dicts.
            rotor_indices_to_del = list()
            for key, vals in directed_rotors.items():
                # An independent loop, so 1D *directed* scans won't be deleted (treated above).
                for pivots_list in vals:
                    new_rotor = {'pivots': pivots_list,  # 1-indexed
                                 'top': list(),  # 1-indexed
                                 'scan': list(),  # 1-indexed
                                 'torsion': list(),  # 0-indexed
                                 'number_of_running_jobs': 0,
                                 'success': None,
                                 'invalidation_reason': '',
                                 'times_dihedral_set': 0,
                                 'trsh_counter': 0,
                                 'trsh_methods': list(),
                                 'scan_path': '',
                                 'directed_scan_type': key,
                                 'directed_scan': dict(),
                                 'dimensions': 0,
                                 'original_dihedrals': list(),
                                 'cont_indices': list(),
                                 }
                    for pivots in pivots_list:
                        for index, rotor_dict in self.rotors_dict.items():
                            if rotor_dict['pivots'] == pivots:
                                new_rotor['top'].append(rotor_dict['top'])
                                new_rotor['scan'].append(rotor_dict['scan'])
                                new_rotor['torsion'].append(rotor_dict['torsion'])
                                new_rotor['dimensions'] += 1
                                if rotor_dict['directed_scan_type'] == 'ess' and index not in rotor_indices_to_del:
                                    # Remove this rotor dict, an ND one will be created instead.
                                    rotor_indices_to_del.append(index)
                                break
                    if new_rotor['dimensions'] != 1:
                        pass  # Todo: consolidate top
                    self.rotors_dict[max(list(self.rotors_dict.keys())) + 1] = new_rotor
            for i in set(rotor_indices_to_del):
                del self.rotors_dict[i]

            # Renumber the keys so iterative looping will make sense.
            new_rotors_dict = dict()
            for i, rotor_dict in enumerate(list(self.rotors_dict.values())):
                new_rotors_dict[i] = rotor_dict
            self.rotors_dict = new_rotors_dict
            self.number_of_rotors = i + 1

            # Replace pivots with four-atom scans.
            for key, vals in directed_rotors.items():
                directed_rotors_scans[key] = list()
                for pivots_list in vals:
                    for rotors_dict in self.rotors_dict.values():
                        if rotors_dict['pivots'] == pivots_list:
                            directed_rotors_scans[key].append(rotors_dict['scan'])
                            break
            self.directed_rotors = directed_rotors_scans

    def set_dihedral(self,
                     scan: list,
                     index: int = 1,
                     deg_increment: Optional[float] = None,
                     deg_abs: Optional[float] = None,
                     count: bool = True,
                     xyz: Optional[dict] = None,
                     chk_rotor_list: bool = True):
        """
        Set the dihedral angle value of the torsion ``scan``.
        Either increment by a given value or set to an absolute value.
        All bonded atoms are rotated as groups. The result is saved to ``self.initial_xyz``.

        Args:
            scan (list): The atom indices representing the dihedral.
            index (int, optional): Whether the atom indices are 1-indexed (pass ``1``) or 0-indexed (pass ``0``).
            deg_increment (float, optional): The dihedral angle increment in degrees.
            deg_abs (float, optional): The absolute desired dihedral angle in degrees.
            count (bool, optional): Whether to increment the rotor's times_dihedral_set parameter. `True` to increment.
            xyz (dict, optional): An alternative xyz to use instead of self.final_xyz.
            chk_rotor_list (bool, optional): Whether to check if the changing dihedral is in the rotor list.

        Raises:
            InputError: If both ``deg_increment`` and ``deg_abs`` are None.
            RotorError: If the rotor could not be identified based on the pivots.
            TypeError: If ``deg_increment`` or ``deg_abs`` are of wrong type.
        """
        if deg_increment is None and deg_abs is None:
            raise InputError('Either deg_increment or deg_abs must be specified.')
        if deg_increment is not None and not isinstance(deg_increment, (int, float)):
            raise TypeError(f'deg_increment must be a float, got {deg_increment} which is a {type(deg_increment)}')
        if deg_abs is not None and not isinstance(deg_abs, (int, float)):
            raise TypeError(f'deg_abs must be a float, got {deg_abs} which is a {type(deg_abs)}')
        pivots = scan[1:3]
        torsion = convert_list_index_0_to_1(scan, direction=-1) if index else scan
        rotor = None
        xyz = xyz or self.final_xyz
        if xyz is None:
            raise ValueError('Cannot set dihedral without xyz')
        if deg_increment is not None:
            deg_abs = calculate_dihedral_angle(coords=xyz, torsion=torsion) + deg_increment
        if is_angle_linear(calculate_angle(coords=xyz, atoms=torsion[:3], index=0)) \
                or is_angle_linear(calculate_angle(coords=xyz, atoms=torsion[1:], index=0)):
            logger.warning(f'Cannot change a dihedral that contains a linear segment. Got torsion:{torsion}, xyz:\n{xyz}')
            return None
        mol = self.mol
        if mol is None:
            mols = molecules_from_xyz(xyz, multiplicity=self.multiplicity, charge=self.charge)
            mol = mols[1] or mols[0]
        if chk_rotor_list:
            for rotor in self.rotors_dict.values():
                if rotor['pivots'] == pivots:
                    break
            if rotor is None:
                raise RotorError(f'Could not identify rotor based of pivots {pivots}:\n{list(self.rotors_dict.values())}')
            if count:
                if rotor['times_dihedral_set'] >= 10:
                    logger.info('\n\n')
                    for i, rotor in self.rotors_dict.items():
                        logger.error(f'Rotor {i} with pivots {rotor["pivots"]} was set '
                                     f'{rotor["times_dihedral_set"]} times')
                    rotor['success'] = False
                    rotor['invalidation_reason'] = f'rotor set too many ({rotor["times_dihedral_set"]}) times'
                    return
                rotor['times_dihedral_set'] += 1
        if deg_increment == 0 and deg_abs is None:
            logger.warning(f'set_dihedral was called with zero increment for {self.label} with pivots {pivots}')
        else:
            new_xyz = modify_coords(coords=xyz,
                                    indices=torsion,
                                    new_value=deg_abs,
                                    modification_type='groups',
                                    mol=mol,
                                    )
            self.initial_xyz = new_xyz

    def determine_symmetry(self) -> None:
        """
        Determine the external symmetry and chirality (optical isomers) of the species.
        """
        xyz = self.get_xyz()
        symmetry, optical_isomers = determine_symmetry(xyz)
        if self.optical_isomers is None:
            self.optical_isomers = self.optical_isomers or optical_isomers
        elif self.optical_isomers != optical_isomers:
            logger.warning(f"User input of optical isomers for {self.label} and ARC's calculation differ: "
                           f"{self.optical_isomers} and {optical_isomers}, respectively. "
                           f"Using the user input of {self.optical_isomers}")
        if self.external_symmetry is None:
            self.external_symmetry = self.external_symmetry or symmetry
        elif self.external_symmetry != symmetry:
            logger.warning(f"User input of external symmetry for {self.label} and ARC's calculation differ: "
                           f"{self.external_symmetry} and {symmetry}, respectively. "
                           f"Using the user input of {self.external_symmetry}")

    def determine_multiplicity(self,
                               smiles: str,
                               adjlist: str,
                               mol: Optional[Molecule],
                               ):
        """
        Determine the spin multiplicity of the species.

        Args:
            smiles (str): The SMILES descriptor .
            adjlist (str): The adjacency list descriptor.
            mol (Molecule): The respective RMG Molecule object.
        """
        if self.charge == 0 and not self.is_ts:
            self.determine_multiplicity_from_descriptors(smiles=smiles, adjlist=adjlist, mol=mol)
        if self.multiplicity is None or self.multiplicity < 1:
            self.determine_multiplicity_from_xyz()
        if self.multiplicity is None and not self.is_ts:
            raise SpeciesError(f'Could not determine multiplicity for species {self.label}')

    def determine_multiplicity_from_descriptors(self,
                                                smiles: str,
                                                adjlist: str,
                                                mol: Optional[Molecule]):
        """
        Determine the spin multiplicity of the species from the chemical descriptors.

        Args:
            smiles (str): The SMILES descriptor .
            adjlist (str): The adjacency list descriptor.
            mol (Molecule): The respective RMG Molecule object.
        """
        if mol is not None and mol.multiplicity >= 1:
            self.multiplicity = mol.multiplicity
        elif adjlist:
            mol = Molecule().from_adjacency_list(adjlist,
                                                 raise_atomtype_exception=False,
                                                 raise_charge_exception=False,
                                                 )
            self.multiplicity = mol.multiplicity
        elif self.mol is not None and self.mol.multiplicity >= 1:
            self.multiplicity = self.mol.multiplicity
        elif smiles:
            mol = Molecule(smiles=smiles)
            self.multiplicity = mol.multiplicity

    def determine_multiplicity_from_xyz(self):
        """
        Determine the spin multiplicity of the species from the xyz.
        """
        xyz = self.get_xyz()
        if xyz is None and len(self.conformers):
            xyz = self.conformers[0]
        if xyz:
            electrons = 0
            for symbol in xyz['symbols']:
                for number, symb in symbol_by_number.items():
                    if symbol == symb:
                        electrons += number
                        break
                else:
                    raise SpeciesError(f'Could not identify atom symbol {symbol}')
            electrons -= self.charge
            if electrons % 2 == 1:
                self.multiplicity = 2
                logger.debug(f'\nMultiplicity not specified for {self.label}, assuming a value of 2')
            else:
                self.multiplicity = 1
                logger.debug(f'\nMultiplicity not specified for {self.label}, assuming a value of 1')

    def make_ts_report(self):
        """A helper function to write content into the .ts_report attribute"""
        self.ts_report = ''
        if self.chosen_ts_method is not None:
            self.ts_report += f'\nTS method summary for {self.label}'
            if self.rxn_label is not None:
                self.ts_report += f' in {self.rxn_label}'
            self.ts_report += ':\n'
            if self.successful_methods:
                self.ts_report += 'Methods that successfully generated a TS guess:\n'
                for successful_method in self.successful_methods:
                    self.ts_report += successful_method + ','
            if self.unsuccessful_methods:
                self.ts_report += '\nMethods that were unsuccessfully in generating a TS guess:\n'
                for unsuccessful_method in self.unsuccessful_methods:
                    self.ts_report += unsuccessful_method + ','
            if not self.ts_guesses_exhausted:
                self.ts_report += f'\nThe method that generated the best TS guess and its output used for the ' \
                                  f'optimization: {self.chosen_ts_method}\n'

    def cluster_tsgs(self):
        """
        Cluster TSGuesses.
        """
        if not self.is_ts or not len(self.ts_guesses):
            return None
        cluster_tsgs = list()
        for tsg in self.ts_guesses:
            for cluster_tsg in cluster_tsgs:
                if cluster_tsg.almost_equal_tsgs(tsg):
                    cluster_tsg.cluster.append(tsg.index)
                    if tsg.method not in cluster_tsg.method:
                        cluster_tsg.method += f' + {tsg.method}'
                        cluster_tsg.execution_time = f'{cluster_tsg.execution_time} + {tsg.execution_time}'
                    break
            else:
                tsg.cluster = [tsg.index]
                cluster_tsgs.append(tsg)
        self.ts_guesses = cluster_tsgs

    def process_completed_tsg_queue_jobs(self, yml_path: str):
        """
        Process YAML files which are the output of running a TS guess job in the queue.

        Args:
            yml_path (str): The path to the output YAML file.
        """
        if not isinstance(yml_path, str) or not os.path.isfile(yml_path):
            return None
        tsg_list = read_yaml_file(yml_path)
        if not isinstance(tsg_list, list) or not all(isinstance(tsg, dict) for tsg in tsg_list):
            return None
        tsgs = [TSGuess(ts_dict=tsg_dict) for tsg_dict in tsg_list]
        for tsg in tsgs:
            if tsg.initial_xyz is not None and not colliding_atoms(tsg.initial_xyz):
                if tsg.index is None:
                    tsg.index = len(self.ts_guesses)
                self.ts_guesses.append(tsg)
        self.cluster_tsgs()

    def mol_from_xyz(self,
                     xyz: Optional[dict] = None,
                     get_cheap: bool = False,
                     ) -> None:
        """
        Make sure atom order in self.mol corresponds to xyz.
        Important for TS searches and for identifying rotor indices.
        This works by generating a molecule from xyz and using the
        2D structure to confirm that the perceived molecule is correct.
        If ``xyz`` is not given, the species xyz attribute will be used.

        Args:
            xyz (dict, optional): Alternative coordinates to use.
            get_cheap (bool, optional): Whether to generate conformers if the species has no xyz data.
        """
        if xyz is None:
            xyz = self.get_xyz(generate=get_cheap, return_format='dict')
        if xyz is None:
            return None

        if self.mol is not None:
            if len(self.mol.atoms) != len(xyz['symbols']):
                raise SpeciesError(f'The number of atoms in the molecule and in the coordinates of {self.label} is different.'
                                   f'\nGot:\n{self.mol.copy(deep=True).to_adjacency_list()}\nand:\n{xyz}')
            # self.mol should have come from another source, e.g., SMILES or yml.
            mol_s, mol_b = molecules_from_xyz(xyz=xyz,
                                              multiplicity=self.multiplicity,
                                              charge=self.charge)
            perceived_mol = mol_b or mol_s
            if perceived_mol is not None:
                allow_nonisomorphic_2d = (self.charge is not None and self.charge) \
                                         or self.mol.has_charge() or perceived_mol.has_charge() \
                                         or (self.multiplicity is not None and self.multiplicity >= 3) \
                                         or self.mol.multiplicity >= 3 or perceived_mol.multiplicity >= 3
                isomorphic = self.check_xyz_isomorphism(mol=perceived_mol,
                                                        xyz=xyz,
                                                        allow_nonisomorphic_2d=allow_nonisomorphic_2d)
                if not isomorphic:
                    logger.warning(f'XYZ and the 2D graph representation for {self.label} are not isomorphic.\nGot '
                                   f'xyz:\n{xyz}\n\nwhich corresponds to {self.mol.copy(deep=True).to_smiles()}\n'
                                   f'{self.mol.copy(deep=True).to_adjacency_list()}\n\nand: '
                                   f'{self.mol.copy(deep=True).to_smiles()}\n'
                                   f'{self.mol.copy(deep=True).to_adjacency_list()}')
                    raise SpeciesError(f'XYZ and the 2D graph representation for {self.label} are not compliant.')
                if not self.keep_mol:
                    self.mol = perceived_mol
        else:
            mol_s, mol_b = molecules_from_xyz(xyz, multiplicity=self.multiplicity, charge=self.charge)
            if mol_b is not None and len(mol_b.atoms) == self.number_of_atoms:
                self.mol = mol_b
            elif mol_s is not None and len(mol_s.atoms) == self.number_of_atoms:
                self.mol = mol_s
            else:
                logger.error(f'Could not infer a 2D graph for species {self.label}')

    def process_xyz(self, xyz_list: Union[list, str, dict]):
        """
        Process the user's input and add either to the .conformers attribute or to .ts_guesses.

        Args:
            xyz_list (list, str, dict): Entries are either string-format, dict-format coordinates or file paths.
                                        (If there's only one entry, it could be given directly, not in a list)
                                        The file paths could direct to either a .xyz file, ARC conformers (w/ or w/o
                                        energies), or an ESS log/input files, making this method extremely flexible.
                                        Internal coordinates (either string or dict) are also allowed and will be
                                        converted into cartesian coordinates.
        """
        if xyz_list is not None:
            if not isinstance(xyz_list, list):
                xyz_list = [xyz_list]
            xyzs, energies = list(), list()
            for xyz in xyz_list:
                xyz_ = ''
                if isinstance(xyz, str):
                    xyz_ = os.path.join(self.project_directory, xyz) \
                        if self.project_directory is not None \
                        and os.path.isfile(os.path.join(self.project_directory, xyz)) else xyz
                if not isinstance(xyz, (str, dict)):
                    raise InputError(f'Each xyz entry in xyz_list must be either a string or a dictionary. '
                                     f'Got:\n{xyz}\nwhich is a {type(xyz)}')
                if isinstance(xyz, dict):
                    xyzs.append(remove_dummies(check_xyz_dict(xyz)))
                    energies.append(None)  # dummy (lists should be the same length)
                elif os.path.isfile(xyz_):
                    file_extension = os.path.splitext(xyz_)[1]
                    if 'txt' in file_extension:
                        # assume this is an ARC conformer file
                        xyzs_, energies_ = process_conformers_file(conformers_path=xyz_)
                        xyzs.extend([remove_dummies(xyz_) for xyz_ in xyzs_])
                        energies.extend(energies_)
                    else:
                        # assume this is an ESS log file
                        xyzs.append(remove_dummies(parse_xyz_from_file(xyz_)))  # also calls standardize_xyz_string()
                        energies.append(None)  # dummy (lists should be the same length)
                elif isinstance(xyz, str):
                    # string which does not represent a (valid) path, treat as a string representation of xyz
                    xyzs.append(remove_dummies(str_to_xyz(xyz)))
                    energies.append(None)  # dummy (lists should be the same length)
            for i, xyz in enumerate(xyzs):
                if colliding_atoms(xyz):
                    raise SpeciesError(f'The following coordinates for species {self.label} have colliding atoms:\n'
                                       f'{xyz_to_str(xyz)}')
                if self.mol is not None:
                    check_atom_balance(xyz, self.mol)
                elif i:
                    check_atom_balance(xyz, xyzs[0])
            if not self.is_ts:
                self.conformers.extend(xyzs)
                self.conformer_energies.extend(energies)
            else:
                tsg_index = len(self.ts_guesses)
                for xyz, energy in zip(xyzs, energies):
                    self.ts_guesses.append(TSGuess(method=f'user guess {tsg_index}',
                                                   xyz=remove_dummies(xyz),
                                                   energy=energy,
                                                   success=True,
                                                   ))
                    # user guesses are always successful in generating a *guess*:
                    self.ts_guesses[tsg_index].success = True
                    tsg_index += 1
            if self.multiplicity is not None and self.charge is not None:
                for xyz in xyzs:
                    consistent = check_xyz(xyz=xyz, multiplicity=self.multiplicity, charge=self.charge)
                    if not consistent:
                        raise SpeciesError(f'Inconsistent combination of number of electrons, multiplicity and charge '
                                           f'for {self.label}.')

    def set_transport_data(self,
                           lj_path: str,
                           opt_path: str,
                           bath_gas: str,
                           opt_level: Level,
                           freq_path: Optional[str] = '',
                           freq_level: Optional[Level] = None):
        """
        Set the species.transport_data attribute after a Lennard-Jones calculation (via OneDMin).

        Args:
            lj_path (str): The path to a oneDMin job output file.
            opt_path (str): The path to an opt job output file.
            bath_gas (str): The oneDMin job bath gas.
            opt_level (Level): The optimization level of theory.
            freq_path (str, optional): The path to a frequencies job output file.
            freq_level (Level, optional): The frequencies level of theory.
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
                epsilon = (float(line.split()[-1]) * 11.96266, 'J/mol')
            elif 'Sigmas[angstrom]' in line:
                # Convert Angstroms to meters
                sigma = (float(line.split()[-1]) * 1e-10, 'm')
        if self.number_of_atoms == 1:
            shape_index = 0
            comment += '; The molecule is monoatomic'
        else:
            if is_linear(coordinates=np.array(self.get_xyz()['coords'])):
                shape_index = 1
                comment += '; The molecule is linear'
            else:
                shape_index = 2
        if self.number_of_atoms > 1:
            dipole_moment = parse_dipole_moment(opt_path) or 0
            if dipole_moment:
                comment += f'; Dipole moment was calculated at the {opt_level.simple()} level of theory'
        else:
            dipole_moment = 0
        polar = self.transport_data.polarizability or (0, 'angstroms^3')
        if freq_path:
            polar = (parse_polarizability(freq_path), 'angstroms^3')
            comment += f'; Polarizability was calculated at the {freq_level.simple()} level of theory'
        comment += '; Rotational Relaxation Collision Number was not determined, default value is 2'
        if original_comment:
            comment += '; ' + original_comment
        self.transport_data = TransportData(
            shapeIndex=shape_index,
            epsilon=epsilon,
            sigma=sigma,
            dipoleMoment=(dipole_moment, 'De'),
            polarizability=polar,
            rotrelaxcollnum=2,  # rotational relaxation collision number at 298 K
            comment=comment
        )

    def check_xyz_isomorphism(self,
                              mol: Optional[Molecule] = None,
                              xyz: Optional[dict] = None,
                              allow_nonisomorphic_2d: Optional[bool] = False,
                              verbose: Optional[bool] = True,
                              ) -> bool:
        """
        Check whether the perception of self.final_xyz or ``xyz`` is isomorphic with self.mol.
        If it is not isomorphic, compliant coordinates will be checked (equivalent to checking isomorphism without
        bond order information, only does not necessitate a molecule object, directly checks bond lengths).

        Args:
            mol (Molecule, optional): A molecule to check instead of self.mol.
            xyz (dict, optional): The coordinates to check instead of self.final_xyz.
            allow_nonisomorphic_2d (bool, optional): Whether to allow non-isomorphic representations to pass this test.
            verbose (bool, optional): Whether to log isomorphism findings and errors.

        Returns: bool
            Whether the perception of self.final_xyz is isomorphic with self.mol, ``True`` if it is.
        """
        mol = mol or self.mol
        xyz = xyz or self.final_xyz
        isomorphic = False

        if mol is not None:

            s_mol, b_mol = None, None

            # 1. Perceive
            try:
                s_mol, b_mol = molecules_from_xyz(xyz, multiplicity=self.multiplicity, charge=self.charge)
            except Exception as e:
                if verbose:
                    logger.error(f'Could not perceive the Cartesian coordinates of species {self.label}. This '
                                 f'might result in inconsistent atom order between the Cartesian and the 2D graph '
                                 f'representations. Got:\n{e}')

            # 2. A. Check isomorphism with bond orders using b_mol
            if b_mol is not None:
                isomorphic = check_isomorphism(mol, b_mol)
                if not isomorphic and verbose:
                    logger.error(f'The Cartesian coordinates of species {self.label} are not isomorphic with the 2D '
                                 f'structure {mol.copy(deep=True).to_smiles()} when considering bond orders.')

            # 2. B. Check isomorphism without bond orders using s_mol (only for charged or high spin multiplicity)
            if not isomorphic and s_mol is not None:
                isomorphic = check_isomorphism(mol, s_mol, convert_to_single_bonds=True)
                if not isomorphic and verbose:
                    logger.error(f'The Cartesian coordinates of species {self.label} with charge {self.charge} '
                                 f'and multiplicity {self.multiplicity} are not isomorphic with the 2D '
                                 f'structure {mol.copy(deep=True).to_smiles()} even without considering bond orders.')

            # 2. C. Check isomorphism without bond orders NOT using s_mol
            if not isomorphic:
                isomorphic = are_coords_compliant_with_graph(xyz=xyz, mol=mol)
                if not isomorphic and verbose:
                    logger.error(f'The Cartesian coordinates of species {self.label} are not compliant with the 2D '
                                 f'structure {mol.copy(deep=True).to_smiles()} even without considering bond orders.')

            # 3. Report and resolve
            if not isomorphic:
                if allow_nonisomorphic_2d:
                    if verbose:
                        logger.warning('Allowing nonisomorphic 2D')
                    isomorphic = True
                elif verbose:
                    if self.conf_is_isomorphic:
                        # conformer was isomorphic, we don't allow nonisomorphism, but the optimized structure isn't
                        logger.warning('Not allowing nonisomorphic 2D (the conformer WAS isomorphic with the 2D graph)')
                    else:
                        logger.warning('Not allowing nonisomorphic 2D')
        else:
            if verbose:
                logger.error(f'Cannot check isomorphism for species {self.label} '
                             f'without the 2D graph connectivity information.')
            if allow_nonisomorphic_2d:
                isomorphic = True
                if verbose:
                    logger.warning('Allowing nonisomorphic 2D')
        return isomorphic

    def label_atoms(self):
        """
        Labels atoms in order.
        The label is stored in the atom.label property.
        """
        for index, atom in enumerate(self.mol.atoms):
            atom.label = str(index)

    def scissors(self,
                 sort_atom_labels: bool = False,
                 ) -> list:
        """
        Cut chemical bonds to create new species from the original one according to the .bdes attribute,
        preserving the 3D geometry other than the scissioned bond.
        If one of the scission-resulting species is a hydrogen atom, it will be returned last, labeled as 'H'.
        Other species labels will be <original species label>_BDE_index1_index2_X, where "X" is either "A" or "B",
        and the indices are 1-indexed.
        
        Args:
            sort_atom_labels (bool, optional): Boolean flag, determines whether sorting is required.

        Returns: list
            The scission-resulting species.
        """
        all_h = True if 'all_h' in self.bdes else False
        if all_h:
            self.bdes.pop(self.bdes.index('all_h'))
        for entry in self.bdes:
            if len(entry) != 2:
                raise SpeciesError(f'Could not interpret entry {entry} in {self.bdes} for BDEs calculations.')
            if not isinstance(entry, (tuple, list)):
                raise SpeciesError(f'`indices` entries must be tuples or lists, '
                                   f'got {entry} which is a {type(entry)} in {self.bdes}')
        self.bdes = [tuple(bde) for bde in self.bdes]
        if all_h:
            for atom1 in self.mol.atoms:
                if atom1.is_hydrogen():
                    for atom2, bond12 in atom1.edges.items():
                        if bond12.is_single():
                            atom_indices = (self.mol.atoms.index(atom2) + 1, self.mol.atoms.index(atom1) + 1)
                            atom_indices_reverse = (atom_indices[1], atom_indices[0])
                            if atom_indices not in self.bdes and atom_indices_reverse not in self.bdes:
                                self.bdes.append(atom_indices)
        if sort_atom_labels:
            self.label_atoms()
        resulting_species = list()
        for index_tuple in self.bdes:
            new_species_list = self._scissors(indices=index_tuple, sort_atom_labels=sort_atom_labels)
            for new_species in new_species_list:
                if new_species.label not in [existing_species.label for existing_species in resulting_species]:
                    # Mainly checks that the H species doesn't already exist.
                    resulting_species.append(new_species)
        return resulting_species

    def _scissors(self,
                  indices: tuple,
                  sort_atom_labels: bool = True,
                  ) -> list:
        """
        Cut a chemical bond to create two new species from the original one, preserving the 3D geometry.

        Args:
            indices (tuple): The atom indices between which to cut (1-indexed, atoms must be bonded).
            sort_atom_labels (bool, optional): Boolean flag, determines whether sorting is required.

        Returns: list
            The scission-resulting species, a list of either one or two species, if the scissored location is linear,
            or one if the scission is in a cycle.
        """
        if any([i < 1 for i in indices]):
            raise SpeciesError(f'Scissors indices must be greater than 0 (1-indexed). Got: {indices}.')
        if not all([isinstance(i, int) for i in indices]):
            raise SpeciesError(f'Scissors indices must be integers. Got: {indices}.')
        if self.final_xyz is None:
            raise SpeciesError(f'Cannot use scissors without the .final_xyz attribute of species {self.label}')
        if len(indices) != 2:
            raise SpeciesError(f'Expected two indices, got {len(indices)}')
        indices = convert_list_index_0_to_1(indices, direction=-1)

        mol_copy = self.mol.copy(deep=True)
        # We are about to change the connectivity of the atoms in the molecule,
        # which invalidates any existing vertex connectivity information; thus we reset it.
        mol_copy.reset_connectivity_values()
        atom1 = mol_copy.atoms[indices[0]]
        atom2 = mol_copy.atoms[indices[1]]
        if not mol_copy.has_bond(atom1, atom2):
            raise SpeciesError('Attempted to remove a nonexistent bond.')
        bond = mol_copy.get_bond(atom1, atom2)
        if not bond.is_single():
            logger.warning(f'Scissors were requested to remove a non-single bond in {self.label}.')
        mol_copy.remove_bond(bond)
        mol_splits, fragment_indices = split_mol(mol_copy)
        if sort_atom_labels:
            for split in mol_splits:
                sort_atoms_in_descending_label_order(split)

        if len(mol_splits) == 1:  # If cutting leads to only one split, then the split is cyclic.
            spc1 = ARCSpecies(label=self.label + '_BDE_' + str(indices[0] + 1) + '_' + str(indices[1] + 1) + '_cyclic',
                              mol=mol_splits[0],
                              multiplicity=mol_splits[0].multiplicity,
                              charge=mol_splits[0].get_net_charge(),
                              compute_thermo=False,
                              e0_only=True)
            spc1.generate_conformers()
            return [spc1]
        elif len(mol_splits) == 2:
            mol1, mol2 = mol_splits
        else:
            logger.warning(f'Could not split {self.label} between indices {indices}.')
            return []

        used_a_label = False
        if len(mol1.atoms) == 1 and mol1.atoms[0].is_hydrogen():
            label1 = 'H'
        else:
            label1 = self.label + '_BDE_' + str(indices[0] + 1) + '_' + str(indices[1] + 1) + '_A'
            used_a_label = True
        if len(mol2.atoms) == 1 and mol2.atoms[0].is_hydrogen():
            label2 = 'H'
        else:
            letter = 'B' if used_a_label else 'A'
            label2 = self.label + '_BDE_' + str(indices[0] + 1) + '_' + str(indices[1] + 1) + '_' + letter

        added_radical = list()
        for mol, label in zip([mol1, mol2], [label1, label2]):
            for atom in mol.atoms:
                theoretical_charge = elements.PeriodicSystem.valence_electrons[atom.symbol] \
                                     - atom.get_total_bond_order() \
                                     - atom.radical_electrons - \
                                     2 * atom.lone_pairs
                if theoretical_charge == atom.charge + 1:
                    # we're missing a radical electron on this atom
                    if label not in added_radical or label == 'H':
                        atom.radical_electrons += 1
                        added_radical.append(label)
                    else:
                        raise SpeciesError(f'Could not figure out which atom should gain a radical '
                                           f'due to scission in {self.label}')
        mol1.update(log_species=False, raise_atomtype_exception=False, sort_atoms=False)
        mol2.update(log_species=False, raise_atomtype_exception=False, sort_atoms=False)

        xyz1, xyz2 = dict(), dict()
        xyz1['symbols'] = tuple(symbol for i, symbol in enumerate(self.final_xyz['symbols']) if i in fragment_indices[0])
        xyz2['symbols'] = tuple(symbol for i, symbol in enumerate(self.final_xyz['symbols']) if i in fragment_indices[1])
        xyz1['isotopes'] = tuple(isotope for i, isotope in enumerate(self.final_xyz['isotopes']) if i in fragment_indices[0])
        xyz2['isotopes'] = tuple(isotope for i, isotope in enumerate(self.final_xyz['isotopes']) if i in fragment_indices[1])
        xyz1['coords'] = tuple(coord for i, coord in enumerate(self.final_xyz['coords']) if i in fragment_indices[0])
        xyz2['coords'] = tuple(coord for i, coord in enumerate(self.final_xyz['coords']) if i in fragment_indices[1])
        xyz1, xyz2 = translate_to_center_of_mass(xyz1), translate_to_center_of_mass(xyz2)

        spc1 = ARCSpecies(label=label1,
                          mol=mol1,
                          xyz=xyz1,
                          multiplicity=mol1.multiplicity,
                          charge=mol1.get_net_charge(),
                          compute_thermo=False,
                          e0_only=True,
                          keep_mol=True)
        spc1.generate_conformers()
        spc1.rotors_dict = None
        spc2 = ARCSpecies(label=label2,
                          mol=mol2,
                          xyz=xyz2,
                          multiplicity=mol2.multiplicity,
                          charge=mol2.get_net_charge(),
                          compute_thermo=False,
                          e0_only=True,
                          keep_mol=True)
        spc2.generate_conformers()
        spc2.rotors_dict = None

        return [spc1, spc2]

    def populate_ts_checks(self):
        """Populate (or restart) the .ts_checks attribute with default (``None``) values."""
        if self.is_ts:
            keys = ['E0', 'e_elect', 'IRC', 'freq', 'NMD']
            self.ts_checks = {key: None for key in keys}
            self.ts_checks['warnings'] = ''


class TSGuess(object):
    """
    A class for representing TS a guess.

    Args:
        index (int, optional): A running index of all TSGuess objects belonging to an ARCSpecies object.
        method (str, optional): The method/source used for the xyz guess.
        method_index (int, optional): A sub-index, used for cases where a single method generates several guesses.
                                      Counts separately for each direction, 'F' and 'R'.
        method_direction (str, optional): The reaction direction used for generating the guess ('F' or 'R').
        constraints (dict, optional): Any constraints to be used when first optimizing this guess
                                      (i.e., keeping bond lengths of the reactive site constant).
        family (str, optional): The RMG family that corresponds to the reaction, if applicable.
        success (bool, optional): Whether the TS guess method succeeded in generating an XYZ guess or not.
        xyz (dict, str, optional): The 3D coordinates guess.
        arc_reaction (ARCReaction, optional): An ARC Reaction object.
        ts_dict (dict, optional): A dictionary to create this object from (used when restarting ARC).
        energy (float, optional): Relative energy of all TS conformers in kJ/mol.
        t0 (datetime.datetime, optional): Initial time of spawning the guess job.
        execution_time (datetime.timedelta, optional): Overall execution time for the TS guess method.
        project_directory (str, optional): The path to the project directory.

    Attributes:
        initial_xyz (dict): The 3D coordinates guess.
        opt_xyz (dict): The 3D coordinates after optimization at the ts_guesses level.
        method (str): The method/source used for the xyz guess.
        method_index (int): A sub-index, used for cases where a single method generates several guesses.
                            Counts separately for each direction, 'F' and 'R'.
        method_direction (str): The reaction direction used for generating the guess ('F' or 'R').
        family (str): The RMG family that corresponds to the reaction, if applicable.
        arc_reaction (ARCReaction): An ARC Reaction object.
        t0 (datetime.datetime, optional): Initial time of spawning the guess job.
        execution_time (str): Overall execution time for the TS guess method.
        success (bool): Whether the TS guess method succeeded in generating an XYZ guess or not.
        energy (float): Relative energy of all TS conformers in kJ/mol.
        index (int): A running index of all TSGuess objects belonging to an ARCSpecies object.
        imaginary_freqs (List[float]): The imaginary frequencies of the TS guess after optimization.
        conformer_index (int): An index corresponding to the conformer jobs spawned for each TSGuess object.
                               Assigned only if self.success is ``True``.
        successful_irc (bool): Whether the IRS run(s) identified this to be the correct TS by isomorphism of the wells.
        successful_normal_mode (bool): Whether a normal mode check was successful.
        errors (str): Problems experienced with this TSGuess. Used for logging.
        cluster (List[int]): Indices of TSGuess object instances clustered together.
    """

    def __init__(self,
                 index: Optional[int] = None,
                 method: Optional[str] = None,
                 method_index: Optional[int] = None,
                 method_direction: Optional[str] = None,
                 constraints: Optional[Dict[List[int], int]] = None,
                 t0: Optional[datetime.datetime] = None,
                 execution_time: Optional[Union[str, datetime.timedelta]] = None,
                 success: Optional[bool] = None,
                 family: Optional[str] = None,
                 xyz: Optional[Union[dict, str]] = None,
                 arc_reaction: Optional = None,
                 ts_dict: Optional[dict] = None,
                 energy: Optional[float] = None,
                 cluster: Optional[List[int]] = None,
                 project_directory: Optional[str] = None,
                 ):

        if ts_dict is not None:
            # Reading from a dictionary
            self.from_dict(ts_dict=ts_dict)
        else:
            # Not reading from a dictionary
            self.index = index
            self.method = method.lower() if method is not None else 'user guess'
            self.method_index = method_index
            self.method_direction = method_direction
            self.constraints = constraints
            self.t0 = t0
            self.execution_time = execution_time if execution_time is not None else execution_time
            self._opt_xyz = None
            self._initial_xyz = None
            self.process_xyz(xyz, project_directory=project_directory)  # populates self.initial_xyz
            self.success = success
            self.energy = energy
            self.cluster = cluster
            if 'user guess' in self.method:
                if self.initial_xyz is None:
                    raise TSError('If no method is specified, an xyz guess must be given')
                self.success = True
                self.execution_time = datetime.timedelta(seconds=0)
            self.arc_reaction = arc_reaction
            self.family = family
            self.imaginary_freqs = None
            self.conformer_index = None
            self.successful_irc = None
            self.successful_normal_mode = None
            self.errors = ''

    def __str__(self) -> str:
        """Return a string representation of the object"""
        str_representation = 'TSGuess('
        str_representation += f'index={self.index}, '
        str_representation += f'method="{self.method}", '
        str_representation += f'method_index={self.method_index}, '
        str_representation += f'method_direction="{self.method_direction}", '
        if self.cluster is not None:
            str_representation += f'cluster="{self.cluster}", '
        str_representation += f'success={self.success})'
        return str_representation

    @property
    def initial_xyz(self):
        """The initial coordinate guess"""
        return self._initial_xyz

    @initial_xyz.setter
    def initial_xyz(self, value):
        """Allow setting the initial coordinate guess"""
        self._initial_xyz = check_xyz_dict(value)

    @property
    def opt_xyz(self):
        """The optimized coordinates"""
        return self._opt_xyz

    @opt_xyz.setter
    def opt_xyz(self, value):
        """Allow setting the initial coordinate guess"""
        self._opt_xyz = check_xyz_dict(value)

    def as_dict(self, for_report: bool = False) -> dict:
        """
        A helper function for dumping this object as a dictionary.

        Args:
            for_report (bool, optional): Whether to generate a concise dictionary representation
                                         for the final_ts_guess_report.

        Returns:
            dict: The dictionary representation.
        """
        ts_dict = dict()
        ts_dict['method'] = self.method
        ts_dict['method_index'] = self.method_index
        if self.method_direction is not None:
            ts_dict['method_direction'] = self.method_direction
        if self.execution_time is not None:
            ts_dict['execution_time'] = str(self.execution_time) if isinstance(self.execution_time, datetime.timedelta) \
                else self.execution_time
        ts_dict['success'] = self.success
        if self.energy is not None:
            ts_dict['energy'] = self.energy
        ts_dict['index'] = self.index
        if self.imaginary_freqs is not None:
            ts_dict['imaginary_freqs'] = [float(f) for f in self.imaginary_freqs]
        ts_dict['conformer_index'] = self.conformer_index
        if self.successful_irc is not None:
            ts_dict['successful_irc'] = self.successful_irc
        if self.successful_normal_mode is not None:
            ts_dict['successful_normal_mode'] = self.successful_normal_mode
        if self.initial_xyz or for_report:
            ts_dict['initial_xyz'] = xyz_to_str(self.initial_xyz)
        if self.opt_xyz or for_report:
            ts_dict['opt_xyz'] = xyz_to_str(self.opt_xyz)
        if not for_report:
            ts_dict['t0'] = str(self.t0.isoformat()) if isinstance(self.t0, datetime.datetime) else self.t0
            if self.cluster is not None:
                ts_dict['cluster'] = self.cluster
            if self.family is not None:
                ts_dict['family'] = self.family
            if self.errors:
                ts_dict['errors'] = self.errors
        return ts_dict

    def from_dict(self, ts_dict: dict):
        """
        A helper function for loading this object from a dictionary in a YAML file for restarting ARC
        """
        self.t0 = datetime.datetime.fromisoformat(ts_dict['t0']) if 't0' in ts_dict and isinstance(ts_dict['t0'], str) \
            else ts_dict['t0'] if 't0' in ts_dict else None
        self.index = ts_dict['index'] if 'index' in ts_dict else None
        self.initial_xyz = ts_dict['initial_xyz'] if 'initial_xyz' in ts_dict else None
        self.process_xyz(self.initial_xyz)  # re-populates self.initial_xyz
        self.opt_xyz = ts_dict['opt_xyz'] if 'opt_xyz' in ts_dict else None
        self.success = ts_dict['success'] if 'success' in ts_dict else None
        self.energy = ts_dict['energy'] if 'energy' in ts_dict else None
        self.cluster = ts_dict['cluster'] if 'cluster' in ts_dict else None
        self.execution_time = timedelta_from_str(ts_dict['execution_time']) if 'execution_time' in ts_dict \
            and isinstance(ts_dict['execution_time'], str) \
            else ts_dict['execution_time'] if 'execution_time' in ts_dict else None
        self.method = ts_dict['method'].lower() if 'method' in ts_dict else 'user guess'
        self.method_index = ts_dict['method_index'] if 'method_index' in ts_dict else None
        self.method_direction = ts_dict['method_direction'] if 'method_direction' in ts_dict else None
        self.imaginary_freqs = ts_dict['imaginary_freqs'] if 'imaginary_freqs' in ts_dict else None
        self.conformer_index = ts_dict['conformer_index'] if 'conformer_index' in ts_dict else None
        self.successful_irc = ts_dict['successful_irc'] if 'successful_irc' in ts_dict else None
        self.successful_normal_mode = ts_dict['successful_normal_mode'] if 'successful_normal_mode' in ts_dict else None
        if 'user guess' in self.method:
            if self.initial_xyz is None:
                raise TSError('If no method is specified, an xyz guess must be given (initial_xyz).')
            self.success = self.success if self.success is not None else True
            self.execution_time = datetime.timedelta(seconds=0)
        self.family = ts_dict['family'] if 'family' in ts_dict else None
        self.errors = ts_dict['errors'] if 'errors' in ts_dict else ''

    def process_xyz(self,
                    xyz: Union[dict, str],
                    project_directory: Optional[str] = None,
                    ):
        """
        Process the user's input. If ``xyz`` represents a file path, parse it.

        Args:
            xyz (dict, str): The coordinates in a dict/string form or a path to a file containing the coordinates.
            project_directory (str, optional): The path to the project directory.

        Raises:
            InputError: If xyz is of the wrong type.
        """
        if xyz is not None:
            if not isinstance(xyz, (dict, str)):
                raise InputError(f'xyz must be either a dictionary or string, got:\n{xyz}\nwhich is a {type(xyz)}')
            self.initial_xyz = check_xyz_dict(xyz, project_directory=project_directory)

    def get_xyz(self,
                return_format: str = 'dict',
                ) -> Optional[Union[dict, str]]:
        """
        Get the highest quality xyz the TSGuess has.
        Returns ``None`` if no xyz can be retrieved.

        Args:
            return_format (str, optional): Whether to output a 'dict' or a 'str' representation of the respective xyz.

        Return:
             Optional[Union[dict, str]]: The xyz coordinates in the requested representation.
        """
        xyz = self.opt_xyz or self.initial_xyz
        if return_format == 'str':
            xyz = xyz_to_str(xyz)
        return xyz

    def almost_equal_tsgs(self, other: 'TSGuess') -> bool:
        """
        Determine whether two TSGuess object instances represent the same geometry.

        Args:
            other (TSGuess): The other TSGuess object instance to compare to.

        Returns:
            bool: Whether the two TSGuess object instances represent the same geometry.
        """
        if self.success != other.success or \
                (self.energy is not None and other.energy is not None
                 and not isclose(self.energy, other.energy, abs_tol=0.1)) or \
                (self.imaginary_freqs is not None and other.imaginary_freqs is not None
                 and (len(self.imaginary_freqs) != len(other.imaginary_freqs)
                      or len(self.imaginary_freqs) == len(other.imaginary_freqs)
                      and any([not isclose(freq_1, freq_2, abs_tol=0.1)
                               for freq_1, freq_2 in zip(self.imaginary_freqs, other.imaginary_freqs)]))):
            return False
        if almost_equal_coords(xyz1=self.get_xyz(), xyz2=other.get_xyz()) \
                or compare_confs(xyz1=self.get_xyz(), xyz2=other.get_xyz(), rmsd_score=False):
            return True
        return False

    def tic(self):
        """
        Initialize self.t0.
        """
        self.t0 = datetime.datetime.now()

    def tok(self):
        """
        Assign the time difference between now and self.t0 into self.execution_time.
        """
        if self.t0 is not None:
            self.execution_time = datetime.datetime.now() - self.t0


def determine_occ(xyz, charge):
    """
    Determines the number of occupied orbitals for an MRCI calculation.

    Todo
    """
    electrons = 0
    for line in xyz.split('\n'):
        if line:
            atom = Atom(element=str(line.split()[0]))
            electrons += atom.number
    electrons -= charge


def determine_rotor_symmetry(label: str,
                             pivots: Union[List[int], str],
                             rotor_path: str = '',
                             energies: Optional[Union[list, np.ndarray]] = None,
                             return_num_wells: bool = False,
                             log: bool = True,
                             ) -> Tuple[int, float, Optional[int]]:
    """
    Determine the rotor symmetry number from a potential energy scan.
    The *worst* resolution for each peak and valley is determined.
    The first criterion for a symmetric rotor is that the highest peak and the lowest peak must be within the
    worst peak resolution (and the same is checked for valleys).
    A second criterion for a symmetric rotor is that the highest and lowest peaks must be within 10% of
    the highest peak value. This is only applied if the highest peak is above 2 kJ/mol.

    Args:
        label (str): The species label (used for error messages).
        pivots (list): A list of two atom indices representing the pivots.
        rotor_path (str): The path to an ESS output rotor scan file.
        energies (list, optional): The list of energies in the scan in kJ/mol.
        return_num_wells (bool, optional): Whether to also return the number of wells, ``True`` to return,
                                           default is ``False``.
        log (bool, optional): Whether to log info, error, and warning messages.

    Raises:
        InputError: If both or none of the rotor_path and energy arguments are given,
        or if rotor_path does not point to an existing file.

    Returns:
        Tuple[int, float, int]:
            int: The symmetry number
            float: The highest torsional energy barrier in kJ/mol.
            int (optional): The number of peaks, only returned if ``return_len_peaks`` is ``True``.
    """
    if not rotor_path and energies is None:
        raise InputError('Expected either rotor_path or energies, got neither')
    if rotor_path and energies is not None:
        raise InputError('Expected either rotor_path or energies, got both')

    if energies is None:
        if not os.path.isfile(rotor_path):
            raise InputError(f'Could not find the path to the rotor file for species {label} {rotor_path}')
        energies = parse_1d_scan_energies(path=rotor_path)[0]

    symmetry = None
    max_e = max(energies)
    if max_e > 2000:
        tol = 0.10 * max_e  # tolerance for the second criterion
    else:
        tol = max_e
    min_e = energies[0]
    for i, e in enumerate(energies):
        # Sometimes the opt level and scan levels mismatch, causing the minimum to be close to 0 degrees, but not at 0.
        if e < min_e:
            min_e = e
    peaks, valleys = list(), list()  # the peaks and valleys of the scan
    worst_peak_resolution, worst_valley_resolution = 0, 0
    for i, e in enumerate(energies):
        # Identify peaks and valleys, and determine the worst resolutions in the scan.
        ip1 = cyclic_index_i_plus_1(i, len(energies))  # i Plus 1
        im1 = cyclic_index_i_minus_1(i)  # i Minus 1
        if i == 0 and energies[im1] == e:
            # If the first and last scan points have same energy, change im1
            im1 -= 1
            logger.debug(f'im1: {im1}, ip1: {ip1}, em1: {energies[im1]}, e: {e}, ep1: {energies[ip1]}')
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
        if log:
            logger.error(f'Rotor of species {label} in pivots {pivots} does not have the same number '
                         f'of peaks ({len(peaks)}) and valleys ({len(valleys)}).')
        if return_num_wells:
            return len(peaks), max_e, len(peaks)  # this works for CC(=O)[O]
        else:
            return len(peaks), max_e, None  # this works for CC(=O)[O]
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
    if log:
        if symmetry not in [1, 2, 3]:
            logger.info(f'Determined symmetry number {symmetry} for rotor of species {label} in pivots {pivots}; '
                        f'you should make sure this makes sense.')
        else:
            logger.info(f'Determined a symmetry number of {symmetry} for rotor of species {label} in pivots '
                        f'{pivots} based on the {reason}.')
    if return_num_wells:
        return symmetry, max_e, len(peaks)
    else:
        return symmetry, max_e, None


def cyclic_index_i_plus_1(i: int,
                          length: int,
                          ) -> int:
    """A helper function for cyclic indexing rotor scans"""
    return i + 1 if i + 1 < length else 0


def cyclic_index_i_minus_1(i: int) -> int:
    """A helper function for cyclic indexing rotor scans"""
    return i - 1 if i - 1 > 0 else -1


def determine_rotor_type(rotor_path: str) -> str:
    """
    Determine whether this rotor should be treated as a HinderedRotor of a FreeRotor
    according to its maximum peak.
    """
    energies = parse_1d_scan_energies(path=rotor_path)[0]
    max_val = max(energies)
    return 'FreeRotor' if max_val < minimum_barrier else 'HinderedRotor'


def enumerate_bonds(mol: Molecule) -> dict:
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


def check_xyz(xyz: dict,
              multiplicity: int,
              charge: int,
              ) -> bool:
    """
    Checks xyz for electronic consistency with the spin multiplicity and charge.

    Args:
        xyz (dict): The species coordinates.
        multiplicity (int): The species spin multiplicity.
        charge (int): The species net charge.

    Returns:
        bool: Whether the input arguments are all in agreement. True if they are.
    """
    symbols = xyz['symbols']
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


def are_coords_compliant_with_graph(xyz: dict,
                                    mol: Molecule,
                                    ) -> bool:
    """
    Check whether the Cartesian coordinates represent the same 2D connectivity as the graph.
    Bond orders are not considered here, this function checks whether the coordinates represent a bond length
    below 120% of the single bond length of the respective elements.

    Args:
        xyz (dict): The Cartesian coordinates.
        mol (Molecule): The 2D graph connectivity information.

    Returns:
        bool: Whether the coordinates are compliant with the 2D connectivity graph. ``True`` if they are.
    """
    checked_atoms = list()
    for atom_index_1, atom1 in enumerate(mol.atoms):
        if atom1.element.symbol != xyz['symbols'][atom_index_1]:
            logger.warning(f'Element order in xyz ({xyz["symbols"]}) differs from mol '
                           f'({[atom.element.symbol for atom in mol.atoms]})')
            return False
        for atom2 in atom1.edges.keys():
            atom_index_2 = mol.atoms.index(atom2)
            if atom_index_2 not in checked_atoms:
                r = calculate_distance(coords=xyz['coords'], atoms=[atom_index_1, atom_index_2])
                single_bond_r = get_single_bond_length(atom1.element.symbol, atom2.element.symbol)
                if r > single_bond_r * 1.2:
                    return False
        checked_atoms.append(atom_index_1)
    return True


def colliding_atoms(xyz: dict,
                    mol: Optional[Molecule] = None,
                    threshold: float = 0.55,
                    ) -> bool:
    """
        Check whether atoms are too close to each other.
        A default threshold of 55% the covalent radii of two atoms is used.
        For example:
        - C-O collide at 55% * 1.42 A = 0.781 A
        - N-N collide at 55% * 1.42 A = 0.781 A
        - C-N collide at 55% * 1.47 A = 0.808 A
        - C-H collide at 55% * 1.07 A = 0.588 A
        - H-H collide at 55% * 0.74 A = 0.588 A

        Args:
            xyz (dict): The Cartesian coordinates.
            mol (Molecule, optional): The corresponding Molecule object instance with formal charge information.
            threshold (float, optional): The collision threshold to use.

        Returns:
            bool: ``True`` if they are colliding, ``False`` otherwise.
    """
    if len(xyz['symbols']) == 1:
        # monoatomic
        return False
    for i in range(len(xyz['symbols']) - 1):
        for j in range(i + 1, len(xyz['symbols'])):
            actual_r = calculate_distance(coords=xyz['coords'], atoms=[i, j], index=0)
            charge_1 = mol.atoms[i].charge if mol is not None else 0
            charge_2 = mol.atoms[j].charge if mol is not None else 0
            single_bond_r = get_single_bond_length(xyz['symbols'][i], xyz['symbols'][j], charge_1, charge_2)
            if actual_r < single_bond_r * threshold:
                return True
    return False


def check_label(label: str,
                is_ts: bool = False,
                verbose: bool = False,
                ) -> Tuple[str, Optional[str]]:
    """
    Check whether a species (or reaction) label is legal, modify it if needed.

    Args:
        label (str): A label.
        is_ts (bool, optional): Whether the species label belongs to a transition state.
        verbose (bool, optional): Whether to log errors.

    Raises:
        TypeError: If the label is not a string type.
        SpeciesError: If label is illegal and cannot be automatically fixed.

    Returns: Tuple[str, Optional[str]]
        - A legal label.
        - The original label if the label was modified, else ``None``.
    """
    if not isinstance(label, str):
        raise TypeError(f'A species label must be a string type, got {label} which is a {type(label)}.')
    if label[:2] == 'TS' and all(char.isdigit() for char in label[2:]) and not is_ts:
        raise SpeciesError(f'A non-TS species cannot be named "TS" with a subsequent index, got {label}')

    char_replacement = {'#': 't',
                        '=': 'd',
                        '(': '[',
                        ')': ']',
                        ' ': '_',
                        '%': 'c',
                        '$': 'd',
                        '*': 's',
                        '@': 'a',
                        '+': 'p',
                        }
    modified = False
    original_label = label
    for char in original_label:
        if char not in valid_chars:
            if verbose:
                logger.error(f'Label {label} contains an invalid character: "{char}"')
            if char in char_replacement.keys():
                label = label.replace(char, char_replacement[char])
            else:
                label = label.replace(char, '_')
            modified = True
    if modified:
        if verbose:
            logger.warning(f'Replaced species label.\n'
                           f'Original label was: "{original_label}".\n'
                           f'New label is: "{label}"')
    else:
        original_label = None
    return label, original_label


def check_atom_balance(entry_1: Union[dict, str, Molecule],
                       entry_2: Union[dict, str, Molecule],
                       verbose: Optional[bool] = True,
                       ) -> bool:
    """
    Check whether the two entries are in atom balance.

    Args:
        entry_1 (Union[dict, str, Molecule]): Either an xyz (dict or str) or an RMG Molecule object.
        entry_2 (Union[dict, str, Molecule]): Either an xyz (dict or str) or an RMG Molecule object.
        verbose (Optional[bool]): Whether to log the differences if found.

    Raises:
        SpeciesError: If both entries are empty.

    Returns:
        bool: Whether ``entry1`` and ``entry2`` are in atomic balance. ``True`` id they are.
    """
    if not entry_1 or not entry_2:
        raise SpeciesError(f'Cannot compare entries. Got:\n{entry_1}\nand\n{entry_2}')
    element_dict_1, element_dict_2 = dict(), dict()
    result = True

    # Count the number of each element.
    for element_dict, entry in zip([element_dict_1, element_dict_2], [entry_1, entry_2]):
        if isinstance(entry, Molecule):
            for atom in entry.atoms:
                symbol = atom.element.symbol
                element_dict[symbol] = element_dict.get(symbol, 0) + 1
        else:
            xyz = check_xyz_dict(entry)
            for symbol in xyz['symbols']:
                element_dict[symbol] = element_dict.get(symbol, 0) + 1

    # Compare elements.
    if len(list(element_dict_1.keys())) != len(list(element_dict_1.keys())):
        result = False
    if result:
        for symbol in element_dict_1.keys():
            if symbol not in list(element_dict_2.keys()):
                result = False
                break
            num_1, num_2 = element_dict_1[symbol], element_dict_2[symbol]
            if num_1 != num_2:
                result = False
                break

    if not result:
        if verbose:
            logger.error(f'\nEntries have different types or numbers of elements, got:\n'
                         f'{element_dict_1}\nand\n{element_dict_2}\n')
        return False

    return result


def split_mol(mol: Molecule) -> Tuple[List[Molecule], List[List[int]]]:
    """
    Split an RMG Molecule object by connectivity gaps while retaining the relative atom order.

    Args:
        mol (Molecule): The Molecule to split.

    Returns:
        Tuple[List[Molecule], List[List[int]]]:
            - Entries are molecular fragments resulting from the split.
            - Entries are lists with indices that correspond to the original atoms that were assigned to each fragment.
    """
    fragments, molecules = list(), list()
    unvisited_indices = list(range(len(mol.atoms)))
    while len(unvisited_indices):
        start = unvisited_indices[0]
        frag_indices = dfs(mol=mol, start=start, sort_result=True)
        unvisited_indices = [index for index in unvisited_indices if index not in frag_indices]
        molecules.append(Molecule(atoms=[mol.atoms[index] for index in frag_indices]))
        fragments.append(frag_indices)
    return molecules, fragments
