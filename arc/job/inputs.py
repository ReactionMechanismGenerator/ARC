#!/usr/bin/env python3
# encoding: utf-8

"""
parameters for input files:

memory (in MB for gaussian, MW for molpro)
method
basis set
slash is '', unless this is gaussian NOT running a composite method, in which case it is '/'
charge
multiplicity/spin
xyz

gaussian:
    job_type_1: '' for sp, irc, or composite methods, 'opt=calcfc', 'opt=(calcfc,ts,noeigen)',
    job_type_2: '' or 'freq iop(7/33=1)' (cannot be combined with CBS-QB3)
                'scf=(tight,direct) int=finegrid irc=(rcfc,forward,maxpoints=100,stepsize=10) geom=check' for irc f
                'scf=(tight,direct) int=finegrid irc=(rcfc,reverse,maxpoints=100,stepsize=10) geom=check' for irc r
    scan: '\nD 3 1 5 8 S 36 10.000000' (with the line break)
    restricted: '' or 'u' for restricted / unrestricted
    `iop(2/9=2000)` makes Gaussian print the geometry nn eee input orientation even for molecules with more
      than 50 atoms (important so it matches the hessian, and so that Arkane can parse the geometry)

qchem:
    job_type_1: 'opt', 'ts', 'sp'
    job_type_2: 'freq'.
    fine: '\n   GEOM_OPT_TOL_GRADIENT 15\n   GEOM_OPT_TOL_DISPLACEMENT 60\n   GEOM_OPT_TOL_ENERGY 5\n   XC_GRID SG-3'
    restricted: 'false' or 'true' for restricted / unrestricted
"""

input_files = {
    'fukui': """#!/usr/bin/env python
# encoding: utf-8

import os
from arc.common import read_yaml_file, save_yaml_file
from arc.species.converter import xyz_to_str, str_to_xyz
from arc.species.species import ARCSpecies
from arc.species.conformers import generate_conformers
import time


yml_path = 'input.yml'
yml_save_path = 'fukui_confs.yml'

data = read_yaml_file(yml_path)

if os.path.isfile(yml_save_path):
    results = read_yaml_file(yml_save_path)
    for i, (label, val) in enumerate(results.items()):
        if 'final_xyz' in list(val.keys()) or 'error' in list(val.keys()):
            data[label] = val
            print(f'copied {label} (number {i}) from {yml_save_path}')
    results = dict()

for i, label in enumerate(data.keys()):
    if 'final_xyz' in list(data[label].keys()):
        print(f'skipping {label}, data already exists')
    elif 'error' not in list(data[label].keys()):
        t0 = time.time()
        print(f'processing {label}, number {i}, smiles: {data[label]["smiles"]}')
        data[label]['error'] = ''
        lowest_confs = None
        xyz = str_to_xyz(data[label]['original_xyz'])
        replace_with = ''
        if 'Br' in xyz['symbols']:
            replace_with = 'Cl' if 'Cl' not in xyz['symbols'] else 'I'
            new_symbols = list()
            for symbol in xyz['symbols']:
                new_symbols.append(symbol if symbol != 'Br' else replace_with)
            xyz['symbols'] = new_symbols

        try:
            spc = ARCSpecies(label=label, smiles=data[label]['smiles'], xyz=xyz)
            lowest_confs = generate_conformers(mol_list=spc.mol_list, label=spc.label,
            charge = spc.charge, multiplicity = spc.multiplicity,
            num_confs_to_return = 1, diastereomers = [spc.get_xyz()])
        except Exception as e:
            data[label]['error'] = str(e)
            print(str(e))
        if lowest_confs is None:
            data[label]['error'] = 'Could not generate confs, got None'
            data[label]['conf_run_time_(s)'] = time.time() - t0
            print('Could not generate confs, got None')
        else:
            data[label]['FF_energy'] = lowest_confs[0]['FF energy']
            if replace_with:
                new_symbols = list()
                for symbol in lowest_confs[0]['xyz']['symbols']:
                    new_symbols.append(symbol if symbol != replace_with else 'Br')
                lowest_confs[0]['xyz']['symbols'] = new_symbols
            data[label]['final_xyz'] = xyz_to_str(lowest_confs[0]['xyz'])
            data[label]['conf_run_time_(s)'] = time.time() - t0
            save_yaml_file(yml_save_path, data)

print('')
print('all done!')

""",
    'gaussian': """%chk=check.chk
%mem={memory}mb
%NProcShared={cpus}

#P {job_type_1} {restricted}{method}{slash}{basis} {job_type_2} {fine} {trsh} iop(2/9=2000)

name

{charge} {multiplicity}
{xyz}

{scan}{scan_trsh}


""",

    'qchem': """$molecule
{charge} {multiplicity}
{xyz}
$end

$rem
   JOBTYPE       {job_type_1}
   METHOD        {method}
   UNRESTRICTED  {restricted}
   BASIS         {basis}{fine}{trsh}{constraint}
$end
{scan}

""",

    'molpro': """***,name
memory,{memory},m;
geometry={{angstrom;
{xyz}}}

basis={basis}

int;

{{hf;{shift}
maxit,1000;
wf,spin={spin},charge={charge};}}

{restricted}{method};
{job_type_1}
{job_type_2}
---;

""",

    'mrci': """***,name
memory,{memory},m;
geometry={{angstrom;
{xyz}}}

gprint,orbitals;

basis={basis}

{{hf;shift,-1.0,-0.5;
maxit,1000;
wf,spin={spin},charge={charge};}}

{{multi;
{occ}noextra,failsafe,config,csf;
wf,spin={spin},charge={charge};
natorb,print,ci;}}

{{mrci;
{occ}wf,spin={spin},charge={charge};}}

E_mrci=energy;
E_mrci_Davidson=energd;

table,E_mrci,E_mrci_Davidson;

---;

""",

    'arkane_input_species': """#!/usr/bin/env python3
# encoding: utf-8

{bonds}externalSymmetry = {symmetry}

spinMultiplicity = {multiplicity}

opticalIsomers = {optical}

energy = {{'{sp_level}': Log('{sp_path}')}}

geometry = Log('{opt_path}')

frequencies = Log('{freq_path}')

{rotors}

""",
    'arkane_hindered_rotor':
        """HinderedRotor(scanLog=Log('{rotor_path}'), pivots={pivots}, top={top}, symmetry={symmetry}, fit='fourier')""",

    'arkane_free_rotor':
        """FreeRotor(pivots={pivots}, top={top}, symmetry={symmetry})""",

    'onedmin': """ 484040 10    ! Rand. no. seed, N_samples
 geo.xyz      ! Name of geometry file; units are Angstroms
 {bath}           ! Bath gas; allowed values are He, Ne, Ar, Kr, H2, N2, O2
 2 5          ! Rmin, Rmax; allowed center of mass range
""",

    'onedmin.molpro.x': """molpro -n 1 --nouse-logfile --no-xml-output -L /opt/molpro2012/molprop_2012_1_Linux_x86_64_i8/lib/ -d /scratch/$USER -o qc.out -s qc.in 
""",

    'onedmin.qc.mol': """***
memory,500,m

nosym
noorient
geometry
GEOMETRY
end

basis,default=avdz,h=vdz
{df-rhf}
{df-rmp2}

set,spin=SPIN

molpro_energy=energy
show[1,e25.15],molpro_energy

---
""",
}
