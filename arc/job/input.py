#!/usr/bin/env python
# encoding: utf-8


##################################################################


"""
parameters for input files:

memory (in mb, gaussian and molpro only)
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
    restricted: '' or 'u' or restricted / unrestricted

qchem:
    job_type_1: 'opt', 'ts', 'sp'
    job_type_2: 'freq'.
    fine: '\n   GEOM_OPT_TOL_GRADIENT 15\n   GEOM_OPT_TOL_DISPLACEMENT 60\n   GEOM_OPT_TOL_ENERGY 5'
    restricted: 'false' or 'true' or restricted / unrestricted
"""

input_files = {
    'gaussian': """%chk=check.chk
%mem={memory}mb
%nproc=8

# {job_type_1} {restricted}{method}{slash}{basis} {job_type_2} {fine} {trsh}

name

{charge} {multiplicity}
{xyz}
{scan}


""",

    'qchem': """$molecule
{charge} {multiplicity}
{xyz}
$end

$rem
   JOBTYPE       {job_type_1}
   METHOD        {method}
   UNRESTRICTED  {restricted}
   BASIS         {basis}{fine}{trsh}
$end

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

    'arkane_species': """#!/usr/bin/env python
# -*- coding: utf-8 -*-

linear = {linear}

externalSymmetry = {symmetry}

spinMultiplicity = {multiplicity}

opticalIsomers = {optical}

energy = {{'{model_chemistry}': Log('{sp_path}')}}

geometry = Log('{opt_path}')

frequencies = Log('{freq_path}')

{rotors}

""",

    'arkane_rotor': """HinderedRotor(scanLog=Log('{rotor_path}'), pivots={pivots}, top={top}, symmetry={symmetry}, fit='best')"""
}
