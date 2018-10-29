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
    'gaussian03': """%chk=check.chk
%mem={memory}mb
%nproc=8

# {job_type_1} {restricted}{method}{slash}{basis} {job_type_2}

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
   BASIS         {basis}{fine}
$end

""",
    'molpro_2012': """***,name
memory,{memory},m;
geometry={angstrom;
{xyz}
}

basis={basis}

int;

{hf;
maxit,1000;
wf,spin={spin},charge={charge};}

{restricted}{method};
{job_type_1}
{job_type_2}
---;

""",
    'molpro_2015': """***,name
memory,{memory},m;
geometry={angstrom;
{xyz}
}

basis={basis}

int;

{hf;
maxit,1000;
wf,spin={spin},charge={charge};}

{restricted}{method};
{job_type_1}
{job_type_2}
---;

""",
}

# TODO: fine opt in gaussian, shift in molpro