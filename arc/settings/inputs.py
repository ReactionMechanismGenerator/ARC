"""
Input files
"""

input_files = {'arkane_input_species': """#!/usr/bin/env python3
# encoding: utf-8

{bonds}externalSymmetry = {symmetry}

spinMultiplicity = {multiplicity}

opticalIsomers = {optical}

energy = Log('{sp_path}')

geometry = Log('{opt_path}')

frequencies = Log('{freq_path}')

{rotors}

""",

    'arkane_input_species_explicit_e': """#!/usr/bin/env python3
# encoding: utf-8

{bonds}externalSymmetry = {symmetry}

spinMultiplicity = {multiplicity}

opticalIsomers = {optical}

energy = {e_elect}

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
