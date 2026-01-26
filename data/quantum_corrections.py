#!/usr/bin/env python3

###############################################################################
#                                                                             #
# RMG - Reaction Mechanism Generator                                          #
#                                                                             #
# Copyright (c) 2002-2020 Prof. William H. Green (whgreen@mit.edu),           #
# Prof. Richard H. West (r.west@neu.edu) and the RMG Team (rmg_dev@mit.edu)   #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the 'Software'),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
###############################################################################

"""
This file provides atomic energy and BAC parameters for several model
chemistries.
"""

# Atom energy corrections to reach gas-phase reference state
# Experimental enthalpy of formation at 0 K, 1 bar for gas phase
# See Gaussian thermo whitepaper at http://gaussian.com/thermo/
# Note: These values are relatively old and some improvement may be possible by using newer values
# (particularly for carbon).
# However, care should be taken to ensure that they are compatible with the BAC values (if BACs are used)
# He, Ne, K, Ca, Ti, Cu, Zn, Ge, Br, Kr, Rb, Ag, Cd, Sn, I, Xe, Cs, Hg, and Pb are taken from CODATA
# Codata: Cox, J. D., Wagman, D. D., and Medvedev, V. A., CODATA Key Values for Thermodynamics, Hemisphere
# Publishing Corp., New York, 1989. (http://www.science.uwaterloo.ca/~cchieh/cact/tools/thermodata.html)

atom_hf = {'H': 51.63, 'He': -1.481,
           'Li': 37.69, 'Be': 76.48, 'B': 136.2, 'C': 169.98, 'N': 112.53, 'O': 58.99, 'F': 18.47, 'Ne': -1.481,
           'Na': 25.69, 'Mg': 34.87, 'Al': 78.23, 'Si': 106.6, 'P': 75.42, 'S': 65.66, 'Cl': 28.59,
           'K': 36.841, 'Ca': 41.014, 'Ti': 111.2, 'Cu': 79.16, 'Zn': 29.685, 'Ge': 87.1, 'Br': 25.26,
           'Kr': -1.481,
           'Rb': 17.86, 'Ag': 66.61, 'Cd': 25.240, 'Sn': 70.50, 'I': 24.04, 'Xe': -1.481,
           'Cs': 16.80, 'Hg': 13.19, 'Pb': 15.17}

# Thermal contribution to enthalpy for the atoms reported by Gaussian thermo whitepaper
# This will be subtracted from the corresponding value in atom_hf to produce an enthalpy used in calculating
# the enthalpy of formation at 298 K
atom_thermal = {'H': 1.01, 'He': 1.481,
                'Li': 1.1, 'Be': 0.46, 'B': 0.29, 'C': 0.25, 'N': 1.04, 'O': 1.04, 'F': 1.05, 'Ne': 1.481,
                'Na': 1.54, 'Mg': 1.19, 'Al': 1.08, 'Si': 0.76, 'P': 1.28, 'S': 1.05, 'Cl': 1.1,
                'K': 1.481, 'Ca': 1.481, 'Ti': 1.802, 'Cu': 1.481, 'Zn': 1.481, 'Ge': 1.768, 'Br': 1.481,
                'Kr': 1.481,
                'Rb': 1.481, 'Ag': 1.481, 'Cd': 1.481, 'Sn': 1.485, 'I': 1.481, 'Xe': 1.481,
                'Cs': 1.481, 'Hg': 1.481, 'Pb': 1.481}

# Spin orbit correction (SOC) in Hartrees
# Values taken from ref 22 of http://dx.doi.org/10.1063/1.477794 and converted to Hartrees
# Values in milli-Hartree are also available (with fewer significant figures) from table VII of
# http://dx.doi.org/10.1063/1.473182
# Iodine SOC calculated as a weighted average of the electronic spin splittings of the lowest energy state.
# The splittings are obtained from Huber, K.P.; Herzberg, G., Molecular Spectra and Molecular Structure. IV.
# Constants of Diatomic Molecules, Van Nostrand Reinhold Co., 1979
# Spin orbit correction for F, Si, Cl, Br and B taken form https://cccbdb.nist.gov/elecspin.asp
SOC = {'H': 0.0, 'N': 0.0, 'O': -0.000355, 'C': -0.000135, 'S': -0.000893, 'P': 0.0, 'I': -0.011547226,
       'F': -0.000614, 'Si': -0.000682, 'Cl': -0.001338, 'Br': -0.005597, 'B': -0.000046}



# Atom Energy Corrections
atom_energies = {    
    #  LevelOfTheory(method='wb97xd3',basis='def2tzvp',software='orca')
    # H :     -0.49996129 +/- 0.00164757 Hartree
    # C :    -37.85029767 +/- 0.00702650 Hartree
    # N :    -54.58838798 +/- 0.00350187 Hartree
    # O :    -75.07558852 +/- 0.00342348 Hartree
    # F :    -99.74175924 +/- 0.00337557 Hartree
    # S :   -398.11092202 +/- 0.00342348 Hartree
    # Cl:   -460.14741391 +/- 0.00316160 Hartree
    # Br:  -2574.17542437 +/- 0.00337557 Hartree
    "LevelOfTheory(method='wb97xd3',basis='def2tzvp',software='orca')": {
        'H': -0.4999612874685611,
        'C': -37.850297667183575,
        'N': -54.58838797943286,
        'O': -75.07558851871958,
        'F': -99.74175924455834,
        'S': -398.11092201981205,
        'Cl': -460.1474139128046,
        'Br': -2574.175424374858
    },
}


# Bond Additivity Corrections
## Petersson-type bond additivity correction parameters
pbac = {
    "LevelOfTheory(method='wb97xd3',basis='def2tzvp',software='orca')": {
            'Br-Br': 2.86765593512125,
            'Br-C': 0.6875215310726311,
            'Br-Cl': 1.5735006932758795,
            'Br-F': 2.794587111390884,
            'Br-H': 2.5884934911270907,
            'Br-O': -2.570040504531607,
            'C#C': -7.793276633596348,
            'C#N': -3.8915409837393606,
            'C#O': -7.431354894566592,
            'C-C': -1.0579713799766048,
            'C-Cl': -0.583875377487877,
            'C-F': 1.3536716648192737,
            'C-H': 0.06982607012553099,
            'C-N': 0.6449560347033947,
            'C-O': -1.2412459131853322,
            'C-S': -1.224162915393716,
            'C=C': -3.7613606024094133,
            'C=N': -1.4636585249683565,
            'C=O': -2.876957692547981,
            'C=S': -3.8185021558926446,
            'Cl-Cl': 0.16580928640831294,
            'Cl-F': 1.047363896901384,
            'Cl-H': 0.33212882996891224,
            'Cl-N': 1.1918237277255987,
            'Cl-O': -0.9561012156646922,
            'Cl-S': -0.3996854084841952,
            'F-F': 1.227417816423643,
            'F-H': -1.5184418463706422,
            'F-O': -0.03834146686445842,
            'F-S': 0.28869609314439043,
            'H-H': 0.175565120519247,
            'H-N': 0.7598392496538244,
            'H-O': -1.2939548326519699,
            'H-S': 1.1341838733630867,
            'N#N': -0.2990722245949918,
            'N-N': 3.4638560587476315,
            'N-O': 1.473886242589889,
            'N=N': 2.7641370858894487,
            'N=O': -2.3918216033247446,
            'O-O': -2.1230774975132376,
            'O-S': -2.601781406629789,
            'O=O': -9.973534873423771,
            'O=S': -3.4928119172623657,
            'S-S': -1.3453438000709452,
            'S=S': -6.890277779076065
        },
}

## Melius-type bond additivity correction parameters
mbac = {
    "LevelOfTheory(method='wb97xd3',basis='def2tzvp',software='orca')": {
        'atom_corr': {
            'Br': -4.1790572284678955,
            'C': 0.054528801547492525,
            'Cl': -2.83920514572373,
            'F': -4.37815812319745,
            'H': -2.0938799081971964,
            'N': -4.999999999999999,
            'O': -2.577254187398695,
            'S': -2.388550629092355
        },
        'bond_corr_length': {
            'Br': 2731.4351837662016,
            'C': 0.1020831006821913,
            'Cl': 377.4987634773345,
            'F': 141.9236854034023,
            'H': 1.2143124729013384,
            'N': 1.1707977527351297e-22,
            'O': 115.28708205711784,
            'S': 315.08628188812366
        },
        'bond_corr_neighbor': {
            'Br': 0.26322638012965055,
            'C': -0.0938789684544172,
            'Cl': 0.28542073272985036,
            'F': 0.13791226932338668,
            'H': -0.1680948853375274,
            'N': -0.38172346940254315,
            'O': -0.07045018330880996,
            'S': -0.14703665911218491
        },
        'mol_corr': -3.8386822655449677
    },
}


