#!/usr/bin/env python
# encoding: utf-8

conformer(
    label = 'R1',
    E0 = (22.2577, 'kJ/mol'),
    modes = [
        IdealGasTranslation(mass=(17.0027, 'amu')),
        LinearRotor(inertia=(0.904473, 'amu*angstrom^2'), symmetry=1),
        HarmonicOscillator(frequencies=([3675.45], 'cm^-1')),
    ],
    spin_multiplicity = 2,
    optical_isomers = 1,
)

conformer(
    label = 'R2',
    E0 = (22.2577, 'kJ/mol'),
    modes = [
        IdealGasTranslation(mass=(17.0027, 'amu')),
        LinearRotor(inertia=(0.904473, 'amu*angstrom^2'), symmetry=1),
        HarmonicOscillator(frequencies=([3675.45], 'cm^-1')),
    ],
    spin_multiplicity = 2,
    optical_isomers = 1,
)

conformer(
    label = 'P1',
    E0 = (-250.574, 'kJ/mol'),
    modes = [
        IdealGasTranslation(mass=(18.0106, 'amu')),
        NonlinearRotor(inertia=([0.611436, 1.17966, 1.7911], 'amu*angstrom^2'), symmetry=2),
        HarmonicOscillator(frequencies=([1615.43, 3782.01, 3887.1], 'cm^-1')),
    ],
    spin_multiplicity = 1,
    optical_isomers = 1,
)

thermo(
    label = 'R1',
    thermo = NASA(
        polynomials = [
            NASAPolynomial(coeffs=[3.49683, 0.000188285, -1.03135e-06, 1.63951e-09, -6.45157e-13, 2675.74, 1.48391],
                           Tmin=(10, 'K'), Tmax=(974.045, 'K')),
            NASAPolynomial(coeffs=[3.44056, -0.000267412, 7.28022e-07, -2.88523e-10, 3.54839e-14, 2719.28, 1.92114],
                           Tmin=(974.045, 'K'), Tmax=(3000, 'K')),
        ],
        Tmin = (10, 'K'), Tmax = (3000, 'K'),
        E0 = (22.2464, 'kJ/mol'), Cp0 = (29.1007, 'J/(mol*K)'), CpInf = (37.4151, 'J/(mol*K)'),
    ),
)

thermo(
    label = 'R2',
    thermo = NASA(
        polynomials = [
            NASAPolynomial(coeffs=[3.49683, 0.000188285, -1.03135e-06, 1.63951e-09, -6.45157e-13, 2675.74, 1.48391],
                           Tmin=(10, 'K'), Tmax=(974.045, 'K')),
            NASAPolynomial(coeffs=[3.44056, -0.000267412, 7.28022e-07, -2.88523e-10, 3.54839e-14, 2719.28, 1.92114],
                           Tmin=(974.045, 'K'), Tmax=(3000, 'K')),
        ],
        Tmin = (10, 'K'), Tmax = (3000, 'K'),
        E0 = (22.2464, 'kJ/mol'), Cp0 = (29.1007, 'J/(mol*K)'), CpInf = (37.4151, 'J/(mol*K)'),
    ),
)

thermo(
    label = 'P1',
    thermo = NASA(
        polynomials = [
            NASAPolynomial(coeffs=[4.00485, -0.000245998, 8.95339e-07, 1.40307e-09, -1.18107e-12, -30136.7, -0.104547],
                           Tmin=(10, 'K'), Tmax=(772.675, 'K')),
            NASAPolynomial(coeffs=[3.50315, 0.00113242, 5.85407e-07, -3.70913e-10, 5.33992e-14, -30022.8, 2.42188],
                           Tmin=(772.675, 'K'), Tmax=(3000, 'K')),
        ],
        Tmin = (10, 'K'), Tmax = (3000, 'K'),
        E0 = (-250.569, 'kJ/mol'), Cp0 = (33.2579, 'J/(mol*K)'), CpInf = (58.2013, 'J/(mol*K)'),
    ),
)
