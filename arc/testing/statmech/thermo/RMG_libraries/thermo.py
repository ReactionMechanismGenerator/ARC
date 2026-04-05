#!/usr/bin/env python
# encoding: utf-8

name = "thermo"
shortDesc = ""
longDesc = """
Calculated using Arkane v3.3.0 using LevelOfTheory(method='wb97xd2023',basis='def2tzvp',software='gaussian').
"""
entry(
    index = 0,
    label = "CHO",
    molecule = 
"""
multiplicity 2
1 O u0 p2 c0 {2,D}
2 C u1 p0 c0 {1,D} {3,S}
3 H u0 p0 c0 {2,S}
""",
    thermo = NASA(
        polynomials = [
            NASAPolynomial(coeffs=[4.0141,-0.000832549,4.56556e-06,1.87695e-10,-2.6565e-12,3769.95,4.13323], Tmin=(10,'K'), Tmax=(629.059,'K')),
            NASAPolynomial(coeffs=[2.71782,0.00469642,-2.14748e-06,4.44347e-10,-3.31095e-14,3986.73,10.2129], Tmin=(629.059,'K'), Tmax=(3000,'K')),
        ],
        Tmin = (10,'K'),
        Tmax = (3000,'K'),
        E0 = (31.3505,'kJ/mol'),
        Cp0 = (33.2579,'J/(mol*K)'),
        CpInf = (58.2013,'J/(mol*K)'),
    ),
    shortDesc = """""",
    longDesc = 
"""
Spin multiplicity: 2
External symmetry: -1.0
Optical isomers: 1

Geometry:
C       0.06174500    0.58096700    0.00000000
O       0.06174500   -0.58771100    0.00000000
H      -0.86443200    1.21588600    0.00000000
""",
)

entry(
    index = 1,
    label = "CH4",
    molecule = 
"""
1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {1,S}
""",
    thermo = NASA(
        polynomials = [
            NASAPolynomial(coeffs=[4.04142,-0.00247777,1.12901e-05,3.70151e-09,-9.36692e-12,-10678.4,-0.426368], Tmin=(10,'K'), Tmax=(616.759,'K')),
            NASAPolynomial(coeffs=[0.680016,0.0116841,-4.57493e-06,7.69448e-10,-3.87648e-14,-10118.5,15.3438], Tmin=(616.759,'K'), Tmax=(3000,'K')),
        ],
        Tmin = (10,'K'),
        Tmax = (3000,'K'),
        E0 = (-88.7702,'kJ/mol'),
        Cp0 = (33.2579,'J/(mol*K)'),
        CpInf = (108.088,'J/(mol*K)'),
    ),
    shortDesc = """""",
    longDesc = 
"""
Spin multiplicity: 1
External symmetry: -1.0
Optical isomers: 1

Geometry:
C       0.00000000    0.00000000    0.00000000
H       0.62869500    0.62869500    0.62869500
H      -0.62869500   -0.62869500    0.62869500
H      -0.62869500    0.62869500   -0.62869500
H       0.62869500   -0.62869500   -0.62869500
""",
)

entry(
    index = 2,
    label = "CH2O",
    molecule = 
"""
1 O u0 p2 c0 {2,D}
2 C u0 p0 c0 {1,D} {3,S} {4,S}
3 H u0 p0 c0 {2,S}
4 H u0 p0 c0 {2,S}
""",
    thermo = NASA(
        polynomials = [
            NASAPolynomial(coeffs=[4.03209,-0.00195503,9.15137e-06,2.77082e-09,-7.98337e-12,-14295.6,3.46613], Tmin=(10,'K'), Tmax=(596.015,'K')),
            NASAPolynomial(coeffs=[1.42019,0.00951794,-4.4814e-06,9.71485e-10,-7.77435e-14,-13876.7,15.6178], Tmin=(596.015,'K'), Tmax=(3000,'K')),
        ],
        Tmin = (10,'K'),
        Tmax = (3000,'K'),
        E0 = (-118.849,'kJ/mol'),
        Cp0 = (33.2579,'J/(mol*K)'),
        CpInf = (83.1447,'J/(mol*K)'),
    ),
    shortDesc = """""",
    longDesc = 
"""
Spin multiplicity: 1
External symmetry: -1.0
Optical isomers: 1

Geometry:
C       0.00000000    0.00000000   -0.52427300
O       0.00000000    0.00000000    0.67079300
H       0.00000000    0.93921500   -1.11035300
H       0.00000000   -0.93921500   -1.11035300
""",
)

entry(
    index = 3,
    label = "CH3",
    molecule = 
"""
multiplicity 2
1 C u1 p0 c0 {2,S} {3,S} {4,S}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
""",
    thermo = NASA(
        polynomials = [
            NASAPolynomial(coeffs=[3.99122,0.000427519,9.38101e-06,-1.09444e-08,4.26317e-12,16127.2,0.20528], Tmin=(10,'K'), Tmax=(646.795,'K')),
            NASAPolynomial(coeffs=[3.24887,0.00501884,-1.26764e-06,3.22015e-11,2.01568e-14,16223.3,3.4632], Tmin=(646.795,'K'), Tmax=(3000,'K')),
        ],
        Tmin = (10,'K'),
        Tmax = (3000,'K'),
        E0 = (134.086,'kJ/mol'),
        Cp0 = (33.2579,'J/(mol*K)'),
        CpInf = (83.1447,'J/(mol*K)'),
    ),
    shortDesc = """""",
    longDesc = 
"""
Spin multiplicity: 2
External symmetry: -1.0
Optical isomers: 1

Geometry:
C       0.00000000    0.00000000    0.00000000
H       0.00000000    1.07942400    0.00000000
H       0.93480900   -0.53971200    0.00000000
H      -0.93480900   -0.53971200    0.00000000
""",
)

