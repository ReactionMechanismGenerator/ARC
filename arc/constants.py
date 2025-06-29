"""
Physical constants for chemistry and molecular modeling.

This module provides fundamental physical constants frequently used in chemistry and related scientific computations.
All constants are defined in SI units unless otherwise noted.

Attributes:
E_h (float): Hartree energy, 4.35974434e-18 J.
E_h_kJmol (float): Hartree energy in kJ/mol, 2625.5 kJ/mol.
F (float): Faraday constant, 96485.3365 C/mol.
G (float): Newtonian gravitational constant, 6.67384e-11 m^3/(kg^1·s^2).
Na (float): Avogadro constant, 6.02214179e23 mol^-1.
R (float): Gas law constant, 8.314472 J/(mol^1·K^1).
a0 (float): Bohr radius, 5.2917721092e-11 m.
c (float): Speed of light in vacuum, 299792458 m/s.
e (float): Elementary charge, 1.602176565e-19 C.
g (float): Standard acceleration due to gravity, 9.80665 m/s^2.
h (float): Planck constant, 6.62606896e-34 J·s.
hbar (float): Reduced Planck constant, 1.054571726e-34 J·s.
kB (float): Boltzmann constant, 1.3806504e-23 J/K.
m_e (float): Electron rest mass, 9.10938291e-31 kg.
m_n (float): Neutron rest mass, 1.674927351e-27 kg.
m_p (float): Proton rest mass, 1.672621777e-27 kg.
amu (float): Atomic mass unit, 1.660538921e-27 kg.
pi (float): Pi, 3.14159...
"""

import math

#: The Hartree energy :math:`E_\mathrm{h}` in :math:`\mathrm{J}`
E_h = 4.35974434e-18

#: The Avogadro constant :math:`N_\mathrm{A}` in :math:`\mathrm{mol^{-1}}`
Na = 6.02214179e23

#: The Hartree energy in kJ/mol
E_h_kJmol = E_h * Na / 1000  # 1 Hartree = 2625.5 kJ/mol

#: The gas law constant :math:`R` in :math:`\mathrm{J/mol \cdot K}`
R = 8.31446261815324

#: The Bohr radius :math:`a_0` in :math:`\mathrm{m}`
a0 = 5.2917721092e-11

#: The atomic mass unit in :math:`\mathrm{kg}`
amu = 1.660538921e-27

#: The speed of light in a vacuum :math:`c` in :math:`\mathrm{m/s}`
c = 299792458

#: The elementary charge :math:`e` in :math:`\mathrm{C}`
e = 1.602176565e-19

#: The Planck constant :math:`h` in :math:`\mathrm{J \cdot s}`
h = 6.62606896e-34

#: The reduced Planck constant :math:`\hbar` in :math:`\mathrm{J \cdot s}`
hbar = 1.054571726e-34

#: The Boltzmann constant :math:`k_\mathrm{B}` in :math:`\mathrm{J/K}`
kB = 1.3806504e-23

#: The mass of an electron :math:`m_\mathrm{e}` in :math:`\mathrm{kg}`
m_e = 9.10938291e-31

#: The mass of a neutron :math:`m_\mathrm{n}` in :math:`\mathrm{kg}`
m_n = 1.674927351e-27

#: The mass of a proton :math:`m_\mathrm{p}` in :math:`\mathrm{kg}`
m_p = 1.672621777e-27

#: :math:`\pi = 3.14159 \ldots`
pi = float(math.pi)

#: Faradays Constant F in C/mol
F = 96485.3321233100184

#: Vacuum permittivity
epsilon_0 = 8.8541878128

bohr_to_angstrom = 0.529177

# Cython does not automatically place module-level variables into the module
# symbol table when in compiled mode, so we must do this manually so that we
# can use the constants from both Cython and regular Python code
globals().update({
    'E_h': E_h,
    'Na': Na,
    'E_h_kJmol': E_h_kJmol,
    'R': R,
    'a0': a0,
    'amu': amu,
    'c': c,
    'e': e,
    'h': h,
    'hbar': hbar,
    'kB': kB,
    'm_e': m_e,
    'm_n': m_n,
    'm_p': m_p,
    'pi': pi,
    'F': F,
    'epsilon_0': epsilon_0,
    'bohr_to_angstrom': bohr_to_angstrom,
})
