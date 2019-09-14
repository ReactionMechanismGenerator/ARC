# Automated Rate Calculator | ARC

[![Build Status](https://travis-ci.org/ReactionMechanismGenerator/ARC.svg?branch=master)](https://travis-ci.org/ReactionMechanismGenerator/ARC)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/cf06bcc72d024b79834c300f39219471)](https://www.codacy.com/app/ReactionMechanismGenerator/ARC?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ReactionMechanismGenerator/ARC&amp;utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/ReactionMechanismGenerator/ARC/branch/master/graph/badge.svg)](https://codecov.io/gh/ReactionMechanismGenerator/ARC)
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)
![Release](https://img.shields.io/badge/version-1.1.0-blue.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3356849.svg)](https://doi.org/10.5281/zenodo.3356849)

<img src="https://github.com/ReactionMechanismGenerator/ARC/blob/master/logo/ARC-logo-small.jpg" alt="arc logo"/>

**ARC - Automated Rate Calculator** is a software for automating
electronic structure calculations relevant for chemical kinetic modeling.

ARC's mission is to provide the kinetics community with a well-documented and extensible code base for automatically calculating species thermochemistry and reaction rate coefficients.

ARC has many <a href="https://reactionmechanismgenerator.github.io/ARC/advanced.html">advanced features</a>,
yet at its core it is simple: It accepts 2D graph representations of chemical species (i.e.,
<a href="https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system">SMILES</a>,
<a href="https://www.inchi-trust.org/">InChI</a>,
or <a href="https://rmg.mit.edu/">RMG</a>'s
<a href="https://reactionmechanismgenerator.github.io/RMG-Py/reference/molecule/adjlist.html">adjacency list</a>),
and  automatically executes, tracks, and processes relevant electronic structure
jobs on user-defined servers. The principal outputs of ARC are thermodynamic properties
(H, S, Cp) and high-pressure limit kinetic rate coefficients for the defined species
and reactions.

Make sure to visit <a href="https://reactionmechanismgenerator.github.io/ARC/index.html">ARC's Documentation</a> page.

Developers and contributors: Visit
<a href="https://github.com/ReactionMechanismGenerator/ARC/wiki">ARC's Developer's Guide</a>
on the wiki page.
