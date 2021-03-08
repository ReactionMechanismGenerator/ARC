# Automated Rate Calculator | ARC

![Build Status](https://github.com/ReactionMechanismGenerator/ARC/actions/workflows/cont_int.yml/badge.svg)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/cf06bcc72d024b79834c300f39219471)](https://www.codacy.com/app/ReactionMechanismGenerator/ARC?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ReactionMechanismGenerator/ARC&amp;utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/ReactionMechanismGenerator/ARC/branch/master/graph/badge.svg)](https://codecov.io/gh/ReactionMechanismGenerator/ARC)
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)
![Release](https://img.shields.io/badge/version-1.1.0-blue.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3356849.svg)](https://doi.org/10.5281/zenodo.3356849)

<img src="https://github.com/ReactionMechanismGenerator/ARC/blob/master/logo/ARC-logo-small.jpg" alt="arc logo"/>

The **A**utomated **R**ate **C**alculator (**ARC**) software is a tool for automating
electronic structure calculations and attaining thermo-kinetic data
relevant for chemical kinetic modeling.

ARC has many <a href="https://reactionmechanismgenerator.github.io/ARC/advanced.html">advanced features</a>,
yet at its core it is simple: It accepts 2D graph representations of chemical species (i.e.,
<a href="https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system">SMILES</a>,
<a href="https://www.inchi-trust.org/">InChI</a>,
or <a href="https://rmg.mit.edu/">RMG</a>'s
<a href="https://reactionmechanismgenerator.github.io/RMG-Py/reference/molecule/adjlist.html">adjacency list</a>),
and  automatically executes, tracks, and processes relevant electronic structure calculation
jobs on user-defined server(s). The principal outputs of ARC are thermodynamic properties
(H, S, Cp) and high-pressure limit kinetic rate coefficients of species and reactions of interest.

## Mission

ARC's mission is to provide the kinetics community with a well-documented and extensible codebase for automatically calculating species thermochemistry and reaction rate coefficients.

## Documentation

Visit out <a href="https://reactionmechanismgenerator.github.io/ARC/index.html">documentation</a> pages for installation instructions, examples, API, advanced features and more.

## Licence

This project is licensed under the MIT License - see the <a href="https://github.com/ReactionMechanismGenerator/ARC/blob/master/LICENSE">LICENSE</a> file for details.

## Contributing

Developers and contributors: Visit
<a href="https://github.com/ReactionMechanismGenerator/ARC/wiki">ARC's Developer's Guide</a>
on the wiki page.

If you have a suggestion or find a bug, please post to our <a href="https://github.com/ReactionMechanismGenerator/ARC/issues">Issues</a> page.

## Questions

If you are having issues, please post to our <a href="https://github.com/ReactionMechanismGenerator/ARC/issues">Issues</a> page. We will do our best to assist.
