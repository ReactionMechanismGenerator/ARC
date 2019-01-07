# ARC - Automatic Rate Calculator

[![Build Status](https://travis-ci.org/ReactionMechanismGenerator/ARC.svg?branch=master)](https://travis-ci.org/ReactionMechanismGenerator/ARC)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/cf06bcc72d024b79834c300f39219471)](https://www.codacy.com/app/ReactionMechanismGenerator/ARC?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ReactionMechanismGenerator/ARC&amp;utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/ReactionMechanismGenerator/ARC/branch/master/graph/badge.svg)](https://codecov.io/gh/ReactionMechanismGenerator/ARC)

<img src="https://github.com/ReactionMechanismGenerator/ARC/blob/master/logo/ARC-logo-small.jpg" alt="arc logo"/>

This program automates quantum chemical calculations. Currently Gaussian, Molpro, and QChem are supported.
The current version should be run locally; it communicates with servers (defined in settings.py) using an RSA key to
spawn the calculations.

Currently ARC can calculate Thermodynamic properties relying on Arkane (see
<a href="http://reactionmechanismgenerator.github.io/RMG-Py/users/arkane/index.html">Arkane's user guide</a>).
We plan to elaborate ARC to automatically identify and compute transition states and eventually calculate kinetic rates
and pressure-dependent networks using Arkane.

Installation instructions are in the Wiki pages of this project.
