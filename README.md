# Automated Rate Calculator | ARC

![Build Status](https://github.com/ReactionMechanismGenerator/ARC/actions/workflows/cont_int.yml/badge.svg)
[![codecov](https://codecov.io/gh/ReactionMechanismGenerator/ARC/branch/main/graph/badge.svg)](https://codecov.io/gh/ReactionMechanismGenerator/ARC)
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)
![Release](https://img.shields.io/badge/version-1.1.0-blue.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3356849.svg)](https://doi.org/10.5281/zenodo.3356849)

<img src="https://github.com/ReactionMechanismGenerator/ARC/blob/main/logo/ARC-logo-small.jpg" alt="ARC logo" width="200"/>

**ARC** (Automated Rate Calculator) automates electronic structure calculations and extracts high-quality thermodynamic and kinetic data for use in chemical kinetic modeling.

ARC has many [advanced features](https://reactionmechanismgenerator.github.io/ARC/advanced.html), but its core functionality is simple:

> It takes 2D graph-based molecular representations—such as  
> [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system), [InChI](https://www.inchi-trust.org/), or [RMG adjacency lists](https://reactionmechanismgenerator.github.io/RMG-Py/reference/molecule/adjlist.html)—and automatically executes, tracks, and processes the relevant electronic structure jobs on user-defined computing resources.

ARC's key outputs are:
- Thermodynamic properties (H, S, Cp)
- High-pressure limit rate coefficients

---

## 🚀 Mission

ARC's mission is to provide the chemical kinetics community with a well-documented, user-friendly, and extensible framework for automatically computing species thermochemistry and reaction kinetics.

---

## 📚 Documentation

See our [documentation site](https://reactionmechanismgenerator.github.io/ARC/index.html) for:

- Installation instructions
- Examples and tutorials
- API reference
- Advanced usage guides

---

## 📜 License

ARC is released under the [MIT License](https://github.com/ReactionMechanismGenerator/ARC/blob/main/LICENSE).

---

## 🤝 Contributing

We welcome contributions!

- To get started, visit the [Developer's Guide](https://github.com/ReactionMechanismGenerator/ARC/wiki)
- Found a bug or have a feature request? [Open an issue](https://github.com/ReactionMechanismGenerator/ARC/issues)

---

## ❓ Questions & Support

For help using ARC, please [open an issue](https://github.com/ReactionMechanismGenerator/ARC/issues) and we’ll do our best to assist.

---

## 📘 Citation

If you use ARC in your work, please cite:

**Text form:**

> A. Grinberg Dana, D. Ranasinghe, H. Wu, C. Grambow, X. Dong, M. Johnson, M. Goldman, M. Liu, W.H. Green, K. Kaplan, C. Pieters,
> *ARC - Automated Rate Calculator*, version 1.1.0, https://github.com/ReactionMechanismGenerator/ARC,  
> DOI: [10.5281/zenodo.3356849](https://doi.org/10.5281/zenodo.3356849)

**BibTeX form:**

```bibtex
@misc{ARC,
  author = {Grinberg Dana, A. and Ranasinghe, D. and Wu, H. and Grambow, C. and Dong, X. and Johnson, M. and Goldman, M. and Liu, M. and Green, W.H. and K. Kaplan and C. Pieters},
  title = {ARC - Automated Rate Calculator, version 1.1.0},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ReactionMechanismGenerator/ARC}},
  doi = {10.5281/zenodo.3356849}
}
