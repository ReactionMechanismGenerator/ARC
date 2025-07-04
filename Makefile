################################################################################
#
#   Makefile for ARC
#
################################################################################

DEVTOOLS_DIR := devtools

.PHONY: all help clean test test-unittests test-functional test-all \
        install-all install-ci install-pyrdl install-rmgdb install-autotst install-gcn \
        install-gcn-cpu install-kinbot install-sella install-xtb install-torchani install-ob \
        lite check-env compile


# Default target
all: help

help:
	@echo "Available targets:"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run unit tests"
	@echo "  test-unittests   Run unit tests with coverage"
	@echo "  test-functional  Run functional tests"
	@echo "  test-all         Run all tests with coverage"
	@echo ""
	@echo "Installation:"
	@echo "  install          Install all external dependencies"
	@echo "  install-ci       Install all external dependencies for CI (no clean)"
	@echo "  install-pyrdl    Install PyRDL"
	@echo "  install-rmgdb    Install RMG-database"
	@echo "  install-autotst  Install AutoTST"
	@echo "  install-gcn      Install TS-GCN (GPU)"
	@echo "  install-gcn-cpu  Install TS-GCN (CPU)"
	@echo "  install-kinbot   Install KinBot"
	@echo "  install-sella    Install Sella"
	@echo "  install-xtb      Install xTB"
	@echo "  install-torchani Install TorchANI"
	@echo "  install-ob       Install OpenBabel"
	@echo ""
	@echo "Maintenance:"
	@echo "  lite             Run lite installation (no tests)"
	@echo "  clean            Clean build artifacts"
	@echo "  delete           Delete cloned repositories (inc. RMG)"
	@echo "  compile          Compile ARC's Cython module (arc.molecule)"
	@echo ""
	@echo "Diagnostics:"
	@echo "  check-env        Show Python environment info"

test: test-unittests

test-unittests:
	pytest arc/ --cov --cov-report=xml -ra -vv

test-functional:
	pytest functional/ -ra -vv

test-all:
	pytest arc/ functional/ --cov --cov-report=xml -ra -vv

install-all: install

install:
	@echo "Installing all external ARC dependencies..."
	bash $(DEVTOOLS_DIR)/install_all.sh --rmg-rms

install-ci:
	@echo "Installing all external ARC dependencies for CI (no clean)..."
	bash $(DEVTOOLS_DIR)/install_all.sh --no-clean

install-lite:
	@echo "Installing ARC's lite version (no external dependencies)..."
	bash $(DEVTOOLS_DIR)/install_all.sh --no-ext --rmg-rms

install-pyrdl:
	bash $(DEVTOOLS_DIR)/install_pyrdl.sh

install-rmgdb:
	bash $(DEVTOOLS_DIR)/install_rmgdb.sh

install-autotst:
	bash $(DEVTOOLS_DIR)/install_autotst.sh

install-gcn:
	bash $(DEVTOOLS_DIR)/install_gcn.sh

install-gcn-cpu:
	bash $(DEVTOOLS_DIR)/install_gcn_cpu.sh

install-kinbot:
	bash $(DEVTOOLS_DIR)/install_kinbot.sh

install-sella:
	bash $(DEVTOOLS_DIR)/install_sella.sh

install-xtb:
	bash $(DEVTOOLS_DIR)/install_xtb.sh

install-torchani:
	bash $(DEVTOOLS_DIR)/install_torchani.sh

install-ob:
	bash $(DEVTOOLS_DIR)/install_ob.sh

lite:
	bash $(DEVTOOLS_DIR)/lite.sh

clean:
	bash $(DEVTOOLS_DIR)/clean.sh
	@ python utilities.py clean

delete:
	bash $(DEVTOOLS_DIR)/delete.sh

check-env:
	@echo "Python binary:"; which python
	@echo "Python version:"; python -V
	@echo "PYTHONPATH:"; echo $$PYTHONPATH


compile:
	@echo "Compiling ARC's Cython module (arc.molecule)…"
	@ python utilities.py check-python
	python setup.py build_ext main --inplace --build-temp .
	@ python utilities.py check-dependencies
