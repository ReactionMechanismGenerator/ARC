################################################################################
#
#   Makefile for ARC
#
################################################################################

# Define variables for common paths
DEVTOOLS_DIR := devtools

.PHONY: all help clean test test-unittests test-functional test-all \
        install-all install-molecule install-rmgdb install-autotst install-gcn \
        install-gcn-cpu install-kinbot install-sella install-xtb install-torchani install-ob \
        lite check-env

# Default target
all: help

# === Help ===
help:
	@echo "Available targets:"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run unit tests (alias for test-unittests)"
	@echo "  test-unittests   Run unit tests with coverage"
	@echo "  test-functional  Run functional tests"
	@echo "  test-all         Run unit and functional tests with coverage"
	@echo ""
	@echo "Installation:"
	@echo "  install-all      Install all external dependencies using devtools scripts"
	@echo "  install-molecule Install molecule dependency"
	@echo "  install-rmgdb    Install RMG-database"
	@echo "  install-autotst  Install AutoTST"
	@echo "  install-gcn      Install TS-GCN (GPU version)"
	@echo "  install-gcn-cpu  Install TS-GCN (CPU version)"
	@echo "  install-kinbot   Install KinBot"
	@echo "  install-sella    Install Sella"
	@echo "  install-xtb      Install xTB"
	@echo "  install-torchani Install TorchANI"
	@echo "  install-ob       Install OpenBabel environment"
	@echo ""
	@echo "Maintenance:"
	@echo "  lite             Run the lite installation script (removes tests)"
	@echo "  clean            Clean build artifacts (__pycache__, testing dirs, .coverage)"
	@echo ""
	@echo "Diagnostics:"
	@echo "  check-env        Check Python binary, version, and PYTHONPATH"


# === Testing ===

test test-unittests:
	pytest arc/ --cov -ra -vv

test-functional:
	pytest functional/ -ra -vv

test-all:
	pytest arc/ functional/ --cov -ra -vv

# === Installers ===

install-all:
	@echo "Installing all external dependencies..."
	bash $(DEVTOOLS_DIR)/install_all.sh

install-molecule:
	bash $(DEVTOOLS_DIR)/install_molecule.sh

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

# === Maintenance ===

lite:
	bash $(DEVTOOLS_DIR)/lite.sh

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf functional ipython
	rm -rf arc/testing/gcn_tst
	rm -f .coverage

# === Diagnostics ===

check-env:
	@echo "Python binary:"
	which python
	@echo "Python version:"
	python -V
	@echo "PYTHONPATH:"
	echo $$PYTHONPATH
