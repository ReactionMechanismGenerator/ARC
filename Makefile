################################################################################
#
#   Makefile for ARC
#
################################################################################

.PHONY : clean gcn gcn-cpu test test-unittests

test test-unittests:
	python3 -m unittest discover -p *Test.py -v

install-all:
	bash devtools/install_all.sh

install-autotst:
	bash devtools/install_autotst.sh

install-gcn:
	bash devtools/install_gcn.sh

install-gcn-cpu:
	bash devtools/install_gcn_cpu.sh

install-kinbot:
	bash devtools/install_kinbot.sh

install-sella:
	bash devtools/install_sella.sh

# xTB is included in the arc_env, install_xtb.sh is not executed by ``make install-all``
install-xtb:
	bash devtools/install_xtb.sh

# TorchANI is included in the arc_env, install_torchani.sh is not executed by ``make install-all``
install-torchani:
	bash devtools/install_torchani.sh

clean:
	find -type d -name __pycache__ -exec rm -rf {} +
	rm -rf testing
	rm -rf arc/testing/gcn_tst
	rm -f .coverage
