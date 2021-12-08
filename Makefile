################################################################################
#
#   Makefile for ARC
#
################################################################################

.PHONY : clean gcn gcn-cpu test test-unittests

test test-unittests:
	python3 -m unittest discover -p *Test.py -v

gcn:
	bash devtools/install_gcn.sh

gcn-cpu:
	bash devtools/install_gcn_cpu.sh

clean:
	find -type d -name __pycache__ -exec rm -rf {} +
	rm -rf testing
	rm -rf arc/testing/gcn_tst
	rm -f .coverage
