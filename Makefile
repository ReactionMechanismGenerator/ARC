################################################################################
#
#   Makefile for ARC
#
################################################################################

test test-unittests:
	nosetests --nocapture --nologcapture --all-modules --verbose --with-coverage --cover-inclusive --cover-package=arc --cover-erase --cover-html --exe --cover-html-dir=testing/coverage

gcn:
	bash devtools/install_gcn.sh

gcn-ci:
	bash devtools/install_gcn_ci.sh
