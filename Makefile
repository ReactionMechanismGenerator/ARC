################################################################################
#
#   Makefile for ARC
#
################################################################################

test test-unittests:
	nosetests --nocapture --nologcapture --all-modules --verbose --with-coverage --cover-inclusive --cover-package=arc --cover-erase --cover-html --exe --cover-html-dir=testing/coverage

gcn:
	bash devtools/install_gcn.sh

gcn-travis:
	bash devtools/install_gcn_travis.sh
