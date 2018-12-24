################################################################################
#
#   Makefile for ARC
#
################################################################################

test test-unittests:
	nosetests --nocapture --nologcapture --all-modules --verbose --with-coverage --cover-inclusive --cover-package=arc --cover-erase --cover-html --cover-html-dir=testing/coverage

