# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
PAPER         =
BUILDDIR      = build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
#ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(SPHINXOPTS) source

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: clean html publish travis_setup

travis_setup:
# This target is intended to be used when automatically compiling documentation with Travis
	@rm -rf $(BUILDDIR)/html
	@git clone --single-branch --branch gh-pages --origin official https://${GH_TOKEN}@github.com/ReactionMechanismGenerator/ARC.git $(BUILDDIR)/html

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html -v
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

clean:
# set aside the git repository info used to push pages to github
	mkdir build_temp
	mv $(BUILDDIR)/html/.git build_temp/.git
	-rm -rf $(BUILDDIR)/*
	mkdir $(BUILDDIR)/html
	mv build_temp/.git $(BUILDDIR)/html/
	rm -rf build_temp

publish: $(BUILDDIR)/html/.git
# Commit changes to gh-pages and push to github, to publish the results!
	touch $(BUILDDIR)/html/.nojekyll
ifdef COMMITMESSAGE # Use the provided COMMITMESSAGE variable for noninteractive use.
	cd $(BUILDDIR)/html; git add -A .; git commit -m "$(COMMITMESSAGE)"; git push official gh-pages
else # Prompt for a commit message
	cd $(BUILDDIR)/html; git add -A .; git commit; git push official gh-pages
endif
