# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

#.PHONY: help Makefile
.PHONY: help clean html doctest setup_github_pages publish travis_setup

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
#%: Makefile
#	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

setup_github_pages:
# Turn the build/html directory into a git repository linked to github ready to push to gh-pages branch!
# Should only need to be run once.
	rm -rf $(BUILDDIR)/html
	git clone --single-branch --branch gh-pages --origin official git@github.com:ReactionMechanismGenerator/ARC.git build/html

publish: $(BUILDDIR)/html/.git
# Commit changes to gh-pages and push to github, to publish the results!
	touch $(BUILDDIR)/html/.nojekyll
ifdef COMMITMESSAGE # Use the provided COMMITMESSAGE variable for noninteractive use.
	cd $(BUILDDIR)/html; git add -A .; git commit -m "$(COMMITMESSAGE)"; git push official gh-pages
else # Prompt for a commit message
	cd $(BUILDDIR)/html; git add -A .; git commit; git push official gh-pages
endif

travis_setup:
# This target is intended to be used when automatically compiling documentation with Travis
	@rm -rf $(BUILDDIR)/html
	@git clone --single-branch --branch gh-pages --origin official https://${GH_TOKEN}@github.com/ReactionMechanismGenerator/ARC.git $(BUILDDIR)/html

clean:
# set aside the git repository info used to push pages to github
	mkdir build_temp
	mv $(BUILDDIR)/html/.git build_temp/.git
	-rm -rf $(BUILDDIR)/*
	mkdir $(BUILDDIR)/html
	mv build_temp/.git $(BUILDDIR)/html/
	rm -rf build_temp

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."
