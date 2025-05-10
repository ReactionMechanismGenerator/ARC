# Minimal Makefile for Sphinx documentation
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
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

# Default target
all: help

# === Help ===
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help clean html latexpdf publish gh_setup

# === Setup for GitHub Pages ===
# Clone gh-pages branch for GitHub Pages publishing
# Ensures the build directory is clean before cloning
gh_setup:
	@echo "Setting up gh-pages branch for publishing..."
	@rm -rf $(BUILDDIR)/html
	@git clone --single-branch --branch gh-pages --origin official https://${GH_TOKEN}@github.com/ReactionMechanismGenerator/ARC.git $(BUILDDIR)/html
	@echo "gh-pages branch cloned into $(BUILDDIR)/html"

# === Build Targets ===
html:
	@echo "Building HTML documentation..."
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html -v
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

latexpdf:
	@echo "Building LaTeX and PDF documentation..."
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@make -C $(BUILDDIR)/latex all-pdf
	@echo
	@echo "Build finished. The PDF is in $(BUILDDIR)/latex."

# === Maintenance ===
clean:
	@echo "Cleaning build artifacts..."
	-rm -rf $(BUILDDIR)/*
	@echo "Build directory $(BUILDDIR) cleaned."

# === Publishing ===
# Publishes the built HTML documentation to the gh-pages branch
# Requires GH_TOKEN environment variable and optionally COMMITMESSAGE
publish: gh_setup html
	@echo "Publishing documentation to gh-pages..."
	touch $(BUILDDIR)/html/.nojekyll
	@cd $(BUILDDIR)/html; \
	git add -A .; \
	if git diff-index --quiet HEAD; then \
		echo "No changes to commit."; \
	else \
		echo "Committing changes..."; \
		if [ -z "${COMMITMESSAGE}" ]; then \
			git commit -m "Update documentation"; \
		else \
			git commit -m "${COMMITMESSAGE}"; \
		fi; \
		echo "Pushing to gh-pages..."; \
		git push official gh-pages; \
	fi
	@echo "Publishing process finished."
