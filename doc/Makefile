# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = build/html
LINKSDIR       = source/link
LINKCHECKDIR  = build/linkcheck
SPHINXAUTOBUILD = sphinx-autobuild

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

ALLSPHINXLIVEOPTS   = $(ALLSPHINXOPTS) -q \
        -p 0 \
        -H 0.0.0.0 \
        -B \
        --delay 1 \
        --ignore "*.swp" \
        --ignore "*.pdf" \
        --ignore "*.log" \
        --ignore "*.out" \
        --ignore "*.toc" \
        --ignore "*.aux" \
        --ignore "*.idx" \
        --ignore "*.ind" \
        --ignore "*.ilg" \
        --ignore "*.tex" \
        --watch source 

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"
	
.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/*

.PHONY: link
link:
	@read -p "Enter a Unique Link Name: " link_name; \
	read -p "Enter the link text the user sees: " link_text; \
	read -p "Enter the URL: " link_url; \
	read -p "Enter the .py file name (use_lower_case_and_underscore of link name): " file_name; \
	echo "The link name is: " $$link_name; \
	echo "The link text is: " $$link_text; \
	echo "The URL is: " $$link_url; \
	echo "Creating the file: " $(LINKSDIR)/$$file_name".py"; \
	echo "Enter the link in content as :xref:\`"$$link_name"\`"; \
	echo "The user will see:" $$link_text; \
	echo "Make sure you build and test the link."; \
  echo "import link\n\nen_us_user_text = \"$$link_text\" \n\n\
links.xref_links.update({\"$$link_name\": (en_us_user_text, \"$$link_url\")})" \
  > $(LINKSDIR)/$$file_name".py" \

.PHONY: html
html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)."


.PHONY: livehtml
livehtml:
	$(SPHINXAUTOBUILD) -b html $(ALLSPHINXLIVEOPTS) $(BUILDDIR)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)."


.PHONY: fasthtml
fasthtml:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR) -j 16
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)."

.PHONY: checklinks
checklinks:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(LINKCHECKDIR)
	@echo
	@echo "Check finished. Report is in $(LINKCHECKDIR)."


.PHONY: latex
latex:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make' in that directory to run these through (pdf)latex" \
	      "(use \`make latexpdf' here to do that automatically)."

.PHONY: latexpdf
latexpdf:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."
LINKCHECKDIR  = build/linkcheck



# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
#SPHINXOPTS    ?=
#SPHINXBUILD   ?= sphinx-build
#SOURCEDIR     = 
#BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
#help:
#	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

#.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
#%: Makefile
#	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
