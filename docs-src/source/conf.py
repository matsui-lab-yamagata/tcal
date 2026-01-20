import os
import subprocess
import sys


sys.path.insert(0, os.path.abspath("../../src"))


project = "tcal"
copyright = "2023, Hiroyuki Matsui, Koki Ozawa"
author = "Hiroyuki Matsui, Koki Ozawa"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.autodoc_pydantic"
]

templates_path = ["_templates"]
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
