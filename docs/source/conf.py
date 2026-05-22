# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add src/ to path so Sphinx can find the package
sys.path.insert(0, os.path.abspath("../../src"))

# ── Project information ───────────────────────────────────────────────────────
project = "ETF Predictor"
copyright = "2024, ETF Predictor Team"
author = "ETF Predictor Team"
release = "0.1.0"

# ── General configuration ────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",       # pull docstrings from code
    "sphinx.ext.napoleon",      # Google / NumPy style docstrings
    "sphinx.ext.viewcode",      # add [source] links
    "sphinx.ext.autosummary",   # auto-generate summary tables
    "myst_parser",              # Markdown support
]

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ── HTML output ──────────────────────────────────────────────────────────────
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "ETF Predictor Documentation"
