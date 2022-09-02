# -- Path setup --------------------------------------------------------------

import os
import sys

package_path = os.path.abspath("../..")
sys.path.insert(
    0, package_path
)  # Insert into sys.path so that we can import adaptive here
# Insert into PYTHONPATH so that jupyter-sphinx will pick it up
os.environ["PYTHONPATH"] = ":".join((package_path, os.environ.get("PYTHONPATH", "")))
docs_path = os.path.abspath(
    ".."
)  # Insert `docs/` such that we can run the logo scripts
sys.path.insert(1, docs_path)

import adaptive  # noqa: E402, isort:skip

# -- Project information -----------------------------------------------------

project = "adaptive"
copyright = "2018-2021, Adaptive Authors"
author = "Adaptive Authors"

# The short X.Y version
version = adaptive.__version__
# The full version, including alpha/beta/rc tags
release = adaptive.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_nb",
    "sphinx_fontawesome",
]
source_parsers = {}
templates_path = ["_templates"]
source_suffix = [".rst", ".md"]
master_doc = "index"
language = None
exclude_patterns = []
pygments_style = "sphinx"
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
htmlhelp_basename = "adaptivedoc"


# -- Extension configuration -------------------------------------------------

default_role = "autolink"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "distributed": ("https://distributed.readthedocs.io/en/stable/", None),
    "holoviews": ("https://holoviews.org/", None),
    "ipyparallel": ("https://ipyparallel.readthedocs.io/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "loky": ("https://loky.readthedocs.io/en/stable/", None),
}
html_js_files = [
    "https://cdn.bokeh.org/bokeh/release/bokeh-2.4.0.min.js",
    "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.4.0.min.js",
    "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.4.0.min.js",
    "https://cdn.bokeh.org/bokeh/release/bokeh-gl-2.4.0.min.js",
    "https://cdn.bokeh.org/bokeh/release/bokeh-mathjax-2.4.0.min.js",
]
html_logo = "_static/logo_docs.png"


# myst-nb configuration
nb_execution_mode = "cache"


def setup(app):
    app.add_css_file("custom.css")  # For the `live_info` widget
