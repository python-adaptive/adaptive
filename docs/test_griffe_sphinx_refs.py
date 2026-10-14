"""Tests for docs/griffe_sphinx_refs.py.

Run with: uv run --group docs pytest docs/test_griffe_sphinx_refs.py -o addopts=""
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from griffe_sphinx_refs import sphinx_refs_to_autorefs  # noqa: E402

PUBLIC = frozenset({"BalancingLearner", "Runner"})


def link(identifier, text):
    return f'<autoref identifier="{identifier}" optional><code>{text}</code></autoref>'


def test_tilde_ref_shows_last_component():
    text = "A `~adaptive.BaseLearner` instance."
    assert (
        sphinx_refs_to_autorefs(text, PUBLIC)
        == f"A {link('adaptive.BaseLearner', 'BaseLearner')} instance."
    )


def test_dotted_ref_keeps_full_path():
    text = "Returns a `pandas.DataFrame`."
    assert (
        sphinx_refs_to_autorefs(text, PUBLIC)
        == f"Returns a {link('pandas.DataFrame', 'pandas.DataFrame')}."
    )


def test_misplaced_tilde_is_handled():
    text = "learner : ~`adaptive.BaseLearner` instance"
    assert (
        sphinx_refs_to_autorefs(text, PUBLIC)
        == f"learner : {link('adaptive.BaseLearner', 'BaseLearner')} instance"
    )


def test_public_name_links_to_adaptive():
    text = "See `BalancingLearner`."
    assert (
        sphinx_refs_to_autorefs(text, PUBLIC)
        == f"See {link('adaptive.BalancingLearner', 'BalancingLearner')}."
    )


def test_literals_and_plain_names_untouched():
    text = "Use ``learner.plot()`` with `retries` and ``Runner``."
    assert sphinx_refs_to_autorefs(text, PUBLIC) == text


def test_numpy_field_headers_are_stripped_not_linked():
    text = "Run it.\n\nParameters\n----------\nlearner : `~adaptive.BaseLearner` instance\n    Uses `~adaptive.Runner`.\n\nReturns\n-------\n`pandas.DataFrame`\n"
    result = sphinx_refs_to_autorefs(text, PUBLIC)
    assert "learner : adaptive.BaseLearner instance\n" in result
    assert "\npandas.DataFrame\n" in result
    assert f"    Uses {link('adaptive.Runner', 'Runner')}." in result


def test_backslash_continuation_in_field_header_is_joined():
    text = "Parameters\n----------\nexecutor : `a.B`,\\\n           `c.D`, optional\n    Description.\n"
    result = sphinx_refs_to_autorefs(text, PUBLIC)
    assert "executor : a.B, c.D, optional\n    Description." in result
