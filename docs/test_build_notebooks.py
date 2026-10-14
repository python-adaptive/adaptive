"""Tests for docs/build_notebooks.py.

Run with: uv run --group docs pytest docs/test_build_notebooks.py -o addopts=""
"""

import json
import sys
from pathlib import Path

from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output

sys.path.insert(0, str(Path(__file__).parent))

import build_notebooks as bn  # noqa: E402

WIDGET_VIEW = {"model_id": "abc", "version_major": 2, "version_minor": 0}


def render(*cells, metadata=None, ipynb_name="page.ipynb"):
    nb = new_notebook(cells=list(cells), metadata=metadata or {})
    return bn.render_notebook(nb, ipynb_name)


def test_hide_cell_wraps_in_details():
    cell = new_code_cell("import adaptive", metadata={"tags": ["hide-cell"]})
    md = render(cell)
    assert '<details class="example" markdown="1">' in md
    assert "```python\nimport adaptive\n```" in md
    assert md.index("<details") < md.index("import adaptive") < md.index("</details>")


def test_remove_input_keeps_output():
    out = new_output("stream", name="stdout", text="hi there\n")
    cell = new_code_cell("print('hi there')", outputs=[out])
    cell.metadata["tags"] = ["remove-input"]
    md = render(cell)
    assert "print(" not in md
    assert "hi there" in md


def test_html_output_wrapped_raw():
    out = new_output("display_data", data={"text/html": "<b>x</b>", "text/plain": "X"})
    md = render(new_code_cell("x", outputs=[out]))
    assert '<div class="nb-output" markdown="0">\n<b>x</b>\n</div>' in md
    assert "X" not in md.replace("<b>x</b>", "")


def test_widget_without_state_falls_back():
    data = {"application/vnd.jupyter.widget-view+json": WIDGET_VIEW, "text/plain": "VBox()"}
    md = render(new_code_cell("w", outputs=[new_output("display_data", data=data)]))
    assert "widget-view+json" not in md
    assert "VBox()" in md


def test_widget_with_state_embeds_view_and_state():
    data = {"application/vnd.jupyter.widget-view+json": WIDGET_VIEW, "text/plain": "VBox()"}
    state = {"state": {"abc": {}}, "version_major": 2, "version_minor": 0}
    metadata = {"widgets": {"application/vnd.jupyter.widget-state+json": state}}
    cell = new_code_cell("w", outputs=[new_output("display_data", data=data)])
    md = render(cell, metadata=metadata)
    assert '<script type="application/vnd.jupyter.widget-view+json">' in md
    assert '<script type="application/vnd.jupyter.widget-state+json">' in md
    assert "@jupyter-widgets/html-manager" in md


def test_error_output_rendered():
    tb = ["\x1b[31mValueError\x1b[0m: boom"]
    out = new_output("error", ename="ValueError", evalue="boom", traceback=tb)
    md = render(new_code_cell("raise ValueError('boom')", outputs=[out]))
    assert "ValueError: boom" in md
    assert "\x1b" not in md


def test_download_link_after_title():
    cells = [new_markdown_cell("# Title\n\nIntro."), new_code_cell("1 + 1")]
    md = render(*cells, ipynb_name="tutorial.Learner1D.ipynb")
    assert md.startswith("# Title\n")
    assert "(tutorial.Learner1D.ipynb)" in md
    assert md.index("(tutorial.Learner1D.ipynb)") < md.index("Intro.")


def test_no_download_link_without_code():
    md = render(new_markdown_cell("# Index\n\nText."))
    assert ".ipynb" not in md


def test_cache_hit_skips_execution(tmp_path, monkeypatch):
    src = tmp_path / "page.md"
    src.write_text("# Page\n\n```{code-cell} ipython3\n1 + 1\n```\n")
    calls = []

    def fake_execute(nb, cwd):
        calls.append(cwd)
        return nb

    monkeypatch.setattr(bn, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(bn, "_execute", fake_execute)
    bn.execute_cached(src)
    bn.execute_cached(src)
    assert len(calls) == 1
    src.write_text(src.read_text() + "\nMore text.\n")
    bn.execute_cached(src)
    assert len(calls) == 2


def test_widget_state_round_trips_html_values():
    value = "<table><tr><th>status</th></tr></table></script>"
    data = {"application/vnd.jupyter.widget-view+json": WIDGET_VIEW, "text/plain": "HTML()"}
    state = {"state": {"abc": {"state": {"value": value}}}, "version_major": 2}
    metadata = {"widgets": {"application/vnd.jupyter.widget-state+json": state}}
    md = render(new_code_cell("w", outputs=[new_output("display_data", data=data)]), metadata=metadata)
    start = md.index('<script type="application/vnd.jupyter.widget-state+json">')
    content = md[md.index(">", start) + 1 : md.index("</script>", start)]
    assert json.loads(content) == state


def test_inline_plotly_bundle_replaced_by_cdn():
    bundle = "<script>window.PlotlyConfig = {};</script>\n<script>/**\n* plotly.js v4.1.1\n*/" + "x" * 2000 + "</script>"
    out = new_output("display_data", data={"text/html": bundle, "text/plain": ""})
    md = render(new_code_cell("learner.plot_3D()", outputs=[out]))
    assert '<script src="https://cdn.plot.ly/plotly-4.1.1.min.js"' in md
    assert "x" * 2000 not in md
