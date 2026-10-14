"""Execute the notebooks in docs/notebooks/ and render them for Zensical.

Every ``docs/notebooks/<path>.md`` (a jupytext MyST notebook) becomes
``docs/<path>.md`` with its outputs embedded as raw HTML, plus a
``docs/<path>.ipynb`` download without outputs. Executed notebooks are cached
in ``.notebook_cache/`` keyed by their source text and library versions.

Usage: python docs/build_notebooks.py [docs/notebooks/<path>.md ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

DOCS_DIR = Path(__file__).parent
SOURCE_DIR = DOCS_DIR / "notebooks"
CACHE_DIR = DOCS_DIR.parent / ".notebook_cache"  # Zensical owns .cache/

# Richest representation first, like nbconvert's HTML exporter.
MIME_PRIORITY = [
    "application/vnd.jupyter.widget-view+json",
    "application/javascript",
    "text/html",
    "text/markdown",
    "image/svg+xml",
    "image/png",
    "image/jpeg",
    "text/latex",
    "text/plain",
]
WIDGET_VIEW = "application/vnd.jupyter.widget-view+json"
WIDGET_STATE = "application/vnd.jupyter.widget-state+json"
WIDGET_LOADER = (
    '<script src="https://cdn.jsdelivr.net/npm/'
    '@jupyter-widgets/html-manager@1.0.15/dist/embed.js" crossorigin="anonymous"></script>'
)
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# adaptive calls plotly.offline.init_notebook_mode(), which inlines plotly.js.
PLOTLY_BUNDLE = re.compile(r"\* plotly\.js v(\d+\.\d+\.\d+)")
HIDE_LABELS = {
    "hide-cell": "Show code cell",
    "hide-input": "Show code",
    "hide-output": "Show output",
}


def _script_json(value: object) -> str:
    # "</" would end the <script> element early; "<\\/" is the same JSON.
    return json.dumps(value).replace("</", "<\\/")


def _raw(html_text: str) -> str:
    return f'<div class="nb-output" markdown="0">\n{html_text.strip()}\n</div>'


def _code_block(text: str, lang: str) -> str:
    fence = "````" if "```" in text else "```"
    return f"{fence}{lang}\n{text.rstrip()}\n{fence}"


def _text_output(text: str) -> str:
    return _code_block(ANSI.sub("", text), " { .text .nb-output-text }")


def _details(label: str, body: str) -> str:
    return (
        f'<details class="example" markdown="1">\n<summary>{label}</summary>\n\n'
        f"{body}\n\n</details>"
    )


def _render_data(data: dict, has_widget_state: bool) -> str:
    for mime in MIME_PRIORITY:
        if mime not in data or (mime == WIDGET_VIEW and not has_widget_state):
            continue
        value = data[mime]
        if mime == WIDGET_VIEW:
            return _raw(f'<script type="{WIDGET_VIEW}">{_script_json(value)}</script>')
        if mime == "application/javascript":
            return _raw(f'<script type="text/javascript">\n{value}\n</script>')
        if mime == "text/html":
            if bundle := PLOTLY_BUNDLE.search(value):
                url = f"https://cdn.plot.ly/plotly-{bundle.group(1)}.min.js"
                value = f'<script src="{url}" charset="utf-8"></script>'
            return _raw(value)
        if mime == "text/markdown":
            return value
        if mime == "image/svg+xml":
            return _raw(value)
        if mime in ("image/png", "image/jpeg"):
            src = f"data:{mime};base64,{value.strip()}"
            return _raw(f'<img src="{src}" alt="output">')
        if mime == "text/latex":
            return value
        return _text_output(value)
    return ""


def _render_output(output: dict, has_widget_state: bool) -> str:
    kind = output["output_type"]
    if kind == "stream":
        return _text_output(output["text"])
    if kind == "error":
        return _text_output("\n".join(output["traceback"]))
    return _render_data(output.get("data", {}), has_widget_state)


def _render_code_cell(cell: dict, has_widget_state: bool) -> str:
    tags = set(cell.get("metadata", {}).get("tags", []))
    if "remove-cell" in tags:
        return ""
    source = "" if "remove-input" in tags else _code_block(cell["source"], "python")
    outputs = (
        []
        if "remove-output" in tags
        else [_render_output(o, has_widget_state) for o in cell.get("outputs", [])]
    )
    # Raw HTML is escaped inside a `markdown="1"` <details>, so hidden cells
    # only hide their source and text outputs; raw outputs follow the block.
    raw = "\n\n".join(o for o in outputs if o.startswith('<div class="nb-output"'))
    text = "\n\n".join(
        o for o in outputs if o and not o.startswith('<div class="nb-output"')
    )
    hide_source = source and tags & {"hide-cell", "hide-input"}
    hide_text = text and tags & {"hide-cell", "hide-output"}
    if hide_source and hide_text:
        return _join(_details(HIDE_LABELS["hide-cell"], f"{source}\n\n{text}"), raw)
    if hide_source:
        source = _details(HIDE_LABELS["hide-input"], source)
    if hide_text:
        text = _details(HIDE_LABELS["hide-output"], text)
    if tags & {"hide-cell", "hide-output"}:
        return _join(source, text, raw)
    return _join(source, "\n\n".join(o for o in outputs if o))


def _join(*parts: str) -> str:
    return "\n\n".join(part for part in parts if part)


def _download_note(ipynb_name: str) -> str:
    return (
        '!!! tip "Run this page yourself"\n'
        f"    Download this page as a Jupyter notebook: [{ipynb_name}]({ipynb_name})."
    )


def render_notebook(nb: nbformat.NotebookNode, ipynb_name: str) -> str:
    """Render an (executed) notebook as Zensical Markdown."""
    widget_state = nb.metadata.get("widgets", {}).get(WIDGET_STATE)
    has_widget_state = bool(widget_state and widget_state.get("state"))
    parts = []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            parts.append(cell.source)
        elif cell.cell_type == "code":
            parts.append(_render_code_cell(cell, has_widget_state))
    md = "\n\n".join(p for p in parts if p.strip())

    if any(cell.cell_type == "code" for cell in nb.cells):
        note = _download_note(ipynb_name)
        title = re.search(r"^# .*$", md, flags=re.MULTILINE)
        if title:
            md = f"{md[: title.end()]}\n\n{note}{md[title.end() :]}"
        else:
            md = f"{note}\n\n{md}"

    if has_widget_state:
        state = _script_json(widget_state)
        md += "\n\n" + _raw(
            f'<script type="{WIDGET_STATE}">{state}</script>\n{WIDGET_LOADER}'
        )
    return md + "\n"


def _execute(nb: nbformat.NotebookNode, cwd: Path) -> nbformat.NotebookNode:
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(cwd)}},
        store_widget_state=True,
    )
    return client.execute()


def _library_versions() -> str:
    # Outputs embed these libraries' versions (e.g. BokehJS CDN URLs).
    packages = ("adaptive", "bokeh", "holoviews", "ipywidgets", "panel", "plotly")
    return " ".join(f"{name}=={version(name)}" for name in packages)


def execute_cached(source: Path) -> nbformat.NotebookNode:
    """Execute a jupytext notebook, reusing the cached result if unchanged."""
    text = source.read_text()
    key = hashlib.sha256(f"{_library_versions()}\n{text}".encode()).hexdigest()
    cached = CACHE_DIR / f"{key}.ipynb"
    if cached.exists():
        return nbformat.read(cached, as_version=4)
    nb = _execute(jupytext.reads(text, fmt="md:myst"), source.parent)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, cached)
    return nb


def build(source: Path) -> Path:
    """Execute and render one notebook; return the written Markdown path."""
    start = time.monotonic()
    target = DOCS_DIR / source.relative_to(SOURCE_DIR)
    target.parent.mkdir(parents=True, exist_ok=True)
    ipynb = target.with_suffix(".ipynb")
    nb = execute_cached(source)
    # Read by docs/overrides/partials/actions.html to link "Edit" to the source.
    front_matter = f"---\nnotebook_source: {source.relative_to(DOCS_DIR)}\n---\n\n"
    target.write_text(front_matter + render_notebook(nb, ipynb.name))
    if any(cell.cell_type == "code" for cell in nb.cells):
        jupytext.write(jupytext.read(source), ipynb)
    print(f"{source.relative_to(DOCS_DIR)}: {time.monotonic() - start:.1f}s")
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="*", type=Path)
    parser.add_argument("-j", "--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    sources = [p.resolve() for p in args.sources] or sorted(SOURCE_DIR.rglob("*.md"))
    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(build, sources))


if __name__ == "__main__":
    main(sys.argv[1:])
