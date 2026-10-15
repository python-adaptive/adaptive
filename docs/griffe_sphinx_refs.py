"""Griffe extension: link Sphinx-style references in docstrings.

The docstrings were written for Sphinx with ``default_role = "autolink"``, so
`` `~adaptive.Runner` `` was a link. This rewrites such references into
optional autorefs, which link when the target is documented (here or in an
inventory) and render as plain code otherwise.
"""

from __future__ import annotations

import re
from typing import Any

import griffe

# `name`, `~dotted.name`, or ~`dotted.name`, but not ``literal``.
SPHINX_REF = re.compile(r"(?<!`)~?`(~?)([A-Za-z_][\w.]*)`(?!`)")
# numpy sections whose unindented lines are "name : type" or "type" headers.
FIELD_SECTIONS = {
    "Parameters",
    "Other Parameters",
    "Returns",
    "Yields",
    "Receives",
    "Raises",
    "Warns",
    "Attributes",
}


def sphinx_refs_to_autorefs(text: str, public: frozenset[str]) -> str:
    """Rewrite Sphinx default-role references in ``text`` to optional autorefs.

    Unindented lines in numpy field sections are types, which mkdocstrings
    renders verbatim, so there the markup is only stripped: griffe then parses
    plain dotted names as expressions and links them itself.
    """
    lines = re.sub(r"\\\n\s+", " ", text).split("\n")
    section = None
    for i, line in enumerate(lines):
        if i + 1 < len(lines) and re.fullmatch(r"-{3,}", lines[i + 1].strip()):
            section = line.strip()
        elif section in FIELD_SECTIONS and line.strip() and not line[0].isspace():
            line = re.sub(r"``([^`]+)``", r"\1", line)
            lines[i] = SPHINX_REF.sub(lambda m: m.group(2), line)
        else:
            lines[i] = _link_refs(line, public)
    return "\n".join(lines)


def _link_refs(text: str, public: frozenset[str]) -> str:
    def replace(match: re.Match) -> str:
        tilde, target = match.groups()
        label = target
        if "." not in target:
            if target not in public:
                return match.group(0)
            target = f"adaptive.{target}"
        elif tilde or match.group(0).startswith("~"):
            label = target.rsplit(".", 1)[-1]
        return f'<autoref identifier="{target}" optional><code>{label}</code></autoref>'

    return SPHINX_REF.sub(replace, text)


class SphinxRefs(griffe.Extension):
    """Apply `sphinx_refs_to_autorefs` to every docstring of the package."""

    def on_package(self, *, pkg: griffe.Module, **kwargs: Any) -> None:
        public = frozenset(str(name) for name in pkg.exports or ())
        self._rewrite(pkg, public, seen=set())

    def _rewrite(self, obj: griffe.Object, public: frozenset[str], seen: set) -> None:
        if obj.path in seen:
            return
        seen.add(obj.path)
        if obj.docstring:
            obj.docstring.value = sphinx_refs_to_autorefs(obj.docstring.value, public)
        for member in obj.members.values():
            if not member.is_alias:
                self._rewrite(member, public, seen)
