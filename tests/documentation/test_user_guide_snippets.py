"""Static checks on the code snippets in documentation/user_guide.md.

Running all ~90 snippets takes well over an hour (several do self-consistent
or Keldysh calculations), so this is deliberately *static*: it parses every
snippet and checks the two things that broke in practice when the guide
drifted from the library, both in a couple of seconds.

1. Every name a snippet reads is one it defines, imports, or inherits from
   an earlier snippet in the same section. This catches a snippet that uses
   `np` without importing numpy, or one that was written as a continuation
   of a neighbour it no longer follows.
2. Every `h.<method>` / `g.<method>` / `geometry.<factory>` the guide names
   -- in prose as well as in code -- exists on the real object. This catches
   a renamed or misspelled entry point, e.g. `geometry.hoenycomb_zigzag_
   ribbon()`, which shipped in the guide for some time.

Neither check runs the physics, so neither can see a wrong argument value or
a snippet that computes the wrong thing. What they do is fail fast, and in
CI, on the class of breakage that makes a snippet unrunnable on the first
copy-paste.
"""
import ast
import builtins
import re

import pytest

from pyqula import geometry

GUIDE = "documentation/user_guide.md"

# The reference chapter is a catalogue of signatures, not a tutorial: its
# fragments illustrate a call on "the h you already have" and are not meant
# to stand alone. Everything before it is.
REFERENCE_CHAPTER = "# Main functions and methods"


def _read_guide():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    return (root / GUIDE).read_text().split("\n")


def _blocks():
    """Every ```python block, with the heading it sits under."""
    lines = _read_guide()
    out, cur, heading, in_reference = [], None, "(top)", False
    for i, l in enumerate(lines):
        if l.startswith("#") and cur is None:
            heading = l.strip()
            if heading == REFERENCE_CHAPTER:
                in_reference = True
        if l.strip().startswith("```python"):
            cur = [i + 2, []]
        elif l.strip() == "```" and cur is not None:
            out.append({"line": cur[0], "code": "\n".join(cur[1]),
                        "heading": heading, "reference": in_reference})
            cur = None
        elif cur is not None:
            cur[1].append(l)
    return out


def _bound_and_read(code):
    """Names a snippet binds, and names it reads, per the AST."""
    tree = ast.parse(code)
    bound, read = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (bound if isinstance(node.ctx, (ast.Store, ast.Del)) else read
             ).add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name) # `except ValueError as e`
        elif isinstance(node, (ast.comprehension,)):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    bound.add(n.id)
    return bound, read


def test_every_snippet_parses():
    """A snippet that is not valid Python cannot be copy-pasted at all."""
    bad = []
    for b in _blocks():
        try:
            ast.parse(b["code"])
        except SyntaxError as e:
            bad.append("L%d: %s" % (b["line"], e))
    assert not bad, "unparseable snippets in the user guide:\n" + "\n".join(bad)


def test_no_snippet_uses_an_undefined_name():
    """Each snippet must define, import, or inherit every name it reads.

    Inheriting means an earlier snippet under the same heading bound it --
    the guide's running-example style, where a follow-up block calls another
    method on the `h` just built. A block that reads a name no neighbour
    ever bound is simply broken for a reader who copies it.
    """
    builtin_names = set(dir(builtins))
    inherited, seen_heading, missing = set(), None, []
    for b in _blocks():
        if b["reference"]:
            continue
        if b["heading"] != seen_heading:
            inherited, seen_heading = set(), b["heading"]
        bound, read = _bound_and_read(b["code"])
        undefined = read - bound - inherited - builtin_names
        if undefined:
            missing.append("L%d (%s): %s" %
                           (b["line"], b["heading"], sorted(undefined)))
        inherited |= bound
    assert not missing, ("user-guide snippets read names nothing defines:\n"
                         + "\n".join(missing))


def test_every_referenced_method_exists():
    """`h.foo()`/`g.foo()`/`geometry.foo()` named anywhere in the guide --
    prose included, since the reference chapter documents methods in prose
    -- must exist on the real object."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    targets = {"h": h, "g": g, "geometry": geometry}
    # g.relax lives on the GrapheneGeometry subclass, documented as such
    allowed_missing = {("g", "relax")}
    bad = []
    for n, line in enumerate(_read_guide(), start=1):
        for m in re.finditer(r"\b(h|g|geometry)\.([A-Za-z_0-9]+)\(", line):
            obj, name = m.group(1), m.group(2)
            if (obj, name) in allowed_missing:
                continue
            if not hasattr(targets[obj], name):
                bad.append("L%d: %s.%s" % (n, obj, name))
    assert not bad, ("the user guide names methods that do not exist:\n"
                     + "\n".join(bad))


@pytest.mark.parametrize("name", ["get_gap", "get_bandwidth", "get_filling",
                                  "get_total_energy", "get_density_matrix",
                                  "get_ipr", "get_vev", "get_single_vev",
                                  "get_several_vev", "get_berry_curvature"])
def test_reference_chapter_documents_these_methods(name):
    """These were public and undocumented until the reference chapter grew
    entries for them; keep the entries from being dropped silently. Some
    share one heading (the get_vev family), so match the method name in any
    `###` entry rather than requiring an entry of its own."""
    headings = [l for l in _read_guide() if l.startswith("### ")]
    documented = any("h.%s(" % name in l for l in headings)
    assert documented, "no reference entry for h.%s in the user guide" % name
