---
name: refresh-docs
description: Refresh pyqula's documentation after a change - recount the test suite, propagate every number that moved, re-run the static user-guide checks, and build the user-guide website and its PDF locally. Use when asked to refresh, rebuild, or update the docs or the PDF guide, after landing a feature or a batch of fixes, or before a release.
---

# Refreshing the documentation

The guide and the README carry a handful of counted facts that drift silently:
nothing fails when they go stale, so they only get fixed when someone looks.
This is the sweep that looks.

Do the steps in order. Each one is cheap except the last.

## 1. Recount the test suite

```bash
OUT=<your session scratchpad>/collect.txt
python -m pytest tests --collect-only -q > "$OUT" 2>&1; echo "exit=$?"; tail -3 "$OUT"
```

**Redirect, never pipe.** `pytest ... | tail` reports the *pipe's* exit status,
so a crashed run or an `unrecognized arguments` error that ran no tests at all
both look like success. This is not hypothetical in this repo -- it once masked
a fatal interpreter abort. Write the output to the session's scratchpad
directory rather than `/tmp`.

Then find every copy of the old number before editing any of them:

```bash
grep -rn "<old count>" --include='*.md' . | grep -v '\.git/'
```

Today the count lives in `CLAUDE.md` only, but it has lived in more than one
file before; grep so they move together. **Leave the 37:34 whole-suite
runtime alone** unless you actually re-measured it on an idle machine -- that
figure was measured three times (33:40, 34:18, 37:34) and re-deriving
it costs more than half an hour of an idle machine.

## 2. Recount anything else the docs assert

The README's tutorial section counts notebooks and how many FUNCTIONALITIES
bullets link to one:

```bash
find jupyter-notebooks -name '*.ipynb' -not -path '*checkpoint*' | wc -l
ls jupyter-notebooks/functionalities/*.ipynb | wc -l
```

If a feature landed since the last sweep, the FUNCTIONALITIES list in
`README.md` (section starts at `# FUNCTIONALITIES #`) may need a bullet, and
the user guide a section on its topic page under `documentation/guide/`
plus an entry on the matching reference page (`reference-hamiltonian.md`,
`reference-geometry.md` or `reference-other.md`). A new page also needs a
line in the `nav` of `mkdocs.yml`.

## 3. Re-run the static guide checks

```bash
python -m pytest tests/documentation -v
```

These parse every snippet of the pages in `documentation/guide/` and check (a) that each name a
snippet reads is one it defines, imports, or inherits from an earlier snippet
in its section of the same page, and (b) that every `h.<method>` / `g.<method>` /
`geometry.<factory>` the guide names -- in prose as well as code -- exists on
the real object. They do not run the physics, so they cannot catch a wrong
argument value; they catch the guide having drifted from the library.

## 4. Build the website and the PDF

The guide is published at https://joselado.github.io/pyqula/ by
`.github/workflows/guide.yml` on every push to master that touches
`documentation/guide/`, `mkdocs.yml` or the PDF builder, so nothing is
committed for it. Before pushing, build both locally to catch what the
workflow would catch:

```bash
pip install mkdocs==1.6.1 mkdocs-material==9.7.7   # in a scratch venv
mkdocs build -d <your session scratchpad>/site     # strict: a broken page link fails the build
python documentation/build_guide_pdf.py <your session scratchpad>/user_guide.pdf
```

The PDF joins the pages in the order of the `nav` in `mkdocs.yml` and needs
pandoc and `xelatex` (the default `pdflatex` fails on the first Greek
letter). It is not tracked in git; the workflow publishes it next to the site.

## 5. Report

Say which numbers moved and what they moved from and to, whether
`tests/documentation` passed, and whether the site and the PDF built. If a count
was already correct, say so rather than implying you changed it.
