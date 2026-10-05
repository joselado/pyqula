---
name: refresh-docs
description: Refresh pyqula's documentation after a change - recount the test suite, propagate every number that moved, re-run the static user-guide checks, and rebuild documentation/user_guide.pdf. Use when asked to refresh, rebuild, or update the docs or the PDF guide, after landing a feature or a batch of fixes, or before a release.
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
the user guide a section plus an entry in `# Main functions and methods`.

## 3. Re-run the static guide checks

```bash
python -m pytest tests/documentation -v
```

These parse every snippet in `user_guide.md` and check (a) that each name a
snippet reads is one it defines, imports, or inherits from an earlier snippet
in its section, and (b) that every `h.<method>` / `g.<method>` /
`geometry.<factory>` the guide names -- in prose as well as code -- exists on
the real object. They do not run the physics, so they cannot catch a wrong
argument value; they catch the guide having drifted from the library.

## 4. Rebuild the PDF

```bash
(cd documentation && bash convert.sh)
```

`xelatex`, not the default `pdflatex`: the guide contains the Greek letters of
the physics prose, and pdflatex fails on the first one it reaches. Confirm the
build actually produced something new rather than leaving a stale file:

```bash
ls -la documentation/user_guide.pdf
```

## 5. Report

Say which numbers moved and what they moved from and to, whether
`tests/documentation` passed, and that the PDF timestamp advanced. If a count
was already correct, say so rather than implying you changed it.
