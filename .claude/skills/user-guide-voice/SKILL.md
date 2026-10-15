---
name: user-guide-voice
description: The maintainer's locked writing voice for every text written for pyqula's users (the user guide, README.md, notebook prose, example comments) and in particular for the user guide, the website pages under documentation/guide/ -- simple English written by a condensed matter physicist, the formula and the physical meaning before the code, no internals; plus the older registers (chapter prose, section intros, catalogue bullets), the spelling decisions, what not to write, and which chapters are the maintainer's own prose to fix rather than rewrite. Load this before writing or rewriting a single sentence of the user guide pages, including when merely adding a section for a new feature, and before writing or changing any text in README.md, in a notebook of jupyter-notebooks/ or in the comments of examples/. Prose written without it reads like a language model wrote it, which is the exact failure this exists to prevent.
---

# Writing the pyqula user guide

`documentation/VOICE.md` is the full reference and is the authority here. **Read it before
writing** -- section 0 first, which sets the register of the website and wins over the
rest (short sentences in simple English, the formula and the physics before the code, no
internals, `documentation/guide/hamiltonian.md` as the model page), then sections 1 to 4, then sections 5 to 8 as a reference while writing, then
run the checklist in section 10 before finishing. When something in it disagrees with a
sentence the maintainer wrote in the guide, the maintainer's sentence wins.

## When the guide has to change

When a change adds or materially changes a user-facing feature, update its page
under `documentation/guide/` (the website, built by MkDocs from `mkdocs.yml`), and `README.md`'s FUNCTIONALITIES list where relevant. The
existing style for a new feature is a short prose section with the physics and the
motivation, a runnable code snippet, a figure when the section computes an
observable (`documentation/figures/README.md`), and -- for anything with a method on `Hamiltonian` or
`Geometry` -- an entry on the reference pages `documentation/guide/reference-*.md`.

## README's FUNCTIONALITIES list

The maintainer's bullets are the model: 3 to 8 words, one noun phrase, no parentheses, no
keyword names ("Metal-superconductor transport", "Frequency resolved topological density").

- A bullet and its `[[notebook]]` link fit on one rendered line, so the text stays at
  75 characters at most. Cut parenthetical lists, backticked keywords and "with/including/
  together with" tails. The cut detail becomes the opening paragraph of the notebook.
- Every bullet links to its own executed notebook in
  `jupyter-notebooks/functionalities/<section>/`. A new feature bullet comes with a new
  notebook, never without one. Two bullets for the same feature become one.
- In the notebook, show at least one heatmap (a color map over k-space, real space,
  $(k,\omega)$ or a parameter plane) instead of, or besides, a line plot whenever the
  physics allows one.
- Do not re-voice the maintainer's own bullets; fix a typo at most.

## What VOICE.md covers, so you know what you are missing without it

- **Who is speaking to whom**: the maintainer writing to a physicist who wants to compute
  something, never reframing the reader as a beginner
- **Three registers**, never mixed inside one paragraph: chapter prose (narrated build-up),
  section intros ("We will now see how..."), and catalogue bullets (definition-first, no
  narration). The typical failure is a catalogue bullet that grows a paragraph of narration,
  or a chapter paragraph that reads like a docstring
- **A calibration table** of phrases per 1000 words, including the ones to keep at zero:
  "worth noting", "notably", "crucially", "importantly", "well-known", "textbook",
  contractions, em-dashes, exclamation marks
- **Spelling decisions**: onsite, mean-field, tight-binding, self-consistent, band structure
- **Sentence length targets**, measured on the maintainer's own chapters
- **How a chapter and a section are built**, as an eight-step shape
- **Which chapters are the maintainer's own prose** -- fix, do not rephrase -- and which are
  Claude-written and may be converted

## Maintainer material stays out of the guide

Pointers into `src/` and `tests/`, benchmark tables, profiling stories, the registry behind
a `mode=` string, the history of a fix, the reason a default was chosen for implementation
rather than physical reasons: none of it helps a reader who wants to know what a method
computes. It belongs in `CLAUDE.md`, in the module docstring, or in `future_development/`.
