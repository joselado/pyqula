# Writing the user guide in the maintainer's voice

How `documentation/user_guide.md` is written, so that a session extending or rewriting it
produces prose that reads like the maintainer's own chapters and not like a language model's
default register. The voice is the one of the maintainer's lectures and lecture notes for the
Advanced Quantum Materials course (github.com/joselado/Advanced_Quantum_Materials_2027),
where it was measured on the recorded lectures, the notebooks and the approved notes; this
file says what that voice looks like when it documents a library, and records the decisions
taken for the guide.

How to use this file: read sections 1 to 4 before touching the guide. Use sections 5 to 8 as a
reference while writing. Run the checklist in section 10 before finishing. When something here
disagrees with a sentence the maintainer wrote in the guide, the maintainer's sentence wins.

---

## 1. Where the voice comes from, and which layers are canonical

| Source | Where | What it is | Status for imitation |
|---|---|---|---|
| The maintainer's guide chapters | commit `77a6fa8` (March 2024) and the edits up to `0117df2` (October 2025): Setting up a Hamiltonian, the first Observables sections, Operators, Superconductivity, the four mean-field sections, Topological invariants, Quantum transport, the first reference entries | The maintainer's own written pyqula prose | **Canonical for the guide register.** Light touch: fix, do not rephrase (rule 19) |
| The course notebooks | `jupyter-notebooks/session1..5.ipynb` of the course repository, the `##` titles and the one sentence under each, and the quiz bullets | The maintainer describing pyqula calculations in one sentence ("We will now see how the Hubbard interaction give rise to a gap opening in an interacting chain", "Let us now see how to compute Fermi surface", "Change the value of the superconducting order (add_swave), what happens to the gap?") | **Canonical for section intros and for the lexicon** |
| The course lecture notes | `lecture_notes/` of the course repository | The maintainer's spoken lectures compiled into a written register and approved | Canonical for the chapter register; the calibration column "notes" in section 3.1 was measured on them |
| README functionality list | `README.md`, FUNCTIONALITIES and EXAMPLES | Maintainer-written headings and snippet comments | Canonical for how a feature is named |
| Guide chapters written by Claude | `421c654`, `6c9606b`, `b0241d5` (July 2026) and the feature commits after them: KPM, Wannierization, spectral functions, response functions, excitons, Keldysh transport, entanglement, classical models, most of the reference chapter | Approved content in a different register | **Not canonical.** This is the material to convert |
| The mandate of `d30babe` (14 September 2026) | its commit message | Physics, not maintainer material: what a method computes and which parameter to change | In force. A voice pass keeps its cuts |
| Functionality notebooks | `jupyter-notebooks/functionalities/` | Claude-written, executed, with plots | Where to point for what a result looks like. Not a voice source |

To tell the two layers apart in a chapter, diff it against `git show 77a6fa8:documentation/
user_guide.md` and `git show 0117df2:documentation/user_guide.md`: a sentence present there is
the maintainer's, everything else is Claude's.

Three facts about the sources that matter before imitating them:

- The maintainer is a fluent non-native English writer, and the 2024 prose carries a few slips
  a copy editor would fix: "teh geometry", "eenrgy", "consists on", "Let now show", a dropped
  article here and there. Fix them. Do not manufacture new ones, and do not smooth the
  plainness that remains ("This can be added to the Hamiltonian as", "Possible inputs").
- The guide is read by search, not end to end. A reader lands on a subsection from the table
  of contents or from a method name, so each subsection has to stand on its own and name its
  method before the code (section 4).
- The snippets are tested statically (`tests/documentation/test_user_guide_snippets.py`
  checks names and method existence, not physics). A prose pass does not touch the code inside
  a block: a "cleaner" argument value silently changes what is computed and nothing catches it.

---

## 2. Who is speaking, to whom

- The maintainer of pyqula writing to a physicist who wants to compute something: a master's
  or PhD student or a postdoc who knows tight-binding models and second quantization and may
  never have opened the library. A colleague who happens to know more, not a manual, and not a
  textbook: the reader is assumed to have seen single-particle solid-state physics and standard
  quantum mechanics, and is never reframed as a beginner.
- The same voice as the lectures, written down: long sentences chained with commas and "and",
  "so", "meaning that"; a few plain phrasings; enthusiasm kept to one adjective; and honesty
  about what is approximate (mean field is uncontrolled, a broadening hides what it cannot
  resolve, a k-mesh converges or it does not, an open problem is called open).
- "We" is the default subject (we will now see, let us, we take). "You" appears when the
  reader's own choice or observation is being traced ("if you want to enforce a certain
  filling", "what you actually see in the band structure"). "One" occasionally, for a generic
  agent ("one can measure"). "I" does not appear in the guide.
- Physics is told as building up: the chain before the honeycomb lattice, first-neighbor
  hopping before the list of hoppings, s-wave before the d-vector, the collinear mean field
  before the non-collinear one. The name comes after the construction ("this is what is called
  the Bogoliubov-de Gennes Hamiltonian"), with one exception forced by search: the method name
  appears before the code block of a subsection, in its first sentence when that reads
  naturally (section 4).
- Everything is tied to what can be computed and to what can be measured: an LDOS map is what
  STM sees, a momentum-resolved spectral function is what ARPES sees, a transmission is what a
  transport measurement sees, a Hall conductivity is how one knows time-reversal symmetry is
  broken.

---

## 3. Three registers inside one guide

| Register | Used for | Feel | Signature |
|---|---|---|---|
| **Chapter prose** | The paragraphs between equations and code; the written-notes register of the lectures | Narrated build-up with the spoken tics stripped; rhythm carried by colons, semicolons and parentheses | "Let us start with the simplest case", "The simplest form of superconductivity is", "What this means is that", "This is what is called" |
| **Section intros and pointers** | The first sentence of a subsection; the pointer to `examples/` and to a notebook; the notebook-cell register of the course | One sentence saying what we will now see | "We will now see how", "Let us now address", "Here we will show how", "See `examples/2d/qpi/main.py` for a runnable version" |
| **Catalogue** | The `Optional arguments` bullets and the whole "Main functions and methods" chapter | One bullet per argument: what it buys you and when to reach for it. Definition-first by design, no narration | "`nk`: number of k-points along the path", "`operator`: color each band by the expectation value of an operator" |

Decide the register per paragraph and never mix inside one. The typical failure is a
catalogue bullet that grows a paragraph of narration (three sentences on which `gmode` reads
`nk`), or a chapter paragraph that reads like a docstring ("Returns a tuple of arrays").

### 3.1 Calibration table

Occurrences per 1000 words. The "notes" column is the written register of the course lecture
notes; the "guide" column was measured on the guide at `d30babe` (36,462 words) and is the
baseline a full pass starts from.

| Phrase | Notes (target) | Guide at `d30babe` | Verdict |
|---|---|---|---|
| of course | 0.1 | 0.05 | fine |
| in particular | 0.1 | 0.08 | fine |
| essentially, effectively | 0.4, 0.2 | | use sparingly |
| let us | 0.5 | 0.03 | too low: the guide never says "let us" |
| we will | 1.2 | 0.19 | too low: section intros should carry it |
| Note that | allowed | 0.16 | fine, it is one of the maintainer's own openers |
| worth noting, notably, crucially, importantly | 0 | 0.03 (one "worth noting") | remove |
| well-known, textbook, literature | 0 | 0.19 (1, 2 and 4) | remove |
| contractions (n't, it's, we'll) | 0 | 0.19 (seven "n't") | remove |
| em-dash, spaced double hyphen | 0 | 0 | keep at zero |
| exclamation marks | 0 | 0 | keep at zero |
| colons, semicolons | 14, 4 | | the notes carry their rhythm on them |
| sentences that are questions | 1% | | a question opens a section now and then, not a paragraph |
| rationalize, give rise to, phenomenology | present | 0, 0.11, 0.03 | use when they fit; do not force them |

Spellings that were split at `d30babe`, and the decision for each (section 7 says why):

| Decided | Instead of | Count at `d30babe` |
|---|---|---|
| onsite | on-site | 49 vs 0 |
| mean-field (as modifier) | mean field Hamiltonian | 47 vs 31 |
| tight-binding (as modifier) | tight binding model | 6 vs 3 |
| self-consistent, self-consistency | selfconsistent | 21 vs 2 |
| band structure | bandstructure | 30 vs 1 |

Sentence length, measured on prose sentences only (no code, equations, headings or bullets):

| Text | Sentences | Mean words | Median |
|---|---|---|---|
| The maintainer's 2024 guide (`77a6fa8`, chapters before the reference) | 79 | 24 | 21 |
| The course lecture notes | | 32 | 30 |
| The pilot chapter "Setting up a Hamiltonian" (14 September 2026) | 35 | 34 | 29 |

The guide's own sentences are shorter than the lecture notes'. The pilot matched the notes
rather than the guide and the maintainer approved its density, so a full pass aims at the
pilot: build-up before each snippet, the result read aloud after it, sentences chained with
commas rather than packed with clauses, and a sentence of 40 words or more only when the
chain reads naturally.

---

## 4. How a chapter and a section are built

**A chapter opens** with one or two sentences in the chapter register: what we are going to
compute, the Hamiltonian in one line if there is a single one for the whole chapter, and what
the sections will build. Not "This chapter covers".

**A section**, most of the time in this order:

1. One sentence saying what we will now see, in the intro register, naming the physics and,
   before the code block and in the first sentence when it reads naturally, the method ("We
   will now see how to add an external Zeeman field to the chain, with `h.add_zeeman()`").
2. The Hamiltonian or the definition, typeset, followed at once by what each term is in words
   ("where $n$ runs over the sites and $s,s'$ over the spin"). The chapter register.
3. The simplest case, and the code for it. Every line of the block carries a trailing comment
   saying what it does physically; the block imports what it uses; it stops at the computed
   arrays and leaves the plotting to the reader.
4. What the result means: what the arrays contain, what to look for, what happens as the
   parameter grows ("the field splits the two spin bands by $2|\vec B|$"; "the band structure
   shows both the electron and the hole states"). Read the plot aloud without the plot. Say
   what the quantity corresponds to in an experiment when there is one.
5. `Optional arguments` as bullets, in the catalogue register.
6. The caveats that change how a number should be read: the broadening, the k-mesh, the
   initial guess of a self-consistent calculation, an approximation that is uncontrolled. Each
   caveat is specific to this calculation; a generic "results should be converged" says nothing.
7. Pointers, one line: the runnable script in `examples/` and the executed notebook in
   `jupyter-notebooks/functionalities/`.
8. Where the guide goes next, when the next section builds on this one ("we will use this
   filling again in the mean-field chapter").

Not every section has all eight; a two-line subsection that documents one keyword has 1, 3 and
5. What no section drops is the physics sentence before the code (a section is not an
argument list) and the method name before the code.

**The reference chapter** keeps its shape: the signature as a heading, one line on what it
computes, the arguments as bullets, a two-line fragment on "the h you already have". It is a
catalogue and reads as one; the narration lives in the chapters and is not repeated there.

**Maintainer material stays out.** Pointers into `src/` and `tests/`, benchmark tables,
profiling stories, the registry behind a `mode=` string, the history of a fix, the reason a
default was chosen for implementation rather than physical reasons: none of it helps a reader
who wants to know what a method computes. It belongs in `CLAUDE.md`, in the module docstring,
or in `future_development/`. This is the mandate of `d30babe`, and a voice pass keeps its cuts.

---

## 5. The rhetorical moves

The moves the maintainer's lectures are built from, with the form each takes in the guide.
The examples are from the maintainer's chapters, lightly fixed, or written here as patterns;
use them as patterns, not as text to reuse.

**5.1 Ask the question first, then answer it.** "How are those expectation values obtained?
By solving a self-consistent problem, and the idea is simple." In the guide the question is
often implicit in the section title; asking it in the first sentence is still the natural
opener when the section is a method ("how do we get a filling of 0.7?").

**5.2 Name the idea before the derivation.** "The mean-field approximation consists in
replacing the four-fermion operator by all the terms that arise by taking the expectation
value of two of the fermions." Then the equation. Then "As a result, the mean-field
Hamiltonian depends on the ground state, and the ground state depends on the mean-field
Hamiltonian", which is the bottom line.

**5.3 Simplest case first, then generalize.** The chain before the ribbon, one hopping before
three, a float before a callable in "Possible inputs", s-wave before the d-vector. A section
that introduces a general machinery (the operator-operator response, the generic mean-field
channel) is preceded by the special case the reader already knows.

**5.4 Define by contrast.** Normal versus anomalous terms; spinless versus spinful; the
electron copy versus the hole copy of the BdG bands; collinear versus non-collinear;
exact diagonalization versus KPM; a real-space versus a reciprocal-space Chern number;
trivial versus topological; in the bulk versus at the edge; gapped versus gapless. One
property separates each pair, and the sentence says which.

**5.5 Say it twice.** In speech, the same statement with one word changed. In the guide, one
sentence with an appositive: "the filling $\nu$, the fraction of the states that are
occupied".

**5.6 Spell it out with "meaning that".** Every equation gets its content in words: "meaning
that the whole band structure is shifted rigidly by $\mu$", "what this means is that the
gap opens regardless of where the chemical potential is".

**5.7 Everyday analogy.** Rare in a guide. When the maintainer's prose has one, keep it; do
not add new ones.

**5.8 Read the plot aloud.** The snippet returns arrays and makes no figure, so the reading
is of the arrays: "the band structure shows flat bands, the Landau levels, and between them
dispersive bands, the states at the two edges of the ribbon". The executed notebook is where
the reader goes for the figure; say so.

**5.9 Tie the mathematics to what can be measured.** "This profile is what one measures with
STM"; "the momentum-resolved spectral function is what ARPES measures"; "the transmission is
the zero-bias conductance in units of $e^2/h$". One clause, not a paragraph.

**5.10 Be honest about approximations.** Specific: "the final solution may be sensitive to the
initial guess, and this matters for systems whose energy landscape has several local minima";
"a broadening below the level spacing of the finite k-mesh shows the mesh, not the physics";
"this is an uncontrolled approximation, and the practical workhorse". Not generic: "results
should be checked for convergence".

**5.11 Point forward and backward inside the guide.** "We will use this term again in the
chapter on superconductivity"; "the filling keyword is the same one that
`get_mean_field_hamiltonian` takes". The guide is one story; a section says what it borrows and
what it defers, with the section name, not a chapter number.

**5.12 Hand the reader an intuition with "you can think of".** "You can think of the Nambu
spinor as a copy of the electron operators stacked on top of a copy of their conjugates";
"you can think of a spinon as the spin sector of an electron, with the charge frozen".

**5.13 Emphasis.** One adjective. "This is the most expensive snippet in the guide" is the
guide's register; "extremely, incredibly, very very" is the lecture's.

---

## 6. Sentence-level habits

- **Length and glue.** Comma-chained sentences with "and", "so", "meaning that", "which is"
  are the voice. Colons, semicolons and parentheses break a chain that gets too long. A
  one-clause sentence is fine; a run of them reads like a manual.
- **Openers.** "Let us", "We will now see", "The simplest", "This is", "Note that", "What
  matters is that", "In the following", "As a result", "The idea is", "Up to now", "So far".
  Not "It is worth noting that", "Notably", "Crucially", "Importantly", "Interestingly".
- **Pronouns.** "We" by default; "you" for the reader's choice and for what they see; "one"
  occasionally ("one can measure"). Never "I", never "the user".
- **Hedges.** "in principle", "roughly", "at least", "in practice", "effectively" (sparingly).
  Not "arguably", "to some extent", "it could be argued".
- **Certainty.** When something is certain, say so plainly: "regardless of", "exactly",
  "just by", "the only thing you need", "always opens up a gap".
- **Causal connectors.** "the reason for this is", "this is because", "as a result", "and
  therefore", "so that".
- **Sequencing.** "first, then, and finally"; "so far, now"; "up to now we have focused on,
  in the following we".
- **Contractions.** None. "Do not", "it is", "let us".
- **Punctuation.** Never an em-dash, never a spaced double hyphen as one. A hyphen in a
  compound modifier is a hyphen; a numeric range is written with "to" (`nk=60` to `nk=140`).
  No exclamation marks.
- **Numbers and code.** Mathematics is typeset ($t_2=0.2$, $\nu=0.5$); a keyword and its
  value are code (`nk=100`, `mode="KPM"`). The prose says what the number does, the code says
  what it is.
- **Method names in prose.** `h.get_bands()`, `g.get_hamiltonian()`, with the object and the
  parentheses, so the reader can search for the reference entry. The maintainer's notebook
  habit of a bare name in parentheses, "(add_rashba)", is for quiz bullets, not for the guide.

---

## 7. Lexicon

Preferred words and constructions, with the alternatives a model would reach for instead.
The first block is the maintainer's vocabulary as it appears in the lectures and the
notebooks; the second block is what the guide adds or decides differently, each with its
reason.

| Use | Instead of | Example |
|---|---|---|
| rationalize | explain, make sense of | "how the phenomenology can be rationalized in terms of the band structure" |
| account for | describe, capture, model | "single-particle physics accounts for much of the phenomena" |
| give rise to, driven by | lead to, produce, induce, cause | "the interaction gives rise to a gap opening"; "gap opening driven by interactions" |
| phenomenology | behaviour, physics, features | "Do you observe the same phenomenology?" |
| emergence, appear, emerge | arise (fine too), manifest | "the emergence of a local moment at each site"; "a gap appears" |
| quantum many-body (as adjective) | strongly correlated, many-particle | "a quantum many-body Hamiltonian" |
| single-particle | one-body, free, non-interacting (used too, less) | "an effective single-particle Hamiltonian" |
| quartic term, four-fermion term | two-body term | "the quartic term is what makes the problem hard" |
| bilinear | quadratic | "the mean field turns the quartic term into a bilinear one" |
| creation and annihilation operators, field operators (always in that order) | ladder operators, second-quantized operators | "$c^\dagger_n$ creates an electron at site $n$" |
| mean-field theory, mean-field Hamiltonian, at the mean-field level | Hartree-Fock (rarely), self-consistent field | "interactions are treated at the mean-field level" |
| self-consistent, solve it self-consistently, self-consistency | iterate to convergence, fixed-point iteration | "the mean-field Hamiltonian depends on the ground state, so the problem is self-consistent" |
| hopping, first-neighbor hopping, hopping between site $i$ and site $j$ | tunneling amplitude, transfer integral | "a hopping to second neighbors bends the band" |
| the Hubbard interaction, the onsite $U$, onsite repulsion | local Coulomb term | "the onsite interaction $U$" |
| sites, site $n$ | atoms (unless a real material), orbitals (only when orbitals are meant) | "two sites connected by a hopping" |
| sublattice imbalance | staggered potential, mass term, Semenoff mass | "the sublattice imbalance opens a gap" |
| electronic structure (umbrella), band structure, dispersion, spectral function, density of states, local density of states | spectrum (used too), DOS profile, LDOS map | "what happens to the electronic structure and why" |
| a gap opens up, gap opening, the gap closes | gapping out, gap formation, gaps the spectrum | "a gap opens up regardless of the chemical potential" |
| reciprocal space, Brillouin zone | momentum space (acceptable), k-space | "opens up gaps in some parts of reciprocal space" |
| in the bulk, at the edge, edge states, in-gap states, zero modes | boundary modes, midgap states | "a gap in the bulk and chiral states at the edges" |
| correlated state, symmetry-broken state, order parameter | ordered phase, broken-symmetry phase | "the superconducting order parameter" |
| topological invariant, non-trivial, protected, robust | topologically nontrivial index, immune | "these edge states are protected" |
| chiral, helical (edge states) | unidirectional, counter-propagating spin-polarized | "helical states at the edges" |
| the thermodynamic limit, the non-interacting limit, the strongly interacting limit, at half filling, in units of $t$ | the bulk limit, weak/strong coupling (used less) | "$U=0$ is the non-interacting limit" |
| exotic, unconventional, sophisticated, non-trivial | novel, cutting-edge, remarkable, elegant | "unconventional superconductors" |
| STM, atomically resolved spectroscopy, ARPES, transport, Hall conductivity | experimental probes, spectroscopic measurements | "this is what one measures with STM" |
| "the way we do this is by", "the reason for this is", "just by", "regardless of", "as long as", "for concreteness", "as a reference" | "by virtue of", "owing to", "notwithstanding", "for the sake of clarity" | |
| "let us now look at", "we will now see how", "here we will show" | "in this section we discuss", "we now turn to", "in this notebook we explore" | |

| Use | Instead of | Why |
|---|---|---|
| onsite energy, onsite interaction, the onsite $U$ | on-site | The method is `h.add_onsite()`, the maintainer's notebook text says "the onsite interaction U", and the guide had 49 "onsite" to no "on-site" |
| mean-field Hamiltonian, tight-binding model, self-consistent calculation, first-neighbor hopping, spin-orbit coupling | mean field Hamiltonian, selfconsistent, tight binding | The written register hyphenates a compound modifier; the bare noun stays open ("the mean field" is fine) |
| band structure | bandstructure | 30 to 1 in the guide; the notebooks' "bandstructure" is the speech spelling |
| Zeeman field | | The term $\vec B\cdot\vec\sigma$ when it is an external field acting on the spin, `h.add_zeeman()` |
| exchange field, magnetic order, magnetization | | The same term when it is the material's own spin splitting, `h.add_exchange()`; the two methods add the same matrix, and the word follows the physics being described |
| spinful, spinless | spin-polarized, with/without spin | pyqula's own words (`has_spin`), and the maintainer's |
| Nambu basis, Nambu spinor, Bogoliubov-de Gennes (BdG) Hamiltonian | particle-hole basis | Spell out Bogoliubov-de Gennes once per chapter, then BdG |
| electron states and hole states, the electron copy and the hole copy | quasiparticle and quasihole branches | "the band structure shows both the electron and the hole states" |
| filling $\nu$, half filling, the Fermi energy, the chemical potential | electron density, doping level | `filling=0.5` is half filling; the Fermi energy is at zero after `h.set_filling()` |
| k-points, k-mesh, k-path, `nk` | momentum grid, sampling density | The API says `nk` and `kpath`; the prose says k-mesh and k-path; "Brillouin zone" is spelled out |
| broadening $\delta$ | smearing, Lorentzian width | The keyword is `delta`; say what it hides when it matters |
| in units of the hopping $t$, in units of the lattice constant | dimensionless, natural units | pyqula sets $t=1$ and the first-neighbor distance to 1; say it once per chapter where a number depends on it |
| a runnable version, an executed notebook | a demo, a worked example | The pointer sentence: "See `examples/...` for a runnable version" |
| modifies the Hamiltonian in place | mutates, returns a new object | The `add_*` methods change `h` in place; the `get_*` methods return a new object and leave `h` alone |

Words that do not appear in the maintainer's prose and should not appear in the guide: delve,
unpack, showcase, highlight (as a verb), underscore, elegant, intriguing, nuanced, robustly,
seamlessly, holistic, landscape (except "energy landscape", which the maintainer uses),
journey, dive, leverage, "under the hood", "out of the box", "battle-tested", "first-class".
Near-absent, and not to be added: beautiful, remarkable.

---

## 8. What not to do

The defaults of a language model that read wrong next to the maintainer's prose. The first
fifteen are the ones found in the course material; the guide adds six of its own.

1. **Em-dashes.** None, anywhere. Rewrite with a comma, colon, semicolon or parentheses.
2. **Appeals to authority.** No "well-known", "standard textbook result", "any standard
   textbook", "the literature", "it is easy to see". State the result.
3. **Year citations and narrative attributions.** No "(1931)", no "first proposed by X, later
   extended by Y". Field eponyms (Haldane model, Kane-Mele model, RKKY interaction, Sancho-Rubio
   decimation, Bethe-Salpeter equation) are vocabulary and stay.
4. **Signposting adverbs.** No "It is worth noting that", "Notably", "Crucially",
   "Importantly", "Interestingly", "Remarkably" as sentence openers. The maintainer's versions
   are "What is important is that", "Note that", "What matters is that", "The interesting
   point is that".
5. **Definition-first paragraphs.** Not "Quasiparticle interference (QPI) maps the
   momentum-space scattering pattern that a defect produces." Build it: the defect scatters,
   the states interfere, the LDOS is modulated, its Fourier transform is the pattern, then
   "this is what is called quasiparticle interference" (section 9).
6. **Label-colon bullets in explanations.** Not "**Key insight:** ..." or "**Physical
   picture:** ...". Bullets are for the argument catalogue and for lists of inputs;
   everything else is prose.
7. **Generic headings.** Not "Introduction", "Overview", "Summary", "Key takeaways". The
   guide's headings name the physics or the quantity ("Including an onsite energy", "Spin
   splitting of an altermagnet").
8. **Compressed encyclopedic sentences.** Not "The half-filled honeycomb Hubbard model
   undergoes a Mott transition at $U_c\approx 2.3t$ within mean-field theory." Narrate: the
   density of states vanishes at the Dirac point, so a small interaction does nothing, and only
   when it becomes large enough does the gap open; the value comes out around 2.3.
9. **Over-polished instructions.** Not "Investigate the dependence of the spectral gap on the
   interaction strength." Write "Increase the interaction. What happens to the gap, and why?"
   (in the rare places the guide asks the reader to try something).
10. **Invented structure.** No closing summary on a chapter that has none, no "as a reminder"
    on a section that recalls nothing, no claim about a method that its code does not support.
11. **Formal hedging.** Not "arguably", "to some extent", "it could be argued". Use "in
    principle", "roughly", "at least", "in practice".
12. **Contractions.** "let's", "don't", "we'll" are speech. The guide says "let us", "do not",
    "we will".
13. **Exclamation marks and emojis.** Never.
14. **Rhetorical flourishes.** Not "full stop", "the skeleton the rest hangs on", "a rich zoo".
    When in doubt, use the plainer phrase.
15. **Reframing the reader.** Not "Consider a reader unfamiliar with..." or "For the
    uninitiated". The reader is a colleague who has seen materials physics and quantum
    mechanics.
16. **Maintainer material.** No `src/` paths, test names, benchmark numbers, registry
    mechanics or fix histories (section 4). If a caveat needs a number to be useful ("the
    `nk=140` above takes minutes; `nk=60` takes about 40 seconds") the number stays and the
    profiling that produced it does not.
17. **A section that is only an argument list.** Every method gets its physics sentence before
    the code, even a one-keyword subsection.
18. **Promising a figure.** The snippets stop at the arrays. Say what to look for in them and
    point at the script and the notebook that plot them; do not write "as shown in the figure".
19. **Re-voicing the maintainer's own sentences.** In the canonical chapters (section 1) the
    opening sentence of a chapter or subsection may be rewritten into the intro register,
    since it is the sentence a reader lands on and the one that names the method. Everything
    after it is kept: the sentences that carry an equation or lead into a code block, and the
    "Possible inputs" bullets, get fixes only (a typo, a dropped article, an agreement, a
    spelling decision from section 7), and the build-up and the reading of the result are
    added around them rather than written over them.
20. **Touching the code while rewriting the prose.** The blocks are copied by the reader and
    tested only for names. A prose pass leaves every block byte for byte, and a change to a
    block is its own commit with its own reason.
21. **Hedging about what the code does.** Not "should return", "may raise", "is expected to".
    Run it or read it, then state it: "returns the k-points and one energy per band", "raises
    `ValueError` naming the accepted modes".

---

## 9. Before and after

A rewrite of one Claude-written guide paragraph into the voice. The "Before" block is the only
place in this file where a definition-first paragraph is shown on purpose.

Before (the Quasiparticle interference section at `d30babe`):

> Quasiparticle interference (QPI) maps the momentum-space scattering pattern that a defect or
> impurity produces, and is what an STM quasiparticle-interference measurement probes.
> `h.get_qpi()` is only available for 2D Hamiltonians; unlike the other observables here it
> does not return arrays. It writes its output to disk, one file per energy in an output
> folder (default `MULTIQPI/`) plus a combined `DOS.OUT`

After:

> Let us now see what a single defect does to the local density of states of a two-dimensional
> system, with `h.get_qpi()`. A defect scatters an electron from a state at momentum $\vec k$
> to a state at $\vec k'$ on the same constant-energy contour, and the two interfere, so that
> the local density of states around the defect is modulated with wavevector
> $\vec q = \vec k - \vec k'$. The Fourier transform of that modulation at each energy is what
> is called the quasiparticle interference pattern, and it is what an STM measurement obtains
> by Fourier transforming a conductance map: a picture of the constant-energy contours of the
> band structure, and of which pairs of states the defect connects. `h.get_qpi()` computes it
> for 2D Hamiltonians; unlike the other observables here it does not return arrays but writes
> one file per energy to a folder (`MULTIQPI/` by default) and the density of states next to
> it in `DOS.OUT`

Note what changed: the object is built (scattering, interference, modulation, Fourier
transform) before it is named; the method name still appears in the first sentence; the STM
link says what the measurement actually does; the output convention is kept because the reader
needs it; nothing was added that the section's own arguments (`mode="pm"` autoconvolves the
spectral weight, `mode="response"` uses the clean band structure) do not support.

And the other direction, a catalogue bullet that had grown narration. Before (the `nk` bullet
of the density of states section):

> nk: number of k-points in the mesh, for `"ED"` and `"KPM"`. `"adaptive"` does not sample a
> mesh at all. It integrates over the Brillouin zone with error-controlled quadrature, tuned by
> `error=1e-1`, and reads `nk` only as a subdivision limit. `"Green"`/`"RG"` pass `nk` to the
> Brillouin-zone sum behind the self-energy, where it matters for `gmode="full"` and (in 2D)
> `gmode="renormalization"` but not for the default `gmode="adaptive"`

After, the bullet keeps one line and the rest becomes a chapter-register paragraph under the
list:

> - nk: number of k-points per direction of the mesh, for `"ED"` and `"KPM"`
>
> The two remaining modes read `nk` differently. `"adaptive"` does not sample a mesh at all:
> it integrates over the Brillouin zone with an error-controlled quadrature, tuned by
> `error=1e-1`, and uses `nk` only as a limit on how far it subdivides. `"Green"` and `"RG"`
> pass `nk` to the Brillouin-zone sum behind the self-energy, where it matters for
> `gmode="full"` and, in two dimensions, for `gmode="renormalization"`, but not for the default
> `gmode="adaptive"`.

---

## 10. Checklist before finishing

- [ ] Register chosen per paragraph (chapter prose, intro or pointer, catalogue) and not mixed
      inside one; catalogue bullets are one line each.
- [ ] Each subsection opens by saying what we will now see and names its method before the
      code; each Hamiltonian or definition is followed by its terms in words.
- [ ] Every snippet is preceded by a physics sentence, followed by what its arrays contain and
      what to look for, and the code inside the block is byte for byte what it was (or the
      change is its own commit).
- [ ] At least one link per section to a measurement, a runnable script or an executed
      notebook, when one exists.
- [ ] Not a single em-dash or spaced double hyphen; no contractions; no "worth noting",
      "notably", "crucially", "well-known", "textbook", "literature", "delve", "leverage"; no
      exclamation marks; no year citations.
- [ ] Spellings follow section 7: onsite, mean-field (modifier), self-consistent,
      tight-binding (modifier), band structure; compound modifiers hyphenated.
- [ ] The maintainer's chapters (section 1) were fixed, not re-voiced, beyond their opening
      sentences.
- [ ] No maintainer material came back in (paths into `src/`, benchmark tables, registry
      mechanics, fix histories).
- [ ] `python -m pytest tests/documentation` passes, run without a pipe.
- [ ] Every `](#...)` anchor in the table of contents still resolves to a heading, and the word
      delta of the pass is reported and defensible: `d30babe` took the guide from 43,512 to
      36,462 words; a canonical chapter grows by its build-up (the pilot chapter went from 675
      to 1,578 words), and a Claude-written chapter should net shrink as the narration leaves
      the bullets, so the guide as a whole does not grow back to where it was.
- [ ] `documentation/convert.sh` was re-run so the PDF matches the Markdown.

---

## 11. Working on the guide

- Rewrite chapter by chapter, and show the first one to the maintainer before doing the rest.
- Read the whole chapter before editing a section of it: sections inherit variables from
  their neighbours inside a chapter (the static test allows a block to use an `h` built two
  blocks earlier), so moving or deleting a block breaks the one after it.
- Grep the finished text, not the diff, for the banned phrases and the split spellings; a
  phrase that survived from an untouched paragraph is still in the guide.
- Count the words before and after (`wc -w documentation/user_guide.md`), and the phrases in
  the calibration table, and put both numbers in the commit message.
- The reference chapter is the last thing to touch, and only for spellings and fixes; its
  shape is not a voice question.
