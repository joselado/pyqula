# Figures of the user guide

Each figure of the user guide is one script here, `<id>.py`, named after its
page (`hamiltonian_landau.py` is on `hamiltonian.md`). The script runs the
calculation with pyqula and plots it, and writes
`documentation/guide/img/<id>.png` (light theme) and `<id>-dark.png` (dark
theme). The website shows the script in a collapsed block under the figure,
so the script is written for a reader: the calculation is the snippet of the
page, with its comments, and the plotting is a few lines after it.

```python
import style
from pyqula import geometry

g = geometry.chain() # geometry of the 1D chain
h = g.get_hamiltonian() # generate the Hamiltonian
(k,e) = h.get_bands() # compute band structure

def draw(ax, c):
    ax.scatter(k, e, s=8, color=c.blue)
    ax.set_xlabel("momentum $k$")
    ax.set_ylabel("energy [$t$]")

style.save(draw, "hamiltonian_chain")
```

`style.save()` calls `draw(ax, c)` once for each theme, so the calculation
runs once. `c` holds the colors of the theme (`c.blue`, `c.orange`, `c.green`,
`c.ink`, `c.ink2`, `c.cmap` for a one-signed heat map, `c.diverging` for a
signed one, gray at zero). With `npanels=2` or `3`, `ax` is a list of axes.
`style.colorbar(mappable, ax, label)` adds a colorbar in the ink of the theme,
and `style.kpath(ax, k, labels, at=None)` puts the high-symmetry labels on the
k-axis of a band structure, evenly spaced unless `at` gives their positions as
fractions of the path (the default 2D path of the honeycomb and triangular
lattices is $\Gamma$, K, M, K', $\Gamma$ at `at=[0,1/3,1/2,2/3,1]`). Importing `style` switches off pyqula's output files, so a
script writes only its two images.

## Conventions

- One figure per section that computes an observable. A section that only
  sets a keyword or a switch has none.
- A heat map whenever the physics has one: a spectral function over
  $(k,\omega)$, an LDOS over real space, a Berry curvature over the Brillouin
  zone, or bands colored by an operator (`h.get_bands(operator="sz")`). Use
  `c.cmap` for a quantity of one sign and `c.diverging` with symmetric limits
  for a signed one.
- Energies in units of the hopping, `energy [$t$]`; axis labels in words with
  the symbol after them; the font sizes are set by `style`, do not override
  them.
- Keep the parameters of the page. When the figure needs another one (a
  larger system, a field along $z$ so that the color can show $S_z$), the
  paragraph under the figure says so.
- A script runs in a minute at most on a laptop. Use the smallest system and
  k-mesh that show the physics.
- The colors are the ones of the dmrgpy guide, checked for color blindness
  in both themes; do not pick new ones.

## Add a figure to a page

````markdown
![One sentence saying what is plotted.](img/<id>.png#only-light)
![One sentence saying what is plotted.](img/<id>-dark.png#only-dark)

The paragraph that reads the figure, with the numbers it shows.

??? example "Code of this figure (documentation/figures/<id>.py)"

    ```python
    --8<-- "<id>.py"
    ```
````

`pymdownx.snippets` copies the script in at build time, so the page always
shows the code that is in git, and a wrong `<id>` fails `mkdocs build`. The
PDF (`documentation/build_guide_pdf.py`) keeps the light image and prints the
path of the script instead of its code. The static snippet test
(`tests/documentation`) parses the included scripts too.

## Recompute

```bash
bash documentation/figures/run_all.sh               # every figure
python3 documentation/figures/hamiltonian_landau.py # one figure
```
