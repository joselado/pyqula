"""Shared matplotlib style of the user-guide figures.

Every documentation/figures/<id>.py computes with pyqula at module level and
then hands a drawing function to save():

    import style
    ...                                 # the calculation, as on the page

    def draw(ax, c):                    # c: colors of the current theme
        ax.plot(x, y, color=c.blue)
        ax.set_xlabel("energy")

    style.save(draw, "<id>")            # writes img/<id>.png and <id>-dark.png

save() calls draw once per theme, so the calculation runs only once. For
several panels pass npanels=2 or 3, and draw receives a list of axes.

The PNG files go to documentation/guide/img/. The background is transparent
and the ink switches with the theme, so the figure sits on the page color.
The categorical colors are the ones of the dmrgpy guide, checked for color
blindness in both themes; heat maps use `c.cmap` (one-signed quantities,
such as a density of states) or `c.diverging` (signed quantities, such as a
Berry curvature or a magnetization, with a neutral gray at zero).

Importing this module switches off the output files of pyqula, so that
running a figure writes nothing but the two PNG files.
"""
import os
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(os.path.dirname(HERE), "guide", "img")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))

from pyqula import filewrite  # noqa: E402
filewrite.set_write(False)

INK = {"light": "#0b0b0b", "dark": "#f0efec"}
INK2 = {"light": "#52514e", "dark": "#c3c2b7"}
NAMES = ["blue", "orange", "green", "yellow", "magenta", "darkgreen",
         "violet", "red"]
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500",
             "#d55181", "#008300", "#9085e9", "#e66767"],
}
SIZES = {1: (6.0, 4.0), 2: (10.0, 4.0), 3: (13.5, 4.0)}
DPI = 150


# blue, neutral gray at zero, red; the gray is visible on the page of
# each theme, so a mostly-zero map is not a white block on the dark page
# and a state with zero weight does not disappear on the light one
DIVERGING = {
    "light": ["#1f5fa8", "#6fa3d6", "#d9d9d6", "#e8836e", "#b2182b"],
    "dark": ["#4f9be8", "#2f5f8f", "#4a4a48", "#9a4535", "#ef6f5a"],
}


def colors(theme):
    """Colors of a theme, as attributes: c.blue, c.ink, c.cmap, ..."""
    from matplotlib.colors import LinearSegmentedColormap
    c = dict(zip(NAMES, SERIES[theme]))
    diverging = LinearSegmentedColormap.from_list("diverging-" + theme,
                                                  DIVERGING[theme])
    c.update(series=SERIES[theme], ink=INK[theme], ink2=INK2[theme],
             cmap="inferno", diverging=diverging, theme=theme)
    return SimpleNamespace(**c)


def apply(theme):
    """Set the rcParams of a theme."""
    ink, ink2 = INK[theme], INK2[theme]
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": ink,
        "axes.labelcolor": ink,
        "axes.titlecolor": ink,
        "axes.facecolor": "none",
        "figure.facecolor": "none",
        "savefig.facecolor": "none",
        "text.color": ink,
        "xtick.color": ink,
        "ytick.color": ink,
        "xtick.labelcolor": ink2,
        "ytick.labelcolor": ink2,
        "legend.labelcolor": ink,
        "axes.prop_cycle": plt.cycler(color=SERIES[theme]),
        "image.cmap": "inferno",
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "mathtext.fontset": "dejavusans",
    })


def colorbar(mappable, ax, label):
    """A colorbar next to ax, with its outline and ticks in the ink color."""
    ink = plt.rcParams["axes.edgecolor"]
    cb = ax.figure.colorbar(mappable, ax=ax)
    cb.set_label(label, color=ink)
    cb.outline.set_edgecolor(ink)
    cb.ax.tick_params(colors=ink)
    return cb


def kpath(ax, k, labels, at=None):
    """Label a k-path with its high-symmetry points.

    `k` is the array returned by h.get_bands(), labels for example
    ["$\\Gamma$", "K", "M", "$\\Gamma$"]. `at` gives the position of each
    label as a fraction of the path; without it the labels are evenly
    spaced. The default path of a 2D lattice is not evenly spaced: on the
    honeycomb and triangular lattices it is Gamma, K, M, K', Gamma at
    at=[0, 1/3, 1/2, 2/3, 1]. Draws a thin vertical line at each inner point.
    """
    import numpy as np
    if at is None:
        at = np.linspace(0, 1, len(labels))
    ticks = np.min(k) + (np.max(k) - np.min(k))*np.array(at)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(ticks[0], ticks[-1])
    for t in ticks[1:-1]:
        ax.axvline(t, color=plt.rcParams["axes.edgecolor"], lw=0.6, alpha=0.4)


def save(draw, figid, npanels=1, **kwargs):
    """Call draw(ax, c) for the light and the dark theme, write both PNGs."""
    os.makedirs(IMG, exist_ok=True)
    for theme in ["light", "dark"]:
        apply(theme)
        fig, ax = plt.subplots(1, npanels, figsize=SIZES[npanels],
                               layout="constrained", **kwargs)
        draw(ax, colors(theme))
        name = figid + ("-dark" if theme == "dark" else "") + ".png"
        fig.savefig(os.path.join(IMG, name), dpi=DPI, transparent=True)
        plt.close(fig)
        print("wrote", os.path.join(IMG, name))
