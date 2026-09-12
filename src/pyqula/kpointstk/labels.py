import numpy as np


# The high-symmetry point labels live in a registry (label -> builder)
# rather than in an if/elif chain, so known_labels below is derived from
# the dispatch instead of being a second list kept in sync by hand, and
# adding a label is one dict entry. Every builder takes the geometry and
# returns the kpoint in reduced coordinates.


def _K(g):
    """Corner of the hexagonal Brillouin zone"""
    from .locate import target_moduli
    from .mapping import unitary
    vm = np.sqrt(3) # moduli
    k = g.k2K(target_moduli(unitary(g.b1),unitary(g.b2),vm))
    return k*2/(3.*np.sqrt(3))


# label -> builder(g). Every label above Z lies in the k3=0 plane; the
# three-dimensional ones were added because a 3d path used to be
# impossible to write down.
_labels = {
  "G":  lambda g: [0.,0.,0.],
  "M":  lambda g: [0.5,0.,0.],
  "K":  _K,
  "K'": lambda g: -_K(g),
  "M1": lambda g: [.5,.0,.0],
  "M2": lambda g: [.0,.5,.0],
  "M3": lambda g: [.5,.5,.0],
  "X":  lambda g: [.5,.0,.0],
  "Y":  lambda g: [.0,.5,.0],
  "Z":  lambda g: [.0,.0,.5], # three dimensional high symmetry points
  "R":  lambda g: [.5,.5,.5],
  "A":  lambda g: [.5,.0,.5],
  "B":  lambda g: [.0,.5,.5],
  }


known_labels = list(_labels) # derived from the dispatch, never by hand


def get_label_names():
    """Return every high-symmetry kpoint label that label2k accepts"""
    return list(_labels)


def label2k(g,kl):
    """Given a kpoint label, return the kpoint"""
    if kl not in _labels:
        raise ValueError("Unrecognized kpoint label '"+str(kl)+"'. Known "
          +"labels: "+str(known_labels)+", or give the kpoint directly as "
          +"a list of three reduced coordinates")
    return _labels[kl](g)
