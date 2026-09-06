import numpy as np

# The named operators are stored in a registry (name -> builder) rather than
# in an if/elif chain, so adding an operator is a single entry and the set of
# valid names can be listed (get_operator_names) and reported in the error
# message when an unknown name is given. Every builder takes the Hamiltonian
# as its first argument and swallows **kwargs, forwarding only the ones the
# underlying routine accepts -- that is what the old chain did, and callers
# rely on passing a generic kwargs bag to h.get_operator.


_registry = None # built on first use, see _operators


def _operators():
    """Return the registry of named operators, name -> builder(h,**kwargs)

    Built once and cached: the builders close over nothing Hamiltonian-
    specific, and get_operator is called inside k-point loops.
    """
    global _registry
    if _registry is not None: return _registry
    from . import operators
    reg = {
      "berry": lambda h,**kw: operators.get_berry(h,**kw),
      "site": lambda h,**kw: operators.get_site(h,**kw),
      "angularmomenta": lambda h,**kw: operators.get_angular_momenta(h,**kw),
      "correlator": lambda h,**kw: operators.get_correlator_ij(h,**kw),
      "valleyberry": lambda h,**kw: operators.get_operator_berry(h,"valley",**kw),
      "szvalleyberry": lambda h,**kw: operators.get_operator_berry(h,"sz_valley",**kw),
      "szberry": lambda h,**kw: operators.get_sz_berry(h,**kw),
      "sx": lambda h,**kw: operators.get_sx(h),
      "sy": lambda h,**kw: operators.get_sy(h),
      "sz": lambda h,**kw: operators.get_sz(h),
      "location": lambda h,**kw: operators.get_location(h,**kw),
      "current": lambda h,**kw: operators.get_current(h),
      "energy": lambda h,**kw: operators.Operator(h),
      "bulk": lambda h,**kw: operators.get_bulk(h,**kw),
      "surface": lambda h,**kw: operators.get_surface(h,**kw),
      "velocity": lambda h,**kw: operators.get_velocity(h),
      "sublattice": lambda h,**kw: operators.get_sublattice(h,mode="both"),
      "sublatticeA": lambda h,**kw: operators.get_sublattice(h,mode="A"),
      "sublatticeB": lambda h,**kw: operators.get_sublattice(h,mode="B"),
      "interface": lambda h,**kw: operators.get_interface(h),
      "spair": lambda h,**kw: operators.get_pairing(h,ptype="s"),
      "deltax": lambda h,**kw: operators.get_pairing(h,ptype="deltax"),
      "deltay": lambda h,**kw: operators.get_pairing(h,ptype="deltay"),
      "deltaz": lambda h,**kw: operators.get_pairing(h,ptype="deltaz"),
      "electron": lambda h,**kw: operators.get_electron(h),
      "hole": lambda h,**kw: operators.get_hole(h),
      "tauz": lambda h,**kw: operators.get_tauz(h),
      "up": lambda h,**kw: operators.get_up(h),
      "dn": lambda h,**kw: operators.get_dn(h),
      "xposition": lambda h,**kw: operators.get_xposition(h),
      "yposition": lambda h,**kw: operators.get_yposition(h),
      "zposition": lambda h,**kw: operators.get_zposition(h),
      "layer": lambda h,**kw: operators.get_layer(h,**kw),
      "valley": lambda h,**kw: operators.get_valley(h,**kw),
      "valley_x": lambda h,**kw: operators.get_valley_taux(h,**kw),
      "valley_y": lambda h,**kw: operators.get_valley_tauy(h,**kw),
      "valley_upper": lambda h,**kw: operators.get_valley_layer(h,n=-1),
      "valley_lower": lambda h,**kw: operators.get_valley_layer(h,n=0),
      "ipr": lambda h,**kw: operators.ipr,
      "potential": lambda h,**kw: operators.get_potential(h,**kw),
      # magnetization: the spin operator restricted to the electron sector
      "mx": lambda h,**kw: h.get_operator("sx")@h.get_operator("electron"),
      "my": lambda h,**kw: h.get_operator("sy")@h.get_operator("electron"),
      "mz": lambda h,**kw: h.get_operator("sz")@h.get_operator("electron"),
      # operators living outside operators.py
      "mass": lambda h,**kw: _mass(h,**kw),
      "ldos": lambda h,**kw: _ldos(h,**kw),
      "unfold": lambda h,**kw: _unfold(h,**kw),
      "singlet": lambda h,**kw: _singlet(h),
      }
    for (name,alias) in _aliases.items(): reg[name] = reg[alias]
    _registry = reg
    return reg


# alternative spellings, alias -> canonical name in the registry above
_aliases = {
        "Berry": "berry",
        "Sx": "sx", "Sy": "sy", "Sz": "sz",
        "Bulk": "bulk",
        "Surface": "surface", "edge": "surface", "Edge": "surface",
        "electrons": "electron",
        "down": "dn",
        "valley_top": "valley_upper", "valley_bottom": "valley_lower",
        "IPR": "ipr",
        }


def _mass(h,**kwargs):
    from .mass import mass_operator
    return mass_operator(h,**kwargs)


def _ldos(h,**kwargs):
    from . import ldos
    return ldos.ldos_projector(h,**kwargs)


def _unfold(h,**kwargs):
    from .unfolding import bloch_projector
    return bloch_projector(h,**kwargs)


def _singlet(h):
    from .sctk.operator import real_singlet
    return real_singlet(h)


def get_operator_names():
    """Return the sorted list of the names accepted by get_operator"""
    return sorted(_operators())


def get_operator(self,name,**kwargs):
      """Return the conventional operator"""
      from . import operators
      from . import potentials
      from .hamiltonians import Hamiltonian
      if type(name) is operators.Operator: return name # return operator
      if isinstance(name, Hamiltonian): # if input is an operator
          return operators.Operator(name) # return operator
      if type(name) is potentials.Potential or callable(name):
          out = self.copy()*0. # initialize
          out.add_onsite(name) # add onsite
          return operators.Operator(out.intra,linear=True) # return operator
      from . import algebra
      if algebra.ismatrix(name): return operators.Operator(name) # raw matrix
      if name is None: return None # return operator
      if name=="None": return None
      reg = _operators() # registry of named operators
      if name in reg: return reg[name](self,**kwargs) # build it
      if self.has_kondo: # heavy-fermion operators are named elsewhere
          from .specialhamiltoniantk import heavyfermion
          return heavyfermion.get_operator(self,name,**kwargs)
      raise ValueError("unknown operator "+str(name)+
              "; get_operator accepts an Operator, a Hamiltonian, a "
              "Potential, a callable, a matrix, None, or one of the names "
              +str(get_operator_names()))


get_scalar_operator = get_operator
