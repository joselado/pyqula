


def clean_hamiltonian(h):
  """Set a Hamiltonian to zero"""
  h.intra *= 0.0
  if h.is_multicell:
    for t in h.hopping: t.m *= 0.0
  else:
    if h.dimensionality==0: pass
    elif h.dimensionality==1: h.inter *= 0.0
    elif h.dimensionality==2: 
      h.tx *= 0.0
      h.ty *= 0.0
      h.txy *= 0.0
      h.txmy *= 0.0
    elif h.dimensionality==3: raise

