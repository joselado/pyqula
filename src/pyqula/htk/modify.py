def modify_hamiltonian_matrices(self,f):
  """Apply a certain function to all the matrices"""
  self.intra = f(self.intra)
  if self.is_multicell: # for multicell hamiltonians
    for i in range(len(self.hopping)): # loop over hoppings
      self.hopping[i].m = f(self.hopping[i].m) # put in nambu form
  else: # conventional way
    if self.dimensionality==0: pass # one dimensional systems
    elif self.dimensionality==1: # one dimensional systems
      self.inter = f(self.inter)
    elif self.dimensionality==2: # two dimensional systems
      self.tx = f(self.tx)
      self.ty = f(self.ty)
      self.txy = f(self.txy)
      self.txmy = f(self.txmy)
    else: raise

