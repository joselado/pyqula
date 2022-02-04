def modify_hamiltonian_matrices(self,f0,use_geometry=False):
    """Apply a certain function to all the matrices"""
    # wrapper function
    if use_geometry: # use geometry
        g = self.geometry # get the geometry
        def f(m,dr): 
            return f0(m,g.r,g.replicas(d=dr)) # use r1 and r2
    else:
        def f(m,dr): return f0(m) # do not use geometry
    # apply the function
    self.intra = f(self.intra,[0,0,0]) # modify intracell
    if self.is_multicell: # for multicell hamiltonians
      for i in range(len(self.hopping)): # loop over hoppings
        # modify Hamiltonian matrix
        self.hopping[i].m = f(self.hopping[i].m,self.hopping[i].dir) 
    else: # conventional way
      if self.dimensionality==0: pass # one dimensional systems
      elif self.dimensionality==1: # one dimensional systems
        self.inter = f(self.inter,[1,0,0])
      elif self.dimensionality==2: # two dimensional systems
        self.tx = f(self.tx,[1,0,0])
        self.ty = f(self.ty,[0,1,0])
        self.txy = f(self.txy,[1,1,0])
        self.txmy = f(self.txmy,[1,-1,0])
      else: raise

