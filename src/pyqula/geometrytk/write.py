import numpy as np



def write_xyz(gin,output_file = "crystal.xyz",units=0.529,nsuper=1):
  """Writes the geometry associatted with a hamiltonian in a file"""
  g = gin.copy() # copy geometry
#  if g.atoms_have_names: g = remove_duplicated(g)
#  else: g = gin.copy()
#  g = gin.copy()
  if g.dimensionality>0: # create supercell
    if nsuper>1:
      g = g.supercell(nsuper)
  x = g.x*units  # x posiions
  y = g.y*units  # y posiions
  z = g.z*units  # z posiions
  fg = open(output_file,"w")
  # get hte names of the atoms
  if g.atoms_have_names:
    names = g.atoms_names
  else:
    names = ["C" for ix in x ] # create names
  # check that there are as many names as positions
  if len(names)!=len(x): raise
  fg.write(str(len(x))+"\nGenerated with Python\n") # number of atoms
  for (n,ix,iy,iz) in zip(names,x,y,z):
    fg.write(n+"   "+str(ix)+ "     "+str(iy)+"   "+str(iz)+"  \n")
  fg.close()






def write_lattice(g,output_file = "LATTICE.OUT"):
  """Writes the lattice in a separte file"""
  open("DIMENSIONALITY.OUT","w").write(str(g.dimensionality))
  if g.dimensionality==0: return
  fg = open(output_file,"w")
  if g.dimensionality==1:
    fg.write(str(g.a1[0])+"   "+str(g.a1[1])+"  "+str(g.a1[2])+"\n")
  elif g.dimensionality==2:
    fg.write(str(g.a1[0])+"   "+str(g.a1[1])+"  "+str(g.a1[2])+"\n")
    fg.write(str(g.a2[0])+"   "+str(g.a2[1])+"  "+str(g.a2[2])+"\n")
  elif g.dimensionality==3:
    fg.write(str(g.a1[0])+"   "+str(g.a1[1])+"  "+str(g.a1[2])+"\n")
    fg.write(str(g.a2[0])+"   "+str(g.a2[1])+"  "+str(g.a2[2])+"\n")
    fg.write(str(g.a3[0])+"   "+str(g.a3[1])+"  "+str(g.a3[2])+"\n")
  else: raise
  fg.close()



def write_sublattice(g,output_file = "SUBLATTICE.OUT"):
  """Writes the geometry associatted with a hamiltonian in a file"""
  if not g.has_sublattice: return
  fg = open(output_file,"w")
  for i in g.sublattice:
    fg.write(str(i)+"  \n")
  fg.close()



def write_positions(g,output_file = "POSITIONS.OUT",nrep=None):
  """Writes the geometry associatted with a hamiltonian in a file"""
  if nrep is not None: g = g.get_supercell(nrep)
  x = g.r[:,0]  # x positions
  y = g.r[:,1]  # y positions
  z = g.r[:,2]  # z positions
  fg = open(output_file,"w")
  fg.write(" # x    y     z   (without spin degree)\n")
  for (ix,iy,iz) in zip(x,y,z):
    fg.write(str(ix)+ "     "+str(iy)+"   "+str(iz)+"  \n")
  fg.close()





def write_vasp(g0,s=1.42,namefile="vasp.vasp"):
    """Turn a geometry into vasp geometry"""
    g = g0.copy() # copy geometry
    if g.dimensionality==3: pass
    elif g.dimensionality==2:
        g.r[:,2] -= np.min(g.r[:,2])
        z = np.max(g.r[:,2]) - np.min(g.r[:,2])
        a3 = np.array([0.,0.,z+9.0])
        g.a3 = a3 # set the lattice vector
        g.dimensionality = 3
        g.get_fractional() # get fractional coordinates
    else: raise # not implemented
    f = open(namefile,"w") # input file
    f.write("Structure\n 1.0\n")
    for i in range(3): f.write(str(s*g.a1[i])+"  ")
    f.write("\n")
    for i in range(3): f.write(str(s*g.a2[i])+"  ")
    f.write("\n")
    for i in range(3): f.write(str(s*g.a3[i])+"  ")
    # write the atoms
    if len(g.r)==len(g.atoms_names):
        atoms_have_names = True
    else: atoms_have_names = False
    if not atoms_have_names: # no name provided
      f.write("\n C\n "+str(len(g.r))+"\n Direct\n")
      # write all the atoms in fractional coordinates
      for ir in g.frac_r:
          for i in range(3): f.write(str(ir[i])+"  ")
          f.write("\n")
    else: # atoms have labels
        namedict = dict() # dictionary
        for (key,n) in zip(g.atoms_names,range(len(g.r))):
            if not key in namedict:
                namedict[key] = [n] # store
            else: namedict[key].append(n) # store
        f.write("\n") # next line
        for key in namedict: f.write(str(key)+"   ")
        f.write("\n") # next line
        for key in namedict: f.write(str(len(namedict[key]))+"   ")
        f.write("\n Direct \n") # next line
        for key in namedict: # loop over types
            ns = namedict[key] # list with atoms
            for ii in ns: # loop over atoms
              for i in range(3): f.write(str(g.frac_r[ii][i])+"  ")
              f.write("\n")
    f.close()




def write_function(self,fun,name="FUNCTION.OUT"):
    """Write a certain function"""
    f = open(name,"w")
    ir = 0
    for r in self.r: # loop over positions
      o = fun(r) # evaluate
      f.write(str(ir)+"  ")
      for io in o:  f.write(str(io)+"  ")
      f.write("\n")
      ir += 1
    f.close() # close file


