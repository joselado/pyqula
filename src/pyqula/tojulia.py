###### functions to generate julia inputs #####


def write_hamiltonian(h):
  """Writes a hamiltonian in julia_hamiltonian.jl as a julia script"""
  def spm(m):
    """Non vanishing elements"""
    from scipy.sparse import coo_matrix as coo
    mc = coo(m) # to coo_matrixi
    data = mc.data # get data
    col = mc.col # get column index
    row = mc.row # get row index
    return (row,col,data.real,data.imag) # return things to print

  if h.dimensionality==1:
    f = open("julia_hamiltonian.jl","w") # open file
    n = len(h.intra) # length of the hamiltonian
    f.write("module julia_hamiltonian\n# dimension\n")  
    f.write("using hamiltonians\n")  
    f.write("function get_hamiltonian()\n")  
    f.write("  h =hamiltonians.hamiltonian1d()\n")  
    f.write("  intra = im*zeros("+str(n)+","+str(n)+")\n") # create intra term
    f.write("  inter = 0.*intra\n\n") # create inter term
    f.write("  # elements of the hamiltonian\n") 
    def write_matrix(m,name):
      """Writes aparticular matrix"""
      row,col,re,im = spm(m)
      for (i,j,a,b) in zip(row,col,re,im):
        f.write("  "+name+"["+str(i+1)+","+str(j+1)+"] = "+
                        str(a)+" + im*"+str(b)+"\n")
      f.write("  ###############\n\n")
    write_matrix(h.intra,"intra")
    write_matrix(h.inter,"inter")

    f.write("  h.intra = intra\n  h.inter = inter\n")  
    f.write("  return h\n")  
    f.write("end\n")  
    f.write("end\n")  
    f.close()  # close file


