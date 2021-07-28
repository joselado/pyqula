# function to modify the Hamiltonian matrices according to criteria


def remove_hopping(self,f):
    """Remove hoppings to site according to criteria from the geometry"""
    g = self.geometry
    def fm(m): # function to modify matrices
        for i in range(len(g.r)): # loop over sites
            if f(g.r[i]): # if this site is removed
                if self.has_spin and not self.has_eh:
                    for j in range(m.shape[0]):
                        m[2*i,j] = 0.
                        m[2*i+1,j] = 0.
                        m[j,2*i+1] = 0.
                        m[j,2*i] = 0.
                elif not self.has_spin and not self.has_eh:
                    for j in range(m.shape[0]):
                        m[j,i] = 0.
                        m[i,j] = 0.
                else: raise
        return m
    self = self.copy()
    self.modify_hamiltonian_matrices(fm)
    return self

