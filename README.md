# SUMMARY #
This is a **Py**thon library to compute **qu**antum-**la**ttice 
tight-binding models in different dimensionalities.


# INSTALLATION #
## With pip (release version) ##
```bash
pip install pyqula
```

## Manual installation (most recent version) ##
Clone the Github repository with

```bash
git clone https://github.com/joselado/pyqula
```

and add the "pyqula/src" path to your Python script with

```python
import sys
sys.path.append(PATH_TO_PYQULA+"/src")
```




# FUNCTIONALITIES #
- Spinless, spinful and Nambu basis for orbitals
- Include magnetism, spin-orbit coupling and superconductivity
- Band structures and density of states
- Selfconsistent mean-field calculations with local/non-local interactions
- Topological characterization of electronic structures
- Green's function formalism for semi-infinite systems
- Spectral functions in infinite geometries
- Kernel polynomial based-methods
- Quantum Transport
- 0d, 1d, 2d and 3d tight binding models 

# EXAMPLES #
A variety of examples can be found in pyqula/examples

## Band structure of a honeycomb lattice
```python
from pyqula import geometry
g = geometry.honeycomb_lattice() # get the geometry object
h = g.get_hamiltonian() # get the Hamiltonian object
h.get_bands() # compute the band structure
```

## Mean field Hubbard model of a zigzag honeycomb ribbon
```python
from pyqula import geometry
from pyqula import scftypes
g = geometry.honeycomb_zigzag_ribbon(10) # create geometry of a zigzag ribbon
h = g.get_hamiltonian() # create hamiltonian of the system
mf = scftypes.guess(h,"ferro",fun=lambda r: [0.,0.,1.])
scf = scftypes.hubbardscf(h,nkp=30,filling=0.5,mf=mf)
h = scf.hamiltonian # get the Hamiltonian
h.get_bands(operator="sz") # calculate band structure
```

## Band structure of twisted bilayer graphene
```python
from pyqula import specialgeometry
from pyqula.specialhopping import twisted_matrix
g = specialgeometry.twisted_bilayer(3)
h = g.get_hamiltonian(mgenerator=twisted_matrix(ti=0.12))
h.get_bands(nk=100)
```

## Chern number of a Chern insulator
```python
from pyqula import geometry
from pyqula import topology
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_rashba(0.3) # Rashba spin-orbit coupling
h.add_zeeman([0.,0.,0.3]) # Zeeman field
c = topology.chern(h) # compute Chern number
print("Chern number is ",c)
```

## Band structure of a nodal line semimetal
```python
from pyqula import geometry
from pyqula import films
g = geometry.diamond_lattice_minimal()
g = films.geometry_film(g,nz=20)
h = g.get_hamiltonian()
h.get_bands()
```

## Surface spectral function of the Haldane model
```python
from pyqula import geometry
from pyqula import kdos
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_haldane(0.05)
kdos.surface(h)
```

## Antiferromagnet-superconductor interface
```python
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon(10) # create geometry of a zigzag ribbon
h = g.get_hamiltonian(has_spin=True) # create hamiltonian of the system
h.add_antiferromagnetism(lambda r: (r[1]>0)*0.5) # add antiferromagnetism
h.add_swave(lambda r: (r[1]<0)*0.3) # add superconductivity
h.get_bands() # calculate band structure
```

## Fermi surface of a Kagome lattice
```python
from pyqula import geometry
from pyqula import spectrum
import numpy as np
g = geometry.kagome_lattice()
h = g.get_hamiltonian()
spectrum.multi_fermi_surface(h,nk=60,energies=np.linspace(-4,4,100),
        delta=0.1,nsuper=1)
```


