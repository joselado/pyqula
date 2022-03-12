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
## Single particle Hamiltonians ##
- Spinless, spinful and Nambu basis for orbitals
- Full non-collinear electron and Nambu formalism
- Include magnetism, spin-orbit coupling and superconductivity
- Band structures with state-resolved expectation values
- Momentum-resolved spectral functions
- Local and full operator-resolved density of states
- 0d, 1d, 2d and 3d tight binding models 
- Electronic structure unfolding in supercells

## Interacting mean-field Hamiltonians ##
- Selfconsistent mean-field calculations with local/non-local interactions
- Both collinear and non-collinear formalism
- Anomalous mean-field for non-collinear superconductors
- Full selfconsistency with all Wick terms for non-collinear superconductors
- Constrained and unconstrained mean-field calculations
- Automatic identification of order parameters for symmetry broken states

## Topological characterization ##
- Berry phases, Berry curvatures, Chern numbers and Z2 invariants
- Operator-resolved Chern numbers and Berry density
- Frequency resolved topological density
- Spatially resolved topological flux
- Real-space Chern density for amorphous systems
- Wilson loop and Green's function formalism

## Spectral functions ##
- Spectral functions in infinite geometries
- Surface spectral functions for semi-infinite systems
- Interfacial spectral function in semi-infinite junctions
- Single impurities in infinite systems
- Operator-resolved spectral functions
- Green's function renormalization algorithm

## Chebyshev kernel polynomial based-algorithms ##
- Local and full spectral functions
- Non-local correlators and Green's functions
- Locally resolved expectation values
- Operator resolved spectral functions
- Reaching system sizes up to 10000000 atoms on a single-core laptop

## Quantum transport ##
- Metal-metal transport
- Metal-superconductor transport
- Fully non-collinear Nambu basis
- Non-equilibrium Green's function formalism
- Operator-resolved transport
- Differential decay rate
- Tunneling and contact scanning probe spectroscopy

# EXAMPLES #
A variety of examples can be found in pyqula/examples


## Band structure of a Kagome lattice
```python
from pyqula import geometry
g = geometry.kagome_lattice() # get the geometry object
h = g.get_hamiltonian() # get the Hamiltonian object
(k,e) = h.get_bands() # compute the band structure
```
![Alt text](images/kagome.png?raw=true "Band structure of a Kagome lattice")


## Valley-resolved band structure of a honeycomb superlattice
```python
from pyqula import geometry
g = geometry.honeycomb_lattice() # get the geometry object
g = g.get_supercell(7) # create a supercell
h = g.get_hamiltonian() # get the Hamiltonian object
(k,e,v) = h.get_bands(operator="valley") # compute the band structure
```
![Alt text](images/valley_supercell.png?raw=true "Valley-resolved band structure of a honeycomb superlattice")



## Non-unitarity of an interacting spin-triplet superconductor
```python
from pyqula import geometry
g = geometry.triangular_lattice() # generate the geometry
h = g.get_hamiltonian() # create Hamiltonian of the system
h.add_exchange([3.,3.,3.]) # add exchange field
h.setup_nambu_spinor() # initialize the Nambu basis
# perform a superconducting non-collinear mean-field calculation
h = h.get_mean_field_hamiltonian(V1=-1.0,filling=0.3,mf="random")
# compute the non-unitarity of the spin-triplet superconducting d-vector
d = h.get_dvector_non_unitarity() # non-unitarity of spin-triplet
```


## Mean-field with local interactions of a zigzag honeycomb ribbon
```python
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon(10) # create geometry of a zigzag ribbon
h = g.get_hamiltonian() # create hamiltonian of the system
h = h.get_mean_field_hamiltonian(U=1.0,filling=0.5,mf="ferro")
(k,e,sz) = h.get_bands(operator="sz") # calculate band structure
```
![Alt text](images/scf_zigzag.png?raw=true "Mean-field with local interactions of a zigzag honeycomb ribbon")

## Band structure of twisted bilayer graphene
```python
from pyqula import specialhamiltonian # special Hamiltonians library
h = specialhamiltonian.twisted_bilayer_graphene() # TBG Hamiltonian
(k,e) = h.get_bands() # compute band structure
```
![Alt text](images/tbg.png?raw=true "Band structure of twisted bilayer graphene")

## Chern number of a Chern insulator
```python
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_rashba(0.3) # Rashba spin-orbit coupling
h.add_zeeman([0.,0.,0.3]) # Zeeman field
c = h.get_chern(h) # compute Chern number
print("Chern number is ",c)
```

## Unfolded electronic structure of a supercell with a defect
```python
from pyqula import geometry
import numpy as np
g = geometry.honeycomb_lattice() # create a honeycomb lattice
n = 3 # size of the supercell
g = g.get_supercell(n,store_primal=True) # create a supercell
h = g.get_hamiltonian() # get the Hamiltonian
fons = lambda r: (np.sum((r - g.r[0])**2)<1e-2)*100 # onsite in the impurity
h.add_onsite(fons) # add onsite energy
kpath = np.array(g.get_kpath(nk=200))*n # enlarged k-path
h.get_kdos_bands(operator="unfold",delta=1e-1,kpath=kpath) # unfolded bands
```
![Alt text](images/unfolded.png?raw=true "Unfolded electronic structure of a supercell with a defect")




## Band structure of a nodal line semimetal slab
```python
from pyqula import geometry
from pyqula import films
g = geometry.diamond_lattice()
g = films.geometry_film(g,nz=20)
h = g.get_hamiltonian()
(k,e) = h.get_bands()
```
![Alt text](images/NLSM.png?raw=true "Band structure of a nodal line semimetal slab")


## Local density of states with atomic orbitals of a honeycomb nanoisland
```python
from pyqula import islands
g = islands.get_geometry(name="honeycomb",n=3,nedges=3) # get an island
h = g.get_hamiltonian() # get the Hamiltonian
h.get_multildos(projection="atomic") # get the LDOS
```
![Alt text](images/ldos_island.png?raw=true "Local density of states with atomic orbitals of a honeycomb nanoisland")



## Surface spectral function of a Chern insulator
```python
from pyqula import geometry
from pyqula import kdos
g = geometry.honeycomb_lattice() # create honeycomb lattice
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_haldane(0.05) # Add Haldane coupling
kdos.surface(h) # surface spectral function
```
![Alt text](images/kdos.png?raw=true "Surface spectral function of a Chern insulator")

## Antiferromagnet-superconductor interface
```python
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon(20) # create geometry of a zigzag ribbon
h = g.get_hamiltonian(has_spin=True) # create hamiltonian of the system
h.add_antiferromagnetism(lambda r: (r[1]>0)*0.5) # add antiferromagnetism
h.add_onsite(lambda r: (r[1]>0)*0.3) # add chemical potential
h.add_swave(lambda r: (r[1]<0)*0.3) # add superconductivity
(k,e,sz) = h.get_bands(operator="sz") # calculate band structure
```
![Alt text](images/AF_SC.png?raw=true "Antiferromagnet-superconductor interface")

## Fermi surface of a triangular lattice supercell
```python
from pyqula import geometry
import numpy as np
g = geometry.triangular_lattice() # create geometry of the system
g = g.get_supercell(2) # create a supercell
h = g.get_hamiltonian() # create hamiltonian of the system
h.get_multi_fermi_surface(energies=np.linspace(-4,4,100),delta=1e-1)
```
![Alt text](images/fermi_surface.png?raw=true "Fermi surface of a triangular lattice supercell")



## Unfolded Fermi surface of a supercell with a defect
```python
from pyqula import geometry
import numpy as np
g0 = geometry.triangular_lattice()
n = 3 # size of the supercell
g = g0.get_supercell(n,store_primal=True) # create a supercell
h = g.get_hamiltonian() # get the Hamiltonian
fons = lambda r: (np.sum((r - g.r[0])**2)<1e-2)*100 # onsite in the impurity
h.add_onsite(fons) # add onsite energy
kpath = np.array(g.get_kpath(nk=200))*n # enlarged k-path
h.get_multi_fermi_surface(nk=50,energies=np.linspace(-4,4,100),
        delta=0.1,nsuper=n,operator="unfold")
```
![Alt text](images/unfolded_FS.png?raw=true "Unfolded Fermi surface of a supercell with a defect")



