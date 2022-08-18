# SUMMARY #
This is a **Py**thon library to compute **qu**antum-**la**ttice 
tight-binding models in different dimensionalities.


# INSTALLATION #
## With pip (release version) ##
```bash
pip install --upgrade pyqula
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

# Tutorials #
Jupyter notebooks with tutorials can be found in the links below (part of the [Jyvaskyla Summer School 2022](https://github.com/joselado/jyvaskyla_summer_school_2022) )
- [Electronic structure](https://github.com/joselado/jyvaskyla_summer_school_2022/blob/main/sessions/session1.ipynb)
- [Superconductivity](https://github.com/joselado/jyvaskyla_summer_school_2022/blob/main/sessions/session2.ipynb)
- [Magnetism](https://github.com/joselado/jyvaskyla_summer_school_2022/blob/main/sessions/session3.ipynb)
- [Moire physics](https://github.com/joselado/jyvaskyla_summer_school_2022/blob/main/sessions/session4.ipynb)
- [Topological matter](https://github.com/joselado/jyvaskyla_summer_school_2022/blob/main/sessions/session5.ipynb)


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
- Hermitian and non-Hermitian mean-field calculations

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
A variety of examples can be found in pyqula/examples. Short examples are shown below


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




## Interaction-driven spin-singlet superconductivity
```python
from pyqula import geometry
import numpy as np
g = geometry.triangular_lattice() # geometry of a triangular lattice
h = g.get_hamiltonian()  # get the Hamiltonian
h.setup_nambu_spinor() # setup the Nambu form of the Hamiltonian
h = h.get_mean_field_hamiltonian(U=-1.0,filling=0.15,mf="swave") # perform SCF
# electron spectral-function
h.get_kdos_bands(operator="electron",nk=400,energies=np.linspace(-1.0,1.0,100))
```

![Alt text](images/scf_SC.png?raw=true "Interaction-driven superconductivity")


## Interaction driven non-unitary spin-triplet superconductor
```python
import numpy as np
from pyqula import geometry
g = geometry.triangular_lattice() # generate the geometry
h = g.get_hamiltonian() # create Hamiltonian of the system
h.add_exchange([0.,0.,1.]) # add exchange field
h.setup_nambu_spinor() # initialize the Nambu basis
# perform a superconducting non-collinear mean-field calculation
h = h.get_mean_field_hamiltonian(V1=-1.0,filling=0.3,mf="random")
# compute the non-unitarity of the spin-triplet superconducting d-vector
d = h.get_dvector_non_unitarity() # non-unitarity of spin-triplet
# electron spectral-function
h.get_kdos_bands(operator="electron",nk=400,energies=np.linspace(-2.0,2.0,400))
```

![Alt text](images/scf_SC_triplet.png?raw=true "Interaction driven non-unitary spin-triplet superconductor")





## Mean-field with local interactions of a zigzag honeycomb ribbon
```python
from pyqula import geometry
g = geometry.honeycomb_zigzag_ribbon(10) # create geometry of a zigzag ribbon
h = g.get_hamiltonian() # create hamiltonian of the system
h = h.get_mean_field_hamiltonian(U=1.0,filling=0.5,mf="ferro")
(k,e,sz) = h.get_bands(operator="sz") # calculate band structure
```
![Alt text](images/scf_zigzag.png?raw=true "Mean-field with local interactions of a zigzag honeycomb ribbon")


## Non-collinear mean-field with local interactions of a square lattice
```python
from pyqula import geometry
g = geometry.square_lattice() # geometry of a square lattice
g = g.get_supercell([2,2]) # generate a 2x2 supercell
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_zeeman([0.,0.,0.1]) # add out-of-plane Zeeman field
h = h.get_mean_field_hamiltonian(U=2.0,filling=0.5,mf="random") # perform SCF
(k,e,c) = h.get_bands(operator="sz") # calculate band structure
m = h.get_magnetization() # get the magnetization
```

![Alt text](images/scf_square.png?raw=true "Non-collinear mean-field interactions with local interactions of a square lattice")


## Interaction-induced non-collinear magnetism in a defective square lattice with spin-orbit coupling
```python
from pyqula import geometry
g = geometry.square_lattice() # geometry of a square lattice
g = g.get_supercell([7,7]) # generate a 7x7 supercell
g = g.remove(i=g.get_central()[0]) # remove the central site
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_rashba(.4) # add Rashba spin-orbit coupling
h = h.get_mean_field_hamiltonian(U=2.0,filling=0.5,mf="random") # perform SCF
(k,e,c) = h.get_bands(operator="sz") # calculate band structure
m = h.get_magnetization() # get the magnetization
```

![Alt text](images/scf_square_vacancy.png?raw=true "Interaction-induced non-collinear magnetism in a defective square lattice with spin-orbit coupling")


## Band structure of twisted bilayer graphene
```python
from pyqula import specialhamiltonian # special Hamiltonians library
h = specialhamiltonian.twisted_bilayer_graphene() # TBG Hamiltonian
(k,e) = h.get_bands() # compute band structure
```

![Alt text](images/tbg.png?raw=true "Band structure of twisted bilayer graphene")

## Band structure of monolayer NbSe2
```python
from pyqula import specialhamiltonian # special Hamiltonians library
h = specialhamiltonian.NbSe2(soc=0.5) # NbSe2 Hamiltonian
(k,e,c) = h.get_bands(operator="sz",kpath=["G","K","M","G"]) # compute bands
```

![Alt text](images/NbSe2.png?raw=true "Band structure of monolayer NbSe2")




## Chern number of an artificial Chern insulator
```python
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_rashba(0.2) # Rashba spin-orbit coupling
h.add_zeeman([0.,0.,0.6]) # Zeeman field
from pyqula import topology
(kx,ky,omega) = h.get_berry_curvature() # compute Berry curvature
c = h.get_chern() # compute the Chern number
```

![Alt text](images/berry_curvature.png?raw=true "Chern number of an artificial Chern insulator")

## Topological phase transition in an artificial topological superconductor
```python
import numpy as np
from pyqula import geometry
g = geometry.chain() # create a chain
g = g.supercell(100) # create a large supercell
g.dimensionality = 0 # make it finite
for J in np.linspace(0.,0.2,50): # loop over exchange couplings
    h = g.get_hamiltonian() # create a new hamiltonian
    h.add_onsite(2.0) # shift the chemical potential
    h.add_rashba(.3) # add rashba spin-orbit coupling
    h.add_exchange([0.,0.,J]) # add exchange coupling
    h.add_swave(.1) # add s-wave superconductivity
    edge = h.get_operator("location",r=g.r[0]) # projector on the edge
    energies = np.linspace(-.2,.2,200) # set of energies
    (e0,d0) = h.get_dos(operator=edge,energies=energies,delta=2e-3) # edge DOS
```

![Alt text](images/TSC.png?raw=true "Topological phase transition in an artificial topological superconductor")


## Spatial distribution of Majorana modes in an artificial topological superconductor
```python
import numpy as np
from pyqula import geometry
g = geometry.chain() # create a chain
g = g.supercell(100) # create a large supercell
g.dimensionality = 0 # make it finite
h = g.get_hamiltonian() # create a new hamiltonian
h.add_onsite(2.0) # shift the chemical potential
h.add_rashba(.3) # add rashba spin-orbit coupling
h.add_exchange([0.,0.,0.15]) # add exchange coupling
h.add_swave(.1) # add s-wave superconductivity
energies = np.linspace(-.15,.15,200) # set of energies
for ri in g.r: # loop over sites
    edge = h.get_operator("location",r=ri) # projector on that site
    (e0,d0) = h.get_dos(operator=edge,energies=energies,delta=2e-3) # local DOS
```

![Alt text](images/TSC_spatial.png?raw=true "Spatial distribution of Majorana modes in an artificial topological superconductor")






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


## Moire band structure of a moire superlattice
```python
from pyqula import geometry
from pyqula import potentials
g = geometry.triangular_lattice() # create geometry
g = g.get_supercell([7,7]) # create a supercell
h = g.get_hamiltonian() # get the Hamiltonian
fmoire = potentials.commensurate_potential(g,n=3,minmax=[0,1]) # moire potential
h.add_onsite(fmoire) # add onsite energy following the moire
h.get_bands(operator=fmoire) # project on the moire
```

![Alt text](images/moire_bands.png?raw=true "Moire band structure of a moire superstructure")


## Unfolded electronic structure of a moire superstructure
```python
from pyqula import geometry
from pyqula import potentials
import numpy as np
g0 = geometry.triangular_lattice() # create geometry
n = 5 # supercell
g = g0.get_supercell(n,store_primal=True) # create a supercell
h = g.get_hamiltonian() # get the Hamiltonian
fmoire = potentials.commensurate_potential(g,n=3,minmax=[0,1]) # moire potential
h.add_onsite(fmoire) # add onsite energy following the moire
kpath = np.array(g.get_kpath(nk=400))*n # enlarged k-path
h.get_kdos_bands(operator="unfold",delta=2e-2,kpath=kpath,
                  energies=np.linspace(-3,-1,300)) # unfolded bands

```

![Alt text](images/unfolding_moire.png?raw=true "Unfolded electronic structure of a moire superstructure")






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


## Band structure of a three dimensional topological insulator slab
```python
from pyqula import geometry
from pyqula import films
import numpy as np
g = geometry.diamond_lattice() # create a diamond lattice
g = films.geometry_film(g,nz=60) # create a thin film
h = g.get_hamiltonian() # generate Hamiltonian
h.add_strain(lambda r: 1.+abs(r[2])*0.8,mode="directional") # add axial strain
h.add_kane_mele(0.1) # add intrinsic spin-orbit coupling
(k,e,c)= h.get_bands(operator="surface") # compute band structure
```

![Alt text](images/3DTI.png?raw=true "Band structure of a three dimensional topological insulator")



## Surface spectral function of a 2D quantum spin-Hall insulator

```python
from pyqula import geometry
g = geometry.honeycomb_lattice() # create a honeycomb lattice
h = g.get_hamiltonian() # generate Hamiltonian
h.add_soc(0.15) # add intrinsic spin-orbit coupling
h.add_rashba(0.1) # add Rashba spin-orbit coupling
h.get_surface_kdos(delta=1e-2) # compute surface spectral function
```

![Alt text](images/2DTI.png?raw=true "Surface spectral function of a 2D quantum spin-Hall insulator")



## Local density of states with atomic orbitals of a honeycomb nanoisland
```python
from pyqula import islands
g = islands.get_geometry(name="honeycomb",n=3,nedges=3) # get an island
h = g.get_hamiltonian() # get the Hamiltonian
h.get_multildos(projection="atomic") # get the LDOS
```
![Alt text](images/ldos_island.png?raw=true "Local density of states with atomic orbitals of a honeycomb nanoisland")



## Interaction-driven magnetism in a honeycomb nanoisland
```python
from pyqula import islands
g = islands.get_geometry(name="honeycomb",n=3,nedges=3) # get an island
h = g.get_hamiltonian() # get the Hamiltonian
h = h.get_mean_field_hamiltonian(U=1.0,filling=0.5,mf="ferro") # perform SCF
m = h.get_magnetization() # get the magnetization in each site
```

![Alt text](images/scf_island.png?raw=true "Interaction-driven magnetism in a honeycomb nanoisland")

## Hofstadter's butterfly of a square lattice
```python
import numpy as np
from pyqula import geometry
g = geometry.square_ribbon(40) # create square ribbon geometry

for B in np.linspace(0.,1.0,300): # loop over magnetic field
    h = g.get_hamiltonian() # create a new hamiltonian
    h.add_orbital_magnetic_field(B) # add an orbital magnetic field
    # calculate DOS projected on the bulk
    (e,d) = h.get_dos(operator="bulk",energies=np.linspace(-4.5,4.5,200))
```

![Alt text](images/hofstadter.png?raw=true "Hofstadter's butterfly of a square lattice")


## Landau levels of a Dirac semimetal
```python
import numpy as np
from pyqula import geometry
g = geometry.honeycomb_ribbon(30) # create a honeycomb ribbon

for B in np.linspace(0.,0.02,100): # loop over magnetic field
    h = g.get_hamiltonian() # create a new hamiltonian
    h.add_orbital_magnetic_field(B) # add an orbital magnetic field
    # calculate DOS projected on the bulk
    (e,d) = h.get_dos(operator="bulk",energies=np.linspace(-1.0,1.0,200),
                       delta=1e-2)
```

![Alt text](images/LL_Dirac.png?raw=true "Landau levels of a Dirac semimetal")



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


## Surface states and Berry curvature of a artificial 2D topological superconductor
```python
from pyqula import geometry
import numpy as np
g = geometry.triangular_lattice() # get the geometry
h = g.get_hamiltonian() # get the Hamiltonian
h.add_onsite(2.0) # shift chemical potential
h.add_rashba(1.0) # Rashba spin-orbit coupling
h.add_zeeman([0.,0.,0.6]) # Zeeman field
h.add_swave(.3) # add superconductivity
(kx,ky,omega) = h.get_berry_curvature() # compute Berry curvature
h.get_surface_kdos(energies=np.linspace(-.4,.4,300)) # surface spectral function
```

![Alt text](images/2DTSC.png?raw=true "Surface states and Berry curvature of a artificial 2D topological superconductor")



# Surface states in a topological superconductor nanoisland

```python
from pyqula import islands
g = islands.get_geometry(name="triangular",shape="flower",
                           r=14.2,dr=2.0,nedges=6) # get a flower-shaped island
h = g.get_hamiltonian() # get the Hamiltonian
h.add_onsite(3.0) # shift chemical potential
h.add_rashba(1.0) # Rashba spin-orbit coupling
h.add_zeeman([0.,0.,0.6]) # Zeeman field
h.add_swave(.3) # add superconductivity
h.get_ldos() # Spatially resolved DOS
```

![Alt text](images/island_TSC.png?raw=true "Surface states in a topological superconductor nanoisland")





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



## Tunneling and Andreev reflection in a metal-superconductor junction

```python
from pyqula import geometry
from pyqula import heterostructures
import numpy as np
g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # create the Hamiltonian
h1 = h.copy() # first lead
h2 = h.copy() # second lead
h2.add_swave(.01) # the second lead is superconducting
es = np.linspace(-.03,.03,100) # set of energies for dIdV
for T in np.linspace(1e-3,1.0,6): # loop over transparencies
    HT = heterostructures.build(h1,h2) # create the junction
    HT.set_coupling(T) # set the coupling between the leads
    Gs = [HT.didv(energy=e) for e in es] # calculate conductance
```

![Alt text](images/andreev.png?raw=true "Tunneling and Andreev reflection in a metal-superconductor junction")




## From tunneling to contact transport of a topological superconductor

```python
from pyqula import geometry
from pyqula import heterostructures
import numpy as np

g = geometry.chain() # create the geometry
h = g.get_hamiltonian() # create teh Hamiltonian
h1 = h.copy() # first lead
h2 = h.copy() # second lead
h2.add_onsite(2.0) # shift chemical potential in the second lead
h2.add_exchange([0.,0.,.3]) # add exchange in the second lead
h2.add_rashba(.3) # add Rashba SOC in the second lead
h2.add_swave(.05) # add s-wave SC in the second lead
es = np.linspace(-.1,.1,100) # grid of energies
for T in np.linspace(1e-3,0.5,10): # loop over transparencies
    HT = heterostructures.build(h1,h2) # create the junction
    HT.set_coupling(T) # set the coupling between the leads
    Gs = [HT.didv(energy=e) for e in es] # calculate transmission
```

![Alt text](images/dIdV_TSC.png?raw=true "From tunneling to contact transport of a topological superconductor")




## Single impurity in an infinite honeycomb lattice
```python
from pyqula import geometry
import numpy as np
from pyqula import embedding
g = geometry.honeycomb_lattice() # create geometry 
h = g.get_hamiltonian() # get the Hamiltonian
hv = h.copy() # copy Hamiltonian to create a defective one
hv.add_onsite(lambda r: (np.sum((r - g.r[0])**2)<1e-2)*100) # add a defect
eb = embedding.Embedding(h,m=hv) # create an embedding object
(x,y,d) = eb.ldos(nsuper=19,energy=0.,delta=1e-2) # compute LDOS
```

![Alt text](images/single_vacancy.png?raw=true "Single impurity in an infinite honeycomb lattice")



## Bound state of a single magnetic impurity in an infinite superconductor
```python
from pyqula import geometry
import numpy as np
from pyqula import embedding
g = geometry.square_lattice() # create geometry
h = g.get_hamiltonian() # get the Hamiltonian
h.add_swave(0.1) # add s-wave superconductivity
h.add_onsite(3.0) # shift chemical potential
hv = h.copy() # copy Hamiltonian to create a defective one
hv.add_exchange(lambda r: [0.,0.,(np.sum((r - g.r[0])**2)<1e-2)*6.]) # add magnetic site
eb = embedding.Embedding(h,m=hv) # create an embedding object
ei = eb.get_energy_ingap_state() # get energy of the impurity state
(x,y,d) = eb.ldos(nsuper=19,energy=ei,delta=1e-3) # compute LDOS
```

![Alt text](images/single_YSR.png?raw=true "Single magnetic impurity in an infinite superconductor")



## Parity switching of a magnetic impurity in an infinite superconductor

```python
from pyqula import geometry
from pyqula import embedding
import numpy as np

g = geometry.square_lattice() # create geometry
for J in np.linspace(0.,4.0,100): # loop over exchange
    h = g.get_hamiltonian() # get the Hamiltonian,spinless
    h.add_onsite(3.0) # shift chemical potential
    h.add_swave(0.2) # add s-wave superconductivity
    hv = h.copy() # copy Hamiltonian to create a defective one
    # add magnetic site
    hv.add_exchange(lambda r: [0.,0.,(np.sum((r - g.r[0])**2)<1e-2)*J])
    eb = embedding.Embedding(h,m=hv) # create an embedding object
    energies = np.linspace(-0.4,0.4,100) # energies
    d = [eb.dos(nsuper=2,delta=1e-2,energy=ei) for ei in energies] # compute DOS
```

![Alt text](images/YSR.png?raw=true "Parity switching of a magnetic impurity in an infinite superconductor")

