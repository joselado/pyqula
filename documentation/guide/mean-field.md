# Interactions at the mean-field level

Up to now every Hamiltonian in this guide has been a single-particle one, meaning that all
its terms are bilinear, a creation and an annihilation operator. We will now see how
electronic interactions, quartic terms with four fermionic operators, can be treated at the
mean-field level: the interaction is replaced by a bilinear term that depends on expectation
values, and since those expectation values depend in turn on the Hamiltonian, the problem is
solved self-consistently. This is an uncontrolled approximation, but it is the practical
workhorse for magnetism, charge order and superconductivity driven by interactions, and
everything in this chapter goes through one method, `h.get_mean_field_hamiltonian()`, which
returns a new Hamiltonian with the converged mean-field terms in it. We start with the
Hubbard model, first collinear and then non-collinear, then let the same interaction produce
superconductivity, then add interactions between neighbors, and finally treat spin-spin
exchange, spin models and the Kondo lattice with the same machinery.

## The collinear Hubbard model

We will start with the simplest interaction term, a local repulsive interaction in a spinful system. Our full Hamiltonian takes the form

$$
H = \sum_{\langle ij\rangle} c^\dagger_i c_{j} + h.c.
+
U\sum_{i} 
c^\dagger_{i,\uparrow} c_{i,\uparrow} 
c^\dagger_{i,\downarrow} c_{i,\downarrow} 
$$

where the first term is the first-neighbor hopping of the previous chapters and $U$ is the
energy cost of putting a spin-up and a spin-down electron on the same site. The interaction term $U\sum_{i}c^\dagger_{i,\uparrow}c_{i,\uparrow}c^\dagger_{i,\downarrow}c_{i,\downarrow}$ is solved at the mean-field level. The mean-field approximation consists in replacing the previous four-fermion operator by all the terms that arise by taking the expectation value of two of the fermions. In particular, in its simplest collinear form, the mean-field term takes the form

$$
H_U^{MF} = 
U\sum_{i} 
\langle c^\dagger_{i,\uparrow} c_{i,\uparrow} \rangle
c^\dagger_{i,\downarrow} c_{i,\downarrow} 
+
c^\dagger_{i,\uparrow} c_{i,\uparrow}
\langle c^\dagger_{i,\downarrow} c_{i,\downarrow} \rangle
$$

where $\langle\rangle$ denotes the ground state expectation value of those operators, meaning
that each spin-down electron feels an onsite energy set by the density of spin-up electrons
on its site, and the other way around. The full Hamiltonian thus takes the form


$$
H^{MF} = 
\sum_{\langle ij\rangle} c^\dagger_i c_{j} + h.c.
+
U\sum_{i} 
\langle c^\dagger_{i,\uparrow} c_{i,\uparrow} \rangle
c^\dagger_{i,\downarrow} c_{i,\downarrow} 
+
c^\dagger_{i,\uparrow} c_{i,\uparrow}
\langle c^\dagger_{i,\downarrow} c_{i,\downarrow} \rangle
$$

As a result, the mean-field Hamiltonian depends on the specific ground state of the system, and the ground state depends of course on the specific mean-field Hamiltonian. The previous circular dependence between the ground state and the mean-field Hamiltonian gives rise to a self-consistent problem. 

This self-consistent condition is solved as follows. We start with an initial guess for the full many-body ground state, that we call $|GS_0\rangle$. With this initial state, we compute the mean-field Hamiltonian $H^{MF}_0$. This mean-field Hamiltonian allows us to compute a new many-body ground state $|GS_1\rangle$, which in turn allows us to compute a new mean-field Hamiltonian $H^{MF}_1$. The previous algorithm is represented as

$$
|GS_0\rangle
\rightarrow
H^{MF}_0
\rightarrow
|GS_1\rangle
\rightarrow
H^{MF}_1
\rightarrow
|GS_2\rangle
\rightarrow
H^{MF}_2
\rightarrow
...
$$

This iterative calculation is performed until $H^{MF}_n = H^{MF}_{n+1}$, at which point the algorithm has converged. In practice the loop stops when two successive mean fields differ by less than a tolerance, `maxerror`, and nothing in the Hamiltonian is symmetric anymore: the mean-field terms are what a symmetry-broken state looks like at the single-particle level, an onsite energy that differs between spins, or between sites, that the original Hamiltonian did not have.

Two important notes can be taken from the previous approach. First, the final solution may be sensitive to the initial guess for the ground state. This guess corresponds to the initialization of the Hamiltonian, and it can be important for systems whose energy landscape has several local minima. A second point is that the update procedure from one iteration to the next can be done adiabatically, or very suddenly. This corresponds to the mixing between solutions, and for systems close to the critical point can lead to tricky convergence.

Let us now show an example of a mean-field calculation, with `h.get_mean_field_hamiltonian()`. We will take now a square lattice, make a 2x2 supercell and include local repulsive interactions at half filling. The obtained ground state is an antiferromagnetic Neel state that opens a gap at half filling

```python
from pyqula import geometry
g = geometry.square_lattice() # geometry of a square lattice
g = g.get_supercell([2,2]) # generate a 2x2 supercell
h = g.get_hamiltonian() # create hamiltonian of the system
h = h.get_mean_field_hamiltonian(U=2.0,filling=0.5,
                            mf="random") # perform SCF
(k,e) = h.get_bands() # calculate band structure
m = h.get_magnetization() # get the magnetization
```

The result is a new Hamiltonian with the converged mean-field terms in it, and the original `h` is left as it was. What you see in its band structure is a gap at zero energy, of about $0.8$ for $U=2$, where the non-interacting square lattice at half filling is a metal; the magnetization is one vector per site, of equal magnitude and alternating sign between the two sublattices of the supercell, along a direction that the random initial guess picked, since nothing in the Hamiltonian selects one. The supercell is needed for the same reason: a Neel state has two inequivalent sites, and a single-site cell could only hold a ferromagnet. The guess `mf=` is one of the names listed by the library (`"random"`, `"antiferro"`, `"ferro"`, `"swave"` and others) or a Hamiltonian whose mean-field terms seed the calculation; `filling` is the keyword of "Setting a filling", enforced at every iteration; and the calculation returns `None` instead of a Hamiltonian when it does not converge, which is discussed at the end of the section on spin-spin exchange together with the parameters that decide it. Increasing $U$ makes the moments larger and the gap wider, up to a gap that grows linearly with $U$ in the strongly interacting limit, where each site holds one localized electron.

See `examples/2d/hubbard_honeycomb/main.py` for the gap as a function of $U$ on the honeycomb lattice, where the density of states vanishes at the Dirac point and the gap only opens above a critical interaction, and `jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/02_collinear_noncollinear.ipynb` for the executed version.


## Non-collinear Hubbard model
Let us now see what happens when the magnetization is not restricted to point along one axis. In the mean-field ansatz considered above, only a single term in the Wick contraction was considered. This term is the collinear term in the z-direction, and allows accounting for solutions that have magnetization in the z-direction. However, in the presence of frustration, external magnetic field or spin-orbit coupling, the magnetization of a system may be non-collinear and pointing in an arbitrary direction. To account for that phenomenology, the mean-field Hamiltonian must include the non-collinear term that takes the form

$$
H_U^{ncMF} = -
U\sum_{i} 
\langle c^\dagger_{i,\downarrow} c_{i,\uparrow} \rangle
c^\dagger_{i,\uparrow} c_{i,\downarrow} 
+ h.c.
$$

which is a spin-flip term on each site, set by the expectation value of the in-plane magnetization. When including this additional term, the full mean-field Hubbard Hamiltonian is rotationally invariant, meaning that it respects SO(3) spin rotational symmetry. This rotationally symmetric form is the default form implemented in the library, which is also why the direction of the Neel moments in the previous section was left to the initial guess.

With the previous point in mind, we now solve a system that develops a non-collinear magnetic state. We take the square lattice considered in the section above, and we add an external magnetic field. The competition between Zeeman energy and antiferromagnetic correlations gives rise to a canted magnetic state

```python
from pyqula import geometry
g = geometry.square_lattice() # geometry of a square lattice
g = g.get_supercell([2,2]) # generate a 2x2 supercell
h = g.get_hamiltonian() # create hamiltonian of the system
h.add_zeeman([0.,0.,0.1]) # add out-of-plane Zeeman field
h = h.get_mean_field_hamiltonian(U=2.0,filling=0.5,
                                  mf="random") # perform SCF
(k,e,c) = h.get_bands(operator="sz") # calculate band structure
m = h.get_magnetization() # get the magnetization
```

The magnetization now has two parts: an antiferromagnetic component, alternating between the sublattices, that lies in the plane perpendicular to the field, and a uniform component along the field, the same on every site. The moments cant toward the field rather than flipping, which is what an antiferromagnet does in a field that is weak compared to its exchange, and a larger field tilts them further. The third array returned by `h.get_bands(operator="sz")` is the spin projection of each state, so that the bands can be colored by it and the uniform component read off from the imbalance between up and down.

See `examples/readme_examples/scf_square/main.py` for this calculation ending in a plot of the bands and of the moments, and the notebook `02_collinear_noncollinear.ipynb` in the folder above.


## Superconducting mean-field

In the cases above we focused on local repulsive interactions that promote collinear or non-collinear magnetism. However, local interactions can promote a different type of symmetry breaking, in particular gauge symmetry breaking associated to superconductivity. The emergence of superconductivity is associated to one of the Wick contractions of the mean-field, the anomalous term, that takes the form


$$
H_U^{aMF} = 
U\sum_{i} 
\langle c_{i,\uparrow} c_{i,\downarrow} \rangle
c^\dagger_{i,\downarrow} c^\dagger_{i,\uparrow} 
+ h.c.
$$

where the expectation value of two annihilation operators, the pairing amplitude, is the
$\Delta_0$ of the s-wave term in the chapter on superconductivity, now obtained from the
interaction rather than put in by hand. The previous term in the mean-field Hamiltonian can become non-zero for $U<0$, and yields an interaction induced superconducting state. This term in the mean-field Hamiltonian is automatically accounted for in Hamiltonians with a Nambu degree of freedom, of course apart from the collinear and non-collinear terms in the mean-field. We show below how an interaction induced superconducting state can be computed with pyqula, with `h.setup_nambu_spinor()` followed by `h.get_mean_field_hamiltonian()`

```python
from pyqula import geometry
import numpy as np
g = geometry.triangular_lattice() # geometry of a triangular lattice
h = g.get_hamiltonian()  # get the Hamiltonian
h.setup_nambu_spinor() # setup the Nambu form of the Hamiltonian
h = h.get_mean_field_hamiltonian(U=-2.0,filling=
                   0.45,mf="swave",nk=40) # perform SCF
# electron spectral-function
h.get_kdos_bands(operator="electron",nk=400,
                   energies=np.linspace(-2.0,2.0,200),delta=0.03)
```

The Nambu spinor has to be set up before the calculation, so that the anomalous contraction has a block of the Hamiltonian to go into; the guess `mf="swave"` seeds a uniform singlet pairing, which the loop then adjusts to its self-consistent value. The electron spectral function, projected with `operator="electron"` on the electron half of the Nambu spinor so that the hole copy of the bands does not appear, shows the normal-state band of the triangular lattice with a gap opened at the Fermi energy, and the gap grows with $|U|$. `h.get_kdos_bands()` is described in the section on momentum-resolved spectral functions.

See `examples/readme_examples/scf_SC/main.py` for a runnable version ending in a plot, and the notebook `04_anomalous_mf_sc.ipynb` in the folder above.


## Long range interactions

Up to now we have considered interacting Hamiltonians that only have local (attractive or repulsive) Hubbard interactions. In the following we are going to consider systems that have many-body interactions also to a certain number of neighbors. Long range interactions are crucial to stabilize specific symmetry broken states, and in particular charge density waves, Peierls instabilities and unconventional superconductivity. The full Hamiltonian we will consider takes the form

$$
H = \sum_{\langle ij\rangle} c^\dagger_i c_{j} + h.c.
+
U\sum_{i}
c^\dagger_{i,\uparrow} c_{i,\uparrow}
c^\dagger_{i,\downarrow} c_{i,\downarrow}
+
V_1\sum_{\langle ij\rangle,s,s'} 
c^\dagger_{i,s} c_{i,s}
c^\dagger_{j,s'} c_{j,s'}
$$

where $U$ parametrizes onsite interactions and $V_1$ interactions between first neighbors; second and third neighbors enter `h.get_mean_field_hamiltonian()` in the same way as `V2` and `V3`, and an interaction with an arbitrary dependence on the distance as `Vr`, a function `Vr(r1,r2)` of the positions of two sites that is the interaction of each pair, exactly as $V_1$ is for a pair of first neighbors, so that a `Vr` equal to $V_1$ on first neighbors and zero beyond them is the same calculation as `V1`. That interaction reaches up to a distance `rcut`, 5 by default in a periodic system, and it is kept or dropped one whole shell of equidistant pairs at a time, so that it has the symmetry of the lattice; in a finite system every pair is kept. The pair of a site with itself is one of those pairs, so `Vr(0)` is an onsite Hubbard term, $U=V_r(0)$, and a function meant for the bonds alone returns zero at zero distance. The previous Hamiltonian gives rise to a variety of terms when performing a mean-field decoupling. By default, pyqula includes all the Wick contractions of the mean-field, and in the presence of Nambu spinors it includes all the anomalous contractions. Let us now briefly elaborate on some of the additional terms that arise due to the first-neighbor interaction $V_1$.

The first term is the charge order term, that takes the form

$$
H^{MF} \sim
\langle c^\dagger_{i,s} c_{i,s} \rangle
c^\dagger_{j,s'} c_{j,s'}
$$

this term can give rise to a different charge imbalance between different sites, and it leads to charge density wave states.

The second term we consider is the bond order, that takes the form

$$
H^{MF} \sim
\langle c^\dagger_{i,s} c_{j,s} \rangle
c^\dagger_{j,s} c_{i,s} + h.c.
$$

which leads to an interaction-enhanced hopping. If this happens in a non-uniform way in the system, the resulting state has a Peierls distortion.

Among the anomalous terms, the mean-field Hamiltonian can generate

$$
H^{MF} \sim
\langle c^\dagger_{i,\uparrow} c^\dagger_{j,\uparrow} \rangle
c_{j,\downarrow} c_{i,\downarrow} + h.c.
$$


$$
H^{MF} \sim
\langle c^\dagger_{i,\downarrow} c^\dagger_{j,\downarrow} \rangle
c_{j,\uparrow} c_{i,\uparrow} + h.c.
$$


$$
H^{MF} \sim
\langle c^\dagger_{i,\uparrow} c^\dagger_{j,\downarrow} \rangle
c_{j,\downarrow} c_{i,\uparrow} + h.c.
$$

where the first two terms correspond to the odd superconducting order, and the third term accounts for both even and odd orders. What this means is that a first-neighbor attraction can pair electrons on neighboring sites, with the same spin or with opposite spins, where the onsite $U$ of the previous section could only pair opposite spins on the same site.

Below, we show an example in which an interaction-induced spin-triplet term is generated. By considering an electronic structure with a large Zeeman splitting and attractive first-neighbor interactions, a state with non-zero $\Delta_{\uparrow\uparrow}$ and $\Delta_{\downarrow\downarrow}$ emerges. 


```python
import numpy as np
from pyqula import geometry
g = geometry.triangular_lattice() # generate the geometry
h = g.get_hamiltonian() # create Hamiltonian of the system
h.add_exchange([0.,0.,1.]) # add exchange field
h.setup_nambu_spinor() # initialize the Nambu basis
# perform a superconducting non-collinear mean-field calculation
h = h.get_mean_field_hamiltonian(V1=-1.5,
                     filling=0.3,mf="random",nk=40)
# electron spectral-function
h.get_kdos_bands(operator="electron",nk=400,
                   energies=np.linspace(-2.0,2.0,200),delta=0.03)
```

The exchange field splits the two spin bands so far apart that singlet pairing between opposite spins has no states to pair at the Fermi energy, and the attractive $V_1$ pairs electrons of the same spin on neighboring sites instead: the converged state has an odd, spin-triplet order parameter, and the electron spectral function shows a gap at the Fermi energy in each of the two spin-split bands. The d-vector of that state, and how far it is from unitary, are the subject of the section on the spin-triplet d-vector in the chapter on superconductivity.

See `examples/readme_examples/scf_SC_triplet/main.py` for this calculation ending in a plot, `examples/2d/meanfield_V1V2V3/main.py` for a charge density wave driven by $V_1$, $V_2$ and $V_3$ in a spinless system, and the notebook `05_full_wick_noncollinear_sc.ipynb` in the folder above.
