# Spin-spin exchange interactions

## Spin-spin exchange interactions

We will now see how a direct coupling between the spins on two sites is treated at the mean-field level, with `h.get_szsz_mean_field_hamiltonian()` and its companions. The interactions considered above are all of density-density type, $c^\dagger_{i,s}c_{i,s}c^\dagger_{j,s'}c_{j,s'}$, meaning that they couple the charge on one site to the charge on another. A magnet made of localized moments is described instead by a coupling $\vec{S}_i\cdot\vec{S}_j$ between the (Pauli) spin operators at two sites,

$$
H = J_z\sum_{\langle ij\rangle} S^z_i S^z_j
+ J_x\sum_{\langle ij\rangle} S^x_i S^x_j
+ J_y\sum_{\langle ij\rangle} S^y_i S^y_j
$$

with $J>0$ the antiferromagnetic (Heisenberg) sign convention and $J<0$ favoring a ferromagnetic instability. Writing $S^z_i=(n_{i,\uparrow}-n_{i,\downarrow})/2$,

$$
S^z_i S^z_j = \tfrac14\left(
n_{i,\uparrow}n_{j,\uparrow} - n_{i,\uparrow}n_{j,\downarrow}
- n_{i,\downarrow}n_{j,\uparrow} + n_{i,\downarrow}n_{j,\downarrow}
\right)
$$

is already a density-density interaction between spin-orbitals, so $S^z_iS^z_j$ is solved at the same mean-field level as the $U$/$V_1$/$V_2$/$V_3$ interactions above, with `h.get_szsz_mean_field_hamiltonian(J1=...)`. `J1`/`J2`/`J3` are the first-, second- and third-neighbor couplings and `Jr` a general distance-dependent one, the same convention as `V1`/`V2`/`V3`/`Vr`. Both `Jr` and `Vr` are evaluated on every pair of sites up to `rcut`, the pair of a site with itself included, so a coupling meant for the bonds alone returns zero at zero distance, and a nonzero value there is an onsite term, described under `h.get_combined_mean_field_hamiltonian()` in the reference. The other two axes are `h.get_sxsx_mean_field_hamiltonian(...)` and `h.get_sysy_mean_field_hamiltonian(...)`. The bare interaction is SU(2) symmetric, so at the same coupling the three differ only in the axis the moment orders along.

```python
from pyqula import geometry
g = geometry.chain() # a chain, prone to ferromagnetic order away from half filling
h = g.get_hamiltonian(has_spin=True)
h = h.get_szsz_mean_field_hamiltonian(J1=-2.0,filling=0.2,
                                       mf="ferroZ", # ferromagnetic Sz-Sz coupling
                                       nk=10,mix=0.3,maxite=300)
m = h.get_magnetization() # uniform moment along z
```

The ferromagnetic coupling on a chain away from half filling orders every site with the same moment along $z$, which `h.get_magnetization()` returns as one vector per site, and the two spin bands of the converged Hamiltonian are split by an exchange field that the calculation found rather than one put in with `h.add_exchange()`.

Let us spell out here what decides whether any of the mean-field calculations in this chapter converges, since all of them return `None` instead of a Hamiltonian when they do not, and the result has to be checked before it is used. Four parameters decide it:

- `nk` (8 by default): the number of k-points per direction of the mesh on which the expectation values are computed
- `mix` (0.1): the fraction of the new mean field mixed into the old one at each iteration
- `maxite` (1000, or 2000 with `use_jax=True`): the maximum number of iterations before giving up, `None` for no limit
- `maxerror` (1e-5): how close two successive mean fields must be for the result to count as converged

The k-mesh is usually the real culprit: a mesh that does not resolve the Fermi surface makes the occupied states flip from one iteration to the next, and no amount of mixing helps. The chain above at `filling=0.2` never converges at `nk=8` and converges at once at `nk=10`, so this is the parameter to change first. A lower `mix` is the adiabatic update of the collinear section, and helps when the calculation oscillates around a solution rather than settling on it. A calculation that never converges gives up after `maxite` iterations and returns `None`, and on a large system 1000 iterations can take a long while, so set a lower `maxite` while exploring parameters and the answer comes back in seconds.

The three axes can be given different couplings in a single anisotropic-exchange calculation, with `h.get_exchange_mean_field_hamiltonian(Jx1=...,Jy1=...,Jz1=...)`:

```python
h = g.get_hamiltonian(has_spin=True)
h = h.get_exchange_mean_field_hamiltonian(Jz1=-1.0,Jx1=-0.5,
                                            filling=0.2,mf="ferroZ",
                                            nk=10,mix=0.3,maxite=300)
```

Density-density interactions and spin-spin exchange can also be solved together in one self-consistent calculation, with `h.get_combined_mean_field_hamiltonian(U=...,V1=...,J1=...,...)`, which is what a model with both charge and magnetic correlations needs. Here $J_1$/$J_2$/$J_3$ (and $J_r$) are isotropic Heisenberg couplings $J(S^x_iS^x_j+S^y_iS^y_j+S^z_iS^z_j)$ on the first, second and third neighbor shells, and $J_{1x}$/$J_{1y}$/$J_{1z}$ add an anisotropy on top of $J_1$ for the first shell only, so that the effective first-neighbor $J_z$ is $J_1+J_{1z}$. All of them default to zero

```python
h = g.get_hamiltonian(has_spin=True)
h = h.get_combined_mean_field_hamiltonian(U=5.0,J1=-1.0,
                                            filling=0.2,mf="ferroZ")
```

For a large, sparse system the diagonalization at every k-point is the expensive step of each iteration, and the same self-consistent loop can get its density matrix instead from a Chebyshev expansion, the kernel polynomial method of its own chapter, never diagonalizing the Bloch Hamiltonian, with `integration="kpm"`:

```python
h = g.get_hamiltonian(has_spin=True)
h = h.get_combined_mean_field_hamiltonian(U=5.0,J1=-1.0,filling=0.2,
                                            mf="ferroZ",integration="kpm")
```

The expansion resolves energies down to roughly the bandwidth divided by `npol`, the number of polynomials, so a magnetic splitting or a superconducting gap smaller than that is smeared out. In practice a self-consistent gap needs this resolution several times below it, and close to a transition, where a gapped and an ungapped solution have nearly the same energy, about ten times below it, or the loop can settle in the wrong one. The Fermi level of each iteration is taken from the Chebyshev recursion of the previous one instead of a search of its own, which makes it exact once the mean field stops changing, and the loop stops only when the filling of its last density matrix is the requested one; note that the path to convergence is then not the one of exact diagonalization, so where a model has several self-consistent states the two routes can settle in different ones from the same guess, and the lower total energy says which one to keep. The recursion behind it runs on the GPU once the device is selected (see [Running on a GPU](gpu.md#running-on-a-gpu)), where `kpm_prec` decides its precision; single precision is the default there and moves the density matrix by about $10^{-8}$, far below what a finite `npol` leaves out, although the change of the mean field from one iteration to the next then stops at about $10^{-6}$ per entry, so a `maxerror` below that needs `kpm_prec="double"`, which keeps the recursion in double precision:

```python
from pyqula import gpu
gpu.set_gpu(True)
h = g.get_hamiltonian(has_spin=True)
h = h.get_combined_mean_field_hamiltonian(U=5.0,J1=-1.0,filling=0.2,
                                            mf="ferroZ",integration="kpm",
                                            npol=400,kpm_prec="single")
```

What comes back is the same self-consistent Hamiltonian as on the CPU to that accuracy, and what changes is the time per iteration; `gpu.set_gpu(False)` returns the rest of a script to the CPU.

What limits this route on a large island is not only the time but the memory: for a dense Hamiltonian, the interaction, the density matrix and the mean field are each a matrix of the size of the whole island, which at $10^5$ sites does not fit in memory. A Hamiltonian built sparse, with `is_sparse=True`, keeps every one of them sparse through the whole loop, holding only the entries that the interaction couples, so the memory grows linearly with the number of sites, and this holds with spin-orbit coupling, with exchange and with the Nambu Hamiltonian of a superconductor alike:

```python
from pyqula import geometry
g = geometry.honeycomb_lattice().get_supercell(12) # a honeycomb island
g.dimensionality = 0 # make it finite
h = g.get_hamiltonian(has_spin=True,is_sparse=True) # keep the matrices sparse
h.add_rashba(0.2) # spin-orbit coupling, so the moments are not collinear
h = h.get_mean_field_hamiltonian(U=3.0,filling=0.5,mf="random",
                                  integration="kpm",npol=150,mix=0.5,
                                  maxerror=1e-3)
```

What comes back is the self-consistent Hamiltonian the dense route gives, to roundoff, and still sparse, so `h.extract("mx")` and its siblings read the exchange field site by site without diagonalizing anything. With a sparse Hamiltonian the tolerance `maxerror` is the change of the mean field per entry it holds, whatever the size of the island; the orientation of the moments, which the Rashba coupling selects only weakly, relaxes slowly under the mixing, so the snippet stops at a change of $10^{-3}$, reached in under a hundred iterations from a random guess, while $2\times10^{-4}$ takes about 400. The time of an iteration grows as the square of the number of sites, since the occupations of every site come from a Chebyshev recursion over the whole island, which is what the truncation below removes. The total energy is here the band energy of the KPM density matrix at zero temperature, $\mathrm{Tr}(H\rho)$, which reaches the sum of the occupied levels as `npol` grows, and a `Vr` or `Jr` on an island needs an explicit `rcut`, since without one every pair of sites interacts. Note that on a small island where the filling times the number of states is not a whole number, the KPM Fermi level holds that fractional number of electrons while exact diagonalization rounds it to whole states, so there the two routes converge to slightly different states. See `examples/0d/kpm_scf_sparse/main.py` for a runnable version with an antiferromagnet and a superconductor.

What makes an iteration grow as the square of the number of sites is that the recursion of every orbital runs over the whole island, while after $n$ steps the vector started on an orbital has only reached the sites within $n$ hops of it. So the occupations around an orbital, computed on the Hamiltonian restricted to the sites within $R$ hops of it, are exact up to the moment $2R$, and beyond it they are those of a finite cluster around the orbital. Whether that cluster is enough is decided by the state and not by the code: in a gapped state the density matrix decays exponentially, over a length of the order of the velocity over the gap, and at a finite temperature over the velocity over the temperature, so the error of the truncation falls exponentially with $R$, while in a metal at zero temperature the density matrix has a power-law tail and the truncation is not controlled. `kpm_radius` sets $R$, in hops between sites, so that a spin flip or a pairing on one site is not a hop:

```python
h = g.get_hamiltonian(has_spin=True,is_sparse=True) # the same island
h.add_rashba(0.2)
h = h.get_mean_field_hamiltonian(U=3.0,filling=0.5,mf="random",
                                  integration="kpm",npol=150,mix=0.5,
                                  maxerror=1e-3,kpm_radius=8)
```

The cost of an iteration is now the number of sites times the size of a ball of $R$ hops, linear in the number of sites, and the Fermi level and the total energy come from the same truncated recursion. The radius is converged as the k-mesh is, by comparing two values: for a collinear antiferromagnet of a half-filled honeycomb island at $U=3$, the exchange field differs from the one of the full recursion by 0.11, 0.023, 0.005 and 0.001 at 2, 4, 6 and 8 hops, meaning that every two hops gain a factor of about five. On six desktop cores, an iteration of the Hubbard model on a square island of $10^5$ sites with Rashba coupling takes about half a minute at `kpm_radius=10` and `npol=100`, and about two minutes with the Nambu spinor of a superconductor, in 1.2 and 2.3 GB of memory.

A loop that refuses to converge under plain mixing can be handed to a nonlinear solver instead, with `use_jax=True`, which treats one iteration $x=f(x)$ of the loop as a root-finding problem for $x-f(x)=0$. `solver="error_gradient"` is the most robust of these on a generic Hamiltonian and the one to reach for first; `"newton"` is the default. These solvers are for the normal state only and do not accept `constrains`.

```python
h = g.get_hamiltonian(has_spin=True)
h = h.get_combined_mean_field_hamiltonian(U=5.0,J1=-1.0,filling=0.2,
                                            mf="ferroZ",use_jax=True,
                                            solver="newton")
```

The accepted names are `"newton"`, `"newton_krylov"`, `"fsolve"`, `"error_gradient"` (also called `"levenberg_marquardt"`), `"lbfgs"`, `"broyden_mixing"` and `"linear_mixing"` (also called `"fixed_point"`), the same for the spinful and the spinless mean-field functions, and an unknown name raises `ValueError` listing them. Without `use_jax=True` the loop is plain mixing, so `solver=` is read only together with it, and passing it alone raises `NotImplementedError` rather than being ignored.

See the notebooks `03_spin_spin_exchange.ipynb`, `11_kpm_scf.ipynb` and `12_jax_scf_solvers.ipynb` in the folder above for executed versions of the three.

All of the spin-spin exchange functions above also work on Bogoliubov-de Gennes (Nambu) Hamiltonians, set up with `h.turn_nambu()` or `h.setup_nambu_spinor()`, where the exchange channels are decoupled in the normal and in the anomalous channel, exactly as $U$/$V_1$/$V_2$/$V_3$ are. Exchange can therefore induce superconducting pairing on its own: an antiferromagnetic isotropic $J$, with no $U$ or $V$ at all, can decouple spontaneously into a purely superconducting singlet-paired state, the resonating-valence-bond mechanism behind exchange-driven superconductivity, while the ferromagnetic sign has no such tendency and stays magnetic. That instability has to be seeded coherently, with `h.add_swave(0.1)` on top of the Hamiltonian passed as `mf`, since a random guess has little overlap with it and usually relaxes back to zero pairing. A state carrying both magnetic and superconducting order can also emerge from an exchange field combined with an attractive $V_1$, as below. The total energy returned with `return_total_energy=True` subtracts the double counting of the pairing mean field as well as of the normal one, so it is the mean-field energy of the paired state, and comparing it with the energy of an unpaired solution of the same interaction tells you which of the two is the ground state at the mean-field level:

```python
h = g.get_hamiltonian(has_spin=True)
h.add_exchange([0.,0.,0.3])
h.turn_nambu()
h = h.get_combined_mean_field_hamiltonian(V1=-1.0,J1z=-0.3,
                                            filling=0.3,mf="random")
```
