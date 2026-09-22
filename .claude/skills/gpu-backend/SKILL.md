---
name: gpu-backend
description: pyqula's CPU/GPU switch (src/pyqula/gpu.py), how a routine is routed onto the device, per-call precision, and the tiered porting plan in documentation/gpu_porting_plan.md. Load this before moving any compute onto the GPU, before adding a jax code path, before touching kpmtk/kpmjax.py, htk/eigenvectorsjax.py or any other *jax.py module, and whenever a task mentions the GPU, CUDA, the device, jax, or making something faster by moving it off the CPU. Each tier of the porting plan wants the maintainer's sign-off before it starts, so check here before proposing GPU work rather than after.
---

# The GPU backend

## One package-wide switch

The CPU/GPU backend is one switch, `src/pyqula/gpu.py`. `gpu.set_gpu(True)` puts every
GPU-capable routine on the device and points jax's default device there, so the jax modules
with no backend branch of their own follow it too. The default is the CPU, on every machine.

A new GPU path routes on `gpu.get_gpu()` rather than growing a switch of its own. Precision
stays per-call: `kpm_prec`, `chi_prec`, `eigh_prec`. The old per-call `kpm_cpugpu` and
`chi_cpugpu` arguments were removed and now raise.

## The porting plan, and the sign-off rule

`documentation/gpu_porting_plan.md` is the maintainer-facing roadmap for moving
compute-heavy paths onto GPU via jax, which is already a hard dependency. Read it before
starting any GPU-related work in this repo.

| Tier | What | State |
| --- | --- | --- |
| 1 | The batched KPM GPU path (`kpmtk/kpmjax.py`, `kpmtk/kpmnumba.py`) | done |
| 2 | Batched dense diagonalization (`htk/eigenvectorsjax.py`) | done |
| 3 | Scoping the forced-CPU jax modules | done |
| 4 | Why sparse/ARPACK-based Green's-function work is harder and lower priority | not started |

**Each tier wants explicit sign-off before it starts.** Propose, do not begin.

## What the GPU is actually good for here

The shape that pays on the device is a batched dense solve: many independent matrices
diagonalized or multiplied at once, large enough that the transfer is amortized. Single
sparse solves and ARPACK-style iterative work are the unfavourable case, which is why
tier 4 sits where it does.

Performance *conclusions* belong in the tracked roadmaps. The hostnames, scratch paths and
job IDs that produced them do not -- see the HPC rule in CLAUDE.md.
