"""The two functions of energy of the perturbative tunneling spectrum of
Ternes, New J. Phys. 17, 063016 (2015), arXiv:1505.04430: the thermally
broadened step Theta and the thermally broadened Kondo logarithm F, here
as kernels on a uniform energy grid.

A kernel is the average of the function over each bin of the grid, so
that a sum over the grid is the integral of the function against a
density that was binned on it. At zero temperature the step is a ramp
over one bin and the logarithm is the difference of its antiderivative,
which is how the singularity of width Gamma0, far below any bin, is
integrated exactly. At a finite temperature both are the zero
temperature kernel convolved with the weight of the derivative of the
thermal step in each bin.

The closed form of the thermal step,

    Theta(x) = 1/2 + (sinh x - x) / (2 (cosh x - 1)),   x = energy / T,

is the derivative with the bias of the current between two Fermi seas,
and the logarithm is

    F(e) = ln(omega0 + |e|) - 1/2 ln(e^2 + Gamma0^2)

convolved with dTheta/de, the reading of eq. 22 of the paper that
reproduces its figures (the dmrgpy package of the same author documents
that reading and its checks; the numbers here agree with it)."""
import numpy as np


def theta(x):
    """Thermally broadened step at x = energy / T: 0 far below zero, 1/2
    at zero, 1 far above, and theta(x) + theta(-x) = 1."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-4
    large = np.abs(x) > 40
    mid = ~small & ~large
    out[small] = 0.5 + x[small]/6.0 - x[small]**3/360.0
    xm = x[mid]
    out[mid] = 0.5 + (np.sinh(xm) - xm)/(2*(np.cosh(xm) - 1))
    xl = np.abs(x[large])
    tail = (xl + 1)*np.exp(-xl)
    out[large] = np.where(x[large] > 0, 1.0 - tail, tail)
    return out


def _log_antiderivative(y, omega0, Gamma0):
    """Antiderivative of ln(omega0 + |y|) - 1/2 ln(y^2 + Gamma0^2), odd
    and continuous at zero"""
    a = np.abs(y)
    up = np.sign(y)*((omega0 + a)*np.log(omega0 + a) - a
                     - omega0*np.log(omega0))
    down = 0.5*(y*np.log(y**2 + Gamma0**2) - 2*y
                + 2*Gamma0*np.arctan(y/Gamma0))
    return up - down


def _thermal_weights(de, T):
    """Weight of the derivative of the thermal step in every bin of width
    de, for the bins -M..M that hold all of it; they add up to 1"""
    M = int(np.ceil(45.*T/de)) + 1
    edges = (np.arange(-M, M + 2) - 0.5)*de
    return np.diff(theta(edges/T))


def _broaden(kernel0, K, de, T):
    """kernel0 is a zero temperature kernel at the points -K..K, or
    -K-M..K+M when T is not zero; return the kernel at T at -K..K"""
    if T == 0.: return kernel0
    return np.convolve(kernel0, _thermal_weights(de, T), mode="valid")


def _points(K, de, T):
    """The points of the zero temperature kernel _broaden needs"""
    M = 0 if T == 0. else (len(_thermal_weights(de, T)) - 1)//2
    return np.arange(-K - M, K + M + 1)*de


def step_kernel(K, de, T):
    """Bin average of the thermal step at the energies k*de, k=-K..K"""
    y = _points(K, de, T)
    return _broaden(np.clip(y/de + 0.5, 0., 1.), K, de, T)


def log_kernel(K, de, T, omega0, Gamma0):
    """Bin average of the thermal Kondo logarithm F at the energies k*de,
    k=-K..K"""
    y = _points(K, de, T)
    k0 = (_log_antiderivative(y + de/2., omega0, Gamma0)
          - _log_antiderivative(y - de/2., omega0, Gamma0))/de
    return _broaden(k0, K, de, T)
