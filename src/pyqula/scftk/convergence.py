"""The error raised when a self-consistent loop stops without converging.

The get_*_mean_field_hamiltonian methods raise it, and so do the public
engines of meanfield.py (meanfield.Vinteraction, VJinteraction, SzSz, ...),
unless they are called with check_convergence=False. The scftk engines
underneath still return their SCF object with .converged=False, for the
library code that handles a failed loop itself. The engines record
.iterations, .residual (the last error reached) and .tol (maxerror) on that
object, which is what the message reports."""


class MeanFieldConvergenceError(RuntimeError):
    """Raised when a mean-field loop stops before reaching maxerror. The
    unconverged SCF object is kept as .scf (its .hamiltonian and
    .total_energy are the last iterate, not a self-consistent answer)."""
    def __init__(self, message, scf=None):
        super().__init__(message)
        self.scf = scf


def _fmt(x, spec):
    try: return format(float(x), spec)
    except (TypeError, ValueError): return "unknown"


def require_converged(scf):
    """Raise MeanFieldConvergenceError unless scf.converged"""
    if scf.converged: return
    ite = getattr(scf, "iterations", None)
    ite = "an unrecorded number of" if ite is None else str(ite)
    res = _fmt(getattr(scf, "residual", None), ".2e")
    tol = _fmt(getattr(scf, "tol", None), ".1e")
    raise MeanFieldConvergenceError("the mean field did not converge in "
            "%s iterations: the error reached %s, above maxerror=%s. Try a "
            "larger maxite, a smaller mix (or a larger one, if the loop is "
            "only slow), another initial guess mf=, or fix the random seed "
            "with np.random.seed() before the call, so that a random guess "
            "is reproducible" % (ite, res, tol), scf=scf)
