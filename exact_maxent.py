"""Exact Z_lambda for frequency-independent multipliers.

If lambda_{q,n} = lambda_q, Parseval gives
    sum_n |rho_{q,n}[R]|^2 = (1/hbar) int_0^{beta hbar} dtau rho_q(tau) rho_{-q}(tau),
so the constraint term is local in imaginary time and

    Z_lambda = tr exp( -beta H_0 - sum_q lambda_q rho_q rho_{-q} ),

an ordinary Gibbs trace with a translation-invariant pair density interaction.
That is diagonalizable for L=12, N=6, so the max-ent problem over equal-time
density constraints can be solved exactly, with no expansion of ln I[psi].
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from tv import hamiltonian, density_mode, thermal, entropy, System


class Lattice:
    def __init__(self, L, N, t, beta):
        self.L, self.N, self.t, self.beta = L, N, t, beta
        H0, states, index = hamiltonian(L, N, t, 0.0)
        self.H0, self.states = H0, states
        self.qs = [2 * np.pi * k / L for k in range(1, L)]
        # |rho_q|^2 is diagonal in the occupation basis
        self.d2 = np.array([np.abs(density_mode(L, states, q)) ** 2 for q in self.qs])
        E0, _ = np.linalg.eigh(beta * H0)
        self.lnI0 = logsumexp(-E0)

    def gibbs(self, u):
        """u: array over self.qs (must satisfy u_k = u_{L-k})."""
        M = self.beta * self.H0 + np.diag(self.d2.T @ u)
        e, U = np.linalg.eigh(M)
        lnZ = logsumexp(-e)
        p = np.exp(-e - lnZ)
        return e, U, lnZ, p

    def struct(self, u):
        e, U, lnZ, p = self.gibbs(u)
        occ = (U ** 2) @ p                      # diagonal of the density matrix
        return self.d2 @ occ / self.N, lnZ, p, U, e

    def solve(self, Sq, x0=None):
        """Minimise the convex dual g(u) = N sum_q u_q S(q) + ln Z(u) - ln I[0]."""
        L = self.L
        half = L // 2
        def expand(x):
            u = np.empty(L - 1)
            for k in range(1, L):
                u[k - 1] = x[min(k, L - k) - 1]
            return u
        def obj(x):
            u = expand(x)
            Smod, lnZ, p, U, e = self.struct(u)
            g = self.N * np.dot(u, Sq) + lnZ - self.lnI0
            resid = self.N * (Sq - Smod)
            grad = np.zeros(half)
            for k in range(1, L):
                grad[min(k, L - k) - 1] += resid[k - 1]
            return g, grad
        x0 = np.zeros(half) if x0 is None else x0
        r = minimize(obj, x0, jac=True, method='L-BFGS-B',
                     options=dict(maxiter=2000, ftol=1e-16, gtol=1e-12))
        return expand(r.x), r


def static_Sq(sysobj):
    """Exact equal-time structure factor of an interacting system."""
    L, N = sysobj.L, sysobj.N
    out = []
    for k in range(1, L):
        q = 2 * np.pi * k / L
        d = density_mode(L, sysobj.states, q)
        M = sysobj.U.conj().T @ (d[:, None] * sysobj.U)
        out.append(float(np.einsum('a,ab->', sysobj.p, np.abs(M) ** 2).real) / N)
    return np.array(out)


def truncated_Fex(lat, u, ref, nmax=1500):
    """beta F_ex in the Gaussian approximation: (1/2) sum ln(1 + 2 N lambda S_0)."""
    ns = np.arange(-nmax, nmax + 1)
    beta, N = lat.beta, lat.N
    tot = 0.0
    for i, q in enumerate(lat.qs):
        S0 = ref.Sqn(q, ns)
        lam = u[i] / beta                                      # lambda has units of energy
        tot += 0.5 * np.sum(np.log1p(2 * N * lam * S0))
        A0 = 2 * (1 - np.cos(q)) * abs(ref.Ekin) / N          # lim omega^2 S_0
        tail = 2 * N * lam * A0 * beta ** 2 / (4 * np.pi ** 2)
        tot += 0.5 * 2 * tail * (np.pi ** 2 / 6 - sum(1.0 / m ** 2 for m in range(1, nmax + 1)))
    return tot
