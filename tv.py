import numpy as np
from itertools import combinations

def basis(L, N):
    states = []
    for occ in combinations(range(L), N):
        s = 0
        for o in occ:
            s |= (1 << o)
        states.append(s)
    states.sort()
    index = {s: i for i, s in enumerate(states)}
    return states, index

def popcount_below(s, a):
    return bin(s & ((1 << a) - 1)).count("1")

def hamiltonian(L, N, t, V):
    states, index = basis(L, N)
    D = len(states)
    H = np.zeros((D, D))
    for i, s in enumerate(states):
        # interaction (diagonal)
        e = 0.0
        for j in range(L):
            k = (j + 1) % L
            if (s >> j) & 1 and (s >> k) & 1:
                e += V
        H[i, i] += e
        # hopping
        for j in range(L):
            k = (j + 1) % L
            for (a, b) in ((j, k), (k, j)):   # c^dag_a c_b
                if not ((s >> b) & 1):
                    continue
                if (s >> a) & 1:
                    continue
                s1 = s & ~(1 << b)
                sgn = (-1) ** popcount_below(s, b)
                sgn *= (-1) ** popcount_below(s1, a)
                s2 = s1 | (1 << a)
                H[index[s2], i] += -t * sgn
    return H, states, index

def kinetic_operator(L, N, t):
    H0, states, index = hamiltonian(L, N, t, 0.0)
    return H0

def density_mode(L, states, q):
    ph = np.exp(-1j * q * np.arange(L))
    d = np.zeros(len(states), dtype=complex)
    for i, s in enumerate(states):
        occ = np.array([(s >> j) & 1 for j in range(L)])
        d[i] = np.dot(ph, occ)
    return d

def thermal(E, beta):
    E0 = E.min()
    w = np.exp(-beta * (E - E0))
    Z = w.sum()
    p = w / Z
    return p, Z, E0

def entropy(p):
    p = p[p > 1e-300]
    return float(-(p * np.log(p)).sum())

class System:
    """Exact spectrum + density matrix elements for one parameter set."""
    def __init__(self, L, N, t, V, beta):
        self.L, self.N, self.t, self.V, self.beta = L, N, t, V, beta
        H, states, index = hamiltonian(L, N, t, V)
        self.states = states
        E, U = np.linalg.eigh(H)
        self.E, self.U = E, U
        self.p, self.Z, self.E0 = thermal(E, beta)
        self.S_vn = entropy(self.p)
        K = kinetic_operator(L, N, t)          # = -t*sum(c^dag c + h.c.)
        self.Ekin = float(np.einsum('a,ia,ij,ja->', self.p, U, K, U, optimize=True).real)
        self.Etot = float((self.p * E).sum())

    def Sqn(self, q, ns, tol=1e-9, wtol=1e-16):
        """S(q, omega_n) for integer array ns, hbar=1."""
        L, N, beta = self.L, self.N, self.beta
        E, U, p = self.E, self.U, self.p
        d = density_mode(L, self.states, q)
        M = U.conj().T @ (d[:, None] * U)
        P = np.abs(M) ** 2
        nu = E[None, :] - E[:, None]
        keep = p > 1e-18
        P = P[keep, :]; nu = nu[keep, :]
        pm = p[keep][:, None]
        pl = (np.exp(-beta * (E - self.E0)) / self.Z)[None, :]
        deg = np.abs(nu) < tol
        w = P * (pm - pl)
        wdeg = (P * pm)[deg].sum()
        sel = (~deg) & (np.abs(w) > wtol)
        nun = nu[sel]; wn = w[sel] * nun
        ns = np.asarray(ns)
        omega = 2 * np.pi * ns / beta
        out = np.empty(len(ns))
        nu2 = nun ** 2
        for s in range(0, len(ns), 64):
            om = omega[s:s + 64][:, None]
            out[s:s + 64] = (wn[None, :] / (om ** 2 + nu2[None, :])).sum(axis=1)
        out /= N
        z = np.where(np.abs(omega) < 1e-14)[0]
        if len(z):
            out[z] += beta * wdeg / N
        return out
