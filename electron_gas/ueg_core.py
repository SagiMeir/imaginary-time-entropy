"""
Uniform electron gas: Matsubara density correlations and the closed-form
correlation free energy, Eq. (Fcorr), evaluated on PIMC input.

Spin-unpolarized jellium in a cubic cell of N electrons. Hartree atomic units,
hbar = m_e = e = 1, so omega_n = 2 pi n / beta and U_q = 4 pi / q^2.
Theta = T / E_F with E_F the Fermi energy of the ideal gas at the same density.

Pipeline of the figure, one density at a time (see section 8, beta_dF_corr_itcf):

  1. ideal reference   S_0(q, w_n) = -chi_0(q, i w_n) / rho on the cell's
                       discrete k-lattice (finite-N Lindhard function);
  2. input             S(q, w_n) from the PIMC imaginary-time correlation
                       function F(q, tau) (RODARE 3923), by exact integration of
                       a cubic spline whose end slopes the f-sum rule fixes;
                       no model enters;
  3. closed form       beta dF_corr = (1/2) sum_{q,n} [ln(S0/S) - S0/S + 1]
                       over |n| <= n_c, with the noise bias of this quadratic
                       functional removed by the jackknife estimate;
  4. finite size       continuum-minus-cell difference of the same functional,
                       evaluated in the random phase approximation.

f_c = (beta dF_corr[cell] + finite-size shift) / (N beta).

The earlier input, kept as a check (estimate_f_c), reconstructs S(q, w_n) from
the PIMC S(q) alone through a static local field correction G(q), solved shell
by shell so that beta^-1 sum_n S(q, w_n) = S_PIMC(q) exactly.

Depends only on numpy and scipy.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit, zeta as hurwitz_zeta


# ----------------------------------------------------------------------
# 1.  State point
# ----------------------------------------------------------------------
class UEG:
    """Spin-unpolarized electron gas at (rs, Theta) in a cell of N electrons."""

    def __init__(self, rs, theta, N):
        self.rs, self.theta, self.N = float(rs), float(theta), int(N)
        self.rho = 3.0 / (4.0 * np.pi * self.rs ** 3)
        self.kF = (9.0 * np.pi / 4.0) ** (1.0 / 3.0) / self.rs
        self.EF = 0.5 * self.kF ** 2
        self.beta = 1.0 / (self.theta * self.EF)
        self.L = (4.0 * np.pi * self.N / 3.0) ** (1.0 / 3.0) * self.rs  # cell edge
        self._mu = None

    @property
    def mu(self):
        """Ideal-gas chemical potential in the thermodynamic limit."""
        if self._mu is None:
            b = self.beta

            def excess_density(mu):
                f = lambda k: k * k * occupation(0.5 * k * k, mu, b)
                return 2.0 * quad(f, 0.0, np.inf, limit=200)[0] / (2 * np.pi ** 2) - self.rho

            self._mu = brentq(excess_density, -50.0 / b - 10.0,
                              5.0 * self.EF + 50.0 / b, xtol=1e-13, rtol=1e-14)
        return self._mu


def occupation(eps, mu, beta):
    """Fermi function 1 / (exp[beta(eps - mu)] + 1), overflow-safe."""
    return expit(-beta * (np.asarray(eps) - mu))


def coulomb(q):
    return 4.0 * np.pi / np.asarray(q, float) ** 2


def matsubara(n, beta):
    return 2.0 * np.pi * np.asarray(n, float) / beta


# ----------------------------------------------------------------------
# 2.  Reciprocal lattice of the cell
# ----------------------------------------------------------------------
def lattice_shells(q_over_kF, ueg):
    """Map published |q| shells onto the cubic reciprocal lattice of the cell.

    Returns (q, qvec, weight): exact lattice magnitudes, one representative
    vector per shell, and the number of lattice vectors in the shell. The sum
    over q != 0 runs over every lattice vector, so each shell is weighted by its
    multiplicity (6, 12, 8, 6, 24, ...), not counted once.
    """
    n2 = np.round((np.asarray(q_over_kF) / q_over_kF[0]) ** 2).astype(int)
    R = int(np.ceil(np.sqrt(n2.max()))) + 1
    rep, mult = {}, {}
    for x in range(-R, R + 1):
        for y in range(-R, R + 1):
            for z in range(-R, R + 1):
                s = x * x + y * y + z * z
                if s:
                    mult[s] = mult.get(s, 0) + 1
                    rep.setdefault(s, (x, y, z))
    qvec = np.array([rep[m] for m in n2], float) * (2.0 * np.pi / ueg.L)
    q = np.linalg.norm(qvec, axis=1)
    if np.max(np.abs(q / (np.asarray(q_over_kF) * ueg.kF) - 1.0)) > 1e-4:
        raise ValueError("q grid is not the reciprocal lattice of this (rs, N) cell")
    return q, qvec, np.array([mult[m] for m in n2], float)


def continuum_q_grid(ueg, q_max_over_kF=12.0, n_q=240):
    """Gauss-Legendre replacement of the cell sum by the continuum integral,
    sum_q f(q) -> (V / 2 pi^2) int dq q^2 f(q), with V = L^3."""
    x, w = np.polynomial.legendre.leggauss(n_q)
    hi = q_max_over_kF * ueg.kF
    q, wq = 0.5 * hi * (x + 1.0), 0.5 * hi * w
    return q, ueg.L ** 3 * q * q * wq / (2.0 * np.pi ** 2)


# ----------------------------------------------------------------------
# 3.  Ideal reference S_0(q, w_n)
# ----------------------------------------------------------------------
def lindhard_finite_N(ueg, qvec, n, k_cutoff_factor=8.0):
    """S_0(q, w_n) by direct sum over the cell's discrete k-lattice, with the
    grand-canonical chemical potential fixed by <N> = N. Shape (len(qvec), len(n)).

    S_0 = (2/N) sum_k (n_k - n_{k+q}) de / (w^2 + de^2),  de = e_{k+q} - e_k.
    The map k -> -k - q takes the lattice onto itself and reverses de, so the
    sum equals (4/N) sum_k n_k de / (w^2 + de^2): only occupied k contribute,
    and the finite k grid is exact for any q, also beyond its edge. (Summing
    n_{k+q} over the grid loses the terms with k near -q once q nears the edge,
    which halves S_0 there and breaks the f-sum rule; at Theta = 1 this
    happens for q above about 8 k_F, i.e. only for the r_s = 5 shells.)
    Exactly degenerate pairs (|k+q| = |k|, common on a cubic lattice) are taken
    in their limit beta n (1 - n), which contributes only at w = 0.
    """
    b = ueg.beta
    dk = 2.0 * np.pi / ueg.L
    m = int(np.ceil(k_cutoff_factor * ueg.kF / dk))
    g = np.arange(-m, m + 1)
    k = dk * np.stack([a.ravel() for a in np.meshgrid(g, g, g, indexing="ij")], axis=1)
    eps = 0.5 * np.sum(k * k, axis=1)

    mu = brentq(lambda mu: 2.0 * occupation(eps, mu, b).sum() - ueg.N,
                -200.0 / b - 5.0, 5.0 * ueg.EF + 200.0 / b, xtol=1e-13)
    occ = occupation(eps, mu, b)

    w = matsubara(n, b)
    zero = np.flatnonzero(np.asarray(n) == 0)
    out = np.empty((len(qvec), len(n)))
    for i, qv in enumerate(np.atleast_2d(qvec)):
        eps_kq = 0.5 * np.sum((k + qv) ** 2, axis=1)
        de = eps_kq - eps
        deg = np.abs(de) < 1e-12 * max(1.0, float(np.max(np.abs(eps))))
        safe = np.where(deg, 1.0, de)
        num = np.where(deg, 0.0, 2.0 * occ * safe)
        acc = num @ (1.0 / (w[None, :] ** 2 + safe[:, None] ** 2))
        if zero.size and deg.any():
            acc[zero[0]] += b * float(np.sum(occ[deg] * (1.0 - occ[deg])))
        out[i] = 2.0 * acc / ueg.N
    return out


def lindhard_continuum(ueg, q, n, n_panels=36, n_grade=25, n_per_panel=16):
    """S_0(q, w_n) in the thermodynamic limit. The angular integral is done
    analytically; the radial one by composite Gauss-Legendre.

    S_0 = 2 / (4 pi^2 rho q) int dk k n(k) ln{[w^2 + (q^2/2 + kq)^2] / [w^2 + (q^2/2 - kq)^2]}

    At w = 0 the logarithm is singular at k = q/2, where q^2/2 - kq vanishes, so
    for each q the panels end there and are graded geometrically toward it
    (n_grade halvings on each side). Panels that miss this point leave the static
    component off by up to 2e-3; the w != 0 components are smooth.
    """
    q = np.atleast_1d(np.asarray(q, float))
    b, mu = ueg.beta, ueg.mu
    w2 = matsubara(n, b) ** 2
    k_hi = 12.0 * max(ueg.kF, np.sqrt(2.0 * max(mu, 0.0) + 20.0 / b))
    xg, wg = np.polynomial.legendre.leggauss(n_per_panel)

    out = np.empty((q.size, w2.size))
    for i, qi in enumerate(q):
        s = 0.5 * qi
        br = list(np.linspace(0.0, k_hi, n_panels + 1))
        if 0.0 < s < k_hi:
            br += [s] + [s - s * 2.0 ** -j for j in range(1, n_grade)] \
                      + [s + (k_hi - s) * 2.0 ** -j for j in range(1, n_grade)]
        br = np.unique(np.clip(br, 0.0, k_hi))
        lo, hi = br[:-1], br[1:]
        k = (0.5 * (hi - lo)[:, None] * xg + 0.5 * (hi + lo)[:, None]).ravel()
        wk = (0.5 * (hi - lo)[:, None] * wg).ravel()
        A, kq = 0.5 * qi * qi, k * qi
        # ln(num / den) = log1p(4 A kq / den), since num - den = 4 A kq
        den = w2[None, :] + ((A - kq) ** 2)[:, None]
        out[i] = (wk * k * occupation(0.5 * k * k, mu, b)) @ np.log1p((4.0 * A * kq)[:, None] / den)
    return 2.0 * out / (4.0 * np.pi ** 2 * ueg.rho * q[:, None])


def rpa(S0, q, rho):
    """S in the random phase approximation, S_0 / (1 + U_q rho S_0)."""
    return S0 / (1.0 + coulomb(q)[:, None] * rho * S0)


# ----------------------------------------------------------------------
# 4.  Static local-field reconstruction of S(q, w_n) from S_PIMC(q)
# ----------------------------------------------------------------------
def reconstruct_static(ueg, q, qvec, S_pimc, n_max=300, lever_min=0.05, n_scan=4000):
    """Solve, shell by shell, for the frequency-independent G(q) with

        chi = chi_0 / [1 - U_q (1 - G) chi_0],   beta^-1 sum_n S(q, w_n) = S_PIMC(q).

    Satisfies the PIMC S(q) and the f-sum rule by construction; neglects only
    the frequency dependence of G. Returns (n, S, S0, G) on n = -n_max .. n_max.

    lever_min: at large q both S_PIMC and S_0 tend to 1, the screening lever
    |U_q chi_0(q, 0)| vanishes, and the fit would push their O(1e-3) difference
    onto G. Past the last shell with lever > lever_min, G is frozen.
    """
    n = np.arange(0, n_max + 1)
    S0 = lindhard_finite_N(ueg, qvec, n)
    chi0 = -ueg.rho * S0
    U = coulomb(q)
    # sum over |n| > n_max of the exact f-sum asymptote S ~ q^2 / w^2
    tail = q ** 2 * 2.0 * (ueg.beta / (2.0 * np.pi)) ** 2 * hurwitz_zeta(2, n_max + 1) / ueg.beta

    def model(i, G):
        s = -(chi0[i] / (1.0 - U[i] * (1.0 - G) * chi0[i])) / ueg.rho
        return (2.0 * s.sum() - s[0]) / ueg.beta + tail[i], s

    # fit up to the last shell with lever > lever_min; beyond it G is frozen, so
    # those shells are not fitted (their S(q) no longer constrains G)
    good = np.abs(U * chi0[:, 0]) > lever_min
    last = int(np.flatnonzero(good)[-1]) if good.any() else q.size - 1
    G = np.empty(q.size)
    S = np.empty_like(S0)
    for i in range(last + 1):
        f = lambda g: model(i, g)[0] - S_pimc[i]
        # S(q) rises monotonically in G up to the pole of the screened response
        g_pole = 1.0 + 1.0 / float(np.max(U[i] * (-chi0[i])))
        grid = np.linspace(-5.0, g_pole - 1e-9, n_scan)
        vals = np.array([f(g) for g in grid])
        ok = np.isfinite(vals)
        j = np.flatnonzero(np.diff(np.sign(vals[ok])) != 0)
        if j.size == 0:
            raise RuntimeError(f"no G reproduces S(q) at q/kF = {q[i] / ueg.kF:.3f}")
        G[i] = brentq(f, grid[ok][j[0]], grid[ok][j[0] + 1], xtol=1e-14, rtol=1e-14)
        S[i] = model(i, G[i])[1]

    G[last + 1:] = G[last]
    for i in range(last + 1, q.size):
        S[i] = model(i, G[i])[1]

    mirror = lambda A: np.concatenate([A[:, :0:-1], A], axis=1)     # S(-n) = S(n)
    return np.arange(-n_max, n_max + 1), mirror(S), mirror(S0), G


# ----------------------------------------------------------------------
# 5.  Closed form, Eq. (Fcorr), with the analytic high-frequency tail
# ----------------------------------------------------------------------
def beta_dF_corr(S, S0, q, weight, n, beta, tail_window):
    """beta dF_corr = (1/2) sum_{q,n} [ln y - y + 1],  y = S_0 / S.

    Beyond |n| = n_max, write S ~ a/w^2 + b/w^4 with a = q^2 fixed by the f-sum
    rule (identical for S and S_0 in the continuum) and b fitted on the
    `tail_window` largest |n|. The summand then falls as -(b - b0)^2 / (4 a^2 w^4)
    and its remainder is summed with the Hurwitz zeta function.
    """
    y = S0 / S
    body = 0.5 * float(np.sum(weight[:, None] * (np.log(y) - y + 1.0)))

    sel = np.argsort(-np.abs(n), kind="stable")[:tail_window]
    w = matsubara(n[sel], beta)
    u = 1.0 / w ** 2
    a = q ** 2
    fit_b = lambda A: ((A[:, sel] * w ** 2 - a[:, None]) @ u) / (u @ u)
    d = (fit_b(S) - fit_b(S0)) / a
    n_max = int(np.max(np.abs(n)))
    scale4 = 2.0 * (beta / (2.0 * np.pi)) ** 4 * hurwitz_zeta(4, n_max + 1)
    return body + float(np.sum(weight * (-0.25 * d * d))) * scale4


def finite_size_shift(ueg, q, weight, n_max=200):
    """(continuum - cell) value of beta dF_corr, both in the random phase
    approximation. Added to the PIMC cell value to reach the thermodynamic limit.
    The error lives at small q, where the cell has no modes below 2 pi / L and
    where the random phase approximation is most accurate."""
    n = np.arange(-n_max, n_max + 1)
    window = max(8, n_max // 5)

    def value(qq, ww):
        S0 = lindhard_continuum(ueg, qq, n)
        return beta_dF_corr(rpa(S0, qq, ueg.rho), S0, qq, ww, n, ueg.beta, window)

    return value(*continuum_q_grid(ueg)) - value(q, weight)


# ----------------------------------------------------------------------
# 6.  One density, end to end
# ----------------------------------------------------------------------
def estimate_f_c(rs, theta, N, q_over_kF, S_pimc, n_max=300, tail_window=80,
                 lever_min=0.05):
    """Closed-form correlation free energy per electron, in Hartree, from a
    PIMC static structure factor on the cell's |q| shells."""
    ueg = UEG(rs, theta, N)
    q, qvec, weight = lattice_shells(q_over_kF, ueg)
    n, S, S0, G = reconstruct_static(ueg, q, qvec, np.asarray(S_pimc, float),
                                     n_max, lever_min)
    if np.any(S <= 0) or np.any(S0 <= 0):
        raise RuntimeError("non-positive S or S_0: the log-functional is undefined")
    cell = beta_dF_corr(S, S0, q, weight, n, ueg.beta, tail_window)
    shift = finite_size_shift(ueg, q, weight)
    per = 1.0 / (N * ueg.beta)
    return dict(rs=rs, theta=theta, N=N, f_c=(cell + shift) * per,
                f_c_cell=cell * per, finite_size_shift=shift * per,
                G_min=float(G.min()), G_max=float(G.max()))


# ----------------------------------------------------------------------
# 7.  Input
# ----------------------------------------------------------------------
def read_gnuplot_table(path):
    """{curve title: (x, y)} from a gnuplot 'set table' file with
    '# Curve title: "..."' headers, as deposited on RODARE."""
    curves, cur = {}, None
    for line in open(path):
        s = line.strip()
        if s.startswith("# Curve title"):
            cur = []
            curves[s.split(":", 1)[1].strip().strip('"')] = cur
        elif s and not s.startswith("#") and cur is not None:
            p = s.split()
            cur.append((float(p[0]), float(p[1])))
    return {t: (np.array(v)[:, 0], np.array(v)[:, 1]) for t, v in curves.items()}


# ----------------------------------------------------------------------
# 8.  S(q, w_n) from PIMC imaginary-time correlation functions (no model)
# ----------------------------------------------------------------------
# n^2 = |q / (2 pi / L)|^2 of the k indices 0..99 of each set of RODARE 10.14278/rodare.3923
# (T. Chuna et al., Phys. Rev. B 112, 125112 (2025)), read from chi_q_rs20_theta1.dat. A repeated value
# marks a file that is a byte-identical copy of the previous shell; 15 magnitudes with n^2 <= 118
# (53, 62, 68, 73, 77, 82, 85, 89, 99, 102, 105, 108, 110, 116, 118) have no file.
K_INDEX_N2 = (
    1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 29, 29, 29,
    30, 32, 32, 32, 33, 34, 35, 36, 37, 38, 40, 40, 40, 41, 42, 43, 44, 45, 46, 48, 48, 49, 50, 51, 52,
    54, 56, 56, 57, 58, 59, 61, 61, 64, 64, 65, 66, 67, 69, 70, 72, 72, 74, 75, 76, 78, 80, 80, 81, 83,
    84, 86, 88, 88, 90, 91, 93, 94, 96, 97, 98, 100, 101, 104, 104, 106, 107, 109, 113, 113, 114, 115,
    117)


def cell_shells(n2_max=118):
    """n^2 of every reciprocal-lattice shell with 0 < n^2 <= n2_max: the integers that are sums of
    three squares. n2_max = 118 gives the 100 shells of the PIMC cell up to 6.81 kF."""
    r = int(np.sqrt(n2_max)) + 1
    s = {a * a + b * b + c * c for a in range(r) for b in range(r) for c in range(r)}
    return np.array(sorted(m for m in s if 0 < m <= n2_max))


def read_itcf_archive(path):
    """Leave-one-out binned F(q, tau) from one RODARE 3923 archive (.tar.gz), read in place.

    Returns (n2, tau, F, D): the n^2 of the distinct shells (sorted), the imaginary times
    tau_j = j beta / P (j = 0..P), the full-data F (shells x P+1, per particle, F(q, 0) = S(q)), and the
    leave-one-out bins as deviations D = F_bin - F (shells x P+1 x M, float32). Duplicate files are
    dropped after checking that they are byte-identical copies.
    """
    import hashlib
    import io
    import re
    import tarfile
    pat = re.compile(r"correlations/JKFData_k_index(\d+)\D")
    rows, digest, n2 = {}, {}, []
    with tarfile.open(path, "r:gz") as tf:
        members = sorted((m for m in tf.getmembers() if pat.search(m.name)),
                         key=lambda m: int(pat.search(m.name).group(1)))
        if len(members) != len(K_INDEX_N2):
            raise ValueError(f"{path}: {len(members)} k-index files, expected {len(K_INDEX_N2)}")
        for k, m in enumerate(members):
            raw = tf.extractfile(m).read()
            h = hashlib.md5(raw).hexdigest()
            m2 = K_INDEX_N2[k]
            if m2 in digest:
                if digest[m2] != h:
                    raise ValueError(f"k index {k}: same |q| as an earlier file but different data")
                continue
            digest[m2] = h
            rows[m2] = np.loadtxt(io.BytesIO(raw), delimiter=",", skiprows=1)
            n2.append(m2)
    n2 = np.array(sorted(n2))
    tau = rows[n2[0]][:, 0]
    F = np.array([rows[m][:, 1] for m in n2])
    D = np.array([(rows[m][:, 2:] - rows[m][:, 1:2]) for m in n2], dtype=np.float32)
    return n2, tau, F, D


def itcf_to_matsubara(F, beta, q2, n):
    """S(q, w_n) = int_0^beta e^{i w_n tau} F(q, tau) dtau from F sampled on tau_j = j beta / P,
    j = 0..P, with F(beta) = F(0).

    F is interpolated by a cubic spline whose end slopes, -q^2/2 at tau = 0 and +q^2/2 at tau = beta,
    are fixed by the f-sum rule, and the spline is integrated exactly against e^{i w_n tau}. Repeated
    integration by parts gives, for n != 0,
        S(w_n) = q^2 / w_n^2 + Re[i (s''(beta) - s''(0)) / w_n^3] + (1/w_n^4) Re sum_k J_k e^{i w_n tau_k},
    with J_k the jump of the spline's third derivative at knot k; n = 0 is the integral of the spline.
    F may carry extra trailing axes (for example the jackknife bins); they are transformed together.
    """
    from scipy.interpolate import CubicSpline
    F = np.asarray(F, float)
    P = F.shape[0] - 1
    tau = np.arange(P + 1) * beta / P
    s0 = np.full(F.shape[1:], -0.5 * q2)
    cs = CubicSpline(tau, F, axis=0, bc_type=((1, s0), (1, -s0)))
    c3 = 6.0 * cs.c[0]                                   # s''' on each interval
    J = c3 - np.roll(c3, 1, axis=0)                      # jumps at the knots (periodic)
    d1 = cs(beta, 1) - cs(0.0, 1)                        # = q^2
    d2 = cs(beta, 2) - cs(0.0, 2)
    n = np.atleast_1d(np.asarray(n))
    out = np.empty((n.size,) + F.shape[1:])
    nz = n != 0
    if nz.any():
        w = (2.0 * np.pi * n[nz] / beta).reshape((-1,) + (1,) * (F.ndim - 1))
        ph = np.exp(2j * np.pi * np.outer(n[nz], np.arange(P)) / P)
        Dk = np.tensordot(ph, J, axes=(1, 0))
        out[nz] = (d1 / w ** 2 - d2 / (1j * w ** 3) + Dk / w ** 4).real
    if (~nz).any():
        out[~nz] = cs.integrate(0.0, beta)
    return out


def beta_dF_corr_itcf(ueg, n2, F, D, n_c=20, n2_max=118):
    """beta dF_corr, Eq. (Fcorr), on the N-electron cell from PIMC F(q, tau), with no model.

    S(q, w_n) for |n| <= n_c comes from itcf_to_matsubara and S_0 from lindhard_finite_N on the same
    lattice. The summand (1/2)[ln y - y + 1], y = S_0/S, is quadratic in S/S_0 - 1, so the statistical
    noise of F biases it downward; the bias is removed term by term with the jackknife estimate
    (M - 1)(mean over leave-one-out bins - full value), exact for the O(1/M) bias of a quadratic.
    Terms with |n| > n_c are dropped: at Theta = 1 they carry less than 0.2% of the sum in the data at
    r_s >= 10 and less than 0.001% in the static-G model, while at r_s <= 5 the data there are dominated
    by noise. The cell sum runs over all shells with n^2 <= n2_max; shells without data are filled by
    linear interpolation in q of the per-vector summand.

    Returns dict: cell (beta dF_corr of the cell), err (jackknife), q_cell, weight_cell, filled,
    q, t (bias-corrected per-vector summand of the shells with data), bias.
    """
    b = ueg.beta
    unit = 2.0 * np.pi / ueg.L / ueg.kF
    q, qvec, _ = lattice_shells(np.sqrt(n2) * unit, ueg)
    n = np.arange(n_c + 1)
    mult = np.where(n == 0, 1.0, 2.0)
    S0 = lindhard_finite_N(ueg, qvec, n)
    M = D.shape[2]
    t_full, t_bins = np.empty(q.size), np.empty((q.size, M))
    for i in range(q.size):
        S = itcf_to_matsubara(F[i], b, q[i] ** 2, n)
        Sb = itcf_to_matsubara(F[i][:, None] + D[i].astype(float), b, q[i] ** 2, n)
        if np.any(S <= 0) or np.any(Sb <= 0):
            raise RuntimeError(f"non-positive S(q, w_n) at q/kF = {q[i] / ueg.kF:.3f}")
        y, yb = S0[i] / S, S0[i][:, None] / Sb
        t_full[i] = 0.5 * np.sum(mult * (np.log(y) - y + 1.0))
        t_bins[i] = 0.5 * np.sum(mult[:, None] * (np.log(yb) - yb + 1.0), axis=0)
    bias = (M - 1) * (t_bins.mean(axis=1) - t_full)
    t = t_full - bias

    n2c = cell_shells(n2_max)
    qc, _, wc = lattice_shells(np.sqrt(n2c) * unit, ueg)
    have = np.isin(n2c, n2)
    idx = np.clip(np.searchsorted(n2, n2c), 0, n2.size - 1)
    fill = lambda v: np.where(have, v[idx], np.interp(qc, q, v))
    totals = np.array([np.sum(wc * fill(t_bins[:, k])) for k in range(M)])
    err = np.sqrt((M - 1) / M * np.sum((totals - totals.mean()) ** 2))
    return dict(cell=float(np.sum(wc * fill(t))), err=float(err), q_cell=qc, weight_cell=wc,
                filled=~have, q=q, t=t, bias=bias)
