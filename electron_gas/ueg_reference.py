"""
Reference correlation free energy of the spin-unpolarized electron gas,

    f_c(rs, Theta) = f_xc(rs, Theta) - f_x(rs, Theta),

the curve the closed form is graded against. Hartree per electron.

f_xc   KSDT parametrization of the exchange-correlation free energy, functional
       form of V. V. Karasiev, T. Sjostrom, J. Dufty and S. B. Trickey,
       Phys. Rev. Lett. 112, 076403 (2014), with the corrected ("corrKSDT")
       parameter set. Fitted to quantum Monte Carlo data; do not use it as a
       reference outside the range of those data.

f_x    exact first-order exchange free energy of the ideal Fermi gas at finite
       temperature, in the form of B. Horovitz and R. Thieberger,
       Physica 71, 99 (1974):
           f_x = -2 rs^3 / (3 pi^2 beta^2) int_{-inf}^{beta mu} I_{-1/2}(eta)^2 deta,
       with I_nu the complete Fermi-Dirac integral and mu the ideal-gas
       chemical potential.

Written from the published formulas. Checked against the independent
implementation in uegpy (github.com/fdmalone/uegpy) to better than 1e-9.
Depends only on numpy and scipy.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit


def fermi_energy(rs):
    return 0.5 * ((9.0 * np.pi / 4.0) ** (1.0 / 3.0) / rs) ** 2


def fermi_integral(nu, eta):
    """I_nu(eta) = int_0^inf x^nu / (exp(x - eta) + 1) dx, for nu = -1/2, 1/2.
    Substituting x = y^2 removes the x^{-1/2} endpoint singularity."""
    f = lambda y: 2.0 * y ** (2.0 * nu + 1.0) * expit(eta - y * y)
    return quad(f, 0.0, np.inf, limit=200)[0]


def chemical_potential(rs, theta):
    """Ideal-gas mu, from n = (sqrt 2 / pi^2) beta^{-3/2} I_{1/2}(beta mu)."""
    ef = fermi_energy(rs)
    beta = 1.0 / (theta * ef)
    n = 3.0 / (4.0 * np.pi * rs ** 3)
    g = lambda mu: np.sqrt(2.0) / np.pi ** 2 * beta ** -1.5 * fermi_integral(0.5, beta * mu) - n
    return brentq(g, -60.0 / beta - 10.0 * ef, 10.0 * ef + 60.0 / beta, xtol=1e-15, rtol=1e-14)


def f_x(rs, theta):
    """Finite-temperature Hartree-Fock exchange free energy per electron."""
    beta = 1.0 / (theta * fermi_energy(rs))
    eta0 = beta * chemical_potential(rs, theta)
    integral = quad(lambda e: fermi_integral(-0.5, e) ** 2, -np.inf, eta0, limit=200)[0]
    return -2.0 * rs ** 3 / (3.0 * np.pi ** 2 * beta ** 2) * integral


def f_xc_ksdt(rs, theta):
    """corrKSDT exchange-correlation free energy per electron, zeta = 0:

        f_xc = -(1/rs) [a(t) + b(t) rs^{1/2} + c(t) rs] / [1 + d(t) rs^{1/2} + e(t) rs]
    """
    t = theta
    th1, ths = np.tanh(1.0 / t), np.tanh(1.0 / np.sqrt(t))
    a = (0.610887 * th1 * (0.75 + 3.04363 * t ** 2 - 0.09227 * t ** 3 + 1.7035 * t ** 4)
         / (1.0 + 8.31051 * t ** 2 + 5.1105 * t ** 4))
    b = ths * (0.342554 + 9.141315 * t ** 2 + 0.448483 * t ** 4) / (1.0 + 18.553096 * t ** 2 + 1.054151 * t ** 4)
    e = th1 * (0.255415 + 0.931933 * t ** 2 + 0.115398 * t ** 4) / (1.0 + 17.234117 * t ** 2 + 0.451437 * t ** 4)
    c = (0.875130 - 0.256320 * np.exp(-0.953988 / t)) * e
    d = ths * (0.725917 + 2.237347 * t ** 2 + 0.280748 * t ** 4) / (1.0 + 4.185911 * t ** 2 + 0.692183 * t ** 4)
    sr = np.sqrt(rs)
    return -(a + b * sr + c * rs) / (rs * (1.0 + d * sr + e * rs))


def f_c_reference(rs, theta):
    """f_c = f_xc(corrKSDT) - f_x(exact)."""
    return f_xc_ksdt(rs, theta) - f_x(rs, theta)


def f_c_reference_curve(rs_values, theta):
    """f_c on many densities at one Theta.

    At fixed Theta, beta mu is independent of rs and f_x scales exactly as 1/rs,
    so the exchange integral is evaluated once and rescaled.
    """
    rs_values = np.asarray(rs_values, float)
    rs0 = float(rs_values[0])
    fx = f_x(rs0, theta) * rs0 / rs_values
    return np.array([f_xc_ksdt(r, theta) for r in rs_values]) - fx
