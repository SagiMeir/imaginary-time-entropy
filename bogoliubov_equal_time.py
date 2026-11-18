"""
Bogoliubov gas: why the equal-time structure factor is not enough.

Fig. 1 of the Letter, with g the coupling constant of
the Bogoliubov Hamiltonian. Units hbar = 2m = 1, energies in units of rho*g, eps_q = q^2,
E_q = sqrt(eps_q^2 + 2 eps_q). For
each mode q (the term of the sum over modes):

    exact          H_q = s(beta E_q) - s(beta eps_q),  s(y) = y/(e^y - 1) - ln(1 - e^-y)
    closed form    Eq. (Hex_explicit) for that mode, summed over Matsubara frequencies
                   (plain sum to |n| <= NMAX for the printed table; with an exact tail for Fig. 1)
    equal time     the classical relation (1/2)(ln r - r + 1), r = S(q)/S_0(q),
                   with S(q) = (eps_q/E_q) coth(beta E_q/2), S_0(q) = coth(beta eps_q/2)

The printed table lists single modes, as a check (it is not in the paper).
Fig. 1 shows the excess entropy summed over all modes, with the density of modes
V q^2 dq / 2 pi^2, per volume, -H_ex / V in units of (2 m rho g / hbar^2)^(3/2),
as a function of beta*rho*g, the only parameter it depends on.

Run:  python3 bogoliubov_equal_time.py   # prints the table and the totals, writes
                                         # fig_bose_equal_time.pdf/.png (a few seconds)
Typography, colours, line width and marker size come from reproduce.py, so the
figure reads as one set with benchmark.pdf and electron_gas/fig_fc_prl.pdf.
"""
import os
import warnings
import numpy as np
from scipy.integrate import quad

from reproduce import _rcparams, C_EXACT, C_CLOSED, LW

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FIG = os.path.join(HERE, 'fig_bose_equal_time')

BETAS = [0.2, 1.0, 5.0]          # beta * rho * g of the printed table
Q_TABLE = [0.25, 0.5, 1.0, 2.0]  # the q of the printed table
NMAX = 400000                    # Matsubara cutoff for the printed table
Q_TAIL = 200.0                   # beyond: q^2 * equal-time term -> -1/(4 q^2), exact -> 0
C_EQUAL = '#009E73'              # equal-time estimate (Okabe-Ito bluish green)
LABEL_OURS = 'Closed form (ours)'
LABEL_CLASSICAL = 'Classical closed form\n(Ariel & Diamant)'   # two rows, so the legend fits the narrow axes
FIGSIZE = (2.50, 1.56)           # width of electron_gas/fig_fc_prl.pdf (Fig. 3); the height fits the axes
AX_ASPECT = 2 / (1 + 5 ** 0.5)   # axes height / width = 1 / golden ratio (axes 2.01 x 1.24 in)
BETA_RANGE = (0.05, 20.0)        # beta * rho * g on the x axis
BETA_CURVE = np.logspace(np.log10(0.05), np.log10(20.0), 61)   # classical curve
BETA_CF = np.logspace(np.log10(0.05), np.log10(20.0), 13)      # closed form, summed numerically


def energies(q):
    eps = q * q
    return eps, np.sqrt(eps * eps + 2.0 * eps)


def s_mode(y):
    """Entropy of one harmonic mode, y = beta * energy."""
    with np.errstate(over='ignore'):
        return y / np.expm1(y) - np.log(-np.expm1(-y))


def h_exact(beta, q):
    eps, E = energies(q)
    return s_mode(beta * E) - s_mode(beta * eps)


def h_equal_time(beta, q):
    eps, E = energies(q)
    r = (eps / E) / np.tanh(beta * E / 2) * np.tanh(beta * eps / 2)
    return 0.5 * (np.log(r) - r + 1.0)


def closed_form_summand(w2, A, B):
    """Summand of Eq. (Hex_explicit) for one mode, with A = eps^2, B = E^2 and
    w2 = (hbar omega_n)^2: S = 2 eps/(w2 + B), S_0 = 2 eps/(w2 + A), and
    d S_0 / d beta at fixed integer n, beta dS_0/dbeta = 4 eps w2/(w2 + A)^2, give
    (1/2)[ln((w2 + A)/(w2 + B)) + 2 (B - A) w2 / ((w2 + B)(w2 + A))]."""
    return 0.5 * (-np.log1p((B - A) / (w2 + A)) + 2 * (B - A) * w2 / ((w2 + B) * (w2 + A)))


def h_closed_form(beta, q, nmax=None):
    """Eq. (Hex_explicit) for one mode. With nmax: the plain sum over |n| <= nmax
    (the printed table). Without: the sum over |n| <= N, with omega_N >= 10 E, plus the
    remainder |n| > N by the Euler-Maclaurin formula, whose integral is exact
    because the summand has the antiderivative (w/2) ln((w^2 + A)/(w^2 + B)) in w."""
    eps, E = energies(q)
    A, B = eps * eps, E * E
    if nmax:
        n = np.arange(-nmax, nmax + 1, dtype=float)
        return np.sum(closed_form_summand((2.0 * np.pi * n / beta) ** 2, A, B))
    N = max(2000, int(np.ceil(10 * beta * E / (2 * np.pi))))
    n = np.arange(-N, N + 1, dtype=float)
    body = np.sum(closed_form_summand((2.0 * np.pi * n / beta) ** 2, A, B))
    dw = 2.0 * np.pi / beta
    w0 = (N + 0.5) * dw
    integral = 0.5 * w0 * np.log1p((B - A) / (w0 * w0 + A)) / dw
    ds_dw = 0.5 * (2 * w0 / (w0 * w0 + A) - 2 * w0 / (w0 * w0 + B)
                   - 4 * B * w0 / (w0 * w0 + B) ** 2 + 4 * A * w0 / (w0 * w0 + A) ** 2)
    return body + 2 * (integral - ds_dw * dw / 24)


def totals(beta):
    """Sums over all modes with weight q^2 (the common factor V / 2 pi^2 dropped)."""
    def integral(f, cuts):
        return sum(quad(lambda q: q * q * f(beta, q), a, b, limit=500)[0]
                   for a, b in zip(cuts[:-1], cuts[1:]))
    exact = integral(h_exact, [0, 1, 5, 30])
    equal = integral(h_equal_time, [0, 1, 5, 30, Q_TAIL]) - 1.0 / (4 * Q_TAIL)
    return exact, equal


def print_table():
    print(f"{'beta':>5} {'q':>5} {'exact':>9} {'closed':>9} {'equal-t':>9} {'dev %':>6}")
    for b in BETAS:
        for q in Q_TABLE:
            ex, cf = h_exact(b, q), h_closed_form(b, q, nmax=NMAX)
            eq = h_equal_time(b, q)
            dev = f"{100 * abs(eq - ex) / abs(ex):.0f}" if abs(ex) > 1e-3 else "--"
            print(f"{b:>5g} {q:>5g} {ex:>9.4f} {cf:>9.4f} {eq:>9.4f} {dev:>6}")
    print("\nsums over all modes (weight q^2):")
    for b in BETAS:
        ex, eq = totals(b)
        print(f"  beta rho g = {b:g}: exact {ex:.6f}, equal-time {eq:.6f}, ratio {eq / ex:.4f}, "
              f"deviation {100 * abs(eq - ex) / abs(ex):.1f}%")


def total_closed_form_ratio(beta, n_q=800):
    """Closed form summed over all modes, divided by the exact sum on the same
    q grid (trapezoid in ln q). Above q_c, where beta q^2 >= 60, the exact terms
    are below e^-60 and the closed form, which equals them (Supplemental Material, Sec. S5.1), is not
    summed."""
    q_c = max(3.0, np.sqrt(60.0 / beta))
    q = np.logspace(-3, np.log10(q_c), n_q)
    lnq = np.log(q)
    cf = np.array([h_closed_form(beta, x) for x in q])
    return np.trapz(q ** 3 * cf, lnq) / np.trapz(q ** 3 * h_exact(beta, q), lnq)


def make_figure():
    import matplotlib
    matplotlib.use('Agg')
    rc, usetex = _rcparams()
    matplotlib.rcParams.update(rc)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter, LogLocator, NullFormatter

    # -H_ex / V in units of (2 m rho g / hbar^2)^(3/2): sum over modes -> (V / 2 pi^2) int dq q^2
    pairs = np.array([totals(b) for b in BETA_CURVE]) / (-2 * np.pi ** 2)
    exact, classical = pairs[:, 0], pairs[:, 1]
    ratio_cf = np.array([total_closed_form_ratio(b) for b in BETA_CF])
    print("closed form / exact, max |deviation from 1| = %.1e" % np.max(np.abs(ratio_cf - 1)))
    ours = ratio_cf * np.array([totals(b)[0] for b in BETA_CF]) / (-2 * np.pi ** 2)
    label_classical = LABEL_CLASSICAL.replace('&', r'\&') if usetex else LABEL_CLASSICAL

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.015, h_pad=0.015)
    ax.set_box_aspect(AX_ASPECT)
    ax.plot(BETA_CURVE, exact, '-', color=C_EXACT, lw=LW, zorder=2, label='Exact')
    ax.plot(BETA_CF, ours, '--', color=C_CLOSED, lw=LW, zorder=3, label=LABEL_OURS)
    ax.plot(BETA_CURVE, classical, '-.', color=C_EQUAL, lw=LW, zorder=2, label=label_classical)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*BETA_RANGE)
    ax.set_ylim(1e-4, 1)
    ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
    ax.xaxis.set_major_formatter(FixedFormatter(['0.1', '1', '10']))
    ax.yaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1, 1]))
    ax.yaxis.set_major_formatter(FixedFormatter([r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$',
                                                 r'$10^{-1}$', r'$1$']))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'$\beta\bar\rho g$', labelpad=1.5)
    ax.set_ylabel(r'$-H_{\mathrm{ex}}/V$')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', length=3)
    ax.tick_params(which='minor', length=1.5)
    ax.legend(frameon=False, loc='lower left', handlelength=2.2, handletextpad=0.6,
              borderaxespad=0.4)
    fig.savefig(OUT_FIG + '.pdf')          # vector, at final size, no rescaling
    fig.savefig(OUT_FIG + '.png', dpi=300)
    w, h = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).size
    print(f"wrote {OUT_FIG}.pdf/.png (usetex={usetex}, axes {w:.2f} x {h:.2f} in)")


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    print_table()
    make_figure()
