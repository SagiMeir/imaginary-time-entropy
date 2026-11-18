"""
Reproduce every number and figure in the spinless-fermion benchmark.

Outputs, all written to ./ :
    entropy_data.json   H_exact, H_closed, H_num vs (beta, W) for the figure
    free_energy.json     exact vs Gaussian-approximation beta*F_ex' for Table S1
    benchmark.pdf/.png   the four-panel figure

    entropy_data_L.json  the same three routes over ring sizes LS
    benchmark_L.pdf/.png  4x4 figure: rows = L at half filling, cols = beta t

Run:  python3 reproduce.py            # tables + data + figure
      python3 reproduce.py --quick    # only beta t = 2 (fast smoke test)
      python3 reproduce.py --figure-only   # redraw from entropy_data.json
      python3 reproduce.py --scan-L        # finite-size sweep + 4x4 figure
      python3 reproduce.py --figure-L-only # redraw from entropy_data_L.json

--scan-L checkpoints after every point and resumes, so it can be interrupted.
See LS below for the cost of each ring size.

Depends only on tv.py and exact_maxent.py (and numpy/scipy/matplotlib).
"""
import json
import shutil
import sys
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from tv import (System, hamiltonian, density_mode, kinetic_operator,
                thermal, entropy)
from exact_maxent import Lattice, static_Sq

# ----------------------------------------------------------------------
# Fixed benchmark parameters
# ----------------------------------------------------------------------
L, N = 12, 6          # the benchmark ring (Fig. 2, benchmark.pdf); half filling
BETAS = [0.5, 1.0, 2.0, 4.0]
WS = [0.25, 0.5, 0.75, 1.0]
NMAX = 150            # Matsubara cutoff for the closed-form sums
HB = 1e-3             # finite-difference step for d/dbeta and matching

# Ring sizes for the finite-size scan (benchmark_L.pdf), all at half filling.
#
# SHELL PARITY.  At half filling the free ring is open-shell when N = L/2 is
# even (L = 8, 12, 16, 20: two degenerate ground states, residual entropy ln 2)
# and closed-shell when N = L/2 is odd (L = 6, 10, 14: unique ground state,
# H -> 0 as T -> 0).  The two classes are not comparable at low temperature:
# H/N at beta t = 4 differs between them by an order of magnitude, and the
# relative error of any estimator diverges on the closed-shell rings because
# the exact entropy it is divided by is going to zero.  Keep a single parity
# in LS.  L = 8, 12, 16, 20 (all N even) is the physically clean sequence.
#
# Cost per (L, beta, W) point, measured; the closed form dominates.
#     L    D=C(L,L/2)   closed form    control      one row (16 points)
#     8        70          0.2 s        0.15 s          ~5 s
#    10       252          1.1 s        0.27 s         ~25 s
#    12       924           60 s         3.9 s         ~17 min
#    14      3432          857 s         206 s          ~4.8 h
#    16     12870        ~4.3 h *       ~3.4 h          ~110 h   (* needs 34 GB)
#    20    184756       out of reach: the dense H alone is 273 GB
# L = 16 also needs Sqn rewritten to bin the Lehmann sum; L = 20 needs a
# momentum-resolved ED, which still would not give the inference routes.
LS = [8, 10, 12, 14]

_KUNIT = {}


def kunit(L, N):
    """Unit-hopping kinetic operator for this ring, built once per (L, N)."""
    if (L, N) not in _KUNIT:
        _KUNIT[(L, N)] = kinetic_operator(L, N, 1.0)
    return _KUNIT[(L, N)]


def kin(sys_):
    """<H_0>_sys with the UNIT-hopping kinetic operator."""
    return float(np.einsum('a,ia,ij,ja->', sys_.p, sys_.U,
                           kunit(sys_.L, sys_.N), sys_.U,
                           optimize=True).real)


# ----------------------------------------------------------------------
# 1.  Closed-form inference (carries the Gaussian approximation)
#     Uses the rescaled reference gamma* and the exact bridge back to gamma=1.
# ----------------------------------------------------------------------
def gamma_star(inter, beta, L=L, N=N):
    """Scale matching E_kin^0(gamma*) = E_kin (the UV sum-rule condition)."""
    Ekin = kin(inter)
    return brentq(lambda g: g * kin(System(L, N, g, 0.0, beta)) - Ekin,
                  0.4, 3.0, xtol=1e-12)


# Which assembly turns the inferred correlations into a total entropy.
#   'H_rel' (default)  H = H_rel(gamma*) + beta gamma* E_kin + ln I[0; gamma*]
#   'H_ex'             H = H_ex(gamma*) + H_0(gamma*)      -- originally published
# The two bracket the exact result and differ substantially below beta t = 2;
# see closed_form_entropy_hrel for why H_rel is the defensible one.
ESTIMATOR = 'H_rel'


def closed_form_entropy_hrel(beta, W, L=L, N=N):
    """H from Eq. (H_rel_main) plus the exact bridge to the physical reference.

    H_rel = -D(P || P_0^gamma) is evaluated directly as a sum of non-positive
    terms, and

        H = H_rel(gamma) + beta gamma E_kin + ln I[0; gamma]

    is an identity (from ln P_0^gamma = -beta gamma H_0 - ln I[0; gamma]), so
    every approximation sits in H_rel itself.  Three reasons this is the right
    assembly, all measured on the ring:

      * weak coupling: the error vanishes as W^2, matching the order to which
        the Gaussian approximation is exact.  The H_ex assembly errs at O(W).
      * high temperature: the error falls roughly as beta^3 toward the
        classical limit, against O(beta) for the H_ex assembly.
      * the summand (1/2)[ln x - x + 1] is stationary at x = 1, so the
        ultraviolet modes -- where gamma* forces S/S0 -> 1 -- contribute no
        noise.  Under noisy S this route is ~50x more precise, and it needs no
        high-frequency tail (the sum converges as 1/n^3, not 1/n).

    No beta-derivative is required, so this costs two spectra per point rather
    than four.
    """
    ns = np.arange(-NMAX, NMAX + 1)
    inter = System(L, N, 1.0, W, beta)
    g = gamma_star(inter, beta, L, N)
    ref = System(L, N, g, 0.0, beta)
    Hrel = 0.0
    for k in range(1, L):
        q = 2 * np.pi * k / L
        x = inter.Sqn(q, ns) / ref.Sqn(q, ns)
        Hrel += 0.5 * np.sum(np.log(x) - x + 1.0)
    lnIg = logsumexp(-beta * ref.E)          # ref.E are gamma * (free levels)
    return Hrel + beta * g * kin(inter) + lnIg, inter.S_vn, g


def closed_form_entropy_hex(beta, W, L=L, N=N):
    """The originally published assembly: Eq. (Hex_explicit) + H_0(gamma*).

    Kept so the published numbers remain reproducible.  Two caveats measured on
    the ring: the Matsubara sum converges only as 1/n_max (the ln(S/S0) term
    leaves an O(w^-2) tail), so NMAX = 150 is NOT converged -- add the tail
    correction, whose coefficient (b0-b)/(2 a w^2) is the same as F_ex's -- and
    the error is O(W) at weak coupling rather than O(W^2).
    """
    ns = np.arange(-NMAX, NMAX + 1)
    inter = System(L, N, 1.0, W, beta)
    g = gamma_star(inter, beta, L, N)
    ref = System(L, N, g, 0.0, beta)
    refp = System(L, N, g, 0.0, beta + HB)
    refm = System(L, N, g, 0.0, beta - HB)
    Hex = 0.0
    for k in range(1, L):
        q = 2 * np.pi * k / L
        S = inter.Sqn(q, ns)
        S0 = ref.Sqn(q, ns)
        dS0 = (refp.Sqn(q, ns) - refm.Sqn(q, ns)) / (2 * HB)
        Hex += 0.5 * np.sum(np.log(S / S0) - beta * (S - S0) / S0 ** 2 * dS0)
    return Hex + ref.S_vn, inter.S_vn, g


def closed_form_entropy(beta, W, L=L, N=N):
    """Dispatch on ESTIMATOR.  Returns (H_closed, H_exact, gamma*)."""
    if ESTIMATOR == 'H_rel':
        return closed_form_entropy_hrel(beta, W, L, N)
    if ESTIMATOR == 'H_ex':
        return closed_form_entropy_hex(beta, W, L, N)
    raise ValueError("ESTIMATOR must be 'H_rel' or 'H_ex'")


# ----------------------------------------------------------------------
# 2.  Numerical inference (exact Z_lambda, no Gaussian approximation)
# ----------------------------------------------------------------------
def numerical_entropy(beta, W, L=L, N=N):
    lat = Lattice(L, N, 1.0, beta)
    inter = System(L, N, 1.0, W, beta)
    u, _ = lat.solve(static_Sq(inter))
    _, _, p, _, _ = lat.struct(u)
    p = p[p > 1e-300]
    return float(-(p * np.log(p)).sum())


# ----------------------------------------------------------------------
# 3.  Excess free energy F_ex' of the q!=0 shifted interaction (constant
#     WN^2/L omitted), exact and in the Gaussian approximation.  Exact value
#     from direct ED of the shifted Hamiltonian; Gaussian-approximation value
#     from the RPA ring sum.
# ----------------------------------------------------------------------
def shifted_hamiltonian(W, beta, L=L, N=N):
    """H_0 + U' with U' = (1/2L) sum_{q!=0} U_q rho_q rho_-q."""
    H0, states, _ = hamiltonian(L, N, 1.0, 0.0)
    qs = [2 * np.pi * k / L for k in range(1, L)]
    d2 = np.array([np.abs(density_mode(L, states, q)) ** 2 for q in qs])
    Uq = np.array([2 * W * np.cos(q) for q in qs])
    Ushift = (1.0 / (2 * L)) * (d2.T @ Uq)
    return H0 + np.diag(Ushift)


def free_energy_row(W, beta, free, lnI0, nmax_tail=5000, L=L, N=N):
    qs = [2 * np.pi * k / L for k in range(1, L)]
    lam = np.array([W * np.cos(q) / L for q in qs])          # lambda_q (energy)

    # exact beta F_ex' = ln I0 - ln Z(shifted)
    Es, _ = np.linalg.eigh(shifted_hamiltonian(W, beta, L, N))
    bFex_ex = lnI0 - logsumexp(-beta * Es)

    # Gaussian approximation: beta F_ex' = (1/2) sum_{q!=0,n} ln(1 + 2N lambda_q S0)
    ns = np.arange(1, nmax_tail + 1)
    bFex_tr = 0.0
    for i, q in enumerate(qs):
        S0 = free.Sqn(q, ns)
        S00 = free.Sqn(q, np.array([0]))[0]
        s = 2.0 * np.log1p(2 * N * lam[i] * S0).sum() + np.log1p(2 * N * lam[i] * S00)
        A0 = 2 * (1 - np.cos(q)) * abs(free.Ekin) / N        # tail coefficient
        c = 2 * N * lam[i] * A0 * (beta / (2 * np.pi)) ** 2
        tail = 2 * c * (np.pi ** 2 / 6 - np.sum(1.0 / ns ** 2))
        bFex_tr += 0.5 * (s + tail)

    inter = System(L, N, 1.0, W, beta)
    constraint = beta * N * float(np.dot(lam, static_Sq(inter)))
    Hrel_ex = constraint - bFex_ex
    return dict(W=W, bFex_ex=float(bFex_ex), bFex_tr=float(bFex_tr),
                err_pct=100 * abs(bFex_tr - bFex_ex) / abs(bFex_ex),
                Hrel_ex=float(Hrel_ex),
                cancel_pct=100 * (1 - abs(Hrel_ex) / abs(constraint)))


# ----------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------
def build_entropy_data(betas):
    out = {}
    for beta in betas:
        for W in WS:
            cf, ex, g = closed_form_entropy(beta, W)
            nm = numerical_entropy(beta, W)
            out[f"{beta}_{W}"] = dict(beta=beta, W=W, gamma_star=g,
                                      H_exact=ex, H_closed=cf, H_num=nm,
                                      relerr_closed=abs(cf - ex) / ex)
            print(f"  beta t={beta:<3} W/t={W:<4}  H_exact={ex:.5f}  "
                  f"closed={cf:.5f} ({100*(cf-ex)/ex:+.1f}%)  num={nm:.5f}",
                  flush=True)
    json.dump(out, open('entropy_data.json', 'w'), indent=1)
    return out


def build_free_energy_table(beta=2.0):
    free = System(L, N, 1.0, 0.0, beta)
    lnI0 = logsumexp(-beta * free.E)
    rows = [free_energy_row(W, beta, free, lnI0) for W in WS]
    json.dump(rows, open('free_energy.json', 'w'), indent=1)
    print(f"\n  free-energy table (beta t={beta}):")
    print("   W/t  bFex'_ex  bFex'_tr   err%   cancel%")
    for r in rows:
        print(f"   {r['W']:<4} {r['bFex_ex']:8.4f}  {r['bFex_tr']:8.4f}  "
              f"{r['err_pct']:5.1f}   {r['cancel_pct']:5.1f}")
    return rows


# ----------------------------------------------------------------------
# 4.  Figure
#     Presentation only.  Nothing below changes a computed value: the exact
#     curve is the same ED routine evaluated on a finer W grid, and the
#     inferred series are read verbatim from entropy_data.json.
# ----------------------------------------------------------------------
FIGSIZE  = (7.0, 1.80)     # two-column figure*, width=\textwidth; height set so the legend sits
                           # just above the panel titles (panel size unchanged from 2.0 in)
BOXASPECT = 1.15 / 1.30    # target panel shape (height / width)
NDENSE   = 25              # points on the dense exact W/t grid, [0, 1]
C_EXACT  = 'black'
C_NUM    = '#0072B2'       # inferred, numerical (pair-density control), small blue dots
C_CLOSED = '#D55E00'       # inferred, closed form, orange open circles (as in fig_fc_prl)
MS       = 4.2            # closed-form circles, size and edge width MEW as in fig_fc_prl
MEW      = 0.9
MS_NUM   = 3.0            # filled dots of the numerical route, drawn above the circles
LW       = 1.2            # width of the exact (solid) and closed-form (dashed) lines
SHADE    = False          # band between the exact curve and the closed form (off)
ERR_FMT  = '{:.2g}%'       # W/t = 1 annotation: two significant figures, as quoted in the text
ERR_LABEL = False          # draw that annotation in each panel of benchmark.pdf (off);
                           # the errors are printed either way
DEV_FMT  = r'$\Delta = {:.4f}$'          # absolute-deviation annotation
ANNOT    = None            # None = percent for H/N, Delta for H_ex/N;
                           # 'percent' or 'delta' forces one everywhere.
                           # Delta is |H_closed - H_exact|/N, which is the SAME
                           # number in both, since H_ex just shifts both curves
                           # by the free ring.  It has no denominator, so it is
                           # immune to H (or H_ex) passing near zero.
USETEX   = None            # None = auto-detect LaTeX; True/False to force


def _rcparams():
    """REVTeX-like serif typography.  Real LaTeX when it is installed,
    otherwise matplotlib's bundled Computer Modern through mathtext."""
    usetex = USETEX
    if usetex is None:
        import subprocess
        usetex = bool(shutil.which('latex') and shutil.which('dvipng'))
        if usetex:                       # matplotlib's preamble needs cm-super
            usetex = subprocess.run(['kpsewhich', 'type1ec.sty'],
                                    capture_output=True).returncode == 0
    rc = {'text.usetex': usetex, 'font.family': 'serif', 'font.size': 8,
          'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
          'xtick.labelsize': 7, 'ytick.labelsize': 7,
          'axes.linewidth': 0.7, 'lines.linewidth': 0.9,
          'xtick.major.width': 0.7, 'xtick.minor.width': 0.7,
          'ytick.major.width': 0.7, 'ytick.minor.width': 0.7,
          'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300}
    if not usetex:
        rc.update({'font.serif': ['cmr10', 'STIXGeneral', 'DejaVu Serif'],
                   'mathtext.fontset': 'cm',
                   'axes.unicode_minus': False,        # cmr10 has no U+2212
                   'axes.formatter.use_mathtext': True})
    return rc, usetex


def dense_W_grid():
    """NDENSE points on [0, 1], snapped so the benchmark couplings lie on it."""
    Wg = np.linspace(0.0, 1.0, NDENSE)
    for W in WS:
        Wg[np.abs(Wg - W).argmin()] = W
    return Wg


def exact_entropy_curves(betas, Wgrid, L=L, N=N):
    """H_exact(W) from ED for each beta on a dense W grid.

    Identical to System(L, N, 1, W, beta).S_vn -- the ring is diagonalised once
    per W and the spectrum reused across temperatures (it does not depend on
    beta), so the values are bit-for-bit the same, just on a finer W grid.
    Only eigenvalues are needed, so this stays affordable well past the size at
    which the inference routes become impractical.
    """
    out = {b: [] for b in betas}
    for W in Wgrid:
        H, _, _ = hamiltonian(L, N, 1.0, float(W))
        E, _ = np.linalg.eigh(H)
        for b in betas:
            p, _, _ = thermal(E, b)
            out[b].append(entropy(p))
    return {b: np.asarray(v) for b, v in out.items()}


def _consistent_decimals(ticks):
    """Fewest decimals that label every tick in this panel distinctly *and*
    exactly -- so a tick can never be drawn at 1.005 and labelled '1.00'."""
    for dec in range(1, 7):
        labs = ['%.*f' % (dec, v) for v in ticks]
        if len(set(labs)) == len(labs) and all(
                abs(float(s) - v) < 1e-9 for s, v in zip(labs, ticks)):
            return dec
    raise ValueError('no faithful y-tick format for %s' % (ticks,))


def make_figure(data):
    import matplotlib
    matplotlib.use('Agg')
    rc, usetex = _rcparams()
    matplotlib.rcParams.update(rc)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import (FixedLocator, MultipleLocator, MaxNLocator,
                                   AutoMinorLocator, FormatStrFormatter)

    Wnodes = np.array([0.0] + WS)          # inferred series: the benchmark grid
    Wdense = dense_W_grid()                # exact series: dense ED grid
    exact = exact_entropy_curves(BETAS, Wdense)
    labels = ['(a)', '(b)', '(c)', '(d)']

    fig, axes = plt.subplots(1, 4, figsize=FIGSIZE, constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.015, h_pad=0.015, wspace=0.035,
                                hspace=0.0)

    errs = {}
    for ax, beta, lab in zip(axes, BETAS, labels):
        Hfree = System(L, N, 1.0, 0.0, beta).S_vn
        ex = exact[beta] / N
        cf = np.array([Hfree] + [data[f"{beta}_{W}"]['H_closed'] for W in WS]) / N
        nm = np.array([Hfree] + [data[f"{beta}_{W}"]['H_num'] for W in WS]) / N

        # Optional band between the two drawn curves (SHADE).  The closed form
        # is connected linearly between its nodes (that is the dashed line
        # itself), so both edges of the band coincide with curves on the plot.
        if SHADE:
            ax.fill_between(Wdense, ex, np.interp(Wdense, Wnodes, cf),
                            color=C_CLOSED, alpha=0.15, linewidth=0, zorder=1)
        ax.plot(Wdense, ex, '-', color=C_EXACT, lw=LW, zorder=2,
                label='Exact')
        # open circles (no fill, so the dashed line shows through) with the
        # small filled dots drawn above them; where the closed form meets the
        # exact result the dot sits inside the circle
        ax.plot(Wnodes, cf, 'o--', color=C_CLOSED, lw=LW, ms=MS, mfc='none',
                mec=C_CLOSED, mew=MEW, zorder=3, label='Inferred, closed form')
        ax.plot(Wnodes, nm, 'o', ls='none', mfc=C_NUM, mec=C_NUM, ms=MS_NUM,
                mew=0, zorder=4, label='Inferred, numerical')

        lo = min(ex.min(), cf.min(), nm.min())
        hi = max(ex.max(), cf.max(), nm.max())
        span = hi - lo
        ax.set_xlim(-0.045, 1.045)
        ax.set_ylim(lo - 0.13 * span, hi + 0.34 * span)
        ax.set_box_aspect(BOXASPECT)

        ax.xaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # steps without 1.5/3 -> no 0.015-style ticks that need a third decimal
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        tk = ax.yaxis.get_major_locator().tick_values(*ax.get_ylim())
        tk = tk[(tk >= ax.get_ylim()[0]) & (tk <= ax.get_ylim()[1])]
        ax.yaxis.set_major_formatter(
            FormatStrFormatter('%%.%df' % _consistent_decimals(tk)))
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.tick_params(which='major', length=3)
        ax.tick_params(which='minor', length=1.5)

        ax.set_title(rf'$\beta t = {beta:g}$', pad=2.5)
        ax.set_xlabel(r'$W/t$', labelpad=1.5)    # on every panel
        ax.text(0.055, 0.95, lab, transform=ax.transAxes, fontsize=8,
                ha='left', va='top')

        # relative error of the closed form at W/t = 1, from the plotted arrays
        rel = 100 * abs(cf[-1] - ex[-1]) / ex[-1]
        errs[beta] = rel
        txt = ERR_FMT.format(rel)
        if ERR_LABEL:
            # 0.13, not 0.05: keeps it off the level of the lowest y-tick label
            ax.text(0.055, 0.13, txt.replace('%', r'\%') if usetex else txt,
                    transform=ax.transAxes, fontsize=6, color='0.45',
                    ha='left', va='bottom')

    axes[0].set_ylabel(r'$H/N$')

    order = ['Exact', 'Inferred, numerical', 'Inferred, closed form']
    hmap = dict(zip(*axes[0].get_legend_handles_labels()[::-1]))
    fig.legend([hmap[k] for k in order], order, loc='outside upper center',
               ncol=3, frameon=False, handlelength=2.2, handletextpad=0.6,
               columnspacing=1.8, borderaxespad=0.0)

    fig.savefig('benchmark.pdf')          # vector, at final size, no rescaling
    fig.savefig('benchmark.png', dpi=300)

    w, h = axes[0].get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()).size
    print(f"\n  wrote benchmark.pdf, benchmark.png "
          f"(usetex={usetex}, panel axes {w:.2f} x {h:.2f} in)")
    print("  closed-form relative error at W/t = 1:")
    for beta in BETAS:
        print(f"    beta t={beta:<4} {errs[beta]:6.3f}%  "
              f"-> {'annotated ' if ERR_LABEL else ''}{ERR_FMT.format(errs[beta])}")
    return errs


# ----------------------------------------------------------------------
# 5.  Finite-size scan: the same benchmark at several L, all at half filling
# ----------------------------------------------------------------------
SCAN_JSON = 'entropy_data_L.json'


def warn_shell_parity(Ls):
    """Mixing N = L/2 parities puts two different shell fillings in one figure."""
    even = [L_ for L_ in Ls if (L_ // 2) % 2 == 0]      # open shell,  g = 2
    odd = [L_ for L_ in Ls if (L_ // 2) % 2 == 1]       # closed shell, g = 1
    if even and odd:
        print(f"  WARNING: mixed shell parity.  L={even} are open-shell "
              f"(N=L/2 even, ground-state degeneracy 2, residual entropy ln 2);"
              f"\n           L={odd} are closed-shell (N=L/2 odd, unique ground "
              f"state, H -> 0 at low T).\n           The beta t = 4 row-to-row "
              f"comparison and the relative-error annotation are not meaningful "
              f"across the two.", flush=True)
    return bool(even and odd)


def build_entropy_data_L(Ls=None, path=SCAN_JSON):
    """Sweep (L, beta, W) for the 4x4 figure.

    Writes after every point and skips points already in the file, so an
    interrupted run resumes where it stopped.  Delete the file to start over.
    """
    import time
    Ls = LS if Ls is None else Ls
    warn_shell_parity(Ls)
    try:
        out = json.load(open(path))
    except (OSError, ValueError):
        out = {}
    todo = [(L_, b, W) for L_ in Ls for b in BETAS for W in WS
            if f"{L_}_{b}_{W}" not in out]
    print(f"  {len(todo)} points to compute, "
          f"{len(Ls) * len(BETAS) * len(WS) - len(todo)} already cached")
    for i, (L_, b, W) in enumerate(todo, 1):
        N_ = L_ // 2                                   # half filling
        t0 = time.time()
        cf, ex, g = closed_form_entropy(b, W, L_, N_)
        nm = numerical_entropy(b, W, L_, N_)
        out[f"{L_}_{b}_{W}"] = dict(L=L_, N=N_, beta=b, W=W, gamma_star=g,
                                    H_exact=ex, H_closed=cf, H_num=nm,
                                    relerr_closed=abs(cf - ex) / ex)
        json.dump(out, open(path, 'w'), indent=1)      # checkpoint every point
        print(f"  [{i}/{len(todo)}] L={L_:<3} N={N_:<2} beta t={b:<4} W/t={W:<5} "
              f"H_exact={ex:.5f} closed={cf:.5f} ({100*(cf-ex)/ex:+.1f}%) "
              f"num={nm:.5f}  [{time.time()-t0:.1f}s]", flush=True)
    return out


def _free_slot(ax, xs, ys):
    """Anchor (in axes fraction) for the annotation: the candidate slot holding
    the fewest plotted points.  Order gives lower-left first so the placement
    stays consistent wherever it can; upper-left sits below the panel letter."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    fx = (np.asarray(xs) - x0) / (x1 - x0)
    fy = (np.asarray(ys) - y0) / (y1 - y0)
    bw, bh, pad = 0.46, 0.20, 0.02
    slots = [(0.055, 0.13, 'left', 'bottom'), (0.945, 0.13, 'right', 'bottom'),
             (0.945, 0.95, 'right', 'top'), (0.055, 0.78, 'left', 'top')]
    best, fewest = slots[0], None
    for sx, sy, ha, va in slots:
        xlo = sx if ha == 'left' else sx - bw
        ylo = sy if va == 'bottom' else sy - bh
        n = int(((fx > xlo - pad) & (fx < xlo + bw + pad) &
                 (fy > ylo - pad) & (fy < ylo + bh + pad)).sum())
        if fewest is None or n < fewest:
            best, fewest = (sx, sy, ha, va), n
        if n == 0:
            break
    return best


def make_figure_L(data, Ls=None, excess=False, annot=None):
    """4x4 grid: rows are ring sizes at half filling, columns temperatures.

    excess=False plots H/N; excess=True plots the excess entropy
    H_ex/N = (H - H_0)/N against the physical free ring at gamma = 1.

    H_ex is the quantity the closed expression actually predicts, and the ln 2
    open-shell degeneracy cancels between H and H_0, so the N = L/2 parity
    offset drops out and the rows become comparable.  The annotation is then
    the ABSOLUTE deviation, not a percentage: H_ex passes through a maximum
    near beta t = 1 and collapses at both temperature ends, so a ratio is
    ill-conditioned there for the same reason it is on H at low T.
    """
    import matplotlib
    matplotlib.use('Agg')
    rc, usetex = _rcparams()
    matplotlib.rcParams.update(rc)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import (FixedLocator, MultipleLocator, MaxNLocator,
                                   AutoMinorLocator, FormatStrFormatter)

    Ls = LS if Ls is None else Ls
    annot = annot or ANNOT or ('delta' if excess else 'percent')
    if annot not in ('percent', 'delta'):
        raise ValueError("annot must be 'percent' or 'delta'")
    missing = [f"{L_}_{b}_{W}" for L_ in Ls for b in BETAS for W in WS
               if f"{L_}_{b}_{W}" not in data]
    if missing:
        raise SystemExit(f"{SCAN_JSON} is missing {len(missing)} points, e.g. "
                         f"{missing[:3]}.  Run:  python3 reproduce.py --scan-L")

    Wnodes = np.array([0.0] + WS)
    Wdense = dense_W_grid()
    letters = '(a) (b) (c) (d) (e) (f) (g) (h) (i) (j) (k) (l) (m) (n) (o) (p)'.split()

    nr, nc = len(Ls), len(BETAS)
    fig, axes = plt.subplots(nr, nc, figsize=(FIGSIZE[0], 1.42 * nr + 0.75),
                             constrained_layout=True, squeeze=False)
    fig.get_layout_engine().set(w_pad=0.015, h_pad=0.015, wspace=0.035,
                                hspace=0.05)
    errs = {}
    for r, L_ in enumerate(Ls):
        N_ = L_ // 2
        exact = exact_entropy_curves(BETAS, Wdense, L_, N_)
        Hfree = {b: System(L_, N_, 1.0, 0.0, b).S_vn for b in BETAS}
        for c, beta in enumerate(BETAS):
            ax = axes[r, c]
            k = f"{L_}_{beta}_%s"
            ex = exact[beta] / N_
            cf = np.array([Hfree[beta]] +
                          [data[k % W]['H_closed'] for W in WS]) / N_
            nm = np.array([Hfree[beta]] +
                          [data[k % W]['H_num'] for W in WS]) / N_
            if excess:                      # subtract the free ring at gamma=1
                off = Hfree[beta] / N_      # ex[0] is that same number exactly
                ex, cf, nm = ex - off, cf - off, nm - off
                ax.axhline(0.0, color='0.7', lw=0.5, zorder=0)

            ax.fill_between(Wdense, ex, np.interp(Wdense, Wnodes, cf),
                            color=C_CLOSED, alpha=0.15, linewidth=0, zorder=1)
            ax.plot(Wdense, ex, '-', color=C_EXACT, lw=0.9, zorder=2,
                    label='exact')
            ax.plot(Wnodes, cf, 's--', color=C_CLOSED, lw=0.9, ms=3.5,
                    mec='w', mew=0.5, zorder=3, label='inferred, closed form')
            ax.plot(Wnodes, nm, 'o', ls='none', mfc='none', mec=C_NUM, ms=3.5,
                    mew=0.8, zorder=4, label='inferred, numerical')

            lo = min(ex.min(), cf.min(), nm.min())
            hi = max(ex.max(), cf.max(), nm.max())
            span = hi - lo
            ax.set_xlim(-0.045, 1.045)
            ax.set_ylim(lo - 0.13 * span, hi + 0.34 * span)
            ax.set_box_aspect(BOXASPECT)

            ax.xaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
            ax.xaxis.set_minor_locator(MultipleLocator(0.25))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            tk = ax.yaxis.get_major_locator().tick_values(*ax.get_ylim())
            tk = tk[(tk >= ax.get_ylim()[0]) & (tk <= ax.get_ylim()[1])]
            ax.yaxis.set_major_formatter(
                FormatStrFormatter('%%.%df' % _consistent_decimals(tk)))
            ax.tick_params(which='both', direction='in', top=True, right=True)
            ax.tick_params(which='major', length=3)
            ax.tick_params(which='minor', length=1.5)
            if r < nr - 1:
                ax.tick_params(labelbottom=False)      # x labels on bottom row

            if r == 0:
                ax.set_title(rf'$\beta t = {beta:g}$', pad=2.5)
            if c == nc - 1:                            # ring size on the right
                ax.yaxis.set_label_position('right')
                ax.set_ylabel(rf'$L = {L_}$', rotation=-90, labelpad=9)
            ax.text(0.055, 0.95, letters[r * nc + c], transform=ax.transAxes,
                    fontsize=8, ha='left', va='top')

            if annot == 'delta':            # absolute deviation at W/t = 1
                errs[(L_, beta)] = abs(cf[-1] - ex[-1])
                txt = DEV_FMT.format(errs[(L_, beta)])
            else:
                errs[(L_, beta)] = 100 * abs(cf[-1] - ex[-1]) / abs(ex[-1])
                txt = ERR_FMT.format(errs[(L_, beta)])
                txt = txt.replace('%', r'\%') if usetex else txt
            sx, sy, ha, va = _free_slot(
                ax, np.r_[Wdense, Wnodes, Wnodes], np.r_[ex, cf, nm])
            ax.text(sx, sy, txt, transform=ax.transAxes, fontsize=6,
                    color='0.45', ha=ha, va=va)

    fig.supxlabel(r'$W/t$', fontsize=8)
    fig.supylabel(r'$H_\mathrm{ex}/N$' if excess else r'$H/N$', fontsize=8)
    order = ['exact', 'inferred, numerical', 'inferred, closed form']
    hmap = dict(zip(*axes[0, 0].get_legend_handles_labels()[::-1]))
    fig.legend([hmap[j] for j in order], order, loc='outside upper center',
               ncol=3, frameon=False, handlelength=2.2, handletextpad=0.6,
               columnspacing=1.8, borderaxespad=0.0)

    stem = 'benchmark_L_excess' if excess else 'benchmark_L'
    fig.savefig(stem + '.pdf')
    fig.savefig(stem + '.png', dpi=300)
    w, h = axes[0, 0].get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()).size
    print(f"\n  wrote {stem}.pdf, {stem}.png "
          f"(usetex={usetex}, panel axes {w:.2f} x {h:.2f} in)")
    print("  closed-form %s at W/t = 1:"
          % ('absolute deviation |H_closed-H_exact|/N' if annot == 'delta'
             else 'relative error'))
    print("     L    " + "".join(f"bt={b:<8}" for b in BETAS))
    fmt = '{:<11.4f}' if annot == 'delta' else '{:<11.2f}'
    for L_ in Ls:
        print(f"    {L_:<5} " + "".join(fmt.format(errs[(L_, b)]) for b in BETAS))
    return errs


if __name__ == '__main__':
    quick = '--quick' in sys.argv
    for a in sys.argv:                       # --estimator=H_rel|H_ex
        if a.startswith('--estimator='):
            ESTIMATOR = a.split('=', 1)[1]
    for a in sys.argv:                       # --L=8,10,12,14 overrides LS
        if a.startswith('--L='):
            LS = [int(x) for x in a[4:].split(',')]
            if any(x % 2 for x in LS):
                sys.exit('--L: half filling needs even L')
    excess = '--excess' in sys.argv       # plot H_ex/N instead of H/N
    annot = next((a[8:] for a in sys.argv if a.startswith('--annot=')), None)
    if '--scan-L' in sys.argv:
        print(f"Finite-size scan, half filling, L = {LS} "
              f"(N = {[x // 2 for x in LS]}):")
        make_figure_L(build_entropy_data_L(LS), LS, excess, annot)
        print("\nDone.")
        sys.exit(0)
    if '--figure-L-only' in sys.argv:
        make_figure_L(json.load(open(SCAN_JSON)), LS, excess, annot)
        print("\nDone.")
        sys.exit(0)
    if '--figure-only' in sys.argv:
        # redraw from the stored numbers; recomputes nothing but the exact curve
        make_figure(json.load(open('entropy_data.json')))
        print("\nDone.")
        sys.exit(0)
    betas = [2.0] if quick else BETAS
    print("Entropy sweep (closed-form + numerical):")
    data = build_entropy_data(betas)
    build_free_energy_table(2.0)
    if not quick:
        make_figure(data)
    print("\nDone.")
