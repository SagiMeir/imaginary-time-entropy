"""
Reproduce Fig. 3: correlation free energy of the warm dense electron gas,
closed form Eq. (Fcorr) on PIMC input vs the QMC parametrization, Theta = 1.

Outputs, written next to this file (not to the working directory):
    fc_data.json         per density: closed form (cell value, finite-size
                         shift, total), reference, deviation, fitted G range
    fig_fc_prl.pdf/.png  the single-column figure

Input: the PIMC imaginary-time correlation functions F(q, tau) of T. Chuna et al.,
Phys. Rev. B 112, 125112 (2025), deposited as RODARE 10.14278/rodare.3923. The four
archives used (r_s = 2, 5, 10, 20; about 100 MB each) are downloaded into
data/rodare3923/ on the first run, checked against their md5 sums, and parsed once
into data/rodare3923/itcf_rs*.npz.

Run:  python3 reproduce_fc.py                # compute + figure (a few minutes the first time)
      python3 reproduce_fc.py --figure-only  # redraw from fc_data.json

Depends on ueg_core.py and ueg_reference.py (numpy/scipy), on matplotlib, and
on ../reproduce.py for the typography shared by every figure of the paper.
"""
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))          # ../reproduce.py, shared style

from ueg_core import (UEG, estimate_f_c, read_itcf_archive, beta_dF_corr_itcf,
                      finite_size_shift)
from ueg_reference import f_c_reference, f_c_reference_curve

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
ITCF_DIR = os.path.join(HERE, 'data', 'rodare3923')
RODARE_3923 = 'https://rodare.hzdr.de/api/files/76190d77-9b80-48dd-aa50-e058a3b0ab38'
ARCHIVES = {                # file of RODARE 10.14278/rodare.3923 and its md5 sum, per r_s
    2.0: ('PIMCUEG_rs2_theta1_correlations_and_MEMestimates.tar.gz', '30d0bbd0d5906ce0c40f703092eb276d'),
    5.0: ('PIMCUEG_rs5_theta1_correlations_and_MEMestimates.tar.gz', '083aa2a57349f33db13778d16a6f2bbf'),
    10.0: ('PIMCUEG_rs10_theta1_correlations_and_MEMestimates.tar.gz', '0f8a9bb479d55394f548f7cdcf980758'),
    20.0: ('PIMCUEG_rs20_theta1_correlations_and_MEMestimates.tar.gz', 'ac58e2dd1f2fab23e4fe777c930a2f19'),
}
N_PART = 34                 # electrons in the PIMC cell
THETA = 1.0
RS_POINTS = (2.0, 5.0, 10.0, 20.0)
# Only densities where the reference is validated against QMC at this Theta.
# The deposit also holds rs = 50, 100, 200; the parametrization is an
# extrapolation there, so those densities are deliberately not compared.

N_C = 20                    # Matsubara cutoff of the closed form on the data, |n| <= N_C

# check: static local field correction fitted to the same S(q) = F(q, 0)
N_MAX = 300                 # explicit Matsubara sum, |n| <= N_MAX
TAIL_WINDOW = 80            # largest-|n| points used to fit the w^-4 tail
LEVER_MIN = 0.05            # freeze G(q) where |U_q chi_0(q,0)| drops below

OUT_JSON = os.path.join(HERE, 'fc_data.json')
OUT_FIG = os.path.join(HERE, 'fig_fc_prl')

# ----------------------------------------------------------------------
# Figure style (presentation only; nothing below changes a computed value)
# Typography, colours and marker encoding follow ../reproduce.py, so this
# figure reads as part of the same set as benchmark.pdf.
# ----------------------------------------------------------------------
FIGSIZE = (2.50, 1.50)      # axes 1.97 x 1.22 in; include at natural size (no width=), so the
                            # fonts match Figs. 1 and 2
AX_ASPECT = 2 / (1 + 5 ** 0.5)  # axes height / width = 1 / golden ratio; the legend still clears the curve
LEGEND = True               # lower right, inside the axes
LABEL_REF = 'QMC parametrization'
LABEL_CLOSED = 'Closed form inference,\nPIMC input'   # two rows, so the legend fits the narrow axes
# Closed-form points: orange open circles, unconnected; the colour and symbol
# of the closed form in the ring figure (C_CLOSED in ../reproduce.py).
C_POINTS = '#D55E00'
SHADE = False               # gap shading; meant for a drawn closed-form line,
                            # so it stays off while the points are unconnected
RS_DENSE = np.logspace(np.log10(1.75), np.log10(23.0), 300)
XLIM = (1.75, 23.0)
YLIM = (-0.158, -0.012)


def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def archive(rs):
    """Path of the RODARE 3923 archive for r_s, downloaded on first use and checked."""
    name, md5 = ARCHIVES[rs]
    path = os.path.join(ITCF_DIR, name)
    if not os.path.exists(path):
        os.makedirs(ITCF_DIR, exist_ok=True)
        print(f"  downloading {name} from RODARE 10.14278/rodare.3923 ...", flush=True)
        req = urllib.request.Request(f'{RODARE_3923}/{name}',
                                     headers={'User-Agent': 'Mozilla/5.0 (compatible; research-script)'})
        with urllib.request.urlopen(req, timeout=900) as r, open(path + '.part', 'wb') as f:
            shutil.copyfileobj(r, f)
        os.replace(path + '.part', path)
    if _md5(path) != md5:
        raise RuntimeError(f"{path}: md5 does not match RODARE 3923")
    return path


def load_itcf(rs):
    """(n2, tau, F, D) for r_s from the parsed cache, or from the archive."""
    cache = os.path.join(ITCF_DIR, f'itcf_rs{rs:g}.npz')
    if os.path.exists(cache):
        d = np.load(cache)
        return d['n2'], d['tau'], d['F'], d['D']
    n2, tau, F, D = read_itcf_archive(archive(rs))
    np.savez(cache, n2=n2, tau=tau, F=F, D=D)
    return n2, tau, F, D


def compute():
    rows = []
    for rs in RS_POINTS:
        t0 = time.time()
        ueg = UEG(rs, THETA, N_PART)
        n2, tau, F, D = load_itcf(rs)
        if abs(tau[-1] / ueg.beta - 1.0) > 1e-5:
            raise RuntimeError(f"r_s = {rs:g}: imaginary-time grid ends at {tau[-1]}, beta = {ueg.beta}")
        r = beta_dF_corr_itcf(ueg, n2, F, D, n_c=N_C)
        shift = finite_size_shift(ueg, r['q_cell'], r['weight_cell'])
        per = 1.0 / (N_PART * ueg.beta)
        ref = f_c_reference(rs, THETA)
        f_c = (r['cell'] + shift) * per
        st = estimate_f_c(rs, THETA, N_PART, np.sqrt(n2) * 2.0 * np.pi / ueg.L / ueg.kF, F[:, 0],
                          n_max=N_MAX, tail_window=TAIL_WINDOW, lever_min=LEVER_MIN)
        row = dict(rs=rs, theta=THETA, N=N_PART, n_c=N_C, n_bins=int(D.shape[2]),
                   n_shells_data=int(n2.size), n_shells_filled=int(r['filled'].sum()),
                   f_c=f_c, f_c_err=r['err'] * per, f_c_cell=r['cell'] * per,
                   finite_size_shift=shift * per, f_c_reference=ref,
                   deviation_percent=100.0 * (f_c - ref) / abs(ref),
                   deviation_err_percent=100.0 * r['err'] * per / abs(ref),
                   f_c_static_G=st['f_c'],
                   deviation_static_G_percent=100.0 * (st['f_c'] - ref) / abs(ref))
        rows.append(row)
        print(f"  rs={rs:<4g} f_c={f_c:.6f} +- {row['f_c_err']:.6f}  reference={ref:.6f}  "
              f"({row['deviation_percent']:+.1f}%)  finite-size shift={row['finite_size_shift']:.6f}"
              f"  static-G check {row['deviation_static_G_percent']:+.1f}%  [{time.time() - t0:.0f}s]", flush=True)
    meta = dict(source='PIMC F(q, tau) of T. Chuna et al., Phys. Rev. B 112, 125112 (2025); '
                       'RODARE 10.14278/rodare.3923',
                N=N_PART, theta=THETA, n_c=N_C, units='Hartree per electron',
                estimator='Eq. (Fcorr) over |n| <= n_c, S(q,w_n) by exact spline transform of F(q,tau) '
                          'with f-sum end slopes, jackknife bias correction, 100 cell shells (n^2 <= 118; '
                          'absent shells interpolated), random-phase finite-size shift',
                check='f_c_static_G: static local field correction fitted to S(q) = F(q,0)')
    json.dump(dict(meta=meta, points=rows), open(OUT_JSON, 'w'), indent=1)
    return rows


def make_figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    from reproduce import (_rcparams, _consistent_decimals, C_EXACT,
                           LW, MS, MEW)
    rc, usetex = _rcparams()
    matplotlib.rcParams.update(rc)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import (FixedLocator, FixedFormatter, LogLocator, NullFormatter,
                                   MaxNLocator, AutoMinorLocator, FuncFormatter)

    rs = np.array([r['rs'] for r in rows])
    fc = np.array([r['f_c'] for r in rows])
    ref = f_c_reference_curve(RS_DENSE, THETA)

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.015, h_pad=0.015)
    if SHADE:
        # Band between the two drawn curves. On the log axis the dashed line is
        # straight between nodes in log(rs), so interpolate there: both edges of
        # the band coincide with curves actually on the plot.
        inside = (RS_DENSE >= rs[0]) & (RS_DENSE <= rs[-1])
        band = np.interp(np.log(RS_DENSE[inside]), np.log(rs), fc)
        ax.fill_between(RS_DENSE[inside], ref[inside], band,
                        color=C_POINTS, alpha=0.15, linewidth=0, zorder=1)
    ax.plot(RS_DENSE, ref, '-', color=C_EXACT, lw=LW, zorder=2, label=LABEL_REF)
    ax.plot(rs, fc, 'o', ls='none', mfc='none', mec=C_POINTS, ms=MS, mew=MEW,
            zorder=3, label=LABEL_CLOSED)

    ax.set_xscale('log')
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_box_aspect(AX_ASPECT)
    ax.set_xlabel(r'$r_s$', labelpad=1.5)
    ax.set_ylabel(r'$f_\mathrm{c}$ (Ha/electron)')
    # label only the simulated densities; an automatic 10^1 decade label
    # would sit on top of the "10" tick
    ax.xaxis.set_major_locator(FixedLocator(list(rs)))
    ax.xaxis.set_major_formatter(FixedFormatter(['%g' % r for r in rs]))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    # y ticks exactly as in benchmark.pdf: no 1.5/3 steps, fewest faithful decimals
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    tk = ax.yaxis.get_major_locator().tick_values(*ax.get_ylim())
    tk = tk[(tk >= ax.get_ylim()[0]) & (tk <= ax.get_ylim()[1])]
    # math mode, so negative ticks get a true minus sign rather than a hyphen
    dec = _consistent_decimals(tk)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: r'$%.*f$' % (dec, v)))
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', length=3)
    ax.tick_params(which='minor', length=1.5)
    if LEGEND:
        ax.legend(frameon=False, loc='lower right', handlelength=1.6,
                  handletextpad=0.6, borderaxespad=0.4)

    fig.savefig(OUT_FIG + '.pdf')          # vector, at final size, no rescaling
    fig.savefig(OUT_FIG + '.png', dpi=300)
    print(f"wrote {OUT_FIG}.pdf/.png (usetex={usetex})")


def print_table(rows):
    print(f"\n{'rs':>4} {'closed form':>12} {'+-':>9} {'reference':>11} {'dev':>7} {'+-':>5}   "
          f"{'cell':>10} {'FS shift':>10} {'FS share':>8} {'static G':>9}")
    for r in rows:
        print(f"{r['rs']:>4g} {r['f_c']:>12.6f} {r['f_c_err']:>9.6f} {r['f_c_reference']:>11.6f} "
              f"{r['deviation_percent']:>6.1f}% {r['deviation_err_percent']:>4.1f}%   {r['f_c_cell']:>10.6f} "
              f"{r['finite_size_shift']:>10.6f} {100 * r['finite_size_shift'] / r['f_c']:>7.0f}% "
              f"{r['deviation_static_G_percent']:>8.1f}%")


if __name__ == '__main__':
    if '--figure-only' in sys.argv:
        rows = json.load(open(OUT_JSON))['points']
    else:
        print(f"Closed-form f_c, N = {N_PART}, Theta = {THETA:g}:")
        rows = compute()
    print_table(rows)
    make_figure(rows)
    print("\nDone.")
