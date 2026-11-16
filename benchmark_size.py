"""
Ring-size dependence of the spinless-fermion benchmark (Supplemental Material).

The four-panel figure of benchmark.pdf with the ring size in place of the
temperature: H/N against W/t at beta t = 1 for L = 8, 10, 12, 14 at half
filling, N = L/2.  Exact curve (ED on the dense W grid of reproduce.py),
numerical route (small blue dots) and closed form (orange open circles).  The
panels carry no error annotation; the closed-form relative
error at W/t = 1 is printed to the terminal.  The L = 12 panel shows the data
of the beta t = 1 panel of benchmark.pdf.

Run:  python3 benchmark_size.py      # writes benchmark_size.pdf/.png (about 2 min)

Like reproduce.py, it reads and writes in the current working directory.  The
inferred values come from entropy_data_L.json, the file of the finite-size scan
(reproduce.py --scan-L); a point missing there is computed with the same
routines and added to the file (about 18 min per point at L = 14).
Everything else -- the closed form, the control, the exact curve, typography,
colours and sizes -- is taken from reproduce.py, so the figure reads as one set
with benchmark.pdf.
"""
import json
import sys
import time
import numpy as np

import reproduce as R

BETA = 1.0
LS = [8, 10, 12, 14]                  # half filling, N = L/2
OUT = 'benchmark_size'


def inferred_data(path=R.SCAN_JSON):
    """The (L, BETA, W) points of the scan; missing ones are computed and saved."""
    try:
        data = json.load(open(path))
    except (OSError, ValueError):
        data = {}
    todo = [(L, W) for L in LS for W in R.WS if f"{L}_{BETA}_{W}" not in data]
    for L, W in todo:
        N = L // 2
        t0 = time.time()
        cf, ex, g = R.closed_form_entropy(BETA, W, L, N)
        nm = R.numerical_entropy(BETA, W, L, N)
        data[f"{L}_{BETA}_{W}"] = dict(L=L, N=N, beta=BETA, W=W, gamma_star=g,
                                       H_exact=ex, H_closed=cf, H_num=nm,
                                       relerr_closed=abs(cf - ex) / ex)
        json.dump(data, open(path, 'w'), indent=1)      # checkpoint every point
        print(f"  computed L={L} W/t={W}  [{time.time() - t0:.0f}s]", flush=True)
    return data


def make_figure_size(data):
    import matplotlib
    matplotlib.use('Agg')
    rc, usetex = R._rcparams()
    matplotlib.rcParams.update(rc)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import (FixedLocator, MultipleLocator, MaxNLocator,
                                   AutoMinorLocator, FormatStrFormatter)

    Wnodes = np.array([0.0] + R.WS)
    Wdense = R.dense_W_grid()
    labels = ['(a)', '(b)', '(c)', '(d)']

    fig, axes = plt.subplots(1, len(LS), figsize=R.FIGSIZE,
                             constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.015, h_pad=0.015, wspace=0.035,
                                hspace=0.0)

    errs = {}
    for ax, L, lab in zip(axes, LS, labels):
        N = L // 2
        Hfree = R.System(L, N, 1.0, 0.0, BETA).S_vn
        ex = R.exact_entropy_curves([BETA], Wdense, L, N)[BETA] / N
        k = f"{L}_{BETA}_%s"
        cf = np.array([Hfree] + [data[k % W]['H_closed'] for W in R.WS]) / N
        nm = np.array([Hfree] + [data[k % W]['H_num'] for W in R.WS]) / N

        if R.SHADE:
            ax.fill_between(Wdense, ex, np.interp(Wdense, Wnodes, cf),
                            color=R.C_CLOSED, alpha=0.15, linewidth=0, zorder=1)
        ax.plot(Wdense, ex, '-', color=R.C_EXACT, lw=R.LW, zorder=2,
                label='Exact')
        ax.plot(Wnodes, cf, 'o--', color=R.C_CLOSED, lw=R.LW, ms=R.MS,
                mfc='none', mec=R.C_CLOSED, mew=R.MEW, zorder=3,
                label='Inferred, closed form')
        ax.plot(Wnodes, nm, 'o', ls='none', mfc=R.C_NUM, mec=R.C_NUM,
                ms=R.MS_NUM, mew=0, zorder=4, label='Inferred, numerical')

        lo = min(ex.min(), cf.min(), nm.min())
        hi = max(ex.max(), cf.max(), nm.max())
        span = hi - lo
        ax.set_xlim(-0.045, 1.045)
        ax.set_ylim(lo - 0.13 * span, hi + 0.34 * span)
        ax.set_box_aspect(R.BOXASPECT)

        ax.xaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        tk = ax.yaxis.get_major_locator().tick_values(*ax.get_ylim())
        tk = tk[(tk >= ax.get_ylim()[0]) & (tk <= ax.get_ylim()[1])]
        ax.yaxis.set_major_formatter(
            FormatStrFormatter('%%.%df' % R._consistent_decimals(tk)))
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.tick_params(which='major', length=3)
        ax.tick_params(which='minor', length=1.5)

        ax.set_title(rf'$L = {L}$', pad=2.5)
        ax.set_xlabel(r'$W/t$', labelpad=1.5)
        ax.text(0.055, 0.95, lab, transform=ax.transAxes, fontsize=8,
                ha='left', va='top')

        # relative error of the closed form at W/t = 1, printed below and not
        # drawn: the panels carry no annotation
        errs[L] = 100 * abs(cf[-1] - ex[-1]) / ex[-1]

    axes[0].set_ylabel(r'$H/N$')

    order = ['Exact', 'Inferred, numerical', 'Inferred, closed form']
    hmap = dict(zip(*axes[0].get_legend_handles_labels()[::-1]))
    fig.legend([hmap[j] for j in order], order, loc='outside upper center',
               ncol=3, frameon=False, handlelength=2.2, handletextpad=0.6,
               columnspacing=1.8, borderaxespad=0.0)

    fig.savefig(OUT + '.pdf')
    fig.savefig(OUT + '.png', dpi=300)

    w, h = axes[0].get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()).size
    print(f"\n  wrote {OUT}.pdf, {OUT}.png "
          f"(usetex={usetex}, panel axes {w:.2f} x {h:.2f} in)")
    print(f"  closed-form relative error at W/t = 1, beta t = {BETA:g}:")
    for L in LS:
        print(f"    L={L:<3} N={L // 2:<2} {errs[L]:6.3f}%  "
              f"({R.ERR_FMT.format(errs[L])})")
    return errs


if __name__ == '__main__':
    print(f"Ring-size benchmark, beta t = {BETA:g}, half filling, L = {LS}:")
    make_figure_size(inferred_data())
    print("\nDone.")
    sys.exit(0)
