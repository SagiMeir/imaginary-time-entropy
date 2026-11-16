# Warm dense electron gas: Fig. 3 and Table S3

Code to reproduce the electron-gas test of the paper: the closed-form
correlation free energy, Eq. (8) of the Letter (`Fcorr` in the code comments),
evaluated on the imaginary-time density correlation functions `F(q,τ)` of path
integral Monte Carlo (PIMC) simulations of the spin-unpolarized uniform
electron gas at `Θ = 1`, `r_s = 2, 5, 10, 20`, and compared with the
parametrization fitted to quantum Monte Carlo (QMC) data. No model enters the
input. A reconstruction of `S(q,ω_n)` from the PIMC `S(q)` with a static local
field correction is computed as a check.

This folder is self-contained apart from the shared figure typography, which it
imports from `../reproduce.py` so every figure of the paper looks the same.

## Files

| file | role |
|------|------|
| `ueg_core.py`       | core: finite-N and continuum Lindhard functions on the Matsubara axis, cell reciprocal lattice, reader of the RODARE 3923 archives, exact Matsubara transform of `F(q,τ)` (`itcf_to_matsubara`), the closed form on the data with its jackknife bias correction (`beta_dF_corr_itcf`), finite-size correction; and, as a check, the static local-field reconstruction of `S(q,ω_n)` from `S(q)` with the `ΔF_corr` functional and its analytic tail. numpy/scipy only. |
| `ueg_reference.py`  | the reference: `f_c = f_xc(corrKSDT) − f_x(exact)`. numpy/scipy only. |
| `reproduce_fc.py`   | driver: regenerates every number and the figure. |
| `data/rodare3923/`  | the PIMC input: four archives of RODARE 3923, downloaded on the first run, and their parsed caches `itcf_rs*.npz` (see Data below). Not stored in the repository. |
| `fc_data.json`      | output: per-density results. |
| `fig_fc_prl.pdf/.png` | output: the figure. |

## Running

```bash
python3 reproduce_fc.py                # compute + figure, about 1 min (plus the download on the first run)
python3 reproduce_fc.py --figure-only  # redraw from fc_data.json, about 1 s
```

Outputs are written next to `reproduce_fc.py` whatever the working directory.
The first run downloads the four archives (430 MB) into `data/rodare3923/`,
checks their md5 sums, and parses each once into `itcf_rs*.npz` (69 MB each).

### Expected output

```
  rs  closed form        +-   reference     dev    +-         cell   FS shift FS share  static G
   2    -0.129938  0.001687   -0.141212    8.0%  1.2%    -0.041533  -0.088405      68%      8.5%
   5    -0.065536  0.000054   -0.074175   11.6%  0.1%    -0.026533  -0.039004      60%     11.6%
  10    -0.037967  0.000018   -0.043711   13.1%  0.0%    -0.017582  -0.020385      54%     13.5%
  20    -0.021508  0.000003   -0.024802   13.3%  0.0%    -0.011001  -0.010507      49%     14.2%
```

Hartree per electron. `+-` is the jackknife statistical error; `dev` is relative
to the reference; `cell` is the value on the `N = 34` simulation cell, `FS shift`
the finite-size correction added to reach the thermodynamic limit, `FS share`
that correction's fraction of the total; `static G` is the deviation of the
check with the static local field correction fitted to the same `S(q) = F(q,0)`.

## Method, per density

1. **Reference `S_0(q,ω_n)`**: the ideal Fermi gas on the same discrete
   k-lattice as the PIMC cell, chemical potential fixed by `⟨N⟩ = 34`.
2. **Input `S(q,ω_n)`**: from the PIMC `F(q,τ)` on 200 imaginary-time slices:
   a cubic spline through the data, with end slopes `∓q²/2` fixed by the f-sum
   rule, integrated exactly against `exp(iω_nτ)` (`itcf_to_matsubara`). It
   reproduces `β⁻¹ Σ_n S(q,ω_n) = S(q)` to `3×10⁻⁶`, and the Matsubara density
   response deposited by Dornheim et al. (RODARE 3088, PRB 110, 075137) to a
   median of 0.06% (`r_s = 20`) and 0.14% (`r_s = 10`) for `l ≤ 29`.
3. **Closed form**: `βΔF_corr = ½ Σ_{q,n} [ln(S_0/S) − S_0/S + 1]` over
   `|n| ≤ N_C = 20`. The summand is quadratic in `S/S_0 − 1`, so the noise of the
   data biases it downward; the bias is removed term by term with the jackknife
   estimate `(M − 1)(mean over leave-one-out bins − full value)` over the
   `M = 1000` bins, which also give the error. Terms with `|n| > N_C` are dropped:
   they carry less than 0.001% of the sum in the static-G check and less than
   0.2% in the data at `r_s ≥ 10`, while at `r_s ≤ 5` the data there are
   dominated by noise. The sum runs over the 100 shells `n² ≤ 118` (`q ≤ 6.81
   k_F`), each weighted by its number of lattice vectors; the 15 shells without
   data (below) are filled by linear interpolation in `q` of the per-vector
   summand.
4. **Finite size**: the continuum-minus-cell difference of the same functional,
   in the random phase approximation, added to the cell value.

## Inputs and their provenance

**Data.** `data/rodare3923/` holds the files
`PIMCUEG_rs{2,5,10,20}_theta1_correlations_and_MEMestimates.tar.gz` of the RODARE
record [10.14278/rodare.3923](https://doi.org/10.14278/rodare.3923) (T. M. Chuna,
N. Barnfield, J. Vorberger, M. P. Friedlander, T. Hoheisel and T. Dornheim, 2025;
licence CC BY 4.0), which accompanies T. Chuna *et al.*, "Estimates of the dynamic
structure factor for the finite temperature electron liquid via analytic
continuation of path integral Monte Carlo data", Phys. Rev. B **112**, 125112
(2025), [doi:10.1103/4d4b-kgtk](https://doi.org/10.1103/4d4b-kgtk). Each archive
holds, for `N = 34` electrons at `Θ = 1`, the leave-one-out binned `F(q,τ)` over
`M = 1000` independent runs of the ISHTAR code, on `τ_j = jβ/200`, one file per
k index (`correlations/JKFData_k_index*.raw`; md5 sums in `reproduce_fc.py`).
The statistical error of `F` is of order `10⁻²` at `r_s = 2`, `10⁻³` at `r_s = 5`
and `10⁻⁴` beyond (Table I of the paper). The archives are not stored in this
repository; `reproduce_fc.py` downloads them. Please cite the paper and the
data set when using them.

*Defects of the deposit, handled by `read_itcf_archive`:* in every set, 15 of
the 100 k indices are byte-identical copies of the preceding shell (same `|q|`
in `chi_q_*.dat`, same data), and 15 magnitudes with `n² ≤ 118` (53, 62, 68, 73,
77, 82, 85, 89, 99, 102, 105, 108, 110, 116, 118) have no file; the `r_s = 2`
`chi_q` file is truncated. `K_INDEX_N2` in `ueg_core.py` maps the k indices to
`n²` (from the `r_s = 20` file), duplicates are dropped after checking that they
are identical, and the absent shells are interpolated (they carry 0.3–0.6% of
the cell sums at `r_s ≥ 5`).

**Reference.** `f_xc` uses the KSDT functional form [V. V. Karasiev *et al.*,
Phys. Rev. Lett. **112**, 076403 (2014)] with the corrected "corrKSDT"
parameters. `f_x` is the exact finite-temperature exchange free energy in the
form of B. Horovitz and R. Thieberger, Physica **71**, 99 (1974). Both are
written from the published formulas and agree with the independent
implementation in [uegpy](https://github.com/fdmalone/uegpy) to `8×10⁻¹²`.
uegpy is not a dependency.

## What is deliberately not compared

The deposit also contains `r_s = 50, 100, 200`. KSDT was fitted to QMC data up
to `r_s = 40`, and the only free-energy data beyond `r_s = 20` (η-ensemble PIMC)
are at `Θ = 2`, not `Θ = 1`. There is no validated reference at those densities,
so they are excluded from the comparison. `RS_POINTS` in `reproduce_fc.py`
controls this.

## Known limitations

- **Statistical noise.** At `r_s = 2` the fermion sign problem leaves the data
  noisy: the jackknife error is 1.2% of `f_c`, cuts `N_C = 10` and `20` agree to
  0.1 percentage points, and larger cuts add terms dominated by noise (the bias
  correction then exceeds the signal). At `r_s ≥ 5` the error is at most 0.1%,
  and `N_C` between 10 and 50 changes `f_c` by at most 0.3%.
- **Momentum cutoff.** With the full imaginary-time dependence, `ΔF_corr` grows
  linearly with the momentum cutoff, because the interaction changes the
  kinetic energy (Supplemental Material, Sec. S7.7, "Large wave numbers and the
  momentum cutoff"). Over the outer shells of the cell the growth is 0.34%
  (`r_s = 20`) and 0.10% (`r_s = 10`) of `|f_c|` per `k_F`, close to the
  predicted 0.35% and 0.14%; at `r_s ≤ 5` it is below the noise. The sum stops
  at the last shell of the cell, `6.81 k_F`.
- **Static-G check.** The static local field correction fitted to `S(q)`
  reproduces `F(q,τ)` poorly at strong coupling (at `τ = β/2` it lies 3–10%
  below the data for `q < 4 k_F` at `r_s = 20`), yet gives `f_c` within 0.9
  percentage points of the data-based value (`static G` column).
- **Finite size.** The correction carries 49–68% of each value and is computed
  in the random phase approximation. Static reconstructions at `N = 34` and
  `N = 66` agree to 1.4% at `r_s = 20`.

## Parameters

Module-level constants at the top of `reproduce_fc.py`: `N_C = 20` (Matsubara
cutoff of the closed form on the data). For the static-G check: `N_MAX = 300`
(Matsubara cutoff), `TAIL_WINDOW = 80` (points used to fit the `ω⁻⁴` tail),
`LEVER_MIN = 0.05` (below this `|U_q χ_0(q,0)|` the `G(q)` fit is
ill-conditioned and `G` is frozen).

## Figure style

The figure takes its typography, tick rules, reference colour, line width and
marker size from `../reproduce.py` (`_rcparams`, `_consistent_decimals`,
`C_EXACT`, `LW`, `MS`, `MEW`), so it reads as one set with `benchmark.pdf`.
`FIGSIZE = (2.50, 1.50)` with `AX_ASPECT` the inverse golden ratio gives axes of
1.97 × 1.22 in; the Letter includes the file at its natural size, so the fonts
match Figs. 1 and 2. The closed-form points are orange open circles without a
connecting line (`C_POINTS`), the symbol of the closed form in the ring figure.
Legend labels, the point colour and an optional gap shading (`SHADE`, off) are
constants at the top of `reproduce_fc.py`.

## Reproducibility notes

Deterministic: no random numbers. Tested with Python 3.11, numpy 1.24 and 2.4,
scipy 1.13 and 1.17, matplotlib 3.8 and 3.10.

With a LaTeX installation (`latex`, `dvipng`, `type1ec.sty`) the figure is
typeset by TeX through `../reproduce.py`'s `_rcparams`, which is how the stored
figure was made; without one it falls back to matplotlib's Computer Modern.
