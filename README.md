# Quantum entropy and free energy from imaginary-time density correlations

Code that produces the results and figures of the paper

> S. Meir, *Imaginary-Time Density Correlations Encode Quantum Entropy and Free Energy*, arXiv:XXXX.XXXXX.

<!-- PLACEHOLDER: when the paper is on arXiv, replace "arXiv:XXXX.XXXXX" above by the link [arXiv:NNNN.NNNNN](https://arxiv.org/abs/NNNN.NNNNN) and delete this comment. -->

The paper relates the relative entropy, the excess entropy and the free energy
of a quantum many-body system to its density-density correlation function
`S(q,ω_n)` and to that of a non-interacting reference, `S_0(q,ω_n)`, in closed
form. This repository evaluates the closed forms for the three tests of the
paper: the Bogoliubov gas, spinless fermions on a ring, and the warm dense
electron gas. For an instantaneous interaction, the closed forms can also be
evaluated from the static structure factor `S(q)` and `S_0(q,ω_n)` alone, with
`S(q,ω_n)` reconstructed through Eqs. (4) and (9) of the Letter. The repository
does this for the ring (`reproduce.py --static`) and for the electron gas (the
static-G check in `electron_gas/`); for the Bogoliubov gas the reconstruction is
exact (Supplemental Material, Sec. S5.3).

## Where each item of the paper comes from

| item | script | output |
|------|--------|--------|
| Fig. 1 (Bogoliubov gas) | `bogoliubov_equal_time.py` | `fig_bose_equal_time.pdf` |
| Fig. 2 (spinless fermions on a ring) | `reproduce.py` | `benchmark.pdf`, `entropy_data.json` |
| `S(q)` as input on the ring (Eq. (9); conclusions of the Letter) | `reproduce.py --static` | `entropy_static.json` |
| Table S1 (ring: exact and Gaussian free energies) | `reproduce.py` | `free_energy.json` |
| Fig. S1 (ring sizes) | `benchmark_size.py` | `benchmark_size.pdf`, from `entropy_data_L.json` |
| Fig. 3 (warm dense electron gas) and Table S2, with the `S(q)`-input check | `electron_gas/reproduce_fc.py` | `electron_gas/fig_fc_prl.pdf`, `electron_gas/fc_data.json` |

The figure and data files in the repository are the ones used in the paper.
Running the scripts reproduces them.

The comments in the code name the equations of the Letter by their labels:

| label in the code | equation of the Letter |
|-------------------|------------------------|
| `H_rel_main`   | Eq. (5), relative entropy |
| `Fex`          | Eq. (6), excess free energy |
| `Hex_explicit` | Eq. (7), excess entropy |
| `Fcorr`        | Eq. (8), correlation free energy |
| `S_reconstructed` | Eq. (9), the multiplier `λ_q` from `S(q)` |

## Requirements

Python 3 with numpy, scipy and matplotlib:

```bash
pip install -r requirements.txt
```

Tested with Python 3.11 (numpy 1.24, scipy 1.13, matplotlib 3.8) and with
numpy 2.4, scipy 1.17 and matplotlib 3.10. matplotlib is needed only for the
figures.

The figures are typeset by LaTeX when `latex`, `dvipng` and `type1ec.sty`
(package cm-super) are installed, and otherwise by matplotlib's Computer Modern
mathtext. `USETEX` in `reproduce.py` selects this (`None` detects it, `True` or
`False` forces it).

## Running

```bash
python3 bogoliubov_equal_time.py      # Fig. 1, a few seconds
python3 reproduce.py                  # Fig. 2 and Table S1: 4 temperatures, free-energy table, figure (about 5 min)
python3 reproduce.py --quick          # smoke test: βt = 2 only, no figure
python3 reproduce.py --figure-only    # redraw benchmark.pdf from entropy_data.json
python3 reproduce.py --static         # closed form from S(q) alone (Eq. (9), SM Sec. S2.7) at the 16 points of Fig. 2,
                                      # writes entropy_static.json (about 1.5 min)
python3 benchmark_size.py             # Fig. S1, from entropy_data_L.json, about 1.5 min
python3 reproduce.py --scan-L         # finite-size sweep behind entropy_data_L.json (resumable, hours)
python3 reproduce.py --figure-L-only  # 4×4 figure of the sweep, from entropy_data_L.json
cd electron_gas && python3 reproduce_fc.py   # Fig. 3 and Table S2, see electron_gas/README.md
```

`reproduce.py --static` evaluates the closed form with the static structure
factor `S(q)` as the only correlation input (Eq. (9) of the Letter;
Supplemental Material, Sec. S2.7): multipliers independent of `n`, one root of
Eq. (9) per `q` (solved in the form of Eq. (S69)), the reconstructed
`S(q,ω_n)` from Eq. (4), and `H_rel` from Eq. (5) with it (Eq. (S71)),
converted to `H` with the same rescaled reference and exact kinetic energy as
Fig. 2. This is the ring part of the statement in the conclusions of the
Letter that `S(q)` and `S_0(q,ω_n)` give nearly the same results: at all 16
points of Fig. 2, `H` from `S(q)` differs from the closed form with the full
`S(q,ω_n)` by at most 0.41% of the exact entropy (`βt = 4`, `W/t = 1`:
−34.21% against −34.62%), and it is never farther from the exact entropy. If
`entropy_data.json` is present, it prints the closed form of Fig. 2 alongside.

`reproduce.py`, `bogoliubov_equal_time.py` and `benchmark_size.py` read and
write in the current working directory. `electron_gas/reproduce_fc.py` writes
next to itself.

**`--quick` overwrites `entropy_data.json`** with the four `βt = 2` points, after
which `--figure-only` fails. Run it in a copy to keep the full data set. With
`--static`, `--quick` likewise overwrites `entropy_static.json`.

The first run of `electron_gas/reproduce_fc.py` downloads its input data
(430 MB) from the RODARE repository; see `electron_gas/README.md`.

## Files

| file | role |
|------|------|
| `tv.py`            | core of the ring benchmark: exact diagonalization of the ring, thermal state, kinetic energy, and the density-density correlation function `S(q,ω_n)` from its Lehmann representation. |
| `exact_maxent.py`  | the numerical inference: builds `Z_λ` as an ordinary thermal trace (valid because the multipliers are frequency-independent) and solves the convex maximum-entropy dual. Also `static_Sq`, the exact equal-time `S(q)`. |
| `reproduce.py`     | driver of the ring benchmark: regenerates its numbers and figures. Depends only on the two files above. |
| `benchmark_size.py` | Fig. S1: the four-panel figure of `benchmark.pdf` with the ring size in place of the temperature, `H/N` against `W/t` at `βt = 1` for `L = 8, 10, 12, 14` at half filling. Reads the inferred values from `entropy_data_L.json` and computes any missing point with the routines of `reproduce.py`. Prints the closed-form relative error at `W/t = 1`: `1.86%`, `1.56%`, `1.37%`, `1.25%`. |
| `bogoliubov_equal_time.py` | Fig. 1, Bogoliubov gas: exact excess entropy, closed form, and the classical relation evaluated with the equal-time ratio `S(q)/S_0(q)`. Prints a table of single modes, as a check (it is not in the paper), and the totals over all modes at `βρ̄g = 0.2, 1, 5`. The figure shows the total over all modes per volume, `−H_ex/V` in units of `(2mρ̄g/ħ²)^{3/2}`, against `βρ̄g` from 0.05 to 20. The closed form equals the exact result within `1.3×10⁻⁶`. |
| `electron_gas/`    | the electron-gas test, with its own README. |

`bogoliubov_equal_time.py`, `benchmark_size.py` and
`electron_gas/reproduce_fc.py` import the figure style from `reproduce.py`, so
all figures share one typography.

## Spinless fermions on a ring

Spinless fermions on a periodic ring with nearest-neighbour hopping `t` and
nearest-neighbour interaction `W`,

```
H = -t Σ_j (c†_j c_{j+1} + h.c.) + W Σ_j n_j n_{j+1},     L = 12 sites, N = 6 fermions.
```

Two inference routes are compared with the exact entropy:

- **numerical**: solves the maximum-relative-entropy problem with the exact
  constrained partition function `Z_λ`, without the Gaussian approximation
  (the dots of Fig. 2);
- **closed form**: evaluates the closed form for the relative entropy,
  Eq. (5) of the Letter, which carries the Gaussian approximation (the circles
  of Fig. 2).

The closed form is evaluated against a reference whose kinetic scale is
rescaled by `γ*`, fixed by matching the kinetic energy (the high-frequency sum
rule); see Appendix D of the End Matter and Sec. S6 of the Supplemental
Material. Its correlation functions are turned into a total entropy by one of
two assemblies, chosen with the module constant `ESTIMATOR` or `--estimator=`:

- **`H_rel`** (default; gives the paper's figure and numbers):
  `H = H_rel(γ*) + β γ* E_kin + ln I[0; γ*]`, with
  `H_rel = ½ Σ_{q≠0,n} [ln x − x + 1]` and `x = S(q,ω_n) / S_0^{γ*}(q,ω_n)`.
  The relation is an identity, so every approximation sits in `H_rel`. It needs
  two spectra per point, no β-derivative and no high-frequency tail: after the
  `γ*` matching the summand decays as `ω_n⁻⁴`.
- **`H_ex`** (an alternative assembly, kept for comparison):
  `H = H_ex(γ*) + H_0(γ*)` from Eq. (7) of the Letter, with `∂S_0/∂β` from
  central finite differences (step `HB`). Its Matsubara sum converges only as
  `1/NMAX`, so `NMAX = 150` is not converged without a tail correction, and its
  error is `O(W)` at weak coupling, against `O(W²)` for `H_rel`.

### Options of `reproduce.py`

- `--estimator=H_rel|H_ex`: closed-form assembly (default `H_rel`);
- `--L=8,10,12,14`: ring sizes for the scan (even `L` only);
- `--excess`, `--annot=percent|delta`: 4×4 figure only: plot `H_ex/N`, and
  force the annotation type.

The full run computes 16 entropy points, the free-energy table and the figure.
The scan grows steeply with `L` (minutes per row at `L = 12`, hours at
`L = 14`; see the cost table above `LS` in `reproduce.py`). The default
`LS = [8, 10, 12, 14]` mixes two shell fillings (`L = 8, 12` open-shell,
`L = 10, 14` closed-shell) and the script prints a warning; across such rows
the `βt = 4` comparison is not meaningful.

### Outputs

Benchmark (`python3 reproduce.py`):

- `entropy_data.json`: for every `(βt, W/t)` of the figure: `beta`, `W`,
  `gamma_star`, `H_exact`, `H_closed`, `H_num`, and
  `relerr_closed = |H_closed − H_exact| / H_exact`;
- `free_energy.json`: exact and Gaussian-approximation `βF_ex′` at `βt = 2`
  (Table S1; `F_ex′` is the excess free energy of the `q≠0` shifted
  interaction, with the constant `WN²/L` omitted), one row per `W`: `bFex_ex`,
  `bFex_tr` (Gaussian approximation), `err_pct`, `Hrel_ex`, `cancel_pct`;
- `benchmark.pdf` / `benchmark.png`: the four-panel figure, `H/N` against `W/t`
  at `βt = 0.5, 1, 2, 4`: exact curve (exact diagonalization on a 25-point `W`
  grid), numerical route (small blue dots), closed form (orange open circles).
  The closed-form relative error at `W/t = 1` is printed to the terminal.

`S(q)` as input (`python3 reproduce.py --static`):

- `entropy_static.json`: for every `(βt, W/t)` of Fig. 2: `beta`, `W`,
  `gamma_star`, `H_exact`, `H_static` (from `S(q)` alone),
  `relerr_static = (H_static − H_exact) / H_exact`, `two_N_lambda_q` (the roots
  `2Nλ_q` of Eq. (9) for `k = 1, …, L − 1`) and `H_closed` (the closed form of
  Fig. 2, copied from `entropy_data.json`).

Finite-size scan (`--scan-L`), all rings at half filling:

- `entropy_data_L.json`: the same quantities for every `(L, βt, W/t)`, with
  `L` and `N` added. The scan saves after every point and resumes where it
  stopped; delete the file to start over;
- `benchmark_L.pdf` / `.png`: 4×4 figure, rows `L`, columns `βt`. With
  `--excess` it is `benchmark_L_excess.pdf` / `.png`, which plots
  `H_ex/N = (H − H_0)/N` and is annotated with the absolute deviation. These
  figures are not in the paper and are not stored in the repository.

### Expected output (`--quick`)

```
Entropy sweep (closed-form + numerical):
  beta t=2.0 W/t=0.25  H_exact=1.95981  closed=1.95031 (-0.5%)  num=1.95981
  beta t=2.0 W/t=0.5   H_exact=1.81503  closed=1.77646 (-2.1%)  num=1.81503
  beta t=2.0 W/t=0.75  H_exact=1.69102  closed=1.60193 (-5.3%)  num=1.69102
  beta t=2.0 W/t=1.0   H_exact=1.58460  closed=1.42024 (-10.4%)  num=1.58460

  free-energy table (beta t=2.0):
   W/t  bFex'_ex  bFex'_tr   err%   cancel%
   0.25  -0.5791   -0.5879    1.5    95.8
   0.5   -1.2079   -1.2615    4.4    92.6
   0.75  -1.8830   -2.0613    9.5    90.0
   1.0   -2.6018   -3.0959   19.0    87.9

Done.
```

With `--estimator=H_ex` the four closed-form values are instead
`1.90507 (-2.8%)`, `1.73679 (-4.3%)`, `1.61698 (-4.4%)` and `1.54138 (-2.7%)`;
everything else is unchanged.

The numerical route reproduces the exact entropy to the optimizer tolerance
(relative deviation at most `4.3×10⁻⁹`, at `βt = 4`, `W/t = 0.75`). The
closed-form (`H_rel`) error grows with coupling and on cooling: `0.007–0.13%` at
`βt = 0.5`, `0.07–1.4%` at `βt = 1`, `0.5–10.4%` at `βt = 2` and `1.7–34.6%` at
`βt = 4`, always below the exact entropy.

### What each quantity means

- **`H_exact`**: total von Neumann entropy `−Σ_m p_m ln p_m` of the interacting
  ring from exact diagonalization. The figure plots `H/N` (total, not excess:
  the `W = 0` endpoint is the free-gas entropy, not zero).
- **`H_closed`**: the closed-form estimate in the assembly set by `ESTIMATOR`.
- **`γ*`**: kinetic rescaling of the free reference, `E_kin^0(γ*) = E_kin`,
  read from the high-frequency tail of `S(q,ω_n)`; about 0.98–1.05 in this
  range.
- **`βF_ex′` (exact)**: from exact diagonalization of the `q≠0` shifted
  Hamiltonian `H_0 + U′`.
- **`βF_ex′` (Gaussian approximation)**: the sum
  `½ Σ_{q≠0,n} ln(1 + 2N λ_q S_0)` at the multipliers `λ_q = W cos q / L`,
  summed to `|n| = 5000` with the `ω_n⁻²` tail beyond added analytically.
- **`Hrel_ex`**: the exact relative entropy at those multipliers,
  `βN Σ_{q≠0} λ_q S(q) − βF_ex′` (exact).
- **`cancel%`**: the fraction to which the two contributions to `H_rel`
  (`βN Σ λ_q S(q)` and `βF_ex′`) cancel; it is why an error of about 1.5% in
  the free energy becomes an error of about 35% in the relative entropy.

### Reproducibility notes

Everything is deterministic (dense `numpy.linalg.eigh`; no random seeds). The
benchmark grid (`L`, `N`, `BETAS`, `WS`), the Matsubara cutoff `NMAX`, the
finite-difference step `HB`, the scan sizes `LS` and `ESTIMATOR` are
module-level constants at the top of `reproduce.py`; the figure constants (size,
colours, annotation format, `USETEX`) sit above `make_figure`.

`NMAX = 150` is converged for the `H_rel` assembly: raising it to 1200 changes
`H` by less than `4×10⁻⁷` of `H` at every benchmark point.

## Data

The electron-gas test uses the path-integral Monte Carlo data of T. Chuna
*et al.*, Phys. Rev. B **112**, 125112 (2025), deposited at
[doi:10.14278/rodare.3923](https://doi.org/10.14278/rodare.3923) under the
licence CC BY 4.0. The data are not stored in this repository;
`electron_gas/reproduce_fc.py` downloads them on the first run. Please cite
that paper and data set when using them.

## Licence

The code is released under the MIT licence; see `LICENSE`. The licence does not
cover the RODARE data, which have their own licence (above).
