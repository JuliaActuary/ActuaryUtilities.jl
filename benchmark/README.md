# Sensitivity benchmarks

Compare the default branch (`master`) with PR #161 using the same harness and
dependencies. The suite includes analytic cashflows, callback gradients and
Hessians, two- and three-curve valuations, fixed and floating contracts,
portfolios, scalar shocks, and Hull–White scenarios. Key-rate grids have 3, 12,
and 30 tenors. Cashflow cases have 60 semiannual payments through year 30;
Hull–White uses 50 scenarios, semiannual steps, and a five-year horizon with a
fresh fixed-seed RNG per evaluation.

## Recorded comparison

- Date: 2026-09-07; macOS ARM64; Julia 1.12.5.
- Baseline: `327e62f85bd3f48f1aaa885c41e8fd30331c81d7` (`master`).
- Updated implementation: `98936f9` (PR #161).
- FinanceCore 2.7.0, FinanceModels 6.4.0, ForwardDiff 1.4.5, DiffResults 1.1.0,
  BenchmarkTools 1.8.0. All manifest entries other than ActuaryUtilities matched.
- One Julia thread and one BLAS thread. Each case runs twice before timing;
  compilation is excluded. Each trial uses `evals=1`, at most 10,000 samples,
  and a 0.3-second budget. Three fresh processes per revision ran sequentially
  in the order master/PR/PR/master/master/PR, without concurrent test runs.
- Reported time is the median of the three trial medians. Allocations are per
  evaluation. The raw trials and complete comparison are in `results/2026-09-07/`.
- Every benchmark's numerical output agrees across revisions within
  `rtol = atol = 1e-10`, including seeded scenario results.

Runtime speedup (`master / PR`; larger is faster):

| Workload | 3 tenors | 12 tenors | 30 tenors |
| --- | ---: | ---: | ---: |
| Analytic duration | 1.00× | 0.97× | 1.00× |
| Analytic bundle | 1.00× | 1.00× | 0.98× |
| Callback duration | 1.07× | 1.05× | 1.02× |
| Callback Hessian bundle | 1.63× | 1.23× | 1.20× |
| Two-curve Hessian bundle | 1.47× | 1.09× | 1.03× |
| Three named curves | 1.27× | 1.12× | 1.09× |
| Fixed contract bundle | 2.29× | 6.55× | 16.09× |
| Floating contract bundle | 1.67× | 4.50× | 12.58× |
| Portfolio bundle | 1.76× | 5.15× | 14.17× |
| Scalar effective duration | 2.21× | 9.91× | 56.61× |
| Scalar spread DV01 | 2.23× | 10.27× | 58.59× |

The Hull–White scenario bundle is 1.75× faster.

All AD cases have lower measured median runtimes. Analytic cases are unchanged
or up to 2.8% (41 ns) slower, with overlapping run ranges and unchanged
allocations. No material runtime regression was observed in this suite.

Allocation tradeoff: callback gradients use 224 additional bytes per call;
the 12-tenor callback Hessian bundle uses 6,848 additional bytes. Other
measured cases allocate the same or fewer bytes. The complete comparison
records byte counts, and the raw trials also record allocation counts.

These measurements cover the included workloads on one machine. Small timing
differences should be treated as noise; they do not establish performance for
every user-supplied valuation or include compilation latency.

## Reproduce

From the updated PR checkout, create a worktree for the recorded baseline and
instantiate the benchmark environment:

```sh
git worktree add ../au-benchmark-master 327e62f85bd3f48f1aaa885c41e8fd30331c81d7
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

To use the recorded direct dependency versions, before copying the environment:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.add([PackageSpec(name="BenchmarkTools", version="1.8.0"), PackageSpec(name="FinanceCore", version="2.7.0"), PackageSpec(name="FinanceModels", version="6.4.0"), PackageSpec(name="ForwardDiff", version="1.4.5"), PackageSpec(name="DiffResults", version="1.1.0")])'
```

Copy both project and manifest to a separate baseline environment, then change
only its ActuaryUtilities path. Inspect the manifests to ensure no other
dependencies changed during resolution.

```sh
mkdir -p /tmp/au-benchmark-master-env
cp benchmark/Project.toml benchmark/Manifest.toml /tmp/au-benchmark-master-env/
julia --project=/tmp/au-benchmark-master-env -e 'using Pkg; Pkg.develop(path="../au-benchmark-master")'
```

Run each revision sequentially using the harness from the PR checkout. Repeat
three times with distinct output filenames and alternate the revision order.

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=/tmp/au-benchmark-master-env --startup-file=no benchmark/sensitivities.jl /tmp/master-1.tsv
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=benchmark --startup-file=no benchmark/sensitivities.jl /tmp/pr-1.tsv
julia --project=benchmark --startup-file=no benchmark/check_values.jl /tmp/master-1.tsv.values /tmp/pr-1.tsv.values
```

`AU_BENCH_SECONDS` changes the time budget per case. `AU_BENCH_FILTER` limits the
suite to names containing that substring; leave it unset for the full comparison.
