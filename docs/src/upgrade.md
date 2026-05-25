# Version Upgrade Guide

## v5.6 to v5.7

### Overview

The key-rate sensitivity API (`duration(KeyRates(), …)`, `convexity(KeyRates(), …)`, `sensitivities`, `IR01`, `CS01`, two-curve variants) now takes the **KRD knot grid as a required positional argument**. There is no longer a `ZeroRateCurve`-specific dispatch that pulled the grid implicitly from `zrc.tenors`.

The AD pathway is also rewritten: it now layers a triangular-hat zero-rate bump on top of the user's curve via `FinanceModels.Yield.TenorShift` rather than rebuilding the curve from AD-tagged rates. This works on any `AbstractYieldModel` — composites, UFR extrapolators, fitted Nelson-Siegel models, etc. — without requiring the user to first convert to `ZeroRateCurve`.

### API Changes

**Breaking — vector / matrix KRD calls now require an explicit `tenors` argument.** The migration is mechanical: insert the knot grid immediately after the curve.

```julia
# v5.6
duration(KeyRates(), zrc, cfs, times)
duration(DV01(), KeyRates(), zrc, cfs, times)
convexity(KeyRates(), zrc, cfs, times)
sensitivities(zrc, cfs, times)
duration(IR01(), KeyRates(), base, credit, cfs, times)
sensitivities(hw, cfs, times)                       # Hull-White MC

# v5.7 (typical migration: use zrc.tenors)
duration(KeyRates(), zrc, zrc.tenors, cfs, times)
duration(DV01(), KeyRates(), zrc, zrc.tenors, cfs, times)
convexity(KeyRates(), zrc, zrc.tenors, cfs, times)
sensitivities(zrc, zrc.tenors, cfs, times)
duration(IR01(), KeyRates(), base, credit, zrc.tenors, cfs, times)
sensitivities(hw, zrc.tenors, cfs, times)
```

You can also pass any other knot grid — KRD buckets are now an explicit modeling choice, not tied to the curve's storage tenors:

```julia
# A curve fit on monthly observations, KRDs reported at FRTB buckets
FRTB = [0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30]
duration(KeyRates(), pv, fitted_curve, FRTB)
```

**Non-breaking — scalar duration / convexity / DV01 calls** fall through to the generic finite-difference scalar path and continue to work without `tenors`:

```julia
duration(zrc, cfs, times)              # still works (FD scalar)
duration(DV01(), zrc, cfs, times)      # still works (FD scalar)
convexity(zrc, cfs, times)             # still works (FD scalar)
duration(zrc) do c; pv(c); end         # still works (FD scalar)
```

Numerical values agree with the v5.6 AD-based scalars to FD precision (~1e-6).

**Per-knot KRDs may shift slightly for non-Linear-spline `ZeroRateCurve` inputs.** The new AD path uses triangular-hat bumps; the old path propagated AD through the curve's spline. For `Spline.Linear()` ZRCs the answers are bitwise identical. For `Spline.MonotoneConvex()` (the default), `PCHIP`, `Cubic`, etc., per-knot KRDs differ by sub-bp on discount factors at typical knot spacing. **Sum of KRDs, scalar modified duration, and parallel-shift sensitivity are all invariant.** The new convention matches the textbook KRD definition and is independent of the curve's interpolator choice.

**Hull-White** no longer requires `hw.curve` to be a `ZeroRateCurve`. Any `AbstractYieldModel` works, and `tenors` is supplied at the `sensitivities(hw, tenors, cfs, times)` call site.

**Two-curve API:** the previous `ArgumentError` on mismatched `base.tenors != credit.tenors` is gone — supply your own knot grid and the two curves can have any storage structure.

## v3 to v4

### Overview 

The shape and API of the package is mostly unchanged. The changes that have made fall into a few categores:

- Accommodating FinanceModels.jl, the next-generation version of Yields.jl.
- Simplifying the API, generally making function calls require more specific arguments to avoid ambiguity
- Accommodating the new `Cashflow` type which makes modeling heterogeneous assets and liabilities simpler.

### API Changes

- Breaking: The functions `europut` and `eurocall` have been moved to `FinanceModels`
- Breaking: Previously, the first argument to `present_value` or `present_values` would be interpreted as a set of `Periodic(1)` one-period forward rates if a vector of real values was passed. Users should explicitly create the yield model first, instead of relying on the implicit conversion:

```julia
# old 
pv([0.05,0.1], cfs)  

# new
using FinanceModels
y = fit(Spline.Linear(),ForwardYields([0.05,0.1]),Bootstrap())
pv(y,cfs)

``` 
