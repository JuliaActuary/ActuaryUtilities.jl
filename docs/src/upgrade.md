# Version Upgrade Guide

## v5.8 to v5.9

Non-breaking unless you relied on the specific edge-case behaviors noted below.

### Risk measures

- **Array inputs to `VaR`, `CTE`, and `Expectation` are now computed as exact order statistics** (the discrete Choquet integral evaluated as a finite weighted sum) instead of adaptive quadrature over the empirical CDF's step function. The exact value of the risk measure is now returned: results may differ from v5.8 in the last few ulps, or visibly at plateaus/quantile boundaries where the quadrature approximation was least accurate. Distribution inputs are unchanged (numerical integration of the distorted CDF).
- **`Expectation` is now exported** — `using ActuaryUtilities` brings it into scope (previously `ActuaryUtilities.RiskMeasures.Expectation`).

### Financial math

- **`spread` solves via Newton + AD** on the pricing residual, converging to machine precision (previously a derivative-free minimization with ~√tolerance precision). It now **throws an `ErrorException` on non-convergence** instead of returning the best-so-far point.
- **`moic` throws an explicit `ArgumentError`** when the input has no positive or no negative cashflows ("moic requires at least one positive (distribution) and one negative (contribution) cashflow"); previously such degenerate input surfaced as an obscure reduce-over-empty-collection error.
- **Analytic fast paths** for `Modified` duration and `convexity` with flat yields (`Real`, `Rate{Periodic}`, `Rate{Continuous}`, `Yield.Constant`): same values as the AD path (equality-tested), substantially faster.
- **`present_values` is now O(n)** (previously O(n²) and recursive — very long cashflow vectors could overflow the stack) and propagates AD dual numbers through its accumulator.
- **Legacy bump-and-reprice `duration(keyrate::KeyRateZero/KeyRatePar, curve, cashflows)`** now derives timepoints from embedded `Cashflow` times (previously it used the vector *indices*, so e.g. semiannual cashflows at 0.5…5.0 got a key-rate grid out to 10 years). Results change for `Cashflow`-vector inputs; plain amount vectors are unaffected. When every timepoint is below 1, the default `krd_points` grid would be empty and now raises an `ArgumentError` asking for an explicit grid (previously an obscure `MethodError`).

### Dependencies & ecosystem

- **Optimization.jl and OptimizationOptimJL.jl are no longer dependencies** (`spread` was their only use).
- **FinanceCore v3 is now supported.** Under FinanceCore v3, a failed `irr`/`internal_rate_of_return` returns `Periodic(NaN, 1)` instead of `nothing` — replace `isnothing(irr(x))` checks with `isnan(rate(irr(x)))`.

## v5.7 to v5.8

Additive release — no existing method changed. A contract / portfolio-aware layer was added on top of the key-rate AD engine, so curve-dependent instruments (e.g. floating-rate bonds) can be passed directly and have their cashflows re-projected under bumped curves:

- **`Effective` and `Spread` duration markers**: `duration(Effective(), contract, curve, tenors)` reprices with coupons re-fixed (the correct interest-rate duration for floaters); `duration(Spread(), contract, curve, tenors)` bumps the discount curve only (discount-margin / credit duration). A contract or a `Vector` of contracts (a portfolio) is accepted.
- **`dv01` verb**: `dv01(Effective()/Spread(), target, [forward, credit,] tenors)` for the dollar versions; `dv01(args...)` is equivalent to `duration(DV01(), args...)` for the existing cashflow/curve forms.
- **`sensitivities(target, [forward, credit,] tenors)`**: one-AD-pass bundle for a contract/portfolio returning `value`, `effective_*`, `spread_*`, and `forward_*` durations / DV01s / key-rate vectors.
- **NamedTuple multi-curve `sensitivities`**: `sensitivities(target, tenors; discount = (; rf, credit, ilp), index = ...)` decomposes sensitivities per named discount role (`rf` ≈ IR01, `credit` ≈ CS01, etc.) plus the `index` (reset) sensitivity, returning `(; value, duration, dv01, key_rate)` per role. A do-block form `sensitivities(valuation, curves::NamedTuple; tenors)` is also available.
- **Helpers**: `zspread` (Newton-solved constant spread to match a market price, with its DV01), `locked_floater` (in-force floater with the current coupon locked until the next reset), and `reproject` (wrap a contract so its coupons are estimated off a given index curve).

## v5.6 to v5.7

### Overview

The key-rate sensitivity API (`duration`, `convexity`, `sensitivities`, with `KeyRates`, `IR01`, `CS01`, and two-curve variants) now carries the KRD knot grid on the `KeyRates` marker itself: `KeyRates(tenors)`. The previous `ZeroRateCurve`-specific dispatch that pulled the grid implicitly from `zrc.tenors` is gone — `KeyRates(tenors)` is now the single uniform way to specify the knot grid.

The AD pathway is also rewritten: it now layers a triangular-hat zero-rate bump on top of the user's curve via `FinanceModels.Yield.TenorShift` rather than rebuilding the curve from AD-tagged rates. This works on any `AbstractYieldModel` — composites, UFR extrapolators, fitted Nelson-Siegel models, etc. — without requiring the user to first convert to `ZeroRateCurve`.

### API Changes

**Breaking — vector / matrix key-rate calls now require `KeyRates(tenors)` carrying the knot grid.** The migration is a one-symbol replacement: `KeyRates()` → `KeyRates(tenors)`, and any trailing `tenors` argument falls out of the call. `sensitivities` also adopts the `KeyRates(tenors)` marker for uniformity.

```julia
# v5.6
duration(KeyRates(), zrc, cfs, times)
duration(DV01(), KeyRates(), zrc, cfs, times)
convexity(KeyRates(), zrc, cfs, times)
sensitivities(zrc, cfs, times)
duration(IR01(), KeyRates(), base, credit, cfs, times)
sensitivities(hw, cfs, times)                       # Hull-White MC

# v5.7 (typical migration: use zrc.tenors as the grid)
tenors = zrc.tenors
duration(KeyRates(tenors), zrc, cfs, times)
duration(DV01(), KeyRates(tenors), zrc, cfs, times)
convexity(KeyRates(tenors), zrc, cfs, times)
sensitivities(KeyRates(tenors), zrc, cfs, times)
duration(IR01(), KeyRates(tenors), base, credit, cfs, times)
sensitivities(KeyRates(tenors), hw, cfs, times)
```

You can pass any knot grid — KRD buckets are now an explicit modeling choice, not tied to the curve's storage tenors:

```julia
# A curve fit on monthly observations, KRDs reported at FRTB buckets
FRTB = [0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30]
duration(KeyRates(FRTB), pv, fitted_curve)
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

**Hull-White** no longer requires `hw.curve` to be a `ZeroRateCurve`. Any `AbstractYieldModel` works, and the knot grid is supplied via `KeyRates(tenors)` at the call site: `sensitivities(KeyRates(tenors), hw, cfs, times)`.

**Two-curve API:** the previous `ArgumentError` on mismatched `base.tenors != credit.tenors` is gone — supply your own knot grid via `KeyRates(tenors)` and the two curves can have any storage structure.

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
y = fit(Spline.Linear(), ForwardYield([0.05,0.1]), Fit.Bootstrap())
pv(y,cfs)

``` 
