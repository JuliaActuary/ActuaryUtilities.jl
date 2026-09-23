# Version Upgrade Guide

## v5.12.0 to v6.0.0

v6 shocks each input in its native form and measures parallel risk without tenor
grids. See [Shock coordinates](@ref) for the full rule.

- **Curve convexity uses continuous-zero shocks.** This changes scalar convexity
  for every yield model, including constant curves, ZeroRateCurve, Nelson–Siegel,
  and custom models. Cashflow and callback results equal the sum of the full
  key-rate convexity matrix. For `[5, 5, 105]` at `[1, 2, 3]` under
  `Yield.Constant(Periodic(0.04, 1))`, convexity changes from **11.26 to 8.40**.
  Scalars and explicit `Rate` inputs retain their compounding conventions.
  See [Convexity Conventions](@ref) for formulas and examples.
- **Dollar risk preserves position sign and zero-value exposure.** DV01, IR01,
  and CS01 use signed value derivatives, so **a net liability now has negative
  DV01, IR01, and CS01**. For `[-1, 1]` at `[0, 1]` under a zero curve, they return
  `0.0001` instead of `NaN`. Contract sensitivity bundles also retain dollar
  exposure at zero value. Normalized duration and convexity remain undefined there.
  **Migration:** aggregate `asset_dv01 + liability_dv01`, not
  `asset_dv01 - liability_dv01`. Remove sign corrections added to compensate for
  the former use of absolute value.
- **Scalar IR01 and CS01 measure the combined rate.** For fixed cashflows,
  `duration(IR01(), base, spread, cfs, times)` and the matching CS01 both equal
  `duration(DV01(), base + spread, cfs, times)`, shocked in the combined rate's
  coordinate. Values change when the inputs mix types: a yield-model component
  makes the sum a yield model with a continuous-zero shock (previously a scalar
  spread was shocked as an annual rate, so CS01 was about IR01 / (1 + spread)), and a
  mixed-compounding `Rate` sum takes the left operand's compounding. Scalar-only
  inputs are unchanged. Use the callback forms
  `duration(IR01(), (b, c) -> ..., base, credit)` when the curves play different roles.
- **Tenor grids appear only where results have a tenor dimension.** Parallel
  measures no longer accept a `tenors` argument; the grid never changed their
  values, and its position let swapped arguments return wrong numbers silently.
  The old calls now throw `MethodError`:

  | v5 call | v6 replacement |
  |:--|:--|
  | `duration(curve, tenors, cfs, times)`, `duration(curve, tenors, cashflows)` | `duration(curve, cfs, times)`, `duration(curve, cashflows)` |
  | `duration(DV01(), curve, tenors, cfs, times)` | `duration(DV01(), curve, cfs, times)` |
  | `duration(IR01(), base, credit, tenors, cfs, times)` (and `CS01`) | `duration(IR01(), base, credit, cfs, times)` |
  | `convexity(curve, tenors, cfs, times)` | `convexity(curve, cfs, times)` |
  | `convexity(base, credit, tenors, cfs, times)` | `convexity(base, credit, cfs, times)` |
  | `duration(valuation, curve, tenors)` | `duration(curve, valuation)` or `duration(curve) do c ... end` |
  | `duration(DV01(), valuation, curve, tenors)` | `duration(DV01(), curve, valuation)` or `duration(DV01(), curve) do c ... end` |
  | `duration(IR01(), valuation, base, credit, tenors)` (and `CS01`) | `duration(IR01(), valuation, base, credit)` or `duration(IR01(), base, credit) do b, c ... end` |
  | `convexity(valuation, curve, tenors)` | `convexity(curve, valuation)` |
  | `convexity(valuation, base, credit, tenors)` | `convexity(valuation, base, credit)` |
  | `duration(Effective(), target, curve, tenors)`, `duration(Spread(), ...)` | `duration(Effective(), target, curve)`, `duration(Spread(), target, curve)` |
  | `dv01(target, curve, tenors)`, `duration(target, curve, tenors)`, `convexity(target, curve, tenors)` | `dv01(target, curve)`, `duration(target, curve)`, `convexity(target, curve)` |
  | two-curve contract forms `(target, forward, credit, tenors)` | `(target, forward, credit)` |

  `KeyRates(tenors)` forms and `sensitivities(...)` bundles keep their grids.
- **The finite-difference `KeyRateDuration` API is removed.** `KeyRate`,
  `KeyRateZero`, `KeyRatePar`, and `krd_points` are gone.
  `duration(KeyRateZero(t), curve, cfs, times, grid)` is the `t` entry of
  `duration(KeyRates(grid), curve, cfs, times)`, which applies the same triangular
  continuous-zero bumps with exact derivatives. Pass the grid explicitly; the former
  default grid of annual knots from year 1 is no longer implied.
- **Embedded cashflow times take precedence.** Analytic key-rate forms now accept
  wrapped `Cashflow` objects with explicit times. Scalar, key-rate, and bundled
  sensitivities use embedded payment times, as do Hull–White default horizons.
  Numeric amounts use the corresponding explicit times. Explicit time vectors must
  cover the collection; trailing entries are ignored.
  **Migration:** to change payment dates, construct updated `Cashflow` objects or
  pass numeric amounts with the desired times.
- Unmarked contract and portfolio duration, DV01, and convexity default to
  `Effective()`, including `duration(DV01(), target, curve)`. Use `Spread()`
  explicitly for spread risk.
- New parallel forms without tenor grids: `duration(DV01(), curve) do c ... end`,
  two-curve callback IR01/CS01, and two-curve convexity blocks
  `(; base, credit, cross)` for callbacks and fixed cashflows.
- Yield-model modified duration and DV01 for fixed cashflows use analytic
  continuous-zero formulas instead of automatic differentiation.
- Callback APIs accept callable structs. Scalar cashflow APIs accept arrays,
  tuples, and finite generators. Arrays are flattened in column-major order;
  generators are collected once before valuation.
- Named cashflow results own independent arrays for each duration role and
  convexity block. Mutating one no longer changes another.
- Hessian calculations reuse value and gradient results through DiffResults.
  Contract duration bundles compute gradients without unused Hessians.

## v5.11.2 to v5.12.0

- ForwardDiff **1.x is now required**. Version 1.0 made Dual comparisons account
  for partials, which the exact zero-stream check needs to preserve cashflow-amount
  derivatives. Support for ForwardDiff 0.10 is removed; Julia 1.10 remains supported.
- Analytic `KeyRates` calculations preserve the numeric type of discounted cashflows,
  including `BigFloat` and automatic-differentiation values from curve parameters.
- Empty cashflow collections **and all-zero cashflow amounts** now have zero value and
  zero risk, including normalized duration and convexity by convention. This applies to
  explicit-cashflow scalar duration/convexity, DV01/IR01/CS01, legacy key rates, and
  `KeyRates` sensitivities (including Hull–White). Previously these inputs could produce
  `NaN`, a `BoundsError`, or a legacy default-grid `ArgumentError`. Results keep
  their usual shapes and use positive zeros.
  `present_values` likewise returns an empty vector or a vector of zeros.
- Covered explicit-cashflow methods accept a time grid longer than the amount
  vector and ignore trailing entries. Previously some scalar methods rejected
  them. Too few times now consistently throws `DimensionMismatch`. Legacy default
  key-rate grids and Hull–White default horizons use only the supplied cashflows'
  times. Empty streams return zero even with a populated time grid; legacy forms
  skip default-grid derivation for zero streams but still validate an explicit grid.
- Zero streams do not evaluate the curve or run simulations. Their numeric types come
  from the amounts and times, plus the tenor grid for key-rate results, **without the
  curve's numeric type**; nonzero streams still promote from discounted cashflows.
  Abstractly typed empty inputs fall back to `Float64`.
- Skipping Hull–White simulation for zero streams leaves the RNG unchanged. In a
  batch using one shared RNG, subsequent contracts therefore receive different
  draws than in prior versions. Use independently assigned RNG streams when
  contract-level reproducibility must be independent of preceding contracts.
- Nonzero amounts that offset to zero present value are not zero streams: normalized
  duration and convexity still produce `NaN`/`Inf`, and dollar exposures are not reset
  to zero. Valuation-function and contract inputs retain their existing behavior.
  Aggregate portfolio values and dollar derivatives before normalizing once. An
  unweighted average of individual durations includes zero-stream entries as zeros;
  it is not a portfolio duration. See [Zero cashflow streams](@ref).
- `VaR` and `CTE` now enforce their documented domain `0 ≤ α < 1` at construction,
  including through the `ValueAtRisk` and `ConditionalTailExpectation` aliases and
  explicitly typed constructors. `WangTransform` requires `0 < α < 1`; its earlier
  documentation incorrectly included the endpoints. Out-of-domain real values
  (including `NaN` and infinities) throw `ArgumentError`. Explicitly typed
  constructors validate the value after conversion. At zero, `VaR` remains the
  essential infimum and `CTE` remains the mean. This establishes behavior for
  invalid parameters; it does not change results at valid confidence levels.

## v5.11.1 to v5.11.2

This release fixes a class of bugs where risk measures returned believable but wrong finite numbers. Two behavior changes matter for existing code.

### VaR now uses the standard lower quantile

The old implementation selected the upper quantile at exact atom boundaries. The new `VaR(α)` is the standard generalized inverse, $\inf\{x : F(x) \ge \alpha\}$, everywhere: distributions, arrays, and discrete laws. Values can decrease by a whole atom at exact boundaries. This is an intentional correctness fix, not a numerical drift:

```julia
VaR(0.5)(Bernoulli(0.5))       # old: 1, new: 0
VaR(0.95)(collect(1.0:1000.0)) # old: 951, new: 950
```

Related details:

- `CTE` semantics are unchanged. It still averages exactly the worst `1-α` of probability mass, with fractional weight on the boundary atom.
- `VaR` on a distribution now returns `Distributions.quantile(risk, α)` and keeps its type — for example an `Int` for a count distribution.
- `VaR(0)` is the essential infimum. For a distribution with support unbounded below it returns `-Inf`; previously it returned an unconverged quadrature number.
- `robustvalue`'s adverse scenario for `VaR` now shifts the rank VaR itself selects. Previously, at an exact boundary the shifted set could exclude that rank, and the reported "robust" value equaled the base value.

### Divergent risks return honest values or throw

The old implementation integrated distorted distribution functions with unchecked quadrature and subtracted the two halves. Divergent integrals produced plausible finite numbers:

```julia
Expectation()(Cauchy())        # old: 0.0,      new: NaN  (mean does not exist)
Expectation()(Pareto(0.5, 1))  # old: ≈2.9e8,   new: Inf  (mean diverges)
CTE(0.95)(Cauchy())            # old: ≈215.14,  new: Inf  (tail mean diverges)
WangTransform(0.9)(Cauchy())   # old: a finite number, new: throws ErrorException
```

The policy: results that are provably divergent from the distribution's `mean` semantics return `NaN`/`±Inf`; a quadrature result that cannot be verified throws instead of returning a number.

Related details:

- `Expectation`, `VaR`, and `CTE` on distributions are now analytic or exact where possible. Values move within the old quadrature tolerance.
- All distortion measures on arrays and on discrete distributions (`DiscreteNonParametric`, `Binomial`, `Poisson`, …) now evaluate as exact or checked weighted sums instead of quadrature over step functions.
- `DualPower` and the internal complementary distortions now use numerically stable forms (`expm1`/`log1p`). The old algebra could silently lose the quadrature error certificate in far tails.
- Empty arrays now throw an `ArgumentError` for every risk measure.

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
- **Legacy bump-and-reprice `duration(keyrate::KeyRateZero/KeyRatePar, curve, cashflows)`** now derives the default key-rate grid from embedded `Cashflow` times rather than the vector *indices*. Pricing always used the embedded times, so values at grid points common to both versions are unchanged; what changes is the default grid *extent* (e.g. semiannual `Cashflow`s at 0.5…5.0 previously implied a grid out to 10 years, so `duration(KeyRate(7), curve, cfs)` returned `0.0` and now — like any timepoint outside the grid — raises an `ArgumentError`). Plain amount vectors are unaffected. When every timepoint is below 1, the default grid would be empty and also raises an `ArgumentError` asking for an explicit `krd_points` (previously an obscure `MethodError`).

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
