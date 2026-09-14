# Financial Math Submodule

Calculate present values, duration, convexity, and related financial measures.

## Quickstart

```julia
using ActuaryUtilities

cfs = [5, 5, 105]
times    = [1, 2, 3]

discount_rate = 0.03

present_value(discount_rate, cfs, times)           # 105.65
duration(Macaulay(), discount_rate, cfs, times)    #   2.86
duration(discount_rate, cfs, times)                #   2.78
convexity(discount_rate, cfs, times)               #  10.62
```

See [Convexity Conventions](@ref) for rate-shock definitions and worked examples.

!!! tip "Floating-rate, multi-curve & portfolios"
    Use `Macaulay` and `Modified` for fixed cashflows. Pass contracts or portfolios
    directly to reproject coupons under curve shocks. `Effective()` selects rate
    risk; `Spread()` selects spread risk. `sensitivities` returns both in one
    calculation. See [Key Rate Sensitivities](@ref) for examples and multiple curves.


## Zero cashflow streams

Empty and all-zero cashflow collections return zero value and risk. Normalized
duration and convexity are zero by convention. This applies to scalar, key-rate,
legacy, and Hull–White cashflow sensitivities. `present_values` returns an empty
vector or a vector of zeros.

Every cashflow needs a corresponding time, but the time grid may be longer.
Unused trailing times are ignored, including when deriving a legacy key-rate grid
or a Hull–White simulation horizon. Too few times throws `DimensionMismatch`.
Empty cashflows are valid with either an empty or populated time grid.

| Cashflow amounts | Value and dollar risk | Normalized risk |
|:--|:--|:--|
| Empty or all exactly zero | Zero | Zero by convention |
| Nonzero amounts with zero net present value | Dollar risk can be nonzero | Undefined (`NaN`/`Inf`) |
| Nonzero present value | Calculated as usual | Calculated as usual |

Scalar DV01 differentiates signed value directly. It remains defined at zero
present value when the valuation has a finite derivative; IR01 and CS01 use the
same calculation. Relative duration and convexity are still undefined there.

```jldoctest zero_value_dollar_risk
julia> using ActuaryUtilities, FinanceModels, FinanceCore

julia> curve = Yield.Constant(Continuous(0.0));

julia> cfs = [-1.0, 1.0]; times = [0.0, 1.0];

julia> pv(curve, cfs, times)
0.0

julia> (duration(DV01(), curve, cfs, times),
        duration(DV01(), curve, c -> pv(c, cfs, times)),
        duration(IR01(), curve, curve, cfs, times),
        duration(CS01(), curve, curve, cfs, times))
(0.0001, 0.0001, 0.0001, 0.0001)

julia> !isfinite(duration(curve, cfs, times))
true
```

The zero check uses exact `iszero` on amounts, including AD partials. Both `0.0`
and `-0.0` count as zero; tiny nonzero amounts do not. Assigned zero duration is a
convention, not the limit as amounts shrink. Explicit tenor grids are still validated.

Zero streams skip curve evaluation and Hull–White simulation. Result types come
from amounts and times, plus the tenor grid for key-rate risk. Abstractly typed
empty inputs fall back to `Float64`. Nonzero streams also incorporate the curve's
numeric type, so a zero stream may return a different type. Derive batch and AD
buffer types from the calculation's inputs rather than the first result.

Skipping Hull–White simulation leaves the RNG unchanged, so a shared-RNG batch
uses different subsequent draws than versions that simulated zero streams.
Independently assigned RNG streams avoid dependence on preceding contracts.

To aggregate portfolio risk, sum values and dollar derivatives before normalizing
once. This also preserves exposures from positions whose net value is zero.
An unweighted average of contract durations is not portfolio duration. A zero
callback or contract value alone does not identify an empty or all-zero stream.

## Curve Transformations

[`FinanceModels.Yield.TenorShift`](https://docs.juliaactuary.org/FinanceModels/dev/) lets you lazily transform any yield curve's zero rates via `curve + (z, t) -> new_rate`. This is useful for scenario analysis (parallel shifts, twists, stresses) without refitting.

### Parallel shift

```@example transformations
using ActuaryUtilities, FinanceModels, FinanceCore

base = Yield.Constant(Continuous(0.05))
shifted = base + (z, t) -> z + Continuous(0.01)   # +100 bp parallel

zero(shifted, 1.0)
```

### Periodic rate arithmetic

Rate arithmetic automatically handles compounding conversion — a `Periodic(0.01, 1)` bump converts to continuous internally:

```@example transformations
shifted_p = base + (z, t) -> z + Periodic(0.01, 1)
zero(shifted_p, 5.0)   # ≈ Continuous(0.05 + log(1.01))
```

### Tenor-dependent twist

A steepener that fades at 30y:

```@example transformations
twist = base + (z, t) -> z + Continuous(0.02 * max(0.0, 1.0 - t / 30.0))
(zero(twist, 1.0), zero(twist, 15.0), zero(twist, 30.0))
```

### PV comparison under stress

```@example transformations
cfs = [5.0, 5.0, 5.0, 105.0]
times = [1.0, 2.0, 3.0, 4.0]

pv_base = present_value(base, cfs, times)
pv_shifted = present_value(shifted, cfs, times)
pct_change = (pv_shifted - pv_base) / pv_base * 100
(; pv_base = round(pv_base, digits=4), pv_shifted = round(pv_shifted, digits=4), pct_change = round(pct_change, digits=2))
```

### Bootstrapped curve + stress

```@example transformations
quotes = ZCBYield.([0.04, 0.05, 0.055, 0.06], [1.0, 3.0, 5.0, 10.0])
fitted = fit(Spline.Linear(), quotes, Fit.Bootstrap())
stressed = fitted + (z, t) -> z + Continuous(0.005)   # +50 bp

dur_base = duration(fitted, cfs, times)
dur_stressed = duration(stressed, cfs, times)
(; dur_base = round(dur_base, digits=4), dur_stressed = round(dur_stressed, digits=4))
```

### Negative rates

Real-world EUR/JPY/CHF scenarios with negative base rates:

```@example transformations
neg = Yield.Constant(Continuous(-0.01))
shifted_neg = neg + (z, t) -> z + Continuous(0.005)
(; rate = zero(shifted_neg, 5.0), df = round(discount(shifted_neg, 5.0), digits=6))
```

## API

### Exported API
```@autodocs
Modules = [ActuaryUtilities.FinancialMath]
Private = false
```

### Unexported API
```@autodocs
Modules = [ActuaryUtilities.FinancialMath]
Public = false
```
