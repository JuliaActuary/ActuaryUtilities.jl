# Financial Math Submodule

Provides a set of common routines in financial maths.

## Quickstart

```julia
cfs = [5, 5, 105]
times    = [1, 2, 3]

discount_rate = 0.03

present_value(discount_rate, cfs, times)           # 105.65
duration(Macaulay(), discount_rate, cfs, times)    #   2.86
duration(discount_rate, cfs, times)                #   2.78
convexity(discount_rate, cfs, times)               #  10.62
```


## Curve Transformations

[`FinanceModels.Yield.TransformedYield`](https://docs.juliaactuary.org/FinanceModels/dev/) lets you lazily transform any yield curve's zero rates via `curve + (z, t) -> new_rate`. This is useful for scenario analysis (parallel shifts, twists, stresses) without refitting.

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