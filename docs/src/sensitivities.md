# Interest-Rate Sensitivities

Calculate durations, DV01s, and convexities for scalar rates, FinanceCore `Rate`s,
FinanceModels [`AbstractYieldModel`](https://github.com/JuliaActuary/FinanceModels.jl)
curves, and contracts. Scalar measures describe a parallel shift. Key-rate measures
decompose risk by tenor with triangular continuous-zero bumps on the original curve.
Callbacks use ForwardDiff through `Yield.TenorShift`; fixed cashflows use analytic
derivatives. See the [autodiff ALM chapter](https://modernfinancialmodeling.com/autodiff_alm)
for background.

Explicit cashflow inputs that are empty or have all-zero amounts return zero value
and risk without evaluating the curve. Nonzero amounts that offset to zero present
value retain their dollar exposures and undefined normalized risk. See
[Zero cashflow streams](@ref) for numeric types and portfolio aggregation.

## Shock coordinates

A one-basis-point shift must move some rate, and the dollar result depends on which
one. Each input is shocked in its own native form:

| Input | What moves by `s` | Modified duration | Convexity weight |
|:------|:------------------|:------------------|:-----------------|
| `Real` `y` (annual effective) | `y` | Macaulay / (1 + y) | `t(t+1) / (1+y)²` |
| `Periodic(y, m)` | the nominal rate `y` | Macaulay / (1 + y/m) | `t(t+1/m) / (1+y/m)²` |
| `Continuous(y)` | `y` | Macaulay | `t²` |
| Any `AbstractYieldModel`, including `Yield.Constant` | every continuous zero rate, in parallel | Macaulay | `t²` |
| `KeyRates(tenors)` | continuous zero rates, through a triangular bump at each tenor | per tenor | per tenor pair |
| `IR01`/`CS01` with fixed cashflows | the combined rate `base + spread`, in its own coordinate | — | — |
| Contracts with `Effective()`/`Spread()` | continuous zero rates of the projection and/or discount curve | — | — |

DV01 is `-∂V/∂s / 10000` in the same coordinate and keeps the position's sign: a
net liability has negative DV01. Scalar curve risk equals the sum of the
corresponding key-rate results, including convexity cross terms (see
[Convexity Conventions](@ref)).

Wrapping a scalar in `Yield.Constant` keeps its discount factors but changes what
moves, so duration and DV01 change by a factor of `1 + y`:

```@example sensitivities
using ActuaryUtilities, FinanceModels, FinanceCore

cfs   = [5.0, 5.0, 5.0, 5.0, 105.0]
times = [1.0, 2.0, 3.0, 4.0, 5.0]

(scalar_dv01 = duration(DV01(), 0.03, cfs, times),
 curve_dv01  = duration(DV01(), Yield.Constant(0.03), cfs, times),
 ratio       = duration(DV01(), Yield.Constant(0.03), cfs, times) / duration(DV01(), 0.03, cfs, times))
```

To measure a scalar yield in the curve coordinate, pass the equivalent continuous
rate:

```@example sensitivities
duration(DV01(), Continuous(log1p(0.03)), cfs, times) ≈ duration(DV01(), Yield.Constant(0.03), cfs, times)
```

## Scalar Measures

Without a `KeyRates` marker, duration, DV01, and convexity return parallel risk.
No tenor grid is involved:

```@example sensitivities
rates  = [0.03, 0.03, 0.03, 0.03, 0.03]
tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
zrc    = ZeroRateCurve(rates, tenors)

(duration  = duration(zrc, cfs, times),
 macaulay  = duration(Macaulay(), zrc, cfs, times),
 dv01      = duration(DV01(), zrc, cfs, times),
 convexity = convexity(zrc, cfs, times))
```

For a flat curve, the scalar measures match an explicitly continuous rate at the
same zero-rate level:

```@example sensitivities
(zrc_dur   = duration(zrc, cfs, times),
 rate_dur  = duration(Continuous(0.03), cfs, times),
 zrc_conv  = convexity(zrc, cfs, times),
 rate_conv = convexity(Continuous(0.03), cfs, times))
```

### Two curves: IR01 and CS01

Fixed cashflows discounted at `base + credit` have equal IR01, CS01, and DV01 of the
combined curve: a one-basis-point move in either component is a one-basis-point move
in the combined rate.

```@example sensitivities
base   = ZeroRateCurve([0.03, 0.03, 0.03, 0.03, 0.03], tenors)
credit = ZeroRateCurve([0.02, 0.02, 0.02, 0.02, 0.02], tenors)

(ir01 = duration(IR01(), base, credit, cfs, times),
 cs01 = duration(CS01(), base, credit, cfs, times),
 dv01 = duration(DV01(), base + credit, cfs, times))
```

The measures separate when the curves play different roles. Pass a callback that
receives `(base, credit)`; each measure applies a parallel shift to one curve:

```@example sensitivities
spread_margin = 0.02
floating_value(b, c) = sum(1:5) do t
    df_prev = t == 1 ? 1.0 : b(t - 1.0)
    coupon  = 100 * (df_prev / b(Float64(t)) - 1 + spread_margin)   # resets on the base curve
    (coupon + (t == 5 ? 100 : 0)) * b(Float64(t)) * c(Float64(t))
end

(ir01 = duration(IR01(), floating_value, base, credit),
 cs01 = duration(CS01(), floating_value, base, credit))
```

Two-curve convexity returns the parallel `base`, `credit`, and `cross` blocks:

```@example sensitivities
(fixed    = convexity(base, credit, cfs, times),
 floating = convexity(floating_value, base, credit))
```

## Key-Rate Decomposition

Pass `KeyRates(tenors)` to decompose risk by tenor. Choose the grid independently of
the curve's own knots, for example the buckets your reporting requires. The grid must
be nonempty, finite, positive, and strictly increasing. The `KeyRates` constructor
validates these requirements, and calculations revalidate the grid in case it has
been mutated.

The first and last bumps stay constant beyond the grid, so sensitivity after the
last tenor belongs to its bucket. Extend the grid to separate exposures at longer
maturities.

```@example sensitivities
# Key rate durations (modified): vector of -∂V/∂rᵢ / V
krds = duration(KeyRates(tenors), zrc, cfs, times)
```

```@example sensitivities
# Key rate DV01s: vector of -∂V/∂rᵢ / 10000
dv01s = duration(DV01(), KeyRates(tenors), zrc, cfs, times)
```

```@example sensitivities
# Key rate convexity matrix: ∂²V/∂rᵢ∂rⱼ / V
conv_matrix = convexity(KeyRates(tenors), zrc, cfs, times)
```

The scalar measures equal the sums of the decomposition:

```@example sensitivities
(duration(zrc, cfs, times) ≈ sum(krds),
 duration(DV01(), zrc, cfs, times) ≈ sum(dv01s),
 convexity(zrc, cfs, times) ≈ sum(conv_matrix))
```

Use `sensitivities` to calculate value, duration or DV01, and convexity together:

```@example sensitivities
result = sensitivities(KeyRates(tenors), zrc, cfs, times)
# result.value       — present value
# result.durations   — key rate durations (modified) — vector
# result.convexities — cross-convexity matrix — matrix
result
```

```@example sensitivities
# For DV01s instead of durations:
dv01_result = sensitivities(DV01(), KeyRates(tenors), zrc, cfs, times)
# dv01_result.value       — present value
# dv01_result.dv01s       — key rate DV01s — vector
# dv01_result.convexities — cross-convexity matrix — matrix
dv01_result
```

Two-curve key-rate forms return per-tenor IR01s and CS01s and per-pair convexity
matrices:

```@example sensitivities
(ir01s = duration(IR01(), KeyRates(tenors), base, credit, cfs, times),
 cs01s = duration(CS01(), KeyRates(tenors), base, credit, cfs, times))
```

```@example sensitivities
twocurve_result = sensitivities(KeyRates(tenors), base, credit, cfs, times)
twocurve_result.base_durations
```

The fixed-cashflow valuation is `V = Σ cf × base(t) × credit(t)`: discount factors
multiply, so continuously compounded zero rates add.

## Market Inputs

When the valuation builds its own curves from market data, differentiate with
respect to named input vectors instead of curve bumps. Each input element is bumped
in the units you pass in:

```@example sensitivities
linear_curve(z) = ZeroRateCurve(z, tenors, Spline.Linear())

inputs = sensitivities((; zeros = [0.02, 0.025, 0.03, 0.035, 0.04], spread = [0.01])) do m
    curve = linear_curve(m.zeros) + Yield.Constant(Continuous(only(m.spread)))
    pv(curve, cfs, times)
end

(value         = inputs.value,
 zero_dv01s    = inputs.key_rate_dv01.zeros,   # per input element
 spread_dv01   = inputs.dv01.spread)           # parallel shift of the whole input
```

With linear zero-rate interpolation, the per-element results equal the `KeyRates`
decomposition on the same knots.

## Callable Valuations

A valuation can be a function or a callable struct that holds its input data:

```@example sensitivities
struct CashflowValuation{C, T}
    amounts::C
    times::T
end
(v::CashflowValuation)(curve) = pv(curve, v.amounts, v.times)

valuation = CashflowValuation(cfs, tenors)
duration(zrc, valuation)
convexity(zrc, valuation)
sensitivities(KeyRates(tenors), valuation, zrc)
```

All callback APIs accept callable objects. Functions also support do-block syntax.

## Using Cashflow Objects

Yield-model cashflow methods accept `Vector{Cashflow}` directly:

```@example sensitivities
cfs_obj = Cashflow.([5.0, 5.0, 5.0, 5.0, 105.0], [1.0, 2.0, 3.0, 4.0, 5.0])

# These are equivalent:
a = duration(zrc, cfs_obj)                                                            # using Cashflow objects
b = duration(zrc, [5.0, 5.0, 5.0, 5.0, 105.0], [1.0, 2.0, 3.0, 4.0, 5.0])  # using amounts + times
(a, b, a ≈ b)
```

This applies to duration, DV01, two-curve IR01/CS01, convexity, and sensitivity bundles.

A `Cashflow` supplies its own amount and payment time. Explicit times apply to
numeric amounts; embedded times take precedence. An explicit time vector must
cover the collection, and trailing entries are ignored:

```@example sensitivities
fallback_times = fill(10.0, length(cfs_obj))
duration(KeyRates(tenors), zrc, cfs_obj, fallback_times) ≈
    duration(KeyRates(tenors), zrc, cfs_obj)
```

Hull–White default simulation horizons also use these resolved payment times. To change payment dates, construct new `Cashflow`
objects or pass numeric amounts with the desired times.

## Other yield models

The same API works with fitted curves, including Nelson–Siegel:

```@example sensitivities
ns        = Yield.NelsonSiegel(1.0, 0.04, -0.02, 0.01)
ns_knots  = [1.0, 2.0, 5.0, 10.0, 20.0]
ns_result = sensitivities(KeyRates(ns_knots), ns,
                          [5.0, 5.0, 5.0, 5.0, 105.0], [1.0, 2.0, 5.0, 10.0, 20.0])
ns_result.durations
```

The Nelson-Siegel parameters stay fixed; only the layered zero-rate bumps move under AD.

## Interest-Sensitive Instruments

For instruments whose cashflows depend on the rate environment (callable bonds, floaters, etc.), use the do-block syntax to pass a custom valuation function:

```@example sensitivities
# Callable bond: key rate durations (vector)
callable_krds = duration(KeyRates(tenors), zrc) do curve
    ncv = pv(curve, cfs, tenors)
    called_value = pv(curve, cfs[1:3], tenors[1:3]) + 102.0 * curve(3.0)
    min(ncv, called_value)
end
```

```@example sensitivities
# Scalar duration (default)
callable_dur = duration(zrc) do curve
    ncv = pv(curve, cfs, tenors)
    called_value = pv(curve, cfs[1:3], tenors[1:3]) + 102.0 * curve(3.0)
    min(ncv, called_value)
end
```

The function receives a curve object and must return a scalar value. ForwardDiff differentiates through the entire valuation, capturing any rate-dependent optionality.

## Two-Curve Key Rates

The two-curve callback forms also accept `KeyRates`. The callback receives the
bumped `(base, credit)` curves.

### Example: Credit-Risky Floating Rate Bond

Fixed cashflows have equal IR01 and CS01 under multiplicative discount composition.
For a floater, base-rate changes also reset coupons, so the sensitivities differ
bucket by bucket:

```@example sensitivities
credit_spread = 0.02
face          = 100.0

floater_result = sensitivities(KeyRates(tenors), base, credit) do base_curve, credit_curve
    total = 0.0
    for t in 1:5
        df_base      = base_curve(Float64(t))
        df_credit    = credit_curve(Float64(t))
        df_base_prev = t == 1 ? 1.0 : base_curve(Float64(t - 1))

        # Coupon resets to risk-free forward rate + fixed credit spread
        fwd = df_base_prev / df_base - 1.0
        total += face * (fwd + credit_spread) * df_base * df_credit

        # Principal at maturity
        t == 5 && (total += face * df_base * df_credit)
    end
    total
end

(IR01 = sum(floater_result.base_durations),
 CS01 = sum(floater_result.credit_durations))
```

Base-rate changes affect coupons and discounting; credit changes affect discounting only.

## Floating-Rate Instruments: Effective vs Spread Duration

Pass a FinanceModels contract or portfolio directly to reproject its cashflows.
The marker selects the risk:

- **Effective (rate) duration** — bump the curve, coupons re-fix → small (≈ time to next reset).
- **Spread (credit) duration** — bump the discount only, coupons fixed → ≈ maturity.

```@example sensitivities
using FinanceModels: Bond
floater = Bond.Floating(0.015, Periodic(1), 5.0, "SOFR")   # SOFR + 150bp, 5y annual

(effective = duration(Effective(), floater, zrc),   # rate duration, yrs — small
 spread    = duration(Spread(),    floater, zrc),   # spread duration, yrs — ≈ maturity
 dv01      = dv01(Effective(),     floater, zrc))   # effective DV01, $/bp
```

Calls without a marker default to `Effective()` for all three verbs, including
portfolios. Request spread risk explicitly with `Spread()`. Two-curve forms take
`(forward, credit)`: coupons project on `forward` and discount on `credit`. The
parallel measures take no tenor grid; use `KeyRates(tenors)` or `sensitivities`
for a key-rate decomposition.

```@example sensitivities
(duration(floater, zrc) ≈ duration(Effective(), floater, zrc),
 dv01(floater, zrc) ≈ dv01(Effective(), floater, zrc),
 convexity(floater, zrc) ≈ convexity(Effective(), floater, zrc))
```

`sensitivities` returns effective, spread, and forward risk together:

```@example sensitivities
s = sensitivities(floater, zrc, tenors)
(effective = s.effective_duration, spread = s.spread_duration, eff_dv01 = s.effective_dv01)
```

For a fixed bond `effective == spread ==` the modified duration. For an **in-force** floater whose current coupon is already fixed, [`locked_floater`](@ref) gives the conventional rate duration ≈ time to next reset:

```@example sensitivities
duration(Effective(), locked_floater(floater, 0.04, 1.0), zrc)   # ≈ 1y, not ≈ 5
```

### Portfolios

For a portfolio, pass a vector of contracts. The calculation sums their values
before normalizing risk:

```@example sensitivities
portfolio = [floater, Bond.Fixed(0.03, Periodic(1), 7.0)]
duration(portfolio, zrc)
```

### Multi-curve: risk-free + credit + ILP + index

Pass named discount layers and a coupon-projection `index` to obtain risk by role:

```@example sensitivities
rf     = zrc
credit = Yield.Constant(Continuous(0.01))
ilp    = Yield.Constant(Continuous(0.004))
r = sensitivities(floater, tenors; discount = (; rf, credit, ilp), index = zrc)
r.duration    # (; rf ≈ IR01, credit ≈ CS01, ilp = "ILP01", index = reset sensitivity)
```

Additional layers can represent liquidity, matching adjustment, or basis spreads.
Use a callback with [`reproject`](@ref) for custom valuations:

```@example sensitivities
sensitivities((; rf, credit, ilp, index = zrc); tenors) do c
    present_value(c.rf + c.credit + c.ilp, reproject(floater, c.index))
end
```

Use [`zspread`](@ref) to fit the discount margin to a market price:

```@example sensitivities
zspread(floater, zrc, 0.99)
```

## Portfolio Sensitivity

DV01s are additive across positions, so a portfolio's DV01 vector equals the sum of individual DV01s:

```@example sensitivities
# Two bonds: 5-year 5% coupon and 5-year 3% coupon
bond1_cfs   = [5.0, 5.0, 5.0, 5.0, 105.0]
bond1_times = [1.0, 2.0, 3.0, 4.0, 5.0]
bond2_cfs   = [3.0, 3.0, 3.0, 3.0, 103.0]
bond2_times = [1.0, 2.0, 3.0, 4.0, 5.0]

# Portfolio DV01 vector
portfolio_dv01 = duration(DV01(), KeyRates(tenors), zrc) do curve
    pv(curve, bond1_cfs, bond1_times) + pv(curve, bond2_cfs, bond2_times)
end

# Equivalently, sum the individual DV01 vectors:
dv01_1 = duration(DV01(), KeyRates(tenors), zrc, bond1_cfs, bond1_times)
dv01_2 = duration(DV01(), KeyRates(tenors), zrc, bond2_cfs, bond2_times)

(portfolio_dv01, dv01_1 .+ dv01_2, portfolio_dv01 ≈ dv01_1 .+ dv01_2)
```

### Example: Portfolio of Floating Rate Bonds

For a floater portfolio, differentiate both coupon amounts and discount factors:

```@example sensitivities
flt_rates  = [0.02, 0.025, 0.03, 0.035, 0.04, 0.042, 0.044, 0.046, 0.048, 0.05]
flt_tenors = collect(1.0:10.0)
flt_zrc    = ZeroRateCurve(flt_rates, flt_tenors)

# 10 floating rate bonds: maturities 1yr to 10yr, face 100 each,
# annual coupons = 1yr forward rate + 50bp credit spread
notionals  = fill(100.0, 10)
maturities = 1:10
flt_spread = 0.005

# The do-block receives the curve and returns the total present value
# of all cashflows across the portfolio.
floater_portfolio = sensitivities(KeyRates(flt_tenors), flt_zrc) do curve
    total = 0.0
    for (notional, mat) in zip(notionals, maturities)
        # For each bond, loop over annual payment dates t = 1, 2, ..., maturity
        for t in 1:mat
            df      = curve(Float64(t))
            df_prev = t == 1 ? 1.0 : curve(Float64(t - 1))

            # 1yr simple forward rate from t-1 to t: F = P(0,t-1)/P(0,t) - 1
            fwd = df_prev / df - 1.0

            # Floating coupon PV: notional × (forward rate + spread) × P(0,t)
            total += notional * (fwd + flt_spread) * df

            # Return principal at maturity
            t == mat && (total += notional * df)
        end
    end
    total
end

(value = floater_portfolio.value,
 total_duration = sum(floater_portfolio.durations))
```

A par floater at reset has near-zero effective duration. A fixed coupon spread
adds duration because those payments do not reset with the curve.

## Stochastic Model Sensitivities

ForwardDiff can differentiate the simulated valuation through Hull–White path
generation. These derivatives describe the Monte Carlo estimate, which remains
subject to sampling and time-discretization error.

### What is being differentiated?

The curve-risk API differentiates continuous-zero bumps at the supplied tenors.
For a stochastic valuation:

```julia
hw = ShortRate.HullWhite(0.1, 0.01, zrc)
sensitivities(KeyRates(tenors), hw, cfs, times; n_scenarios=500, rng=Xoshiro(42))
```

the chain of differentiation is:

1. ForwardDiff perturbs the zero-rate bump at tenor `i`
2. The perturbed `curve` changes the forward curve `f(0, t)`
3. Hull-White recalibrates `θ(t)` from the new forwards
4. All simulated paths shift (same random draws, different drift)
5. The value derivative, divided by `-V`, gives KRD at tenor `i`

Mean reversion `a` and volatility `σ` stay fixed. These KRDs measure the
portfolio's response to curve shocks.

### Model parameter sensitivities (vega, mean-reversion sensitivity)

Parameter sensitivities measure changes in `a` or `σ`:

| | Curve KRDs (`-(∂V/∂rᵢ)/V`) | Parameter sensitivities (`∂V/∂a`, `∂V/∂σ`) |
|---|---|---|
| **What moves** | Market zero rates | Model calibration parameters |
| **Use case** | Hedging with bonds/swaps | Model risk, calibration stability |

The curve-risk methods do not calculate parameter sensitivities. One way to
estimate them is finite differences with matched random draws:

```@example sensitivities
using FinanceModels: ShortRate, simulate
using FinanceCore: discount
using Random: Xoshiro

mc_rates  = [0.03, 0.03, 0.03, 0.03, 0.03]
mc_tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
mc_cfs    = [5.0, 5.0, 5.0, 5.0, 105.0]

function mc_value(a, σ)
    curve     = ZeroRateCurve(mc_rates, mc_tenors)
    hw        = ShortRate.HullWhite(a, σ, curve)
    scenarios = simulate(hw; n_scenarios = 1000, timestep = 1/12, horizon = 6.0, rng = Xoshiro(42))
    sum(pv(sc, mc_cfs, mc_tenors) for sc in scenarios) / 1000
end

# Finite-difference sensitivities
ε       = 1e-5
dV_da   = (mc_value(0.1 + ε, 0.01) - mc_value(0.1 - ε, 0.01)) / (2ε)   # mean reversion
dV_dσ   = (mc_value(0.1, 0.01 + ε) - mc_value(0.1, 0.01 - ε)) / (2ε)   # volatility (vega)

(dV_da, dV_dσ)
```

### Hull-White: sensitivities w.r.t. the initial term structure

A Hull–White model calibrates its drift θ(t) to the initial yield curve. Calculate
the simulated value's sensitivity to curve bumps:

```@example sensitivities
# Key rate sensitivities of E[V] under Hull-White dynamics
hw_curve = ZeroRateCurve(mc_rates, mc_tenors)
hw       = ShortRate.HullWhite(0.1, 0.01, hw_curve)
hw_result = sensitivities(KeyRates(mc_tenors), hw, mc_cfs, mc_tenors;
                          n_scenarios = 500,
                          timestep    = 1/12,
                          horizon     = 6.0,
                          rng         = Xoshiro(42))

(durations = hw_result.durations,
 sum_durations = sum(hw_result.durations))
```

This uses nested AD: curve-risk derivatives pass through the forward-rate
derivatives used to calibrate Hull–White drift. ForwardDiff's
[tag system](https://github.com/JuliaDiff/ForwardDiff.jl/issues/83) separates them.

### Comparison: deterministic vs model-based sensitivities

Compare simulated sensitivities with direct discounting for fixed cashflows:

```@example sensitivities
# Deterministic: discount directly off the initial curve
det_result = sensitivities(KeyRates(mc_tenors), hw_curve, mc_cfs, mc_tenors)

# Model-based: average across simulated rate paths (computed above as hw_result)
(det_durations  = det_result.durations,
 hw_durations   = hw_result.durations,
 sum_det        = sum(det_result.durations),
 sum_hw         = sum(hw_result.durations))
```

For fixed cashflows, exact risk-neutral valuation gives
`V = Σ cf_i × P(0, t_i)` ([Glasserman, 2003, Ch. 7](https://link.springer.com/book/10.1007/978-0-387-21617-1)).
A model calibrated to each shocked curve should therefore reproduce both total
and bucket risk under those same shocks. Sampling, time discretization, and the
implementation of curve shocks can cause differences in numerical estimates.
Agreement in total alone does not validate the individual buckets.

Pathwise AD differentiates the simulated estimate through drift calibration,
path generation, and valuation. See
[Giles & Glasserman (2006)](https://people.maths.ox.ac.uk/~gilesm/files/mc_greeks.pdf).

!!! note
    Reuse the same random draws for every AD evaluation. Changing the draws
    between evaluations makes the value and derivatives inconsistent.

## Choosing Interpolation

`ZeroRateCurve` accepts an optional third argument for the interpolation method:

```@example sensitivities
interp_rates  = [0.02, 0.03, 0.04, 0.05]
interp_tenors = [1.0, 3.0, 5.0, 10.0]

zrc_default = ZeroRateCurve(interp_rates, interp_tenors)                       # default: MonotoneConvex
zrc_pchip   = ZeroRateCurve(interp_rates, interp_tenors, Spline.PCHIP())       # PCHIP
zrc_lin     = ZeroRateCurve(interp_rates, interp_tenors, Spline.Linear())      # linear
zrc_cub     = ZeroRateCurve(interp_rates, interp_tenors, Spline.Cubic())       # cubic spline
zrc_aki     = ZeroRateCurve(interp_rates, interp_tenors, Spline.Akima())       # Akima

# All can be passed into the same sensitivities API:
interp_cfs = [3.0, 3.0, 3.0, 103.0]
(default_durs = sensitivities(KeyRates(interp_tenors), zrc_default, interp_cfs, interp_tenors).durations,
 linear_durs  = sensitivities(KeyRates(interp_tenors), zrc_lin,     interp_cfs, interp_tenors).durations)
```

Interpolation controls the base curve between its quoted points. The supported
choices include monotone convex, PCHIP, linear, Akima, and cubic interpolation.
See the [FinanceModels interpolation guide](https://docs.juliaactuary.org/FinanceModels/dev/interpolation/)
for their smoothness and shape constraints.

`KeyRates` applies the same triangular bumps for every interpolation method.
Changing the base interpolation can change discounted cashflow weights, but it
does not change the bump shape or tenor grid.

## Validating AD vs Bump-and-Reprice

Compare AD with central finite differences using the same shocks. The finite
difference has O(ε²) truncation error:

```@example sensitivities
val_rates  = [0.02, 0.03, 0.04, 0.05]
val_tenors = [1.0, 3.0, 5.0, 10.0]
val_zrc    = ZeroRateCurve(val_rates, val_tenors)
val_cfs    = [3.0, 3.0, 3.0, 103.0]

# AD (exact) — use KeyRates(tenors) for the per-tenor vector
ad_dv01 = duration(DV01(), KeyRates(val_tenors), val_zrc, val_cfs, val_tenors)

# Finite difference (bump-and-reprice)
ε = 1e-5
fd_dv01 = map(1:4) do i
    rates_up      = copy(val_rates); rates_up[i] += ε
    rates_dn      = copy(val_rates); rates_dn[i] -= ε
    v_up = pv(ZeroRateCurve(rates_up, val_tenors), val_cfs, val_tenors)
    v_dn = pv(ZeroRateCurve(rates_dn, val_tenors), val_cfs, val_tenors)
    -(v_up - v_dn) / (2ε) / 10_000
end

(; ad_dv01, fd_dv01, max_abs_diff = maximum(abs.(ad_dv01 .- fd_dv01)))
```

## Validating AD with TenorShift

Use `TenorShift` to compare the observed price change with the derivative's
prediction for a small finite shock:

```@example sensitivities
# AD: total DV01 across all tenors — already in dollar-per-1bp units
total_dv01 = sum(duration(DV01(), KeyRates(val_tenors), val_zrc, val_cfs, val_tenors))

# TenorShift: actual PV change under a +1 bp parallel shift
pv_base    = present_value(val_zrc, val_cfs, val_tenors)
shifted    = val_zrc + (z, t) -> z + Continuous(0.0001)  # +1 bp
pv_shifted = present_value(shifted, val_cfs, val_tenors)
actual_change = -(pv_shifted - pv_base)

# DV01 is per-1bp, so it directly predicts the 1bp PV change
predicted_change = total_dv01

(; predicted_change = round(predicted_change, digits = 6),
   actual_change    = round(actual_change,    digits = 6),
   ratio            = round(actual_change / predicted_change, digits = 6))
```

The ratio approaches one as the shock shrinks. DV01 is a first-order estimate;
finite price changes also include convexity and higher-order terms.
