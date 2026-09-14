# Key Rate Sensitivities

Calculate key-rate durations, DV01s, and convexities for any FinanceModels
[`AbstractYieldModel`](https://github.com/JuliaActuary/FinanceModels.jl).

Sensitivities measure triangular continuous-zero bumps on the original curve.
Callbacks use ForwardDiff through `Yield.TenorShift`; fixed cashflows use analytic
derivatives. See the [autodiff ALM chapter](https://modernfinancialmodeling.com/autodiff_alm)
for background.

## API shape: curve + explicit KRD knots

Every key-rate API takes the **curve** and an **explicit `tenors` vector**:

```julia
duration(KeyRates(knots), curve, cfs, times)
```

Choose the risk grid independently of the curve's own knots. For a `ZeroRateCurve`,
you can use `zrc.tenors`; otherwise choose the buckets your reporting requires.

**Requirements on `tenors`**: nonempty, finite, sorted ascending, distinct, and strictly positive. The `KeyRates` constructor validates these requirements, and calculations revalidate the grid in case it has been mutated.

Explicit cashflow inputs that are empty or have all-zero amounts return zero value
and risk without evaluating the curve. Nonzero amounts that offset to zero present
value retain their dollar exposures and undefined normalized risk. See
[Zero cashflow streams](@ref) for numeric types and portfolio aggregation.

**Endpoint extrapolation:** the first and last bumps stay constant beyond the
grid. Sensitivity after the last tenor belongs to its bucket. Extend the grid
to separate exposures at longer maturities.

## Basic Usage

```@example sensitivities
using ActuaryUtilities, FinanceModels, FinanceCore

rates  = [0.03, 0.03, 0.03, 0.03, 0.03]
tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
zrc    = ZeroRateCurve(rates, tenors)

cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

# Scalar modified duration (sum of KRDs)
dur = duration(zrc, tenors, cfs, tenors)
```

```@example sensitivities
# Scalar DV01
dv01_scalar = duration(DV01(), zrc, tenors, cfs, tenors)
```

```@example sensitivities
# Scalar convexity
conv = convexity(zrc, tenors, cfs, tenors)
```

To get the full key-rate decomposition (vectors/matrices), use `KeyRates(tenors)`:

```@example sensitivities
# Key rate durations (modified): vector of -∂V/∂rᵢ / V
krds = duration(KeyRates(tenors), zrc, cfs, tenors)
```

```@example sensitivities
# Key rate DV01s: vector of -∂V/∂rᵢ / 10000
dv01s = duration(DV01(), KeyRates(tenors), zrc, cfs, tenors)
```

```@example sensitivities
# Key rate convexity matrix: ∂²V/∂rᵢ∂rⱼ / V
conv_matrix = convexity(KeyRates(tenors), zrc, cfs, tenors)
```

Use `sensitivities` to calculate value, duration or DV01, and convexity together:

```@example sensitivities
result = sensitivities(KeyRates(tenors), zrc, cfs, tenors)
# result.value       — present value
# result.durations   — key rate durations (modified) — vector
# result.convexities — cross-convexity matrix — matrix
result
```

```@example sensitivities
# For DV01s instead of durations:
dv01_result = sensitivities(DV01(), KeyRates(tenors), zrc, cfs, tenors)
# dv01_result.value       — present value
# dv01_result.dv01s       — key rate DV01s — vector
# dv01_result.convexities — cross-convexity matrix — matrix
dv01_result
```

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
a = duration(zrc, tenors, cfs_obj)                                                            # using Cashflow objects
b = duration(zrc, tenors, [5.0, 5.0, 5.0, 5.0, 105.0], [1.0, 2.0, 3.0, 4.0, 5.0])  # using amounts + times
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

Legacy default key-rate grids and Hull–White default simulation horizons also use
these resolved payment times. To change payment dates, construct new `Cashflow`
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

## Scalar vs Key-Rate Decomposition

Without `KeyRates`, duration, DV01, and convexity return scalar parallel risk.

For an `AbstractYieldModel`, scalar duration and convexity use an additive
parallel shift in continuously compounded zero-rate space. This is the same
shock coordinate used by the tenor-aware and key-rate forms. Plain scalar and
explicit `Rate` inputs continue to use their own compounding conventions.

Scalar convexity equals the sum of **all** key-rate matrix entries, including
cross terms. See [Convexity Conventions](@ref) for the derivation and examples.

To obtain the per-tenor decomposition, pass `KeyRates(tenors)` as the first argument:

```@example sensitivities
# Scalar (default) — same as sum of key-rate decomposition
scalar_dur   = duration(zrc, tenors, cfs, tenors)
scalar_dv01  = duration(DV01(), zrc, tenors, cfs, tenors)
scalar_conv  = convexity(zrc, tenors, cfs, tenors)

# Key-rate decomposition
vector_dur   = duration(KeyRates(tenors), zrc, cfs, tenors)
vector_dv01  = duration(DV01(), KeyRates(tenors), zrc, cfs, tenors)
matrix_conv  = convexity(KeyRates(tenors), zrc, cfs, tenors)

(scalar_dur, sum(vector_dur))
```

The scalar value equals the sum of the key-rate decomposition:

```@example sensitivities
duration(zrc, tenors, cfs, tenors) ≈ sum(duration(KeyRates(tenors), zrc, cfs, tenors))
convexity(zrc, cfs, tenors) ≈ convexity(zrc, tenors, cfs, tenors)
convexity(zrc, tenors, cfs, tenors) ≈ sum(convexity(KeyRates(tenors), zrc, cfs, tenors))
```

For a flat curve, the scalar measures match an explicitly continuous rate at
the same zero-rate level:

```@example sensitivities
flat_cfs    = [5.0, 5.0, 5.0, 5.0, 105.0]
flat_tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
flat_zrc    = ZeroRateCurve(fill(0.03, 5), flat_tenors)

(zrc_dur   = duration(flat_zrc, flat_cfs, flat_tenors),
 rate_dur  = duration(Continuous(0.03), flat_cfs, flat_tenors),
 zrc_conv  = convexity(flat_zrc, flat_cfs, flat_tenors),
 rate_conv = convexity(Continuous(0.03), flat_cfs, flat_tenors))
```

Request Macaulay duration with its marker:

```@example sensitivities
duration(Macaulay(), 0.03, flat_cfs, flat_tenors)
```

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

## Two-Curve Decomposition

Decompose sensitivities into base (risk-free) and credit spread components using `IR01` and `CS01`:

```@example sensitivities
base   = ZeroRateCurve([0.03, 0.03, 0.03, 0.03, 0.03], tenors)
credit = ZeroRateCurve([0.02, 0.02, 0.02, 0.02, 0.02], tenors)

# Scalar IR01 and CS01
ir01 = duration(IR01(), base, credit, tenors, cfs, tenors)
cs01 = duration(CS01(), base, credit, tenors, cfs, tenors)
(ir01, cs01)
```

```@example sensitivities
# Key-rate decomposition (vectors)
ir01s = duration(IR01(), KeyRates(tenors), base, credit, cfs, tenors)
cs01s = duration(CS01(), KeyRates(tenors), base, credit, cfs, tenors)
(ir01s, cs01s)
```

```@example sensitivities
# Two-curve convexity — scalars by default
conv_2c = convexity(base, credit, tenors, cfs, tenors)
# conv_2c.base, conv_2c.credit, conv_2c.cross (all scalars)
```

```@example sensitivities
# Key-rate decomposition (matrices)
conv_2c_kr = convexity(KeyRates(tenors), base, credit, cfs, tenors)
# conv_2c_kr.base, conv_2c_kr.credit, conv_2c_kr.cross (all matrices)
```

```@example sensitivities
# Full two-curve sensitivities (always key-rate decomposition)
twocurve_result = sensitivities(KeyRates(tenors), base, credit, cfs, tenors)
twocurve_result.base_durations
```

The fixed-cashflow valuation is `V = Σ cf × base(t) × credit(t)`: discount factors
multiply, so continuously compounded zero rates add.

### Example: Credit-Risky Floating Rate Bond

Fixed cashflows have equal IR01 and CS01 under this discount composition. For a
floater, base-rate changes also reset coupons, so the sensitivities can differ:

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

duration(Effective(), floater, zrc, tenors)   # rate duration, yrs — small
duration(Spread(),    floater, zrc, tenors)   # spread duration, yrs — ≈ maturity
dv01(Effective(),     floater, zrc, tenors)   # effective DV01, $/bp
```

Single-curve calls without a marker default to `Effective()` for all three verbs,
including portfolios. Request spread risk explicitly with `Spread()`.

```@example sensitivities
(duration(floater, zrc, tenors) ≈ duration(Effective(), floater, zrc, tenors),
 dv01(floater, zrc, tenors) ≈ dv01(Effective(), floater, zrc, tenors),
 convexity(floater, zrc, tenors) ≈ convexity(Effective(), floater, zrc, tenors))
```

`sensitivities` returns effective, spread, and forward risk together:

```@example sensitivities
s = sensitivities(floater, zrc, tenors)
(effective = s.effective_duration, spread = s.spread_duration, eff_dv01 = s.effective_dv01)
```

For a fixed bond `effective == spread ==` the modified duration. For an **in-force** floater whose current coupon is already fixed, [`locked_floater`](@ref) gives the conventional rate duration ≈ time to next reset:

```@example sensitivities
duration(Effective(), locked_floater(floater, 0.04, 1.0), zrc, tenors)   # ≈ 1y, not ≈ 5
```

### Portfolios

For a portfolio, pass a vector of contracts. The calculation sums their values
before normalizing risk:

```@example sensitivities
portfolio = [floater, Bond.Fixed(0.03, Periodic(1), 7.0)]
duration(portfolio, zrc, tenors)
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
