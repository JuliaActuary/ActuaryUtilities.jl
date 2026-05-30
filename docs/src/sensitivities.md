# Key Rate Sensitivities

Compute key rate durations, DV01s, and convexities via automatic differentiation against any [`AbstractYieldModel`](https://github.com/JuliaActuary/FinanceModels.jl) — `ZeroRateCurve`, `NelsonSiegel`, a fitted spline, a user-defined composite, anything that defines `discount(curve, t)`.

The AD pathway layers a triangular-hat zero-rate bump on top of the user's curve via `Yield.TenorShift`, then takes ForwardDiff gradients w.r.t. the bump magnitudes. The user's curve is preserved at all non-knot points; no resampling or refitting occurs. See the [autodiff ALM chapter](https://modernfinancialmodeling.com/autodiff_alm) for background.

## API shape: curve + explicit KRD knots

Every key-rate API takes the **curve** and an **explicit `tenors` vector**:

```julia
duration(KeyRates(knots), curve, cfs, times)
```

The KRD knot grid is a separate modeling choice from any tenor structure baked into the curve itself. For a `ZeroRateCurve`, `zrc.tenors` is the natural default; for other curves you supply your preferred bucket convention (Bloomberg, FRTB, BMA SBA, etc.).

**Requirements on `tenors`**: sorted ascending, distinct, strictly positive. These preconditions are not checked at runtime — a malformed grid produces wrong gradients silently.

**Endpoint extrapolation**: the hat bump is flat outside the knot range. Bumping `tenors[1]` perturbs all cashflows at `t ≤ tenors[1]` equally; bumping `tenors[end]` perturbs all cashflows at `t ≥ tenors[end]` equally. For long-duration insurance liabilities (LTC, deferred / payout annuities), extend the grid past your longest cashflow if you want that sensitivity decomposed.

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
dv01 = duration(DV01(), zrc, tenors, cfs, tenors)
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

For a complete set of key-rate results in a single AD pass, use `sensitivities`:

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

## Using Cashflow Objects

Any `AbstractYieldModel + tenors` method accepts `Vector{Cashflow}` directly, eliminating the need to manually split into amounts and times:

```@example sensitivities
cfs_obj = Cashflow.([5.0, 5.0, 5.0, 5.0, 105.0], [1.0, 2.0, 3.0, 4.0, 5.0])

# These are equivalent:
a = duration(zrc, tenors, cfs_obj)                                                            # using Cashflow objects
b = duration(zrc, tenors, [5.0, 5.0, 5.0, 5.0, 105.0], [1.0, 2.0, 3.0, 4.0, 5.0])  # using amounts + times
(a, b, a ≈ b)
```

The same dispatch works with all method variants — `KeyRates(tenors)`, `DV01()`, two-curve `IR01()`/`CS01()`, `convexity`, and `sensitivities`.

## Any AbstractYieldModel — no resampling required

Because the AD path is curve-agnostic, you can compute key rates directly against any `AbstractYieldModel` — a fitted Nelson-Siegel, a user-defined composite, a UFR extrapolator, etc. There is no need to first convert to `ZeroRateCurve`:

```@example sensitivities
ns        = Yield.NelsonSiegel(1.0, 0.04, -0.02, 0.01)
ns_knots  = [1.0, 2.0, 5.0, 10.0, 20.0]
ns_result = sensitivities(KeyRates(ns_knots), ns,
                          [5.0, 5.0, 5.0, 5.0, 105.0], [1.0, 2.0, 5.0, 10.0, 20.0])
ns_result.durations
```

The Nelson-Siegel parameters stay fixed; only the layered zero-rate bumps move under AD.

## Scalar vs Key-Rate Decomposition

By default, `duration` and `convexity` (without `KeyRates`) return **scalars** — the total modified duration, DV01, or convexity. This is consistent with the yield-based API (`duration(0.03, cfs, times)`).

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
```

For a flat curve, the scalar modified duration matches the yield-based API:

```@example sensitivities
flat_cfs    = [5.0, 5.0, 5.0, 5.0, 105.0]
flat_tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
flat_zrc    = ZeroRateCurve(fill(0.03, 5), flat_tenors)

(zrc_dur     = duration(flat_zrc, flat_tenors, flat_cfs, flat_tenors),
 yield_dur   = duration(0.03, flat_cfs, flat_tenors))
```

For Macaulay duration, use the scalar yield API directly — there is no `ZeroRateCurve` dispatch:

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

The default two-curve valuation uses multiplicative discount factors: `V = Σ cf × base(t) × credit(t)`, which corresponds to additive rates.

### Example: Credit-Risky Floating Rate Bond

For fixed cashflows, IR01 and CS01 are identical because base and credit rates enter additively. A **credit-risky floating rate bond** breaks this symmetry — its coupons reset to the risk-free forward rate plus a fixed credit spread, so bumping base rates changes both coupon amounts and discount factors (partially canceling), while bumping credit rates only affects discounting:

```@example sensitivities
spread = 0.02
face   = 100.0

floater_result = sensitivities(KeyRates(tenors), base, credit) do base_curve, credit_curve
    total = 0.0
    for t in 1:5
        df_base      = base_curve(Float64(t))
        df_credit    = credit_curve(Float64(t))
        df_base_prev = t == 1 ? 1.0 : base_curve(Float64(t - 1))

        # Coupon resets to risk-free forward rate + fixed credit spread
        fwd = df_base_prev / df_base - 1.0
        total += face * (fwd + spread) * df_base * df_credit

        # Principal at maturity
        t == 5 && (total += face * df_base * df_credit)
    end
    total
end

(IR01 = sum(floater_result.base_durations),
 CS01 = sum(floater_result.credit_durations))
```

Bumping base rates changes both the floating coupon amounts and the discount factors (partially canceling), while bumping credit rates only affects discounting. This asymmetry is why the IR01/CS01 decomposition matters for instruments with rate-dependent cashflows.

## Floating-Rate Instruments: Effective vs Spread Duration

The do-block above re-projects the floating coupons by hand. You can instead pass a `FinanceModels` contract — or a vector of contracts (a portfolio) — directly: the familiar markers select the risk and the verb selects the units. A floater has two durations and both matter:

- **Effective (rate) duration** — bump the curve, coupons re-fix → small (≈ time to next reset).
- **Spread (credit) duration** — bump the discount only, coupons fixed → ≈ maturity.

```@example sensitivities
using FinanceModels: Bond
floater = Bond.Floating(0.015, Periodic(1), 5.0, "SOFR")   # SOFR + 150bp, 5y annual

duration(Effective(), floater, zrc, tenors)   # rate duration, yrs — small
duration(Spread(),    floater, zrc, tenors)   # spread duration, yrs — ≈ maturity
dv01(Effective(),     floater, zrc, tenors)   # effective DV01, $/bp
```

`sensitivities` returns the whole picture (years, DV01s, and key-rate vectors) in one AD pass:

```@example sensitivities
s = sensitivities(floater, zrc, tenors)
(effective = s.effective_duration, spread = s.spread_duration, eff_dv01 = s.effective_dv01)
```

For a fixed bond `effective == spread ==` the modified duration. For an **in-force** floater whose current coupon is already fixed, [`locked_floater`](@ref) gives the conventional rate duration ≈ time to next reset:

```@example sensitivities
duration(Effective(), locked_floater(floater, 0.04, 1.0), zrc, tenors)   # ≈ 1y, not ≈ 5
```

### Portfolios

A vector of contracts is a target too — valued by summation in one AD pass (value-weighted):

```@example sensitivities
portfolio = [floater, Bond.Fixed(0.03, Periodic(1), 7.0)]
duration(portfolio, zrc, tenors)
```

### Multi-curve: risk-free + credit + ILP + index

Pass the discount as a stack of named layers plus the coupon-projection `index`; sensitivities come back per role, no `Dict` to assemble:

```@example sensitivities
rf     = zrc
credit = Yield.Constant(Continuous(0.01))
ilp    = Yield.Constant(Continuous(0.004))
r = sensitivities(floater, tenors; discount = (; rf, credit, ilp), index = zrc)
r.duration    # (; rf ≈ IR01, credit ≈ CS01, ilp = "ILP01", index = reset sensitivity)
```

ILP / matching-adjustment / basis are just additional named layers. For arbitrary valuations, the do-block form with [`reproject`](@ref) hides the model `Dict`:

```@example sensitivities
sensitivities((; rf, credit, ilp, index = zrc); tenors) do c
    present_value(c.rf + c.credit + c.ilp, reproject(floater, c.index))
end
```

And [`zspread`](@ref) solves the discount margin to a market price:

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

# Compute portfolio DV01 vector in a single AD pass
portfolio_dv01 = duration(DV01(), KeyRates(tenors), zrc) do curve
    pv(curve, bond1_cfs, bond1_times) + pv(curve, bond2_cfs, bond2_times)
end

# Equivalently (but two AD passes):
dv01_1 = duration(DV01(), KeyRates(tenors), zrc, bond1_cfs, bond1_times)
dv01_2 = duration(DV01(), KeyRates(tenors), zrc, bond2_cfs, bond2_times)

(portfolio_dv01, dv01_1 .+ dv01_2, portfolio_dv01 ≈ dv01_1 .+ dv01_2)
```

### Example: Portfolio of Floating Rate Bonds

Floating rate bonds have coupons that reset to the prevailing market rate, so their cashflows depend on the rate curve itself. The do-block captures this dependency through AD — differentiating through both the discount factors and the coupon amounts in a single pass:

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

Without the spread, a floater prices at par and has near-zero duration (coupons offset discount factor changes). The spread introduces duration because its fixed cashflows are rate-sensitive — similar to a portfolio of small fixed-rate annuities layered on top of the par-valued floaters.

## Stochastic Model Sensitivities

ForwardDiff's dual numbers propagate through the full Monte Carlo simulation pipeline in FinanceModels.jl, including the Euler-Maruyama path generation. This means you can compute exact sensitivities of expected present values under stochastic short-rate models — differentiating through thousands of simulated rate paths in a single AD pass.

### What is being differentiated?

The `sensitivities` function always differentiates with respect to the **zero rates in the `ZeroRateCurve`** — these are the market-observable inputs. When you wrap a stochastic model inside the do-block:

```julia
hw = ShortRate.HullWhite(0.1, 0.01, zrc)
sensitivities(KeyRates(tenors), hw, cfs, times; n_scenarios=500, rng=Xoshiro(42))
```

the chain of differentiation is:

1. ForwardDiff perturbs zero rate `rᵢ`
2. The perturbed `curve` changes the forward curve `f(0, t)`
3. Hull-White recalibrates `θ(t)` from the new forwards
4. All simulated paths shift (same random draws, different drift)
5. The expected PV changes → this change is the KRD at tenor `i`

The stochastic model parameters (`a`, `σ`) are **not** being differentiated — they are constants in this computation. The KRDs answer: *"if the market yield curve shifts, how does my model-valued portfolio respond?"* This is the relevant question for hedging with market instruments (bonds, swaps), which is the primary use case for key rate durations.

### Model parameter sensitivities (vega, mean-reversion sensitivity)

A separate question is: *"how does expected PV change if I change the model parameters `a` or `σ`?"* These are **model risk** sensitivities, useful for understanding calibration sensitivity and model uncertainty. They are conceptually different from curve KRDs:

| | Curve KRDs (`∂V/∂rᵢ`) | Model Greeks (`∂V/∂a`, `∂V/∂σ`) |
|---|---|---|
| **What moves** | Market zero rates | Model calibration parameters |
| **Use case** | Hedging with bonds/swaps | Model risk, calibration stability |
| **Hedgeable?** | Yes (with market instruments) | No (not directly tradeable) |

Model parameter sensitivities (`∂V/∂a`, `∂V/∂σ`) are **not currently supported** by the AD pathway. The `simulate` function in FinanceModels.jl uses `Float64` arrays internally for simulation paths, which prevents ForwardDiff dual numbers from propagating through the model parameters. Dual numbers flow through the *curve rates* (because `build_model` and `θ(t)` calibration handle generic numeric types), but `a` and `σ` must be plain `Float64`.

For model parameter sensitivities, use finite differences as a workaround:

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

!!! note
    Supporting AD through model parameters would require parameterizing the element type of internal simulation arrays on the model parameter types in FinanceModels.jl. This is a potential future enhancement.

### Hull-White: sensitivities w.r.t. the initial term structure

A Hull-White model calibrates its drift θ(t) to match an initial yield curve. When that curve is a `ZeroRateCurve`, you can compute how the Monte Carlo expected value responds to movements in the initial zero rates:

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

This involves nested AD: the outer ForwardDiff differentiates w.r.t. zero rates, while Hull-White's θ(t) calibration internally uses ForwardDiff to compute instantaneous forward rates from the curve. ForwardDiff's [tag system](https://github.com/JuliaDiff/ForwardDiff.jl/issues/83) disambiguates the two differentiation passes automatically.

### Comparison: deterministic vs model-based sensitivities

The deterministic `ZeroRateCurve` and Hull-White MC valuations produce the same total duration for fixed cashflows (a consequence of the [risk-neutral pricing theorem](https://en.wikipedia.org/wiki/Risk-neutral_measure)), but decompose it across tenors differently:

```@example sensitivities
# Deterministic: discount directly off the initial curve
det_result = sensitivities(KeyRates(mc_tenors), hw_curve, mc_cfs, mc_tenors)

# Model-based: average across simulated rate paths (computed above as hw_result)
(det_durations  = det_result.durations,
 hw_durations   = hw_result.durations,
 sum_det        = sum(det_result.durations),
 sum_hw         = sum(hw_result.durations))
```

**Why the totals match:** For fixed cashflows, E[V] = Σ cf_i × P(0, t_i) under any risk-neutral model ([Glasserman, 2003, Ch. 7](https://link.springer.com/book/10.1007/978-0-387-21617-1)), so a parallel shift of all zero rates produces the same ΔV regardless of whether we compute it by direct discounting or via Monte Carlo. This implies Σ KRD_det = Σ KRD_HW.

**Why the decomposition differs:** The two approaches construct discount factors through different mathematical pathways. `ZeroRateCurve` with linear interpolation gives `df(t) = exp(-r_interp(t) × t)`, where bumping rate_j only affects the interpolated rate near tenor j — producing localized KRDs. Hull-White constructs discount factors by integrating a calibrated short-rate ODE: bumping rate_j changes the forward curve f(0,t), which changes θ(t) = ∂f/∂t + a·f + σ²(1−e^{−2at})/2a everywhere, altering the short-rate path at all times via the mean-reversion dynamics ([Brigo & Mercurio, 2006, Ch. 3](https://link.springer.com/book/10.1007/978-3-540-34604-3)). This creates non-local sensitivity even in the σ→0 limit — it is the model's parametric structure, not stochastic volatility, that redistributes duration.

This phenomenon is well-established in derivatives pricing as "model-dependent Greeks": different models calibrated to the same curve produce identical prices but different sensitivities. The pathwise differentiation technique used here ([Giles & Glasserman, 2006](https://people.maths.ox.ac.uk/~gilesm/files/mc_greeks.pdf)) computes exact derivatives of the Monte Carlo estimate in a single forward pass, capturing the full chain of dependencies from initial curve through θ(t) calibration through path simulation to valuation.

!!! note
    The fixed `rng` seed ensures reproducibility: the same random draws are used for every AD perturbation, giving exact pathwise derivatives. Without a fixed seed, each call would use different paths, introducing MC noise into the gradient.

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

**MonotoneConvex** (`Spline.MonotoneConvex()`, default): Finance-aware interpolation ([Hagan & West, 2006](https://doi.org/10.1080/13504860600829233)). Guarantees positive continuous forward rates, best KRD locality among smooth methods, and fastest AD performance.

**PCHIP** (`Spline.PCHIP()`): Smooth forward curves (C1), local sensitivity, monotonicity-preserving. Good general-purpose alternative.

**Linear** (`Spline.Linear()`): Perfectly local KRDs (zero sensitivity outside adjacent intervals), but kinks in the forward curve at tenor points.

**Akima** (`Spline.Akima()`): Alternative to PCHIP with different behavior near inflection points. Slightly more non-local leakage than PCHIP.

**Cubic spline** (`Spline.Cubic()`): Smoothest (C2), but bumps have non-local effects. KRDs at distant tenors may be negative. Use when smoothness matters most.

See the [FinanceModels interpolation guide](https://docs.juliaactuary.org/FinanceModels/dev/interpolation/) for detailed benchmarks and tradeoff analysis. On a flat curve, all methods produce identical results.

## Validating AD vs Bump-and-Reprice

AD sensitivities can be cross-validated against traditional finite-difference (bump-and-reprice) results. The AD approach gives exact derivatives in a single pass, while FD has O(ε²) truncation error:

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

## Validating AD with TransformedYield

AD gives the instantaneous rate of change (the derivative), while `TransformedYield` lets you apply an actual finite shift and observe the PV change. Comparing the two is a useful sanity check — the AD-predicted change should closely match the actual change for small shifts:

```@example sensitivities
# AD: total DV01 across all tenors — already in dollar-per-1bp units
total_dv01 = sum(duration(DV01(), KeyRates(val_tenors), val_zrc, val_cfs, val_tenors))

# TransformedYield: actual PV change under a +1 bp parallel shift
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

The ratio is very close to 1.0, confirming AD and TransformedYield agree. The small deviation is due to convexity — DV01 is a first-order (linear) approximation, while the actual PV change includes higher-order effects.
