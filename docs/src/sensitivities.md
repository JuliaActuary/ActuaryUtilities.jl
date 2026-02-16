# Key Rate Sensitivities

Compute key rate durations, DV01s, and convexities via automatic differentiation using `ZeroRateCurve` from [FinanceModels.jl](https://github.com/JuliaActuary/FinanceModels.jl).

This approach uses ForwardDiff to differentiate through the curve construction, giving exact (machine-precision) sensitivities in a single pass. See the [autodiff ALM chapter](https://modernfinancialmodeling.com/autodiff_alm) for background.

## Basic Usage

Construct a `ZeroRateCurve` with continuously-compounded zero rates and tenor times, then pass it to `duration`, `convexity`, or `sensitivities`:

```julia
using ActuaryUtilities, FinanceModels

rates = [0.03, 0.03, 0.03, 0.03, 0.03]
tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
zrc = ZeroRateCurve(rates, tenors)

cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

# Scalar modified duration (default)
dur = duration(zrc, cfs, tenors)

# Scalar DV01
dv01 = duration(DV01(), zrc, cfs, tenors)

# Scalar convexity
conv = convexity(zrc, cfs, tenors)
```

To get the full key-rate decomposition (vectors/matrices), use `KeyRates()`:

```julia
# Key rate durations (modified): vector of -∂V/∂rᵢ / V
krds = duration(KeyRates(), zrc, cfs, tenors)

# Key rate DV01s: vector of -∂V/∂rᵢ / 10000
dv01s = duration(DV01(), KeyRates(), zrc, cfs, tenors)

# Key rate convexity matrix: ∂²V/∂rᵢ∂rⱼ / V
conv_matrix = convexity(KeyRates(), zrc, cfs, tenors)
```

For a complete set of key-rate results in a single AD pass, use `sensitivities`:

```julia
result = sensitivities(zrc, cfs, tenors)
# result.value       — present value
# result.durations   — key rate durations (modified) — vector
# result.convexities — cross-convexity matrix — matrix

# For DV01s instead of durations:
dv01_result = sensitivities(DV01(), zrc, cfs, tenors)
# dv01_result.value       — present value
# dv01_result.dv01s       — key rate DV01s — vector
# dv01_result.convexities — cross-convexity matrix — matrix
```

## Scalar vs Key-Rate Decomposition

By default, `duration` and `convexity` with a `ZeroRateCurve` return **scalars** — the total modified duration, DV01, or convexity. This is consistent with the yield-based API (`duration(0.03, cfs, times)`).

To obtain the per-tenor decomposition, pass `KeyRates()` as the first argument:

```julia
# Scalar (default) — same as sum of key-rate decomposition
duration(zrc, cfs, tenors)                          # scalar
duration(DV01(), zrc, cfs, tenors)                   # scalar
convexity(zrc, cfs, tenors)                          # scalar

# Key-rate decomposition
duration(KeyRates(), zrc, cfs, tenors)               # vector
duration(DV01(), KeyRates(), zrc, cfs, tenors)       # vector
convexity(KeyRates(), zrc, cfs, tenors)              # matrix
```

The scalar value equals the sum of the key-rate decomposition:

```julia
duration(zrc, cfs, tenors) ≈ sum(duration(KeyRates(), zrc, cfs, tenors))
```

For a flat curve, the scalar modified duration matches the yield-based API:

```julia
using ActuaryUtilities

cfs = [5.0, 5.0, 5.0, 5.0, 105.0]
tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
zrc = ZeroRateCurve(fill(0.03, 5), tenors)

duration(zrc, cfs, tenors)                            # ≈ 4.57
duration(0.03, cfs, tenors)                            # ≈ 4.57 (same)
```

For Macaulay duration, use the scalar yield API directly — there is no `ZeroRateCurve` dispatch:

```julia
duration(Macaulay(), 0.03, cfs, tenors)
```

## Interest-Sensitive Instruments

For instruments whose cashflows depend on the rate environment (callable bonds, floaters, etc.), use the do-block syntax to pass a custom valuation function:

```julia
# Callable bond: key rate durations (vector)
callable_krds = duration(KeyRates(), zrc) do curve
    ncv = pv(curve, cfs, tenors)
    called_value = pv(curve, cfs[1:3], tenors[1:3]) + 102.0 * curve(3.0)
    min(ncv, called_value)
end

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

```julia
base = ZeroRateCurve([0.03, 0.03, 0.03, 0.03, 0.03], tenors)
credit = ZeroRateCurve([0.02, 0.02, 0.02, 0.02, 0.02], tenors)

# Scalar IR01 and CS01
ir01 = duration(IR01(), base, credit, cfs, tenors)
cs01 = duration(CS01(), base, credit, cfs, tenors)

# Key-rate decomposition (vectors)
ir01s = duration(IR01(), KeyRates(), base, credit, cfs, tenors)
cs01s = duration(CS01(), KeyRates(), base, credit, cfs, tenors)

# Two-curve convexity — scalars by default
conv = convexity(base, credit, cfs, tenors)
# conv.base, conv.credit, conv.cross (all scalars)

# Key-rate decomposition (matrices)
conv_kr = convexity(KeyRates(), base, credit, cfs, tenors)
# conv_kr.base, conv_kr.credit, conv_kr.cross (all matrices)

# Full two-curve sensitivities (always key-rate decomposition)
result = sensitivities(base, credit, cfs, tenors)
```

The default two-curve valuation uses multiplicative discount factors: `V = Σ cf × base(t) × credit(t)`, which corresponds to additive rates.

### Example: Credit-Risky Floating Rate Bond

For fixed cashflows, IR01 and CS01 are identical because base and credit rates enter additively. A **credit-risky floating rate bond** breaks this symmetry — its coupons reset to the risk-free forward rate plus a fixed credit spread, so bumping base rates changes both coupon amounts and discount factors (partially canceling), while bumping credit rates only affects discounting:

```julia
base = ZeroRateCurve([0.03, 0.03, 0.03, 0.03, 0.03], tenors)
credit = ZeroRateCurve([0.02, 0.02, 0.02, 0.02, 0.02], tenors)

# Floating rate bond: coupon = risk-free forward + 200bp credit spread
# Discounted at the combined base + credit rate
spread = 0.02
face = 100.0

result = sensitivities(base, credit) do base_curve, credit_curve
    total = 0.0
    for t in 1:5
        df_base = base_curve(Float64(t))
        df_credit = credit_curve(Float64(t))
        df_base_prev = t == 1 ? 1.0 : base_curve(Float64(t - 1))

        # Coupon resets to risk-free forward rate + fixed credit spread
        fwd = df_base_prev / df_base - 1.0
        total += face * (fwd + spread) * df_base * df_credit

        # Principal at maturity
        t == 5 && (total += face * df_base * df_credit)
    end
    total
end

sum(result.base_durations)    # IR01 — small, coupon reset offsets base rate sensitivity
sum(result.credit_durations)  # CS01 — larger, credit spread only affects discounting
```

Bumping base rates changes both the floating coupon amounts and the discount factors (partially canceling), while bumping credit rates only affects discounting. This asymmetry is why the IR01/CS01 decomposition matters for instruments with rate-dependent cashflows.

## Portfolio Sensitivity

DV01s are additive across positions, so a portfolio's DV01 vector equals the sum of individual DV01s:

```julia
zrc = ZeroRateCurve(rates, tenors)

# Compute portfolio DV01 vector in a single AD pass
# bond1_cfs, bond2_cfs are Vector{Cashflow} (from FinanceCore)
portfolio_dv01 = duration(DV01(), KeyRates(), zrc) do curve
    pv(curve, bond1_cfs) + pv(curve, bond2_cfs)
end

# Equivalently (but two AD passes):
dv01_1 = duration(DV01(), KeyRates(), zrc, bond1_cfs, bond1_times)
dv01_2 = duration(DV01(), KeyRates(), zrc, bond2_cfs, bond2_times)
portfolio_dv01 ≈ dv01_1 .+ dv01_2
```

### Example: Portfolio of Floating Rate Bonds

Floating rate bonds have coupons that reset to the prevailing market rate, so their cashflows depend on the rate curve itself. The do-block captures this dependency through AD — differentiating through both the discount factors and the coupon amounts in a single pass:

```julia
using ActuaryUtilities, FinanceModels

rates = [0.02, 0.025, 0.03, 0.035, 0.04, 0.042, 0.044, 0.046, 0.048, 0.05]
tenors = collect(1.0:10.0)
zrc = ZeroRateCurve(rates, tenors)

# 10 floating rate bonds: maturities 1yr to 10yr, face 100 each,
# annual coupons = 1yr forward rate + 50bp credit spread
notionals = fill(100.0, 10)
maturities = 1:10
spread = 0.005

# The do-block receives the curve and returns the total present value
# of all cashflows across the portfolio.
# The curve is constructed once per AD evaluation — valuing all 10 bonds
# inside a single do-block avoids rebuilding the curve for each bond.
result = sensitivities(zrc) do curve
    total = 0.0
    for (notional, mat) in zip(notionals, maturities)
        # For each bond, loop over annual payment dates t = 1, 2, ..., maturity
        for t in 1:mat
            # Discount factors: df = P(0,t), df_prev = P(0,t-1)
            df = curve(Float64(t))
            df_prev = t == 1 ? 1.0 : curve(Float64(t - 1))

            # 1yr simple forward rate from t-1 to t: F = P(0,t-1)/P(0,t) - 1
            fwd = df_prev / df - 1.0

            # Floating coupon PV: notional × (forward rate + spread) × P(0,t)
            total += notional * (fwd + spread) * df

            # Return principal at maturity
            t == mat && (total += notional * df)
        end
    end
    total
end

result.value       # portfolio present value (≈ 10 × 100 + spread premium)
result.durations   # key rate durations — small, since floaters reset
```

Without the spread, a floater prices at par and has near-zero duration (coupons offset discount factor changes). The spread introduces duration because its fixed cashflows are rate-sensitive — similar to a portfolio of small fixed-rate annuities layered on top of the par-valued floaters.

## Stochastic Model Sensitivities

ForwardDiff's dual numbers propagate through the full Monte Carlo simulation pipeline in FinanceModels.jl, including the Euler-Maruyama path generation. This means you can compute exact sensitivities of expected present values under stochastic short-rate models — differentiating through thousands of simulated rate paths in a single AD pass.

### What is being differentiated?

The `sensitivities` function always differentiates with respect to the **zero rates in the `ZeroRateCurve`** — these are the market-observable inputs. When you wrap a stochastic model inside the do-block:

```julia
hw = ShortRate.HullWhite(0.1, 0.01, zrc)
sensitivities(hw, cfs, times; n_scenarios=500, rng=Xoshiro(42))
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

```julia
using FinanceModels: ShortRate, simulate
using FinanceCore: discount
using Random: Xoshiro

rates = [0.03, 0.03, 0.03, 0.03, 0.03]
tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

function mc_value(a, σ)
    curve = ZeroRateCurve(rates, tenors)
    hw = ShortRate.HullWhite(a, σ, curve)
    scenarios = simulate(hw; n_scenarios=1000, timestep=1/12, horizon=6.0, rng=Xoshiro(42))
    sum(pv(sc, cfs, tenors) for sc in scenarios) / 1000
end

# Finite-difference sensitivities
ε = 1e-5
dV_da = (mc_value(0.1 + ε, 0.01) - mc_value(0.1 - ε, 0.01)) / (2ε)   # mean reversion
dV_dσ = (mc_value(0.1, 0.01 + ε) - mc_value(0.1, 0.01 - ε)) / (2ε)   # volatility (vega)
```

!!! note
    Supporting AD through model parameters would require parameterizing the element type of internal simulation arrays on the model parameter types in FinanceModels.jl. This is a potential future enhancement.

### Hull-White: sensitivities w.r.t. the initial term structure

A Hull-White model calibrates its drift θ(t) to match an initial yield curve. When that curve is a `ZeroRateCurve`, you can compute how the Monte Carlo expected value responds to movements in the initial zero rates:

```julia
using ActuaryUtilities, FinanceModels
using FinanceModels: ShortRate, simulate
using FinanceCore: discount
using Random

rates = [0.03, 0.03, 0.03, 0.03, 0.03]
tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
zrc = ZeroRateCurve(rates, tenors)

cfs = [5.0, 5.0, 5.0, 5.0, 105.0]
times = [1.0, 2.0, 3.0, 4.0, 5.0]

# Key rate sensitivities of E[V] under Hull-White dynamics
hw = ShortRate.HullWhite(0.1, 0.01, zrc)
hw_result = sensitivities(hw, cfs, times; n_scenarios=500, timestep=1/12, horizon=6.0, rng=Xoshiro(42))

hw_result.durations   # key rate durations under stochastic dynamics
hw_result.convexities # cross-convexity matrix
```

This involves nested AD: the outer ForwardDiff differentiates w.r.t. zero rates, while Hull-White's θ(t) calibration internally uses ForwardDiff to compute instantaneous forward rates from the curve. ForwardDiff's [tag system](https://github.com/JuliaDiff/ForwardDiff.jl/issues/83) disambiguates the two differentiation passes automatically.

### Comparison: deterministic vs model-based sensitivities

The deterministic `ZeroRateCurve` and Hull-White MC valuations produce the same total duration for fixed cashflows (a consequence of the [risk-neutral pricing theorem](https://en.wikipedia.org/wiki/Risk-neutral_measure)), but decompose it across tenors differently:

```julia
# Deterministic: discount directly off the initial curve
det_result = sensitivities(zrc, cfs, tenors)

# Model-based: average across simulated rate paths
hw = ShortRate.HullWhite(0.1, 0.01, zrc)
hw_result = sensitivities(hw, cfs, times; n_scenarios=1000, timestep=1/12, horizon=6.0, rng=Xoshiro(42))

det_result.durations  # [0.04, 0.09, 0.13, 0.16, 4.15]  (localized at each tenor)
hw_result.durations   # [-1.01, 1.04, 1.70, 1.85, 0.99]  (redistributed across tenors)

sum(det_result.durations) # 4.57  — total modified duration (= duration(zrc, cfs, tenors))
sum(hw_result.durations)  # 4.57  — same total (risk-neutral guarantee)
```

**Why the totals match:** For fixed cashflows, E[V] = Σ cf_i × P(0, t_i) under any risk-neutral model ([Glasserman, 2003, Ch. 7](https://link.springer.com/book/10.1007/978-0-387-21617-1)), so a parallel shift of all zero rates produces the same ΔV regardless of whether we compute it by direct discounting or via Monte Carlo. This implies Σ KRD_det = Σ KRD_HW.

**Why the decomposition differs:** The two approaches construct discount factors through different mathematical pathways. `ZeroRateCurve` with linear interpolation gives `df(t) = exp(-r_interp(t) × t)`, where bumping rate_j only affects the interpolated rate near tenor j — producing localized KRDs. Hull-White constructs discount factors by integrating a calibrated short-rate ODE: bumping rate_j changes the forward curve f(0,t), which changes θ(t) = ∂f/∂t + a·f + σ²(1−e^{−2at})/2a everywhere, altering the short-rate path at all times via the mean-reversion dynamics ([Brigo & Mercurio, 2006, Ch. 3](https://link.springer.com/book/10.1007/978-3-540-34604-3)). This creates non-local sensitivity even in the σ→0 limit — it is the model's parametric structure, not stochastic volatility, that redistributes duration.

This phenomenon is well-established in derivatives pricing as "model-dependent Greeks": different models calibrated to the same curve produce identical prices but different sensitivities. The pathwise differentiation technique used here ([Giles & Glasserman, 2006](https://people.maths.ox.ac.uk/~gilesm/files/mc_greeks.pdf)) computes exact derivatives of the Monte Carlo estimate in a single forward pass, capturing the full chain of dependencies from initial curve through θ(t) calibration through path simulation to valuation.

!!! note
    The fixed `rng` seed ensures reproducibility: the same random draws are used for every AD perturbation, giving exact pathwise derivatives. Without a fixed seed, each call would use different paths, introducing MC noise into the gradient.

## Choosing Interpolation

`ZeroRateCurve` accepts an optional third argument for the interpolation method:

```julia
zrc = ZeroRateCurve(rates, tenors)                              # default: MonotoneConvex
zrc_pchip = ZeroRateCurve(rates, tenors, Spline.PCHIP())        # PCHIP
zrc_lin = ZeroRateCurve(rates, tenors, Spline.Linear())          # linear
zrc_cub = ZeroRateCurve(rates, tenors, Spline.Cubic())           # cubic spline
zrc_aki = ZeroRateCurve(rates, tenors, Spline.Akima())           # Akima
```

**MonotoneConvex** (`Spline.MonotoneConvex()`, default): Finance-aware interpolation ([Hagan & West, 2006](https://doi.org/10.1080/13504860600829233)). Guarantees positive continuous forward rates, best KRD locality among smooth methods, and fastest AD performance.

**PCHIP** (`Spline.PCHIP()`): Smooth forward curves (C1), local sensitivity, monotonicity-preserving. Good general-purpose alternative.

**Linear** (`Spline.Linear()`): Perfectly local KRDs (zero sensitivity outside adjacent intervals), but kinks in the forward curve at tenor points.

**Akima** (`Spline.Akima()`): Alternative to PCHIP with different behavior near inflection points. Slightly more non-local leakage than PCHIP.

**Cubic spline** (`Spline.Cubic()`): Smoothest (C2), but bumps have non-local effects. KRDs at distant tenors may be negative. Use when smoothness matters most.

See the [FinanceModels interpolation guide](https://docs.juliaactuary.org/FinanceModels/dev/interpolation/) for detailed benchmarks and tradeoff analysis. On a flat curve, all methods produce identical results.

## Validating AD vs Bump-and-Reprice

AD sensitivities can be cross-validated against traditional finite-difference (bump-and-reprice) results. The AD approach gives exact derivatives in a single pass, while FD has O(ε²) truncation error:

```julia
using ActuaryUtilities, FinanceModels, Test

rates = [0.02, 0.03, 0.04, 0.05]
tenors = [1.0, 3.0, 5.0, 10.0]
zrc = ZeroRateCurve(rates, tenors)
cfs = [3.0, 3.0, 3.0, 103.0]

# AD (exact) — use KeyRates() for the per-tenor vector
ad_dv01 = duration(DV01(), KeyRates(), zrc, cfs, tenors)

# Finite difference (bump-and-reprice)
ε = 1e-5
for i in 1:4
    rates_up = copy(rates); rates_up[i] += ε
    rates_dn = copy(rates); rates_dn[i] -= ε
    v_up = pv(ZeroRateCurve(rates_up, tenors), cfs, tenors)
    v_dn = pv(ZeroRateCurve(rates_dn, tenors), cfs, tenors)
    fd_dv01 = -(v_up - v_dn) / (2ε) / 10_000
    @test ad_dv01[i] ≈ fd_dv01 atol = 1e-4
end
```
