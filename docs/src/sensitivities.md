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

# Key rate durations (modified): vector of -∂V/∂rᵢ / V
krds = duration(zrc, cfs, tenors)

# Key rate DV01s: vector of -∂V/∂rᵢ / 10000
dv01s = duration(DV01(), zrc, cfs, tenors)

# Key rate convexity matrix: ∂²V/∂rᵢ∂rⱼ / V
conv = convexity(zrc, cfs, tenors)
```

For a complete set of results in a single AD pass, use `sensitivities`:

```julia
result = sensitivities(zrc, cfs, tenors)
# result.value       — present value
# result.durations   — key rate durations (modified)
# result.dv01s       — key rate DV01s
# result.convexities — cross-convexity matrix
```

## Interest-Sensitive Instruments

For instruments whose cashflows depend on the rate environment (callable bonds, floaters, etc.), use the do-block syntax to pass a custom valuation function:

```julia
# Callable bond: issuer calls at par + 2 after year 3
callable_dur = duration(zrc) do curve
    ncv = sum(cf * curve(t) for (cf, t) in zip(cfs, tenors))
    called_value = sum(cf * curve(t) for (cf, t) in zip(cfs[1:3], tenors[1:3])) +
                   102.0 * curve(3.0)
    min(ncv, called_value)
end
```

The function receives a curve object and must return a scalar value. ForwardDiff differentiates through the entire valuation, capturing any rate-dependent optionality.

## Two-Curve Decomposition

Decompose sensitivities into base (risk-free) and credit spread components using `IR01` and `CS01`:

```julia
base = ZeroRateCurve([0.03, 0.03, 0.03, 0.03, 0.03], tenors)
credit = ZeroRateCurve([0.02, 0.02, 0.02, 0.02, 0.02], tenors)

# Base curve DV01s
ir01s = duration(IR01(), base, credit, cfs, tenors)

# Credit spread DV01s
cs01s = duration(CS01(), base, credit, cfs, tenors)

# Two-curve convexity (base, credit, and cross matrices)
conv = convexity(base, credit, cfs, tenors)
# conv.base, conv.credit, conv.cross

# Full two-curve sensitivities
result = sensitivities(base, credit, cfs, tenors)
```

The default two-curve valuation uses multiplicative discount factors: `V = Σ cf × base(t) × credit(t)`, which corresponds to additive rates.

## Portfolio Sensitivity

DV01s are additive across positions, so a portfolio's DV01 vector equals the sum of individual DV01s:

```julia
zrc = ZeroRateCurve(rates, tenors)

# Compute portfolio DV01 in a single AD pass
portfolio_dv01 = duration(DV01(), zrc) do curve
    sum(cf * curve(t) for (cf, t) in zip(bond1_cfs, bond1_times)) +
    sum(cf * curve(t) for (cf, t) in zip(bond2_cfs, bond2_times))
end

# Equivalently (but two AD passes):
dv01_1 = duration(DV01(), zrc, bond1_cfs, bond1_times)
dv01_2 = duration(DV01(), zrc, bond2_cfs, bond2_times)
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

result = sensitivities(zrc) do curve
    total = 0.0
    for (notional, mat) in zip(notionals, maturities)
        for t in 1:mat
            df = curve(Float64(t))
            df_prev = t == 1 ? 1.0 : curve(Float64(t - 1))
            fwd = df_prev / df - 1.0          # 1yr simple forward rate
            total += notional * (fwd + spread) * df  # PV of floating coupon
            t == mat && (total += notional * df)      # principal at maturity
        end
    end
    total
end

f(zrc,notionals,maturities) = sensitivities(zrc) do curve
    total = 0.0
    for (notional, mat) in zip(notionals, maturities)
        for t in 1:mat
            df = curve(Float64(t))
            df_prev = t == 1 ? 1.0 : curve(Float64(t - 1))
            fwd = df_prev / df - 1.0          # 1yr simple forward rate
            total += notional * (fwd + spread) * df  # PV of floating coupon
            t == mat && (total += notional * df)      # principal at maturity
        end
    end
    total
end

@benchmark f($zrc,$notionals,$maturities)

result.value       # portfolio present value (≈ 10 × 100 + spread premium)
result.durations   # key rate durations — small, since floaters reset
result.dv01s       # key rate DV01s
```

Without the spread, a floater prices at par and has near-zero duration (coupons offset discount factor changes). The spread introduces duration because its fixed cashflows are rate-sensitive — similar to a portfolio of small fixed-rate annuities layered on top of the par-valued floaters.

## Stochastic Model Sensitivities

ForwardDiff's dual numbers propagate through the full Monte Carlo simulation pipeline in FinanceModels.jl, including the Euler-Maruyama path generation. This means you can compute exact sensitivities of expected present values under stochastic short-rate models — differentiating through thousands of simulated rate paths in a single AD pass.

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
hw_result = sensitivities(zrc) do curve
    hw = ShortRate.HullWhite(0.1, 0.01, curve)
    scenarios = simulate(hw; n_scenarios=500, timestep=1/12, horizon=6.0, rng=Xoshiro(42))
    sum(sum(cf * discount(sc, t) for (cf, t) in zip(cfs, times)) for sc in scenarios) / 500
end

hw_result.durations   # key rate durations under stochastic dynamics
hw_result.dv01s       # key rate DV01s
hw_result.convexities # cross-convexity matrix
```

This involves nested AD: the outer ForwardDiff differentiates w.r.t. zero rates, while Hull-White's θ(t) calibration internally uses ForwardDiff to compute instantaneous forward rates from the curve. ForwardDiff's [tag system](https://github.com/JuliaDiff/ForwardDiff.jl/issues/83) disambiguates the two differentiation passes automatically.

### Comparison: deterministic vs model-based sensitivities

The deterministic `ZeroRateCurve` and Hull-White MC valuations produce the same total duration for fixed cashflows (a consequence of the [risk-neutral pricing theorem](https://en.wikipedia.org/wiki/Risk-neutral_measure)), but decompose it across tenors differently:

```julia
# Deterministic: discount directly off the initial curve
det_result = sensitivities(zrc, cfs, tenors)

# Model-based: average across simulated rate paths
hw_result = sensitivities(zrc) do curve
    hw = ShortRate.HullWhite(0.1, 0.01, curve)
    scenarios = simulate(hw; n_scenarios=1000, timestep=1/12, horizon=6.0, rng=Xoshiro(42))
    sum(sum(cf * discount(sc, t) for (cf, t) in zip(cfs, times)) for sc in scenarios) / 1000
end

det_result.durations  # [0.04, 0.09, 0.13, 0.16, 4.15]  (localized at each tenor)
hw_result.durations   # [-1.01, 1.04, 1.70, 1.85, 0.99]  (redistributed across tenors)

sum(det_result.durations) # 4.57  — total modified duration
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
zrc_linear = ZeroRateCurve(rates, tenors)                    # default: linear
zrc_linear = ZeroRateCurve(rates, tenors, Spline.Linear())   # explicit linear
zrc_cubic  = ZeroRateCurve(rates, tenors, Spline.Cubic())    # cubic spline
```

**Linear interpolation** (`Spline.Linear()`): Key rate bumps have local effect — bumping one tenor only affects adjacent intervals. This produces intuitive, well-localized KRDs.

**Cubic spline** (`Spline.Cubic()`): Smoother curve, but bumps have non-local effects due to the global nature of cubic splines. KRDs at distant tenors may be slightly negative. Use cubic when curve smoothness matters more than KRD locality.

On a flat curve, both methods produce identical results.
