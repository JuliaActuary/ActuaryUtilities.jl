module FinancialMath

import ..FinanceCore
import ..FinanceModels
import ..ForwardDiff
import ..ActuaryUtilities: duration
import ..Optimization
import ..OptimizationOptimJL
import Random

export irr, internal_rate_of_return, spread,
    pv, present_value, price, present_values,
    breakeven, moic,
    Macaulay, Modified, DV01, IR01, CS01, KeyRates, KeyRatePar, KeyRateZero, KeyRate, duration, convexity,
    sensitivities

"""
    present_values(interest, cashflows, timepoints)

Efficiently calculate a vector representing the present value of the given cashflows at each period prior to the given timepoint.

# Examples
```julia-repl
julia> present_values(0.00, [1,1,1])
[3,2,1]

julia> present_values(ForwardYield([0.1,0.2]), [10,20],[0,1]) # after `using FinanceModels`
2-element Vector{Float64}:
 28.18181818181818
 18.18181818181818
```

"""
function present_values(interest, cashflows, times = eachindex(cashflows))
    return present_values_accumulator(interest, cashflows, times)
end

function present_values_accumulator(interest, cashflows, times, pvs = [0.0])
    from_time = length(times) == 1 ? 0.0 : times[end - 1]
    pv = FinanceCore.discount(interest, from_time, last(times)) * (first(pvs) + last(cashflows))
    pvs = pushfirst!(pvs, pv)

    if length(cashflows) > 1

        new_cfs = @view cashflows[1:(end - 1)]
        new_times = @view times[1:(end - 1)]
        return present_values_accumulator(interest, new_cfs, new_times, pvs)
    else
        # last discount and return
        return pvs[1:(end - 1)] # end-1 get rid of trailing 0.0
    end
end


"""
    price(...)

The absolute value of the `present_value(...)`. 

# Extended help

Using `price` can be helpful if the directionality of the value doesn't matter. For example, in the common usage, duration is more interested in the change in price than present value, so `price` is used there.
"""
price(x1, x2) = FinanceCore.present_value(x1, x2) |> abs
price(x1, x2, x3) = FinanceCore.present_value(x1, x2, x3) |> abs

"""
    breakeven(yield, cashflows::Vector)
    breakeven(yield, cashflows::Vector,times::Vector)

Calculate the time when the accumulated cashflows breakeven given the yield.

Assumptions:

- cashflows occur at the end of the period
- cashflows evenly spaced with the first one occuring at time zero if `times` not given

Returns `nothing` if cashflow stream never breaks even.

```julia
julia> breakeven(0.10, [-10,1,2,3,4,8])
5

julia> breakeven(0.10, [-10,15,2,3,4,8])
1

julia> breakeven(0.10, [-10,-15,2,3,4,8]) # returns the `nothing` value


```
"""
function breakeven(y, cashflows, timepoints = (eachindex(cashflows) .- 1))
    accum = 0.0
    last_neg = nothing

    # `amount` and `timepoint` allow to generically handle `Cashflow`s and amount/time vectors
    accum += FinanceCore.amount(cashflows[1])
    if accum >= 0 && isnothing(last_neg)
        last_neg = FinanceCore.timepoint(cashflows[1], timepoints[1])
    end

    for i in 2:length(cashflows)
        # accumulate the flow from each timepoint to the next
        a = FinanceCore.timepoint(cashflows[i - 1], timepoints[i - 1])
        b = FinanceCore.timepoint(cashflows[i], timepoints[i])
        accum *= FinanceCore.accumulation(y, a, b)
        accum += FinanceCore.amount(cashflows[i])

        if accum >= 0 && isnothing(last_neg)
            last_neg = b
        elseif accum < 0
            last_neg = nothing
        end
    end

    return last_neg

end


abstract type Duration end

struct Macaulay <: Duration end
struct Modified <: Duration end
"""
    DV01 <: Duration

Dollar Value of 01. The dollar change in value for a 1 basis point (0.01%) parallel shift in rates.

`DV01 = -∂V/∂r / 10000`, so a DV01 of 0.045 means the position loses \$0.045 per \$100 notional for a 1bp rate increase.

See also: [`IR01`](@ref), [`CS01`](@ref)
"""
struct DV01 <: Duration end

"""
    IR01 <: Duration

Interest Rate 01. The dollar change in value for a 1 basis point parallel shift in the risk-free (base) curve, holding the credit spread constant.

Requires both a base curve and credit spread to be specified. For a flat additive decomposition, `IR01 ≈ CS01 ≈ DV01`.

See also: [`CS01`](@ref), [`DV01`](@ref)
"""
struct IR01 <: Duration end

"""
    CS01 <: Duration

Credit Spread 01. The dollar change in value for a 1 basis point parallel shift in the credit spread, holding the risk-free (base) curve constant.

Requires both a base curve and credit spread to be specified. For a flat additive decomposition, `CS01 ≈ IR01 ≈ DV01`.

See also: [`IR01`](@ref), [`DV01`](@ref)
"""
struct CS01 <: Duration end

"""
    KeyRates <: Duration

Dispatch type that requests the full key-rate decomposition (vector of durations or matrix
of convexities) instead of the default scalar summary.

Use with `duration` and `convexity` when a `ZeroRateCurve` is the rate input:

```julia
duration(KeyRates(), zrc, cfs, times)            # vector of key rate durations
duration(DV01(), KeyRates(), zrc, cfs, times)     # vector of key rate DV01s
convexity(KeyRates(), zrc, cfs, times)            # matrix of key rate convexities
```

Without `KeyRates()`, these functions return a scalar (the sum of the decomposition).

See also: [`DV01`](@ref), [`IR01`](@ref), [`CS01`](@ref)
"""
struct KeyRates <: Duration end

abstract type KeyRateDuration <: Duration end


"""
    KeyRatePar(timepoint,shift=0.001) <: KeyRateDuration

Shift the par curve by the given amount at the given timepoint. Use in conjunction with `duration` to calculate the key rate duration. 

Unlike other duration statistics which are computed using analytic derivatives, `KeyRateDuration`s are computed via a shift-and-compute the yield curve approach.

`KeyRatePar` is more commonly reported (than [`KeyRateZero`](@ref)) in the fixed income markets, even though the latter has more analytically attractive properties. See the discussion of KeyRateDuration in the FinanceModels.jl docs.

"""
struct KeyRatePar{T, R} <: KeyRateDuration
    timepoint::T
    shift::R
    KeyRatePar(timepoint, shift = 0.001) = new{typeof(timepoint), typeof(shift)}(timepoint, shift)
end

"""
    KeyRateZero(timepoint,shift=0.001) <: KeyRateDuration

Shift the par curve by the given amount at the given timepoint. Use in conjunction with `duration` to calculate the key rate duration.

Unlike other duration statistics which are computed using analytic derivatives, `KeyRateDuration` is computed via a shift-and-compute the yield curve approach.

`KeyRateZero` is less commonly reported (than [`KeyRatePar`](@ref)) in the fixed income markets, even though the latter has more analytically attractive properties. See the discussion of KeyRateDuration in the FinanceModels.jl docs.
"""
struct KeyRateZero{T, R} <: KeyRateDuration
    timepoint::T
    shift::R
    KeyRateZero(timepoint, shift = 0.001) = new{typeof(timepoint), typeof(shift)}(timepoint, shift)
end

"""
    KeyRate(timepoints,shift=0.001)

A convenience constructor for [`KeyRateZero`](@ref). 

## Extended Help
[`KeyRateZero`](@ref) is chosen as the default constructor because it has more attractive properties than [`KeyRatePar`](@ref):

- rates after the key `timepoint` remain unaffected by the `shift`
  - e.g. this causes a 6-year zero coupon bond would have a negative duration if the 5-year par rate was used


"""
KeyRate = KeyRateZero

"""
    duration(Macaulay(),interest_rate,cfs,times)
    duration(Modified(),interest_rate,cfs,times)
    duration(DV01(),interest_rate,cfs,times)
    duration(IR01(),base_curve,credit_spread,cfs,times)
    duration(CS01(),base_curve,credit_spread,cfs,times)
    duration(interest_rate,cfs,times)             # Modified Duration
    duration(interest_rate,valuation_function)    # Modified Duration

Calculates the Macaulay, Modified, DV01, IR01, or CS01 duration. `times` may be ommitted and the valuation will assume evenly spaced cashflows starting at the end of the first period.

`cfs` can be an `AbstractVector{<:Cashflow}` (from FinanceCore), in which case `times` is extracted automatically and should be omitted.

Note that the calculated duration will depend on the periodicity convention of the `interest_rate`: a `Periodic` yield (or yield model with that convention) will be a slightly different computed duration than a `Continous` which follows from the present value differing according to the periodicity.

When not given `Modified()` or `Macaulay()` as an argument, will default to `Modified()`.

- Modified duration: the relative change per point of yield change.
- Macaulay: the cashflow-weighted average time.
- DV01: the absolute change per basis point (hundredth of a percentage point).
- IR01: the absolute change per basis point shift in the risk-free (base) curve, holding credit spread constant.
- CS01: the absolute change per basis point shift in the credit spread, holding the risk-free (base) curve constant.

# Examples

Using vectors of cashflows and times
```julia-repl
julia> times = 1:5;

julia> cfs = [0,0,0,0,100];

julia> duration(0.03,cfs,times)
4.854368932038835

julia> duration(Periodic(0.03,1),cfs,times)
4.854368932038835

julia> duration(Continuous(0.03),cfs,times)
5.0

julia> duration(Macaulay(),0.03,cfs,times)
5.0

julia> duration(Modified(),0.03,cfs,times)
4.854368932038835

julia> convexity(0.03,cfs,times)
28.277877274012614

```

Using any given value function: 

```julia-repl
julia> lump_sum_value(amount,years,i) = amount / (1 + i ) ^ years
julia> my_lump_sum_value(i) = lump_sum_value(100,5,i)
julia> duration(0.03,my_lump_sum_value)
4.854368932038835
julia> convexity(0.03,my_lump_sum_value)
28.277877274012617

```
"""
function duration(::Macaulay, yield, cfs, times)
    return sum(FinanceCore.timepoint.(cfs, times) .* price.(yield, cfs, times) / price(yield, cfs, times))
end

function duration(::Modified, yield, cfs, times)
    D(i) = price(i, cfs, times)
    return duration(yield, D)
end

function duration(yield, valuation_function::T) where {T <: Function}
    D(i) = log(valuation_function(i + yield))
    return δV = -ForwardDiff.derivative(D, 0.0)
end

function duration(yield, cfs, times)
    return duration(Modified(), yield, vec(cfs), times)
end

# timepoints are used to make the function more generic
# with respect to allowing Cashflow objects
function duration(yield, cfs)
    times = FinanceCore.timepoint.(cfs, 1:length(cfs))
    return duration(Modified(), yield, cfs, times)
end

function duration(::DV01, yield, cfs, times)
    return duration(DV01(), yield, i -> price(i, vec(cfs), times))
end
function duration(d::Duration, yield, cfs)
    times = FinanceCore.timepoint.(cfs, 1:length(cfs))
    return duration(d, yield, vec(cfs), times)
end

function duration(::DV01, yield, valuation_function::Y) where {Y <: Function}
    return duration(yield, valuation_function) * valuation_function(yield) / 10000
end

"""
    duration(IR01(), base_curve, credit_spread, cfs, times)
    duration(IR01(), base_curve, credit_spread, cfs)

Calculate the IR01 (Interest Rate 01): the dollar change in value for a 1 basis point parallel shift in the risk-free (base) curve, holding the credit spread constant.

The total discount rate is assumed to be `base_curve + credit_spread`. For a flat additive decomposition (e.g. scalar rates), `IR01 ≈ CS01 ≈ DV01`.

# Examples

```julia-repl
julia> cfs = [5, 5, 5, 105];

julia> times = 1:4;

julia> duration(IR01(), 0.03, 0.02, cfs, times)
0.03465054893498076

julia> duration(IR01(), 0.03, 0.02, cfs, times) ≈ duration(DV01(), 0.05, cfs, times)
true
```
"""
function duration(::IR01, base_curve, credit_spread, cfs, times)
    return duration(DV01(), base_curve, i -> price(i + credit_spread, vec(cfs), times))
end

function duration(::IR01, base_curve, credit_spread, cfs)
    times = FinanceCore.timepoint.(cfs, 1:length(cfs))
    return duration(IR01(), base_curve, credit_spread, vec(cfs), times)
end

"""
    duration(CS01(), base_curve, credit_spread, cfs, times)
    duration(CS01(), base_curve, credit_spread, cfs)

Calculate the CS01 (Credit Spread 01): the dollar change in value for a 1 basis point parallel shift in the credit spread, holding the risk-free (base) curve constant.

The total discount rate is assumed to be `base_curve + credit_spread`. For a flat additive decomposition (e.g. scalar rates), `CS01 ≈ IR01 ≈ DV01`.

# Examples

```julia-repl
julia> cfs = [5, 5, 5, 105];

julia> times = 1:4;

julia> duration(CS01(), 0.03, 0.02, cfs, times)
0.03465054893498076

julia> duration(CS01(), 0.03, 0.02, cfs, times) ≈ duration(DV01(), 0.05, cfs, times)
true
```
"""
function duration(::CS01, base_curve, credit_spread, cfs, times)
    return duration(DV01(), credit_spread, s -> price(base_curve + s, vec(cfs), times))
end

function duration(::CS01, base_curve, credit_spread, cfs)
    times = FinanceCore.timepoint.(cfs, 1:length(cfs))
    return duration(CS01(), base_curve, credit_spread, vec(cfs), times)
end

"""
    convexity(yield,cfs,times)
    convexity(yield,valuation_function)

Calculates the convexity.
    - `yield` should be a fixed effective yield (e.g. `0.05`).
    - `times` may be omitted and it will assume `cfs` are evenly spaced beginning at the end of the first period.

# Examples

Using vectors of cashflows and times
```julia-repl
julia> times = 1:5
julia> cfs = [0,0,0,0,100]
julia> duration(0.03,cfs,times)
4.854368932038834
julia> duration(Macaulay(),0.03,cfs,times)
5.0
julia> duration(Modified(),0.03,cfs,times)
4.854368932038835
julia> convexity(0.03,cfs,times)
28.277877274012614

```

Using any given value function: 

```julia-repl
julia> lump_sum_value(amount,years,i) = amount / (1 + i ) ^ years
julia> my_lump_sum_value(i) = lump_sum_value(100,5,i)
julia> duration(0.03,my_lump_sum_value)
4.854368932038835
julia> convexity(0.03,my_lump_sum_value)
28.277877274012617

```

"""
function convexity(yield, cfs, times)
    return convexity(yield, i -> price(i, cfs, times))
end

function convexity(yield, cfs)
    times = 1:length(cfs)
    return convexity(yield, i -> price(i, cfs, times))
end

function convexity(yield, valuation_function::T) where {T <: Function}
    v(x) = abs(valuation_function(yield + x))
    ∂²P = ForwardDiff.derivative(y -> ForwardDiff.derivative(v, y), 0.0)
    return ∂²P / v(0.0)
end


"""
    duration(keyrate::KeyRateDuration,curve,cashflows)    
    duration(keyrate::KeyRateDuration,curve,cashflows,timepoints)
    duration(keyrate::KeyRateDuration,curve,cashflows,timepoints,krd_points)

Calculate the key rate duration by shifting the **zero** (not par) curve by the kwarg `shift` at the timepoint specified by a KeyRateDuration(time).

The approach is to carve up the curve into `krd_points` (default is the unit steps between `1` and  the last timepoint of the casfhlows). The 
zero rate corresponding to the timepoint within the `KeyRateDuration` is shifted by `shift` (specified by the `KeyRateZero` or `KeyRatePar` constructors. A new curve is created from the shifted rates. This means that the 
"width" of the shifted section is ± 1 time period, unless specific points are specified via `krd_points`.

The `curve` may be any FinanceModels.jl curve (e.g. does not have to be a curve constructed via `FinanceModels.Zero(...)`).

!!! Experimental: Due to the paucity of examples in the literature, this feature does not have unit tests like the rest of JuliaActuary functionality. Additionally, the API may change in a future major/minor version update.

# Examples


```julia-repl
julia> riskfree_maturities = [0.5, 1.0, 1.5, 2.0];

julia> riskfree    = [0.05, 0.058, 0.064,0.068];

julia> rf_curve = FinanceModels.Zero(riskfree,riskfree_maturities);

julia> cfs = [10,10,10,10,10];

julia> duration(KeyRate(1),rf_curve,cfs)
8.932800152336995

```

# Extended Help

Key Rate Duration is not a well specified topic in the literature and in practice. The reference below suggest that shocking the par curve is more common 
in practice, but that the zero curve produces more consistent results. Future versions may support shifting the par curve.

References: 
- [Quant Finance Stack Exchange: To compute key rate duration, shall I use par curve or zero curve?](https://quant.stackexchange.com/questions/33891/to-compute-key-rate-duration-shall-i-use-par-curve-or-zero-curve)
- (Financial Exam Help 123](http://www.financialexamhelp123.com/key-rate-duration/)

"""
function duration(keyrate::KeyRateDuration, curve, cashflows, timepoints, krd_points)
    shift = keyrate.shift
    curve_up = _krd_new_curve(keyrate, curve, krd_points)
    curve_down = _krd_new_curve(opposite(keyrate), curve, krd_points)
    price = FinanceCore.pv(curve, cashflows, timepoints)
    price_up = FinanceCore.pv(curve_up, cashflows, timepoints)
    price_down = FinanceCore.pv(curve_down, cashflows, timepoints)


    return (price_down - price_up) / (2 * shift * price)

end

opposite(kr::KeyRateZero) = KeyRateZero(kr.timepoint, -kr.shift)
opposite(kr::KeyRatePar) = KeyRatePar(kr.timepoint, -kr.shift)

"""
    _tent_bump(shift, τ, krd_points)

Return a closure `(z, t) -> z + Continuous(bump)` implementing the Ho (1992)
tent function for key-rate duration bump-and-reprice:

- **First KRD point:** flat `shift` for `t ≤ τ`, linear ramp to 0 at next neighbor.
- **Last KRD point:** linear ramp from 0 at previous neighbor, flat `shift` for `t ≥ τ`.
- **Interior:** triangle with peak `shift` at `τ`, zero at both neighbors.
"""
function _tent_bump(shift, τ, krd_points)
    idx = findfirst(==(τ), krd_points)
    n = length(krd_points)

    τ_left = idx > 1 ? krd_points[idx-1] : nothing
    τ_right = idx < n ? krd_points[idx+1] : nothing

    return function (z, t)
        if τ_left === nothing && τ_right === nothing
            # Single KRD point: flat shift everywhere
            bump = shift
        elseif τ_left === nothing
            # First point: flat left, ramp right
            if t <= τ
                bump = shift
            elseif t >= τ_right
                bump = oftype(shift, 0)
            else
                bump = shift * (τ_right - t) / (τ_right - τ)
            end
        elseif τ_right === nothing
            # Last point: ramp left, flat right
            if t >= τ
                bump = shift
            elseif t <= τ_left
                bump = oftype(shift, 0)
            else
                bump = shift * (t - τ_left) / (τ - τ_left)
            end
        else
            # Interior: triangle peak at τ
            if t <= τ_left || t >= τ_right
                bump = oftype(shift, 0)
            elseif t <= τ
                bump = shift * (t - τ_left) / (τ - τ_left)
            else
                bump = shift * (τ_right - t) / (τ_right - τ)
            end
        end
        return z + FinanceCore.Continuous(bump)
    end
end

_ensure_yield_model(curve::FinanceModels.Yield.AbstractYieldModel) = curve
_ensure_yield_model(curve::FinanceCore.Rate) = FinanceModels.Yield.Constant(curve)
_ensure_yield_model(curve::Real) = FinanceModels.Yield.Constant(curve)

function _krd_new_curve(keyrate::KeyRateZero, curve, krd_points)
    bump = _tent_bump(keyrate.shift, keyrate.timepoint, krd_points)
    base = _ensure_yield_model(curve)
    return FinanceModels.Yield.TransformedYield(base, bump)
end

function _krd_new_curve(keyrate::KeyRatePar, curve, krd_points)
    curve_times = krd_points
    shift = keyrate.shift

    pars = FinanceModels.par.(curve, curve_times)

    zero_index = findfirst(==(keyrate.timepoint), curve_times)

    target_rate = pars[zero_index]
    pars[zero_index] += FinanceModels.Rate(shift, target_rate.compounding)

    new_curve = FinanceModels.fit(FinanceModels.Spline.Linear(), FinanceModels.ParYield.(pars, curve_times), FinanceModels.Fit.Bootstrap())

    return new_curve
end

function duration(keyrate::KeyRateDuration, curve, cashflows, timepoints)
    krd_points = 1:maximum(timepoints)
    return duration(keyrate, curve, cashflows, timepoints, krd_points)

end

function duration(keyrate::KeyRateDuration, curve, cashflows)
    timepoints = eachindex(cashflows)
    krd_points = 1:maximum(timepoints)
    return duration(keyrate, curve, cashflows, timepoints, krd_points)

end

""" 
    spread(curve1,curve2,cashflows)

Return the solved-for constant spread to add to `curve1` in order to equate the discounted `cashflows` with `curve2`

# Examples

```julia-repl
spread(0.04, 0.05, cfs)
Rate{Float64, Periodic}(0.010000000000000009, Periodic(1))
```
"""
function spread(curve1, curve2, cashflows, times = eachindex(cashflows))
    times = FinanceCore.timepoint.(cashflows, times)
    cashflows = FinanceCore.amount.(cashflows)
    pv2 = FinanceCore.pv(curve2, cashflows, times)


    function f(s, p)
        return abs2(FinanceCore.pv(curve1 + FinanceCore.Periodic(only(s), 1), cashflows, times) - pv2)
    end

    s0 = zeros(1)

    prob = Optimization.OptimizationProblem(f, s0, nothing)
    sol = Optimization.solve(prob, OptimizationOptimJL.NelderMead())
    return FinanceCore.Periodic(only(sol.u), 1)

end

"""
    moic(cashflows<:AbstractArray)

The multiple on invested capital ("moic") is the un-discounted sum of distributions divided by the sum of the contributions. The function assumes that negative numbers in the array represent contributions and positive numbers represent distributions.

# Examples

```julia-repl
julia> moic([-10,20,30])
5.0
```

"""
function moic(cfs::T) where {T <: AbstractArray}
    returned = sum(FinanceCore.amount(cf) for cf in cfs if FinanceCore.amount(cf) > 0)
    invested = -sum(FinanceCore.amount(cf) for cf in cfs if FinanceCore.amount(cf) < 0)
    return returned / invested
end

## Cashflow extraction helper

function _extract_cfs_times(cfs::AbstractVector{<:FinanceCore.Cashflow})
    return FinanceCore.amount.(cfs), FinanceCore.timepoint.(cfs)
end

## Scalar do-block forwarding for AbstractYieldModel
#
# Forwards `duration(vf, curve)` and `convexity(vf, curve)` (no tenors) to the
# generic finite-difference scalar path that works on any yield-like input.

function duration(valuation_fn::Function, yield::FinanceModels.Yield.AbstractYieldModel)
    return duration(yield, valuation_fn)
end
function convexity(valuation_fn::Function, yield::FinanceModels.Yield.AbstractYieldModel)
    return convexity(yield, valuation_fn)
end

## Key rate sensitivities via AD on zero-rate bumps (Yield.TenorShift)
#
# KRDs are computed by layering a triangular-hat zero-rate bump on top of the
# user's curve via `FinanceModels.Yield.TenorShift`, then taking ForwardDiff
# gradients/hessians w.r.t. the bump magnitudes. This works on any
# `AbstractYieldModel` — no curve-internal field is required, and there is no
# special dispatch for `ZeroRateCurve`. Callers always pass `tenors` (the KRD
# knot grid) explicitly; for a ZRC, the natural choice is `zrc.tenors`.

const AYM = FinanceModels.Yield.AbstractYieldModel

# Triangular hats with flat extrapolation outside the knot range.
# At a knot τᵢ the bump equals bᵢ; between τᵢ and τᵢ₊₁ it is linear.
function _hat_bump(tenors, bumps, t)
    t <= first(tenors) && return first(bumps)
    t >= last(tenors)  && return last(bumps)
    i = searchsortedlast(tenors, t)
    w = (t - tenors[i]) / (tenors[i+1] - tenors[i])
    return (one(w) - w) * bumps[i] + w * bumps[i+1]
end

# Layer a hat-function zero-rate bump over `curve` lazily.
_bumped(curve, tenors, bumps) = FinanceModels.Yield.TenorShift(
    curve,
    (z, t) -> z + FinanceCore.Continuous(_hat_bump(tenors, bumps, t)),
)

# Single-curve AD helper: value, gradient, optional hessian w.r.t. zero-rate bumps.
function _keyrate_ad(curve::AYM, tenors::AbstractVector, valuation_fn; order = 1)
    f(b) = valuation_fn(_bumped(curve, tenors, b))
    z = zeros(length(tenors))
    v = f(z)
    grad = ForwardDiff.gradient(f, z)
    order >= 2 || return (; value = v, gradient = grad)
    hess = ForwardDiff.hessian(f, z)
    return (; value = v, gradient = grad, hessian = hess)
end

# Two-curve AD helper: bumps for base and credit are concatenated, one AD pass, partitioned.
function _keyrate_ad(base::AYM, credit::AYM, tenors::AbstractVector, valuation_fn; order = 1)
    n = length(tenors)
    function f(combined)
        base_shift   = _bumped(base,   tenors, @view combined[1:n])
        credit_shift = _bumped(credit, tenors, @view combined[(n+1):(2n)])
        valuation_fn(base_shift, credit_shift)
    end
    z = zeros(2n)
    v = f(z)
    grad = ForwardDiff.gradient(f, z)
    base_grad = grad[1:n]
    credit_grad = grad[(n+1):(2n)]
    if order >= 2
        hess = ForwardDiff.hessian(f, z)
        return (;
            value = v,
            base_gradient = base_grad,
            credit_gradient = credit_grad,
            base_hessian = hess[1:n, 1:n],
            credit_hessian = hess[(n+1):(2n), (n+1):(2n)],
            cross_hessian = hess[1:n, (n+1):(2n)],
        )
    end
    return (; value = v, base_gradient = base_grad, credit_gradient = credit_grad)
end

# Standard valuation for fixed cashflows
_standard_valuation(cfs, times) = curve -> sum(cf * curve(t) for (cf, t) in zip(cfs, times))

# Two-curve standard valuation (additive on rates → multiplicative on discount factors)
_standard_valuation_2curve(cfs, times) = (base, credit) -> sum(cf * base(t) * credit(t) for (cf, t) in zip(cfs, times))

## AbstractYieldModel + tenors: KRD / IR01 / CS01 / convexity / sensitivities
#
# These dispatches accept any `FinanceModels.Yield.AbstractYieldModel` plus an
# explicit `tenors` vector specifying the KRD knot grid. Internally the AD path
# layers a hat-function zero-rate bump over the user's curve via
# `Yield.TenorShift`; the user's curve is never resampled or rebuilt.
#
# `ZeroRateCurve` inputs go through the same path — it has no special dispatch.
# Pass `zrc.tenors` (or any other knot grid) explicitly at the call site.
#
# Tenor grid is required (no default) because KRD bucket conventions vary
# (Bloomberg, FRTB, BMA SBA, etc.); downstream should choose explicitly.

"""
    duration(valuation_fn, curve::AbstractYieldModel, tenors) -> scalar
    duration(curve::AbstractYieldModel, tenors, cfs, times) -> scalar
    duration(curve::AbstractYieldModel, tenors, cfs::AbstractVector{<:Cashflow}) -> scalar

Scalar modified duration for any `AbstractYieldModel` evaluated against a KRD
knot grid. Equivalent to `sum(duration(KeyRates(), ..., tenors))`.

See [`duration(::KeyRates, ...)`](@ref) for the full key-rate decomposition.

# Example
```julia
duration(pv, my_composite_curve, [0.25, 1, 5, 10, 30])
```
"""
function duration(valuation_fn::Function, curve::AYM, tenors)
    return sum(duration(KeyRates(), valuation_fn, curve, tenors))
end
function duration(curve::AYM, tenors, cfs, times)
    return sum(duration(KeyRates(), curve, tenors, cfs, times))
end
function duration(curve::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow})
    amounts, times = _extract_cfs_times(cfs)
    return duration(curve, tenors, amounts, times)
end

"""
    duration(::KeyRates, valuation_fn, curve::AbstractYieldModel, tenors) -> Vector
    duration(::KeyRates, curve::AbstractYieldModel, tenors, cfs, times) -> Vector
    duration(::KeyRates, curve::AbstractYieldModel, tenors, cfs::AbstractVector{<:Cashflow}) -> Vector

Key-rate durations (modified) for any `AbstractYieldModel`, computed by
layering a triangular-hat zero-rate bump at each tenor in `tenors` over the
user's curve via `Yield.TenorShift`, then taking the AD gradient w.r.t. the
bump magnitudes. The user's curve is preserved at all non-knot points.

# Tenor grid

`tenors` is the KRD knot grid — a separate modeling choice from any tenor
structure baked into the curve itself. For a `ZeroRateCurve`, passing
`zrc.tenors` is the natural choice but not required; you can evaluate
key-rate durations on any grid (e.g. Bloomberg `{0.25, 1, 2, 5, 10, 30}`,
FRTB `{0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30}`, etc.) without re-fitting.

The grid must be sorted ascending, distinct, and strictly positive. These
preconditions are not checked at runtime — a malformed grid produces wrong
gradients silently.

# Bump shape and endpoint extrapolation

The bump at the i-th knot is a triangular hat centered at `tenors[i]` with
support `[tenors[i-1], tenors[i+1]]`. Outside the knot range it is flat:
bumping `tenors[1]` perturbs all cashflows at `t ≤ tenors[1]` equally, and
bumping `tenors[end]` perturbs all cashflows at `t ≥ tenors[end]` equally.
For long-duration insurance liabilities (LTC, deferred / payout annuities),
the last-knot KRD absorbs all super-tenor sensitivity — extend the grid
past your longest cashflow if you want that decomposed.

For a `ZeroRateCurve` with `Spline.Linear()` the result matches AD over
the curve's own rates exactly. For other splines the bump kernel is
hat-shaped rather than spline-shaped, so per-knot KRDs shift slightly;
the sum of KRDs (= scalar modified duration) is invariant either way.

# Example
```julia
duration(KeyRates(), pv, curve, [0.25, 1, 5, 10, 30])

duration(KeyRates(), curve, [0.25, 1, 5, 10, 30]) do c
    pv(c)
end
```
"""
function duration(::KeyRates, valuation_fn::Function, curve::AYM, tenors)
    ad = _keyrate_ad(curve, tenors, valuation_fn)
    return -ad.gradient ./ ad.value
end
function duration(::KeyRates, curve::AYM, tenors, cfs, times)
    return duration(KeyRates(), _standard_valuation(cfs, times), curve, tenors)
end
function duration(::KeyRates, curve::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow})
    amounts, times = _extract_cfs_times(cfs)
    return duration(KeyRates(), curve, tenors, amounts, times)
end
duration(valuation_fn::Function, ::KeyRates, curve::AYM, tenors) =
    duration(KeyRates(), valuation_fn, curve, tenors)

"""
    duration(::DV01, valuation_fn, curve::AbstractYieldModel, tenors) -> scalar
    duration(::DV01, curve::AbstractYieldModel, tenors, cfs, times) -> scalar
    duration(::DV01, ::KeyRates, valuation_fn, curve::AbstractYieldModel, tenors) -> Vector
    duration(::DV01, ::KeyRates, curve::AbstractYieldModel, tenors, cfs, times) -> Vector

DV01 (scalar or per-knot vector) for any `AbstractYieldModel`. Equivalent to
the `KeyRates` variants of `duration` but in dollars per basis point.
"""
function duration(::DV01, valuation_fn::Function, curve::AYM, tenors)
    return sum(duration(DV01(), KeyRates(), valuation_fn, curve, tenors))
end
function duration(::DV01, curve::AYM, tenors, cfs, times)
    return sum(duration(DV01(), KeyRates(), curve, tenors, cfs, times))
end
function duration(::DV01, curve::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow})
    amounts, times = _extract_cfs_times(cfs)
    return duration(DV01(), curve, tenors, amounts, times)
end
duration(valuation_fn::Function, ::DV01, curve::AYM, tenors) =
    duration(DV01(), valuation_fn, curve, tenors)

function duration(::DV01, ::KeyRates, valuation_fn::Function, curve::AYM, tenors)
    ad = _keyrate_ad(curve, tenors, valuation_fn)
    return -ad.gradient ./ 10_000
end
function duration(::DV01, ::KeyRates, curve::AYM, tenors, cfs, times)
    return duration(DV01(), KeyRates(), _standard_valuation(cfs, times), curve, tenors)
end
function duration(::DV01, ::KeyRates, curve::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow})
    amounts, times = _extract_cfs_times(cfs)
    return duration(DV01(), KeyRates(), curve, tenors, amounts, times)
end
duration(valuation_fn::Function, ::DV01, ::KeyRates, curve::AYM, tenors) =
    duration(DV01(), KeyRates(), valuation_fn, curve, tenors)

"""
    duration(::IR01, valuation_fn, base::AbstractYieldModel, credit::AbstractYieldModel, tenors) -> scalar
    duration(::IR01, base::AbstractYieldModel, credit::AbstractYieldModel, tenors, cfs, times) -> scalar
    duration(::IR01, ::KeyRates, valuation_fn, base, credit, tenors) -> Vector
    duration(::IR01, ::KeyRates, base, credit, tenors, cfs, times) -> Vector
    duration(::CS01, ...) -> ...

Two-curve IR01/CS01 for any `AbstractYieldModel` pair sharing a `tenors`
grid. IR01 bumps the base (risk-free) curve only; CS01 bumps the credit
(spread) curve only.
"""
function duration(::IR01, valuation_fn::Function, base::AYM, credit::AYM, tenors)
    return sum(duration(IR01(), KeyRates(), valuation_fn, base, credit, tenors))
end
function duration(::IR01, base::AYM, credit::AYM, tenors, cfs, times)
    return sum(duration(IR01(), KeyRates(), base, credit, tenors, cfs, times))
end
function duration(::IR01, base::AYM, credit::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow})
    amounts, times = _extract_cfs_times(cfs)
    return duration(IR01(), base, credit, tenors, amounts, times)
end
duration(valuation_fn::Function, ::IR01, base::AYM, credit::AYM, tenors) =
    duration(IR01(), valuation_fn, base, credit, tenors)

function duration(::IR01, ::KeyRates, valuation_fn::Function, base::AYM, credit::AYM, tenors)
    ad = _keyrate_ad(base, credit, tenors, valuation_fn)
    return -ad.base_gradient ./ 10_000
end
function duration(::IR01, ::KeyRates, base::AYM, credit::AYM, tenors, cfs, times)
    return duration(IR01(), KeyRates(), _standard_valuation_2curve(cfs, times), base, credit, tenors)
end
function duration(::IR01, ::KeyRates, base::AYM, credit::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow})
    amounts, times = _extract_cfs_times(cfs)
    return duration(IR01(), KeyRates(), base, credit, tenors, amounts, times)
end
duration(valuation_fn::Function, ::IR01, ::KeyRates, base::AYM, credit::AYM, tenors) =
    duration(IR01(), KeyRates(), valuation_fn, base, credit, tenors)

function duration(::CS01, valuation_fn::Function, base::AYM, credit::AYM, tenors)
    return sum(duration(CS01(), KeyRates(), valuation_fn, base, credit, tenors))
end
function duration(::CS01, base::AYM, credit::AYM, tenors, cfs, times)
    return sum(duration(CS01(), KeyRates(), base, credit, tenors, cfs, times))
end
function duration(::CS01, base::AYM, credit::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow})
    amounts, times = _extract_cfs_times(cfs)
    return duration(CS01(), base, credit, tenors, amounts, times)
end
duration(valuation_fn::Function, ::CS01, base::AYM, credit::AYM, tenors) =
    duration(CS01(), valuation_fn, base, credit, tenors)

function duration(::CS01, ::KeyRates, valuation_fn::Function, base::AYM, credit::AYM, tenors)
    ad = _keyrate_ad(base, credit, tenors, valuation_fn)
    return -ad.credit_gradient ./ 10_000
end
function duration(::CS01, ::KeyRates, base::AYM, credit::AYM, tenors, cfs, times)
    return duration(CS01(), KeyRates(), _standard_valuation_2curve(cfs, times), base, credit, tenors)
end
function duration(::CS01, ::KeyRates, base::AYM, credit::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow})
    amounts, times = _extract_cfs_times(cfs)
    return duration(CS01(), KeyRates(), base, credit, tenors, amounts, times)
end
duration(valuation_fn::Function, ::CS01, ::KeyRates, base::AYM, credit::AYM, tenors) =
    duration(CS01(), KeyRates(), valuation_fn, base, credit, tenors)

"""
    convexity(valuation_fn, curve::AbstractYieldModel, tenors) -> scalar
    convexity(curve::AbstractYieldModel, tenors, cfs, times) -> scalar
    convexity(::KeyRates, valuation_fn, curve::AbstractYieldModel, tenors) -> Matrix
    convexity(::KeyRates, curve::AbstractYieldModel, tenors, cfs, times) -> Matrix
    convexity(base::AbstractYieldModel, credit::AbstractYieldModel, tenors, cfs, times) -> NamedTuple
    convexity(::KeyRates, base, credit, tenors, cfs, times) -> NamedTuple

Key-rate convexity (matrix) and scalar convexity for any `AbstractYieldModel`
or pair. Mirrors `duration` but returns ∂²V/∂rᵢ∂rⱼ rather than ∂V/∂rᵢ.
"""
function convexity(valuation_fn::Function, curve::AYM, tenors)
    return sum(convexity(KeyRates(), valuation_fn, curve, tenors))
end
function convexity(curve::AYM, tenors, cfs, times)
    return sum(convexity(KeyRates(), curve, tenors, cfs, times))
end
function convexity(curve::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow})
    amounts, times = _extract_cfs_times(cfs)
    return convexity(curve, tenors, amounts, times)
end

function convexity(::KeyRates, valuation_fn::Function, curve::AYM, tenors)
    ad = _keyrate_ad(curve, tenors, valuation_fn; order = 2)
    return ad.hessian ./ ad.value
end
function convexity(::KeyRates, curve::AYM, tenors, cfs, times)
    return convexity(KeyRates(), _standard_valuation(cfs, times), curve, tenors)
end
function convexity(::KeyRates, curve::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow})
    amounts, times = _extract_cfs_times(cfs)
    return convexity(KeyRates(), curve, tenors, amounts, times)
end
convexity(valuation_fn::Function, ::KeyRates, curve::AYM, tenors) =
    convexity(KeyRates(), valuation_fn, curve, tenors)

function convexity(valuation_fn::Function, base::AYM, credit::AYM, tenors)
    kr = convexity(KeyRates(), valuation_fn, base, credit, tenors)
    return (; base = sum(kr.base), credit = sum(kr.credit), cross = sum(kr.cross))
end
function convexity(base::AYM, credit::AYM, tenors, cfs, times)
    return convexity(_standard_valuation_2curve(cfs, times), base, credit, tenors)
end
function convexity(base::AYM, credit::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow})
    amounts, times = _extract_cfs_times(cfs)
    return convexity(base, credit, tenors, amounts, times)
end

function convexity(::KeyRates, valuation_fn::Function, base::AYM, credit::AYM, tenors)
    ad = _keyrate_ad(base, credit, tenors, valuation_fn; order = 2)
    return (;
        base = ad.base_hessian ./ ad.value,
        credit = ad.credit_hessian ./ ad.value,
        cross = ad.cross_hessian ./ ad.value,
    )
end
function convexity(::KeyRates, base::AYM, credit::AYM, tenors, cfs, times)
    return convexity(KeyRates(), _standard_valuation_2curve(cfs, times), base, credit, tenors)
end
function convexity(::KeyRates, base::AYM, credit::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow})
    amounts, times = _extract_cfs_times(cfs)
    return convexity(KeyRates(), base, credit, tenors, amounts, times)
end
convexity(valuation_fn::Function, ::KeyRates, base::AYM, credit::AYM, tenors) =
    convexity(KeyRates(), valuation_fn, base, credit, tenors)

"""
    sensitivities(valuation_fn, curve::AbstractYieldModel, tenors) -> NamedTuple
    sensitivities(curve::AbstractYieldModel, tenors, cfs, times) -> NamedTuple
    sensitivities(::DV01, valuation_fn, curve::AbstractYieldModel, tenors) -> NamedTuple
    sensitivities(base::AbstractYieldModel, credit::AbstractYieldModel, tenors, cfs, times) -> NamedTuple
    sensitivities(::DV01, base, credit, tenors, cfs, times) -> NamedTuple

Bundled value + key-rate durations (or DV01s) + convexity matrix for any
`AbstractYieldModel` or pair, in a single AD pass.
"""
function sensitivities(valuation_fn::Function, curve::AYM, tenors)
    ad = _keyrate_ad(curve, tenors, valuation_fn; order = 2)
    return (;
        value = ad.value,
        durations = -ad.gradient ./ ad.value,
        convexities = ad.hessian ./ ad.value,
    )
end
function sensitivities(curve::AYM, tenors, cfs, times)
    return sensitivities(_standard_valuation(cfs, times), curve, tenors)
end
function sensitivities(curve::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow})
    amounts, times = _extract_cfs_times(cfs)
    return sensitivities(curve, tenors, amounts, times)
end

function sensitivities(::DV01, valuation_fn::Function, curve::AYM, tenors)
    ad = _keyrate_ad(curve, tenors, valuation_fn; order = 2)
    return (;
        value = ad.value,
        dv01s = -ad.gradient ./ 10_000,
        convexities = ad.hessian ./ ad.value,
    )
end
function sensitivities(::DV01, curve::AYM, tenors, cfs, times)
    return sensitivities(DV01(), _standard_valuation(cfs, times), curve, tenors)
end
function sensitivities(::DV01, curve::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow})
    amounts, times = _extract_cfs_times(cfs)
    return sensitivities(DV01(), curve, tenors, amounts, times)
end
sensitivities(valuation_fn::Function, ::DV01, curve::AYM, tenors) =
    sensitivities(DV01(), valuation_fn, curve, tenors)

function sensitivities(valuation_fn::Function, base::AYM, credit::AYM, tenors)
    ad = _keyrate_ad(base, credit, tenors, valuation_fn; order = 2)
    return (;
        value = ad.value,
        base_durations = -ad.base_gradient ./ ad.value,
        credit_durations = -ad.credit_gradient ./ ad.value,
        convexities = (;
            base = ad.base_hessian ./ ad.value,
            credit = ad.credit_hessian ./ ad.value,
            cross = ad.cross_hessian ./ ad.value,
        ),
    )
end
function sensitivities(base::AYM, credit::AYM, tenors, cfs, times)
    return sensitivities(_standard_valuation_2curve(cfs, times), base, credit, tenors)
end
function sensitivities(base::AYM, credit::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow})
    amounts, times = _extract_cfs_times(cfs)
    return sensitivities(base, credit, tenors, amounts, times)
end

function sensitivities(::DV01, valuation_fn::Function, base::AYM, credit::AYM, tenors)
    ad = _keyrate_ad(base, credit, tenors, valuation_fn; order = 2)
    return (;
        value = ad.value,
        base_dv01s = -ad.base_gradient ./ 10_000,
        credit_dv01s = -ad.credit_gradient ./ 10_000,
        convexities = (;
            base = ad.base_hessian ./ ad.value,
            credit = ad.credit_hessian ./ ad.value,
            cross = ad.cross_hessian ./ ad.value,
        ),
    )
end
function sensitivities(::DV01, base::AYM, credit::AYM, tenors, cfs, times)
    return sensitivities(DV01(), _standard_valuation_2curve(cfs, times), base, credit, tenors)
end
function sensitivities(::DV01, base::AYM, credit::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow})
    amounts, times = _extract_cfs_times(cfs)
    return sensitivities(DV01(), base, credit, tenors, amounts, times)
end
sensitivities(valuation_fn::Function, ::DV01, base::AYM, credit::AYM, tenors) =
    sensitivities(DV01(), valuation_fn, base, credit, tenors)

## Hull-White convenience methods
#
# `hw.curve` can be any `AbstractYieldModel` — the AD path uses TenorShift
# bumps over the curve via the AYM-based `sensitivities` impls above.

const HW = FinanceModels.ShortRate.HullWhite

# Fixed cashflows: sensitivities(hw, tenors, cfs, times; ...)
function sensitivities(hw::HW, tenors, cfs, times;
                       n_scenarios=1000, timestep=1/12, horizon=nothing,
                       rng=Random.default_rng())
    h = horizon === nothing ? maximum(times) + 1.0 : Float64(horizon)
    sensitivities(hw.curve, tenors) do curve
        hw_new = FinanceModels.ShortRate.HullWhite(hw.a, hw.σ, curve)
        scenarios = FinanceModels.simulate(hw_new; n_scenarios, timestep, horizon=h, rng)
        sum(FinanceCore.pv(sc, cfs, times) for sc in scenarios) / n_scenarios
    end
end

# Do-block: sensitivities(hw, tenors; ...) do scenarios ... end
function sensitivities(valuation_fn::Function, hw::HW, tenors;
                       n_scenarios=1000, timestep=1/12, horizon=30.0,
                       rng=Random.default_rng())
    sensitivities(hw.curve, tenors) do curve
        hw_new = FinanceModels.ShortRate.HullWhite(hw.a, hw.σ, curve)
        scenarios = FinanceModels.simulate(hw_new; n_scenarios, timestep, horizon, rng)
        valuation_fn(scenarios)
    end
end

# DV01 variants
function sensitivities(::DV01, hw::HW, tenors, cfs, times;
                       n_scenarios=1000, timestep=1/12, horizon=nothing,
                       rng=Random.default_rng())
    h = horizon === nothing ? maximum(times) + 1.0 : Float64(horizon)
    sensitivities(DV01(), hw.curve, tenors) do curve
        hw_new = FinanceModels.ShortRate.HullWhite(hw.a, hw.σ, curve)
        scenarios = FinanceModels.simulate(hw_new; n_scenarios, timestep, horizon=h, rng)
        sum(FinanceCore.pv(sc, cfs, times) for sc in scenarios) / n_scenarios
    end
end

function sensitivities(valuation_fn::Function, ::DV01, hw::HW, tenors;
                       n_scenarios=1000, timestep=1/12, horizon=30.0,
                       rng=Random.default_rng())
    sensitivities(DV01(), hw.curve, tenors) do curve
        hw_new = FinanceModels.ShortRate.HullWhite(hw.a, hw.σ, curve)
        scenarios = FinanceModels.simulate(hw_new; n_scenarios, timestep, horizon, rng)
        valuation_fn(scenarios)
    end
end

end
