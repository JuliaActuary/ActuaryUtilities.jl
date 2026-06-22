module FinancialMath

import ..FinanceCore
import ..FinanceCore: irr, internal_rate_of_return, pv, present_value
import ..FinanceModels
import ..ForwardDiff
import ..ActuaryUtilities: duration
import Random

export irr, internal_rate_of_return, spread,
    pv, present_value, price, present_values,
    breakeven, moic,
    Macaulay, Modified, DV01, IR01, CS01, Effective, Spread, KeyRates, KeyRatePar, KeyRateZero, KeyRate, duration, convexity,
    sensitivities, dv01, zspread, locked_floater, reproject

"""
    present_values(interest, cashflows, timepoints)

Efficiently calculate a vector representing the present value of the given cashflows at each period prior to the given timepoint.

# Examples
```julia-repl
julia> present_values(0.00, [1,1,1])
3-element Vector{Float64}:
 3.0
 2.0
 1.0

julia> present_values(0.05, [10,10,110], [1,2,3])
3-element Vector{Float64}:
 113.61624014685238
 109.297052154195
 104.76190476190476
```

"""
function present_values(interest, cashflows, times = eachindex(cashflows))
    length(cashflows) == length(times) || throw(DimensionMismatch("cashflows and times must have equal length"))
    n = length(cashflows)
    # single reverse scan: pvs[k] is the value at times[k-1] (time zero for k = 1)
    # of cashflows k..n. O(n) and non-recursive (the prior implementation was
    # O(n²) with recursion depth n), and the element type follows the data so
    # AD dual numbers propagate.
    acc = zero(FinanceCore.discount(interest, first(times)) * first(cashflows))
    pvs = Vector{typeof(acc)}(undef, n)
    @inbounds for k in n:-1:1
        from = k == 1 ? zero(times[k]) : times[k - 1]
        acc = FinanceCore.discount(interest, from, times[k]) * (acc + cashflows[k])
        pvs[k] = acc
    end
    return pvs
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

```julia-repl
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

`DV01 = -∂V/∂r / 10000`, so a DV01 of 0.045 means the position loses \\\$0.045 per \\\$100 notional for a 1bp rate increase.

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
    Effective <: Duration

Effective (rate) duration / convexity for a curve-dependent contract (e.g. a
floating-rate bond): reprices under a shifted curve with the projected cashflows
RE-COMPUTED, so a floating coupon re-fixes. The correct interest-rate duration for
floating-rate instruments; `Modified`/`Macaulay` are valid only for curve-independent
(fixed) cashflows. `duration(Effective(), contract, curve, tenors)`.

See also: [`Spread`](@ref), [`sensitivities`](@ref), [`locked_floater`](@ref).
"""
struct Effective <: Duration end

"""
    Spread <: Duration

Spread (credit) duration: bumps the discount curve only, holding the projected
(index) cashflows fixed. For a floating-rate bond this is ≈ time to maturity — the
discount-margin / credit sensitivity.

See also: [`Effective`](@ref), [`sensitivities`](@ref).
"""
struct Spread <: Duration end

"""
    KeyRates(tenors) <: Duration

Marker type carrying the key-rate knot grid `tenors` for use with [`duration`](@ref),
[`convexity`](@ref), and [`sensitivities`](@ref). Requests the full key-rate
decomposition (vector of durations, matrix of convexities) instead of the default
scalar summary.

`tenors` is any `AbstractVector{<:Real}` of positive knot times. The knot grid is
carried with the measurement intent — "key rate durations at these tenors" lives in
one object.

```julia
tenors = [1.0, 2.0, 5.0, 10.0, 30.0]
duration(KeyRates(tenors), curve, cfs, times)            # vector of key rate durations
duration(DV01(), KeyRates(tenors), curve, cfs, times)    # vector of key rate DV01s
convexity(KeyRates(tenors), curve, cfs, times)           # matrix of key rate convexities
sensitivities(KeyRates(tenors), curve, cfs, times)       # value + durations + convexities
```

See also: [`DV01`](@ref), [`IR01`](@ref), [`CS01`](@ref)
"""
struct KeyRates{T<:AbstractVector{<:Real}} <: Duration
    tenors::T
    function KeyRates(tenors::T) where {T<:AbstractVector{<:Real}}
        isempty(tenors)   && throw(ArgumentError("KeyRates tenors must be non-empty"))
        issorted(tenors)  || throw(ArgumentError("KeyRates tenors must be sorted ascending"))
        allunique(tenors) || throw(ArgumentError("KeyRates tenors must be distinct"))
        all(>(0), tenors) || throw(ArgumentError("KeyRates tenors must be strictly positive"))
        return new{T}(tenors)
    end
end

abstract type KeyRateDuration <: Duration end


"""
    KeyRatePar(timepoint,shift=0.001) <: KeyRateDuration

Shift the par curve by the given amount at the given timepoint. Use in conjunction with `duration` to calculate the key rate duration. 

Unlike other duration statistics which are computed using analytic derivatives, `KeyRateDuration`s are computed via a shift-and-compute the yield curve approach.

`KeyRatePar` is more commonly reported (than [`KeyRateZero`](@ref)) in the fixed income markets, even though [`KeyRateZero`](@ref) has more analytically attractive properties. See the discussion of KeyRateDuration in the FinanceModels.jl docs.

"""
struct KeyRatePar{T, R} <: KeyRateDuration
    timepoint::T
    shift::R
    KeyRatePar(timepoint, shift = 0.001) = new{typeof(timepoint), typeof(shift)}(timepoint, shift)
end

"""
    KeyRateZero(timepoint,shift=0.001) <: KeyRateDuration

Shift the **zero** curve by the given amount at the given timepoint. Use in conjunction with `duration` to calculate the key rate duration.

Unlike other duration statistics which are computed using analytic derivatives, `KeyRateDuration` is computed via a shift-and-compute the yield curve approach.

`KeyRateZero` is less commonly reported (than [`KeyRatePar`](@ref)) in the fixed income markets, even though zero-curve shifts have more analytically attractive properties (rates beyond the shifted timepoint are unaffected). See the discussion of KeyRateDuration in the FinanceModels.jl docs.
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
  - e.g. shifting the 5-year par rate would (incorrectly) give a 6-year zero coupon bond a negative key rate duration, while a 5-year zero-rate shift leaves it unaffected


"""
const KeyRate = KeyRateZero

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

When not given `Modified()` or `Macaulay()` as an argument, will default to `Modified()`.

- Modified duration: the relative change per point of yield change.
- Macaulay: the cashflow-weighted average time.
- DV01: the absolute change per basis point (hundredth of a percentage point).
- IR01: the absolute change per basis point shift in the risk-free (base) curve, holding credit spread constant.
- CS01: the absolute change per basis point shift in the credit spread, holding the risk-free (base) curve constant.

# Periodicity convention

The Modified duration returned depends on the space in which the parallel rate shock is applied, and this differs between plain rates and yield *models*:

- A scalar (e.g. `0.04`) or a `Rate` is shocked in its own compounding space. A scalar is treated as `Periodic(0.04, 1)`, so Modified = Macaulay / (1 + 0.04); in general a `Periodic(y, m)` rate gives Modified = Macaulay / (1 + y/m), and a `Continuous(y)` rate gives Modified = Macaulay.
- A yield model (e.g. `Yield.Constant(0.04)` from FinanceModels) composes the shock in continuous-zero space, so Modified = Macaulay under the curve's own discounting, regardless of the compounding convention stored in the model.

The same inputs therefore produce two different numbers by design:

```julia-repl
julia> times = 1:5; cfs = [0,0,0,0,100];

julia> duration(0.04, cfs, times)                  # Periodic(1) shock: Macaulay / 1.04
4.8076923076923075

julia> duration(Yield.Constant(0.04), cfs, times)  # continuous-zero shock: Macaulay
5.0
```

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
28.277877274012635

```

Using any given value function: 

```julia-repl
julia> lump_sum_value(amount,years,i) = amount / (1 + i ) ^ years
julia> my_lump_sum_value(i) = lump_sum_value(100,5,i)
julia> duration(0.03,my_lump_sum_value)
4.854368932038835
julia> convexity(0.03,my_lump_sum_value)
28.277877274012642

```
"""
function duration(::Macaulay, yield, cfs, times)
    return sum(FinanceCore.timepoint.(cfs, times) .* price.(yield, cfs, times) / price(yield, cfs, times))
end

function duration(::Modified, yield, cfs, times)
    D(i) = price(i, cfs, times)
    return duration(yield, D)
end

# ── Analytic Modified-duration fast paths for flat yields ───────────────────
#
# Each method below is exactly equal to the generic AD path (locked by
# equality tests vs `duration(yield, i -> price(i, cfs, times))`); the only
# difference between yield types is the space in which the parallel shock `i`
# is applied by `i + yield`:
#
# * `Real` y: nominal `Periodic(1)` space → V(i) = Σ cf·(1+y+i)^(-t),
#   so Modified = Macaulay / (1 + y).
# * `Rate{Periodic(m)}`: the rate's own nominal space →
#   Modified = Macaulay / (1 + y/m).
# * `Rate{Continuous}`: the continuous rate itself → Modified = Macaulay.
# * `Yield.Constant`: model arithmetic composes in continuous-zero space
#   (`Constant(i) + Constant(y)` adds continuous rates), so Modified = Macaulay
#   under the curve's own discounting, regardless of the stored compounding.
#
# Macaulay here is the signed cashflow-weighted time Σ t·cf·d / Σ cf·d, which
# matches the generic path's d/di log|V| for any sign of V.

# Macaulay (cashflow-weighted average time) is the identity-weighted ratio.
# Shares the guarded `@inbounds` accumulation kernel `_weighted_ratio` (defined
# alongside the convexity fast paths below) with the convexity statistics.
_macaulay_ratio(yield, cfs, times) = _weighted_ratio(yield, identity, cfs, times)

function duration(::Modified, yield::Real, cfs::AbstractVector, times)
    return _macaulay_ratio(yield, cfs, times) / (1 + yield)
end
function duration(::Modified, yield::FinanceCore.Rate{<:Real, FinanceCore.Periodic}, cfs::AbstractVector, times)
    m = yield.compounding.frequency
    return _macaulay_ratio(yield, cfs, times) / (1 + FinanceCore.rate(yield) / m)
end
function duration(::Modified, yield::FinanceCore.Rate{<:Real, FinanceCore.Continuous}, cfs::AbstractVector, times)
    return _macaulay_ratio(yield, cfs, times)
end
function duration(::Modified, yield::FinanceModels.Yield.Constant{<:FinanceCore.Rate}, cfs::AbstractVector, times)
    return _macaulay_ratio(yield.rate, cfs, times)
end

function duration(yield, valuation_function::T) where {T <: Function}
    # `abs`: duration is defined on the magnitude of value, consistent with
    # `price` (used by the cashflow forms) and the `convexity` sibling — a
    # negative-valued (liability) valuation function is a valid input
    D(i) = log(abs(valuation_function(i + yield)))
    return δV = -ForwardDiff.derivative(D, 0.0)
end

# Element access for cashflow vectors that may be either numeric or
# wrapped `FinanceCore.Cashflow` values. The scalar duration / convexity
# fast paths use this so they work uniformly across both representations.
@inline _cf_value(c::FinanceCore.Cashflow) = FinanceCore.amount(c)
@inline _cf_value(c) = c

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
0.035459505041623596

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
0.035459505041623596

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
28.277877274012635

```

Using any given value function: 

```julia-repl
julia> lump_sum_value(amount,years,i) = amount / (1 + i ) ^ years
julia> my_lump_sum_value(i) = lump_sum_value(100,5,i)
julia> duration(0.03,my_lump_sum_value)
4.854368932038835
julia> convexity(0.03,my_lump_sum_value)
28.277877274012642

```

"""
function convexity(yield, cfs, times)
    return convexity(yield, i -> price(i, cfs, times))
end

function convexity(yield, cfs)
    times = FinanceCore.timepoint.(cfs, 1:length(cfs))
    return convexity(yield, cfs, times)
end

# ── Analytic convexity fast paths for flat yields ───────────────────────────
#
# Exactly equal to the generic nested-AD path (locked by equality tests vs
# `convexity(yield, i -> price(i, cfs, times))`). As with the Modified-duration
# fast paths above, the weight and divisor follow from where `yield + x`
# applies the shock:
#
# * `Real` y: V(x) = Σ cf·(1+y+x)^(-t) → Σ cf·d·t(t+1) / V / (1+y)²
# * `Rate{Periodic(m)}`: V(x) = Σ cf·(1+(y+x)/m)^(-mt) → Σ cf·d·t(t+1/m) / V / (1+y/m)²
# * `Rate{Continuous}`: V(x) = Σ cf·e^(-(y+x)t) → Σ cf·d·t² / V
# * `Yield.Constant`: shock composes in continuous-zero space as log(1+x), so
#   V(x) = Σ cf·d·(1+x)^(-t) → Σ cf·d·t(t+1) / V (no divisor).
#
# The ratio uses the signed V, matching the generic path's |V|-normalized
# second derivative for any sign of V (signs cancel).

# Shared accumulation kernel: Σ weight(t)·cf·d / Σ cf·d. `weight = identity`
# gives the Macaulay ratio (Modified-duration fast paths above); the t(t+1)/t²
# weights below give the convexity statistics.
function _weighted_ratio(yield, weight, cfs, times)
    # @inbounds below indexes `times` by `eachindex(cfs)` — a silent mismatch
    # would read out of bounds rather than zip-truncate
    length(cfs) == length(times) || throw(DimensionMismatch("cfs and times must have equal length"))
    t1 = FinanceCore.timepoint(first(cfs), first(times))
    z = _cf_value(first(cfs)) * FinanceCore.discount(yield, t1)
    V = zero(z)
    Vw = zero(weight(t1) * z)
    @inbounds for k in eachindex(cfs)
        t = FinanceCore.timepoint(cfs[k], times[k])
        cfd = _cf_value(cfs[k]) * FinanceCore.discount(yield, t)
        V += cfd
        Vw += weight(t) * cfd
    end
    return Vw / V
end

function convexity(yield::Real, cfs::AbstractVector, times)
    return _weighted_ratio(yield, t -> t * (t + 1), cfs, times) / (1 + yield)^2
end
function convexity(yield::FinanceCore.Rate{<:Real, FinanceCore.Periodic}, cfs::AbstractVector, times)
    m = yield.compounding.frequency
    return _weighted_ratio(yield, t -> t * (t + 1 / m), cfs, times) / (1 + FinanceCore.rate(yield) / m)^2
end
function convexity(yield::FinanceCore.Rate{<:Real, FinanceCore.Continuous}, cfs::AbstractVector, times)
    return _weighted_ratio(yield, t -> t * t, cfs, times)
end
function convexity(yield::FinanceModels.Yield.Constant{<:FinanceCore.Rate}, cfs::AbstractVector, times)
    return _weighted_ratio(yield.rate, t -> t * (t + 1), cfs, times)
end
# disambiguation vs `convexity(curve::AYM, tenors, cfs::AbstractVector{<:Cashflow})`:
# a Cashflow vector in the third position means (tenors, cashflows), not (cfs, times)
function convexity(yield::FinanceModels.Yield.Constant{<:FinanceCore.Rate}, tenors::AbstractVector, cfs::AbstractVector{<:FinanceCore.Cashflow})
    return convexity(yield, tenors, _extract_cfs_times(cfs)...)
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

!!! warning "Experimental"
    Due to the paucity of examples in the literature, this feature does not have unit tests like the rest of JuliaActuary functionality. Additionally, the API may change in a future major/minor version update.

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
    idx === nothing && throw(
        ArgumentError(
            "KeyRateDuration timepoint $τ is not a point of the krd_points grid $krd_points; pass krd_points containing the shifted timepoint"
        )
    )
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
    return FinanceModels.Yield.TenorShift(base, bump)
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

function _default_krd_points(timepoints)
    mt = maximum(timepoints)
    mt >= 1 || throw(
        ArgumentError(
            "the default krd_points grid 1:maximum(timepoints) is empty because all timepoints are < 1; pass krd_points explicitly"
        )
    )
    return 1:mt
end

function duration(keyrate::KeyRateDuration, curve, cashflows, timepoints)
    return duration(keyrate, curve, cashflows, timepoints, _default_krd_points(timepoints))
end

function duration(keyrate::KeyRateDuration, curve, cashflows)
    # extract embedded Cashflow times where present; otherwise the index is the time
    timepoints = FinanceCore.timepoint.(cashflows, eachindex(cashflows))
    return duration(keyrate, curve, cashflows, timepoints, _default_krd_points(timepoints))
end

""" 
    spread(curve1,curve2,cashflows)

Return the solved-for constant spread to add to `curve1` in order to equate the discounted `cashflows` with `curve2`

The spread is found via a damped Newton iteration on the pricing residual and is solved to machine precision; an `ErrorException` is thrown if the solve does not converge within `maxiter` iterations.

!!! note
    For mixed-sign cashflows the pricing residual can have more than one exact root (e.g. a duration-neutral asset/liability pair); the root reached from a starting spread of zero is returned.

# Examples

```julia-repl
julia> spread(0.04, 0.05, fill(10.0, 10))
Periodic(0.010000000000000009, 1)
```
"""
function spread(curve1, curve2, cashflows, times = eachindex(cashflows); tol = 1.0e-12, maxiter = 100)
    times = FinanceCore.timepoint.(cashflows, times)
    cashflows = FinanceCore.amount.(cashflows)
    pv2 = FinanceCore.pv(curve2, cashflows, times)

    # Newton + AD on the smooth pricing residual — converges to machine
    # precision in a handful of iterations, vs. the previous derivative-free
    # simplex minimization of the squared residual, whose attainable precision
    # was only ~sqrt of the function tolerance. The step is damped because the
    # residual is not monotone for mixed-sign cashflows: a duration-neutral
    # portfolio has f′(0) ≈ 0, and an undamped step would launch the iterate
    # out of the valid spread domain (s > -1).
    f(s) = FinanceCore.pv(curve1 + FinanceCore.Periodic(s, 1), cashflows, times) - pv2
    ftol = tol * max(one(pv2), abs(pv2))
    max_step = 0.25
    s = 0.0
    fs = f(s)
    converged = abs(fs) < ftol
    iters = 0
    while !converged && iters < maxiter
        d = ForwardDiff.derivative(f, s)
        step = fs / d
        if !isfinite(step) || abs(step) > max_step
            step = isnan(step) ? max_step : copysign(max_step, step)
        end
        s = max(s - step, -0.999)
        fs = f(s)
        converged = abs(fs) < ftol
        iters += 1
    end
    converged || throw(ErrorException("spread did not converge in $maxiter iterations (last residual = $fs)"))
    return FinanceCore.Periodic(s, 1)
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
    has_pos = any(cf -> FinanceCore.amount(cf) > 0, cfs)
    has_neg = any(cf -> FinanceCore.amount(cf) < 0, cfs)
    has_pos && has_neg || throw(
        ArgumentError(
            "moic requires at least one positive (distribution) and one negative (contribution) cashflow"
        )
    )
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
_valuation(cfs, times) = curve -> sum(cf * curve(t) for (cf, t) in zip(cfs, times))

# Two-curve standard valuation (additive on rates → multiplicative on discount factors)
_valuation2(cfs, times) = (base, credit) -> sum(cf * base(t) * credit(t) for (cf, t) in zip(cfs, times))

# ─── Closed-form KRD for the vanilla cashflow case ──────────────────────
#
# When the valuation function is just `Σ cf_k · disc(curve, t_k)`, the
# gradient and Hessian of V(b) w.r.t. the hat-bump vector b are linear /
# quadratic in the hat weights at each cashflow time. The triangular hats
# only support 1–2 pillars per time, so each t_k writes at most 2 entries
# of the gradient and a 2×2 Hessian block. Total work is O(N_cf),
# independent of the number of KRD pillars — no ForwardDiff Dual
# arithmetic over an N-wide partials vector. Numerically equivalent to
# `_keyrate_ad` for these inputs; just much cheaper for typical bond /
# liability cashflow vectors.

# Active hat pair at `t`. Returns (i, w_i, j, w_j) such that the hat sum
# at t equals `w_i * b[i] + w_j * b[j]`. At/beyond the endpoints only one
# hat is active (the other weight is 0 and j == i).
@inline function _active_hats(tenors, t)
    n = length(tenors)
    if t <= first(tenors)
        return 1, one(float(t)), 1, zero(float(t))
    elseif t >= last(tenors)
        return n, one(float(t)), n, zero(float(t))
    else
        i = searchsortedlast(tenors, t)
        w_right = (t - tenors[i]) / (tenors[i + 1] - tenors[i])
        w_left  = one(w_right) - w_right
        return i, w_left, i + 1, w_right
    end
end

# Single- and two-curve analytic KRD are the L = 1 and L = 2 cases of the
# multi-curve kernel `_ncurve_analytic` (just below). For the vanilla
# `Σ cf · ∏ disc` valuation the per-role gradients and all three Hessian blocks
# (base, credit, cross) coincide — see the kernel's note — so the two-curve
# adapter aliases the single shared gradient / Hessian into the base / credit /
# cross names. Downstream callers only broadcast (`./`) these, never mutate them
# in place, so the aliasing is safe and the public two-curve forms still hand
# back distinct output buffers.
_keyrate_analytic(curve::AYM, tenors::AbstractVector, cfs::AbstractVector, times; order = 1) =
    _ncurve_analytic((; curve), tenors, cfs, times; order)

function _keyrate_analytic(base::AYM, credit::AYM, tenors::AbstractVector,
                           cfs::AbstractVector, times; order = 1)
    an = _ncurve_analytic((; base, credit), tenors, cfs, times; order)
    order >= 2 || return (; value = an.value,
                            base_gradient = an.gradient, credit_gradient = an.gradient)
    return (; value = an.value,
              base_gradient = an.gradient, credit_gradient = an.gradient,
              base_hessian  = an.hessian,  credit_hessian = an.hessian, cross_hessian = an.hessian)
end

# N-curve analytic. `curves::NamedTuple{roles}` of L curves with a shared tenor
# grid; the discount is the product ∏_layers disc_layer(t). All roles must be
# discount-role layers (multiplicatively composed); do not pass `:index`.
# Under multiplicative composition every per-role gradient and every (role,
# role) Hessian block carry identical values, so the helper returns a single
# shared gradient vector and a single shared Hessian matrix. The single-, two-,
# and N-curve public wrappers all delegate here and alias these across their
# role positions.
function _ncurve_analytic(curves::NamedTuple, tenors::AbstractVector,
                              cfs::AbstractVector, times; order = 1)
    L = length(curves)
    n = length(tenors)
    T = float(promote_type(eltype(cfs), eltype(times)))
    grad_shared = zeros(T, n)
    hess_shared = order >= 2 ? zeros(T, n, n) : nothing
    V = zero(T)
    @inbounds for k in eachindex(cfs)
        t = times[k]
        # `prod` over the curve tuple is unrolled and type-stable even when the
        # roles have different concrete types (e.g. a ZeroRateCurve base with a
        # flat Constant credit). A `for c in values(curves)` loop would make `c`
        # non-concrete for a heterogeneous tuple and box `discount(c, t)` once
        # per cashflow — an O(N_cf) allocation hit on the two-curve IR01/CS01 path.
        d = prod(c -> FinanceCore.discount(c, t), values(curves))
        cfd = cfs[k] * d
        V += cfd
        i, wi, j, wj = _active_hats(tenors, t)
        grad_shared[i] -= t * cfd * wi
        if i != j
            grad_shared[j] -= t * cfd * wj
        end
        if order >= 2
            tt = t * t * cfd
            hess_shared[i, i] += tt * wi * wi
            if i != j
                ij = tt * wi * wj
                hess_shared[i, j] += ij
                hess_shared[j, i] += ij
                hess_shared[j, j] += tt * wj * wj
            end
        end
    end
    if order >= 2
        return (; value = V, gradient = grad_shared, hessian = hess_shared)
    else
        return (; value = V, gradient = grad_shared)
    end
end

# Normalize the three Hessian blocks of a two-curve AD/analytic result into the
# `(; base, credit, cross)` convexity NamedTuple. Each `./` allocates a fresh
# array, so callers always receive distinct output buffers even when the
# analytic inputs alias a single shared matrix.
_conv_blocks(r) = (; base   = r.base_hessian   ./ r.value,
                     credit = r.credit_hessian ./ r.value,
                     cross  = r.cross_hessian  ./ r.value)

## AbstractYieldModel + KeyRates(tenors): KRD / IR01 / CS01 / convexity / sensitivities
#
# These dispatches accept any `FinanceModels.Yield.AbstractYieldModel`. The
# KRD knot grid is carried by `KeyRates(tenors)`. Internally the AD path layers
# a hat-function zero-rate bump over the user's curve via `Yield.TenorShift`;
# the user's curve is never resampled or rebuilt.
#
# `ZeroRateCurve` inputs go through the same path — it has no special dispatch.
#
# Tenor grid is required (no default) because KRD bucket conventions vary
# (Bloomberg, FRTB, BMA SBA, etc.); downstream should choose explicitly.

"""
    duration(valuation_fn, curve::AbstractYieldModel, tenors) -> scalar
    duration(curve::AbstractYieldModel, tenors, cfs, times) -> scalar
    duration(curve::AbstractYieldModel, tenors, cfs::AbstractVector{<:Cashflow}) -> scalar

Scalar modified duration for any `AbstractYieldModel` evaluated against a KRD
knot grid. Equivalent to `sum(duration(KeyRates(tenors), ...))`.

Use [`KeyRates`](@ref) to obtain the per-knot vector decomposition.

# Example
```julia
duration(pv, my_composite_curve, [0.25, 1, 5, 10, 30])
```
"""
function duration(valuation_fn::Function, curve::AYM, tenors)
    return sum(duration(KeyRates(tenors), valuation_fn, curve))
end
function duration(curve::AYM, tenors, cfs, times)
    return sum(duration(KeyRates(tenors), curve, cfs, times))
end
duration(curve::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(curve, tenors, _extract_cfs_times(cfs)...)

"""
    duration(kr::KeyRates, valuation_fn, curve::AbstractYieldModel) -> Vector
    duration(kr::KeyRates, curve::AbstractYieldModel, cfs, times) -> Vector
    duration(kr::KeyRates, curve::AbstractYieldModel, cfs::AbstractVector{<:Cashflow}) -> Vector

Key-rate durations (modified) for any `AbstractYieldModel`, computed by
layering a triangular-hat zero-rate bump at each tenor in `kr.tenors` over
the user's curve via `Yield.TenorShift`, then taking the AD gradient w.r.t.
the bump magnitudes. The user's curve is preserved at all non-knot points.

# Tenor grid

`kr.tenors` is the KRD knot grid — a separate modeling choice from any
tenor structure baked into the curve itself. You can evaluate key-rate
durations on any grid (e.g. Bloomberg `{0.25, 1, 2, 5, 10, 30}`, FRTB
`{0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30}`, etc.) without re-fitting the
underlying curve.

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

For a linearly-interpolated zero-rate curve the result matches AD over
the curve's own rates exactly. For other splines the bump kernel is
hat-shaped rather than spline-shaped, so per-knot KRDs shift slightly;
the sum of KRDs (= scalar modified duration) is invariant either way.

# Example
```julia
duration(KeyRates([0.25, 1, 5, 10, 30]), pv, curve)

duration(KeyRates([0.25, 1, 5, 10, 30]), curve) do c
    pv(c)
end
```
"""
function duration(kr::KeyRates, valuation_fn::Function, curve::AYM)
    ad = _keyrate_ad(curve, kr.tenors, valuation_fn)
    return -ad.gradient ./ ad.value
end
function duration(kr::KeyRates, curve::AYM, cfs, times)
    an = _keyrate_analytic(curve, kr.tenors, cfs, times)
    return -an.gradient ./ an.value
end
duration(kr::KeyRates, curve::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(kr, curve, _extract_cfs_times(cfs)...)

"""
    duration(::DV01, valuation_fn, curve::AbstractYieldModel, tenors) -> scalar
    duration(::DV01, curve::AbstractYieldModel, tenors, cfs, times) -> scalar
    duration(::DV01, kr::KeyRates, valuation_fn, curve::AbstractYieldModel) -> Vector
    duration(::DV01, kr::KeyRates, curve::AbstractYieldModel, cfs, times) -> Vector

DV01 (scalar or per-knot vector) for any `AbstractYieldModel`. Equivalent to
the `KeyRates` variants of `duration` but in dollars per basis point.
"""
function duration(::DV01, valuation_fn::Function, curve::AYM, tenors)
    return sum(duration(DV01(), KeyRates(tenors), valuation_fn, curve))
end
function duration(::DV01, curve::AYM, tenors, cfs, times)
    return sum(duration(DV01(), KeyRates(tenors), curve, cfs, times))
end
duration(::DV01, curve::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(DV01(), curve, tenors, _extract_cfs_times(cfs)...)

function duration(::DV01, kr::KeyRates, valuation_fn::Function, curve::AYM)
    ad = _keyrate_ad(curve, kr.tenors, valuation_fn)
    return -ad.gradient ./ 10_000
end
function duration(::DV01, kr::KeyRates, curve::AYM, cfs, times)
    an = _keyrate_analytic(curve, kr.tenors, cfs, times)
    return -an.gradient ./ 10_000
end
duration(::DV01, kr::KeyRates, curve::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(DV01(), kr, curve, _extract_cfs_times(cfs)...)

"""
    duration(::IR01, valuation_fn, base::AbstractYieldModel, credit::AbstractYieldModel, tenors) -> scalar
    duration(::IR01, base::AbstractYieldModel, credit::AbstractYieldModel, tenors, cfs, times) -> scalar
    duration(::IR01, kr::KeyRates, valuation_fn, base, credit) -> Vector
    duration(::IR01, kr::KeyRates, base, credit, cfs, times) -> Vector
    duration(::CS01, ...) -> ...

Two-curve IR01/CS01 for any `AbstractYieldModel` pair sharing a tenor
grid. IR01 bumps the base (risk-free) curve only; CS01 bumps the credit
(spread) curve only.
"""
function duration(::IR01, valuation_fn::Function, base::AYM, credit::AYM, tenors)
    return sum(duration(IR01(), KeyRates(tenors), valuation_fn, base, credit))
end
function duration(::IR01, base::AYM, credit::AYM, tenors, cfs, times)
    return sum(duration(IR01(), KeyRates(tenors), base, credit, cfs, times))
end
duration(::IR01, base::AYM, credit::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(IR01(), base, credit, tenors, _extract_cfs_times(cfs)...)

function duration(::IR01, kr::KeyRates, valuation_fn::Function, base::AYM, credit::AYM)
    ad = _keyrate_ad(base, credit, kr.tenors, valuation_fn)
    return -ad.base_gradient ./ 10_000
end
function duration(::IR01, kr::KeyRates, base::AYM, credit::AYM, cfs, times)
    an = _keyrate_analytic(base, credit, kr.tenors, cfs, times)
    return -an.base_gradient ./ 10_000
end
duration(::IR01, kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(IR01(), kr, base, credit, _extract_cfs_times(cfs)...)

function duration(::CS01, valuation_fn::Function, base::AYM, credit::AYM, tenors)
    return sum(duration(CS01(), KeyRates(tenors), valuation_fn, base, credit))
end
function duration(::CS01, base::AYM, credit::AYM, tenors, cfs, times)
    return sum(duration(CS01(), KeyRates(tenors), base, credit, cfs, times))
end
duration(::CS01, base::AYM, credit::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(CS01(), base, credit, tenors, _extract_cfs_times(cfs)...)

function duration(::CS01, kr::KeyRates, valuation_fn::Function, base::AYM, credit::AYM)
    ad = _keyrate_ad(base, credit, kr.tenors, valuation_fn)
    return -ad.credit_gradient ./ 10_000
end
function duration(::CS01, kr::KeyRates, base::AYM, credit::AYM, cfs, times)
    an = _keyrate_analytic(base, credit, kr.tenors, cfs, times)
    return -an.credit_gradient ./ 10_000
end
duration(::CS01, kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(CS01(), kr, base, credit, _extract_cfs_times(cfs)...)

# Do-block-first forwarders (support `f(args...) do x; ...; end` syntax)
duration(vf::Function, kr::KeyRates, curve::AYM)                          = duration(kr,          vf, curve)
duration(vf::Function, ::DV01,       curve::AYM, tenors)                  = duration(DV01(),      vf, curve, tenors)
duration(vf::Function, ::DV01, kr::KeyRates, curve::AYM)                  = duration(DV01(),      kr, vf, curve)
duration(vf::Function, ::IR01, base::AYM, credit::AYM, tenors)            = duration(IR01(),      vf, base, credit, tenors)
duration(vf::Function, ::IR01, kr::KeyRates, base::AYM, credit::AYM)      = duration(IR01(),      kr, vf, base, credit)
duration(vf::Function, ::CS01, base::AYM, credit::AYM, tenors)            = duration(CS01(),      vf, base, credit, tenors)
duration(vf::Function, ::CS01, kr::KeyRates, base::AYM, credit::AYM)      = duration(CS01(),      kr, vf, base, credit)

"""
    convexity(valuation_fn, curve::AbstractYieldModel, tenors) -> scalar
    convexity(curve::AbstractYieldModel, tenors, cfs, times) -> scalar
    convexity(kr::KeyRates, valuation_fn, curve::AbstractYieldModel) -> Matrix
    convexity(kr::KeyRates, curve::AbstractYieldModel, cfs, times) -> Matrix
    convexity(base::AbstractYieldModel, credit::AbstractYieldModel, tenors, cfs, times) -> NamedTuple
    convexity(kr::KeyRates, base, credit, cfs, times) -> NamedTuple
    convexity(kr::KeyRates, curves::NamedTuple, cfs, times) -> NamedTuple{roles}{roles}

Key-rate convexity (matrix) and scalar convexity for any `AbstractYieldModel`,
pair, or named tuple of discount-role curves. Mirrors `duration` but returns
∂²V/∂rᵢ∂rⱼ rather than ∂V/∂rᵢ.

For the `NamedTuple` form, every named curve must be a discount-role layer
(multiplicatively composed); do not pass `:index`. Per-role and per-pair
outputs alias a single shared matrix — values coincide by construction
under multiplicative composition. `copy` if you need independent buffers.

The scalar forms (first two signatures) return the parallel-shift second
derivative ∂²V/∂s² under a *continuous-rate* shock — matching the matrix
forms exactly under partition of unity of the KRD hats. `tenors` is accepted
for API symmetry but is not used by the scalar derivative computation.

If you also want the durations / DV01s, prefer [`sensitivities`](@ref) — it returns
the value, gradient, and Hessian from one AD pass at the same cost.
"""
# Continuous-shock parallel-shift convexity via a single scalar second
# derivative. Under partition of unity of the KRD hat functions (`_hat_bump`
# above), `sum(convexity(KeyRates(tenors), …))` equals ∂²V/∂s² for parallel
# shift `s` by the chain rule — the matrix path returns the right number but
# pays O(N² AD work + dense Hessian allocation) for what is an O(1) scalar
# second derivative. This helper performs the scalar derivative directly on a
# `TenorShift`-bumped curve, matching the matrix-sum form exactly while
# avoiding the per-pillar Hessian.
function _parallel_continuous_convexity(curve::AYM, valuation_fn)
    bumped(s) = FinanceModels.Yield.TenorShift(curve, (z, t) -> z + FinanceCore.Continuous(s))
    v(s) = abs(valuation_fn(bumped(s)))
    ∂²V = ForwardDiff.derivative(s2 -> ForwardDiff.derivative(v, s2), 0.0)
    return ∂²V / v(0.0)
end

convexity(valuation_fn::Function, curve::AYM, _tenors) =
    _parallel_continuous_convexity(curve, valuation_fn)
convexity(curve::AYM, _tenors, cfs, times) =
    _parallel_continuous_convexity(curve, c -> sum(_cf_value(cfs[k]) * FinanceCore.discount(c, times[k]) for k in eachindex(cfs)))
convexity(curve::AYM, _tenors, cfs::AbstractVector{<:FinanceCore.Cashflow}) =
    convexity(curve, _tenors, _extract_cfs_times(cfs)...)

function convexity(kr::KeyRates, valuation_fn::Function, curve::AYM)
    ad = _keyrate_ad(curve, kr.tenors, valuation_fn; order = 2)
    return ad.hessian ./ ad.value
end
function convexity(kr::KeyRates, curve::AYM, cfs, times)
    an = _keyrate_analytic(curve, kr.tenors, cfs, times; order = 2)
    return an.hessian ./ an.value
end
convexity(kr::KeyRates, curve::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = convexity(kr, curve, _extract_cfs_times(cfs)...)

function convexity(valuation_fn::Function, base::AYM, credit::AYM, tenors)
    cv = convexity(KeyRates(tenors), valuation_fn, base, credit)
    return (; base = sum(cv.base), credit = sum(cv.credit), cross = sum(cv.cross))
end
function convexity(base::AYM, credit::AYM, tenors, cfs, times)
    # static cashflows: the analytic helper computes the same blocks as the
    # (2n)×(2n) ForwardDiff Hessian the do-block form pays for, in O(N_cf)
    an = _keyrate_analytic(base, credit, tenors, cfs, times; order = 2)
    return (;
        base = sum(an.base_hessian) / an.value,
        credit = sum(an.credit_hessian) / an.value,
        cross = sum(an.cross_hessian) / an.value,
    )
end
convexity(base::AYM, credit::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow}) = convexity(base, credit, tenors, _extract_cfs_times(cfs)...)

function convexity(kr::KeyRates, valuation_fn::Function, base::AYM, credit::AYM)
    ad = _keyrate_ad(base, credit, kr.tenors, valuation_fn; order = 2)
    return _conv_blocks(ad)
end
function convexity(kr::KeyRates, base::AYM, credit::AYM, cfs, times)
    an = _keyrate_analytic(base, credit, kr.tenors, cfs, times; order = 2)
    return _conv_blocks(an)
end
convexity(kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = convexity(kr, base, credit, _extract_cfs_times(cfs)...)

# Multi-curve NamedTuple cashflow form. Per-role and per-role-pair convexities
# for static cashflows. All L² blocks alias one shared N×N matrix — values
# coincide by construction under multiplicative discount composition.
function convexity(kr::KeyRates, curves::NamedTuple, cfs, times)
    an = _ncurve_analytic(curves, kr.tenors, cfs, times; order = 2)
    roles = keys(curves)
    L = length(roles)
    normalized = an.hessian ./ an.value
    return NamedTuple{roles}(ntuple(_ -> NamedTuple{roles}(ntuple(_ -> normalized, L)), L))
end
convexity(kr::KeyRates, curves::NamedTuple, cfs::AbstractVector{<:FinanceCore.Cashflow}) =
    convexity(kr, curves, _extract_cfs_times(cfs)...)

# Do-block-first forwarders (support `f(args...) do x; ...; end` syntax)
convexity(vf::Function, kr::KeyRates, curve::AYM)                 = convexity(kr, vf, curve)
convexity(vf::Function, kr::KeyRates, base::AYM, credit::AYM)     = convexity(kr, vf, base, credit)

"""
    sensitivities(kr::KeyRates, valuation_fn, curve::AbstractYieldModel) -> NamedTuple
    sensitivities(kr::KeyRates, curve::AbstractYieldModel, cfs, times) -> NamedTuple
    sensitivities(::DV01, kr::KeyRates, valuation_fn, curve::AbstractYieldModel) -> NamedTuple
    sensitivities(kr::KeyRates, base::AbstractYieldModel, credit::AbstractYieldModel, cfs, times) -> NamedTuple
    sensitivities(::DV01, kr::KeyRates, base, credit, cfs, times) -> NamedTuple

Bundled value + key-rate durations (or DV01s) + convexity matrix for any
`AbstractYieldModel` or pair, in a single AD pass. The knot grid is carried
by [`KeyRates`](@ref).
"""
function sensitivities(kr::KeyRates, valuation_fn::Function, curve::AYM)
    ad = _keyrate_ad(curve, kr.tenors, valuation_fn; order = 2)
    return (;
        value = ad.value,
        durations = -ad.gradient ./ ad.value,
        convexities = ad.hessian ./ ad.value,
    )
end
function sensitivities(kr::KeyRates, curve::AYM, cfs, times)
    an = _keyrate_analytic(curve, kr.tenors, cfs, times; order = 2)
    return (;
        value       = an.value,
        durations   = -an.gradient ./ an.value,
        convexities = an.hessian ./ an.value,
    )
end
sensitivities(kr::KeyRates, curve::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = sensitivities(kr, curve, _extract_cfs_times(cfs)...)

function sensitivities(::DV01, kr::KeyRates, valuation_fn::Function, curve::AYM)
    ad = _keyrate_ad(curve, kr.tenors, valuation_fn; order = 2)
    return (;
        value = ad.value,
        dv01s = -ad.gradient ./ 10_000,
        convexities = ad.hessian ./ ad.value,
    )
end
function sensitivities(::DV01, kr::KeyRates, curve::AYM, cfs, times)
    an = _keyrate_analytic(curve, kr.tenors, cfs, times; order = 2)
    return (;
        value       = an.value,
        dv01s       = -an.gradient ./ 10_000,
        convexities = an.hessian ./ an.value,
    )
end
sensitivities(::DV01, kr::KeyRates, curve::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = sensitivities(DV01(), kr, curve, _extract_cfs_times(cfs)...)

function sensitivities(kr::KeyRates, valuation_fn::Function, base::AYM, credit::AYM)
    ad = _keyrate_ad(base, credit, kr.tenors, valuation_fn; order = 2)
    return (;
        value = ad.value,
        base_durations = -ad.base_gradient ./ ad.value,
        credit_durations = -ad.credit_gradient ./ ad.value,
        convexities = _conv_blocks(ad),
    )
end
function sensitivities(kr::KeyRates, base::AYM, credit::AYM, cfs, times)
    an = _keyrate_analytic(base, credit, kr.tenors, cfs, times; order = 2)
    return (;
        value             = an.value,
        base_durations    = -an.base_gradient   ./ an.value,
        credit_durations  = -an.credit_gradient ./ an.value,
        convexities       = _conv_blocks(an),
    )
end
sensitivities(kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = sensitivities(kr, base, credit, _extract_cfs_times(cfs)...)

# Multi-curve NamedTuple cashflow form. One AD-free pass returns per-role
# durations + per-role-pair N×N convexity blocks. All per-role durations
# alias one shared vector; all L² Hessian blocks alias one shared matrix —
# values coincide by construction under multiplicative discount composition.
function sensitivities(kr::KeyRates, curves::NamedTuple, cfs, times)
    an = _ncurve_analytic(curves, kr.tenors, cfs, times; order = 2)
    roles = keys(curves)
    L = length(roles)
    dur_normalized  = -an.gradient ./ an.value
    conv_normalized =  an.hessian  ./ an.value
    durations   = NamedTuple{roles}(ntuple(_ -> dur_normalized, L))
    convexities = NamedTuple{roles}(ntuple(_ -> NamedTuple{roles}(ntuple(_ -> conv_normalized, L)), L))
    return (; value = an.value, durations, convexities)
end
sensitivities(kr::KeyRates, curves::NamedTuple, cfs::AbstractVector{<:FinanceCore.Cashflow}) =
    sensitivities(kr, curves, _extract_cfs_times(cfs)...)

function sensitivities(::DV01, kr::KeyRates, valuation_fn::Function, base::AYM, credit::AYM)
    ad = _keyrate_ad(base, credit, kr.tenors, valuation_fn; order = 2)
    return (;
        value = ad.value,
        base_dv01s = -ad.base_gradient ./ 10_000,
        credit_dv01s = -ad.credit_gradient ./ 10_000,
        convexities = _conv_blocks(ad),
    )
end
function sensitivities(::DV01, kr::KeyRates, base::AYM, credit::AYM, cfs, times)
    an = _keyrate_analytic(base, credit, kr.tenors, cfs, times; order = 2)
    return (;
        value        = an.value,
        base_dv01s   = -an.base_gradient   ./ 10_000,
        credit_dv01s = -an.credit_gradient ./ 10_000,
        convexities  = _conv_blocks(an),
    )
end
sensitivities(::DV01, kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = sensitivities(DV01(), kr, base, credit, _extract_cfs_times(cfs)...)

# Do-block-first forwarders (support `f(args...) do x; ...; end` syntax)
sensitivities(vf::Function, kr::KeyRates, curve::AYM)                    = sensitivities(kr, vf, curve)
sensitivities(vf::Function, ::DV01, kr::KeyRates, curve::AYM)            = sensitivities(DV01(), kr, vf, curve)
sensitivities(vf::Function, kr::KeyRates, base::AYM, credit::AYM)        = sensitivities(kr, vf, base, credit)
sensitivities(vf::Function, ::DV01, kr::KeyRates, base::AYM, credit::AYM) = sensitivities(DV01(), kr, vf, base, credit)

## ── Contract / portfolio-aware, re-projecting duration & sensitivities ────────
#
# A contract (or a vector of contracts = a portfolio) is a "target": it is valued
# by RE-PROJECTING under a bumped curve, so a floater's coupons re-fix automatically
# (its projection reads the model) while a fixed bond's do not. `_contract_keys`
# (empty vs not) is the only discriminator — no `ValuationStyle` trait. The risk
# factor stays the familiar vocabulary: `Effective` (rate) / `Spread` (credit) /
# `KeyRates`; units are the verb (`duration` yrs / `dv01` $ / `convexity`).

_contract_keys(c::FinanceModels.Bond.Floating) = (c.key,)
_contract_keys(c::FinanceCore.Composite)        = (_contract_keys(c.a)..., _contract_keys(c.b)...)
_contract_keys(c::FinanceModels.Forward)        = _contract_keys(c.instrument)
_contract_keys(::FinanceCore.AbstractContract)  = ()

const _Contractish = Union{FinanceCore.AbstractContract, AbstractVector{<:FinanceCore.AbstractContract}}

"""
    reproject(contract, index_curve)

Wrap `contract` so its coupons are estimated off `index_curve`: returns the contract
itself if it reads no model, else a `Projection` mapping the contract's keys to
`index_curve`. Lets a multi-curve valuation avoid hand-writing the model `Dict`.
"""
reproject(c::FinanceCore.AbstractContract, index) =
    isempty(_contract_keys(c)) ? c :
    FinanceModels.Projection(c, Dict(k => index for k in _contract_keys(c)), FinanceModels.CashflowProjection())

# value of a contract/portfolio under a single curve (coupons + discount = curve) …
_cvalue(c::FinanceCore.AbstractContract, curve) = FinanceCore.present_value(curve, reproject(c, curve))
_cvalue(cs::AbstractVector{<:FinanceCore.AbstractContract}, curve) = sum(_cvalue(c, curve) for c in cs)
# … and two curves: estimate coupons on `fwd`, discount on `credit`.
_cvalue2(c::FinanceCore.AbstractContract, fwd, credit) = FinanceCore.present_value(credit, reproject(c, fwd))
_cvalue2(cs::AbstractVector{<:FinanceCore.AbstractContract}, fwd, credit) = sum(_cvalue2(c, fwd, credit) for c in cs)

"""
    sensitivities(target, curve, tenors) -> NamedTuple
    sensitivities(target, forward, credit, tenors) -> NamedTuple

One-AD-pass bundle for a (possibly curve-dependent) `target` — a `FinanceModels`
contract or a vector of contracts (portfolio) — re-projecting cashflows under bumped
curves. Coupons are estimated on `forward`, discounted on `credit` (pass a single
`curve` for both). Returns, over the `tenors` key-rate grid:

  - `value`
  - `effective_duration` / `effective_dv01` / `effective_key_rate` — bump both curves
    (coupons re-fix): the interest-rate duration (≈ next reset for a floater).
  - `spread_duration` / `spread_dv01` / `spread_key_rate` — bump the discount only:
    the discount-margin / credit duration (≈ maturity for a floater).
  - `forward_duration` / `forward_dv01` / `forward_key_rate` — bump the index only;
    `effective = forward + spread` (first order).

Durations in years; DV01s in dollars per 1bp. For a fixed bond `effective == spread ==`
the modified duration and `forward == 0`. See [`duration`](@ref) with [`Effective`](@ref)/
[`Spread`](@ref), [`dv01`](@ref), [`zspread`](@ref), [`locked_floater`](@ref).
"""
function sensitivities(target::_Contractish, forward::AYM, credit::AYM, tenors)
    s = sensitivities(KeyRates(tenors), (f, c) -> _cvalue2(target, f, c), forward, credit)
    v = s.value
    eff = s.base_durations .+ s.credit_durations
    spr = s.credit_durations
    fwd = s.base_durations
    return (; value = v,
        effective_duration = sum(eff), effective_dv01 = sum(eff) * v / 10_000, effective_key_rate = eff,
        spread_duration    = sum(spr), spread_dv01    = sum(spr) * v / 10_000, spread_key_rate    = spr,
        forward_duration   = sum(fwd), forward_dv01   = sum(fwd) * v / 10_000, forward_key_rate   = fwd)
end
sensitivities(target::_Contractish, curve::AYM, tenors) = sensitivities(target, curve, curve, tenors)

"""
    duration(Effective(), target, curve, tenors)          # rate duration, yrs
    duration(Spread(),    target, curve, tenors)          # spread duration, yrs
    duration(Effective(), KeyRates(tenors), target, curve) # key-rate vector
    dv01(Effective()/Spread(), target, curve, tenors)     # the dollar versions

Effective (rate) and spread (credit) duration / DV01 for a contract or portfolio,
re-projecting cashflows under bumped curves. Two-curve forms take `(forward, credit)`.
See [`sensitivities`](@ref) for the full one-pass bundle.
"""
duration(::Effective, target::_Contractish, forward::AYM, credit::AYM, tenors) = sensitivities(target, forward, credit, tenors).effective_duration
duration(::Effective, target::_Contractish, curve::AYM, tenors) = duration(Effective(), target, curve, curve, tenors)
duration(::Spread,    target::_Contractish, forward::AYM, credit::AYM, tenors) = sensitivities(target, forward, credit, tenors).spread_duration
duration(::Spread,    target::_Contractish, curve::AYM, tenors) = duration(Spread(), target, curve, curve, tenors)
duration(::Effective, kr::KeyRates, target::_Contractish, curve::AYM) = sensitivities(target, curve, kr.tenors).effective_key_rate
duration(::Spread,    kr::KeyRates, target::_Contractish, curve::AYM) = sensitivities(target, curve, kr.tenors).spread_key_rate
# default (no marker) on a contract/portfolio = effective
duration(target::_Contractish, curve::AYM, tenors) = duration(Effective(), target, curve, tenors)
duration(kr::KeyRates, target::_Contractish, curve::AYM) = duration(Effective(), kr, target, curve)

# Effective convexity: parallel-shift second derivative of the contract's
# present value under a continuous-rate shock. Routes through the O(1) scalar
# helper above (`_parallel_continuous_convexity`), which is numerically
# equivalent to the prior `sum(convexity(KeyRates(tenors), …))` matrix-sum form
# under partition of unity but avoids the O(N²) Hessian.
convexity(::Effective, target::_Contractish, curve::AYM, _tenors) =
    _parallel_continuous_convexity(curve, c -> _cvalue(target, c))

"""
    dv01(args...)

Dollar value of a 1bp move. `dv01(args...)` ≡ `duration(DV01(), args...)` for the
cashflow/curve forms, with `dv01(Effective()/Spread(), target, [forward, credit,] tenors)`
giving the floating-rate dollar durations (years × value ÷ 10⁴).
"""
dv01(::Effective, target::_Contractish, forward::AYM, credit::AYM, tenors) = sensitivities(target, forward, credit, tenors).effective_dv01
dv01(::Effective, target::_Contractish, curve::AYM, tenors) = dv01(Effective(), target, curve, curve, tenors)
dv01(::Spread,    target::_Contractish, forward::AYM, credit::AYM, tenors) = sensitivities(target, forward, credit, tenors).spread_dv01
dv01(::Spread,    target::_Contractish, curve::AYM, tenors) = dv01(Spread(), target, curve, curve, tenors)
dv01(args...; kwargs...) = duration(DV01(), args...; kwargs...)

# ── Multi-curve: N named curves, per-role sensitivities in one AD pass ─────────
function _ncurve_ad(valuation, curves::NamedTuple, tenors)
    roles = keys(curves); k = length(roles); n = length(tenors)
    f(B) = valuation(NamedTuple{roles}(ntuple(i -> _bumped(curves[i], tenors, view(B, (i-1)*n+1:i*n)), k)))
    z = zeros(k * n)
    v = f(z)
    g = ForwardDiff.gradient(f, z)
    return v, NamedTuple{roles}(ntuple(i -> g[(i-1)*n+1:i*n], k))
end

"""
    sensitivities(valuation, curves::NamedTuple; tenors) -> (; value, duration, dv01, key_rate)
    sensitivities(target, tenors; discount::NamedTuple, index) -> same

Multi-curve sensitivities: differentiate `valuation(curves)` w.r.t. each named curve
in `curves` in a single AD pass, returning a per-role `duration`/`dv01`/`key_rate`
NamedTuple. The structured form assembles `discount = sum(discount layers)` and projects
the contract's coupons on `index` — e.g. `discount = (; rf, credit, ilp)` gives `r.duration.rf`
(≈ IR01), `.credit` (≈ CS01), `.ilp` ("ILP01"), and `.index` (the reset sensitivity). ILP /
matching-adjustment / basis are just additional named curves.
"""
function sensitivities(valuation, curves::NamedTuple; tenors)
    v, grads = _ncurve_ad(valuation, curves, tenors)
    roles = keys(curves)
    return (; value = v,
        duration = NamedTuple{roles}(map(g -> -sum(g) / v, values(grads))),
        dv01     = NamedTuple{roles}(map(g -> -sum(g) / 10_000, values(grads))),
        key_rate = NamedTuple{roles}(map(g -> -g ./ v, values(grads))))
end
sensitivities(curves::NamedTuple, valuation::Function; tenors) = sensitivities(valuation, curves; tenors)  # do-block form
function sensitivities(target::_Contractish, tenors::AbstractVector; discount::NamedTuple, index)
    layers = keys(discount)
    return sensitivities(merge(discount, (; index = index)); tenors) do c
        _cvalue2(target, c.index, reduce(+, getfield(c, r) for r in layers))
    end
end

"""
    zspread(contract, credit, market_price; forward=credit) -> (; zspread, zspread_dv01)

Constant continuously-compounded spread `s` on the `credit` (discount) curve such that
the model price equals `market_price`, with coupons estimated on `forward` (held fixed).
Returns the spread and its sensitivity (\\\$/1bp parallel move of `credit + s`). Newton + AD.
"""
function zspread(contract::FinanceCore.AbstractContract, credit::AYM, market_price; forward::AYM = credit, s0 = 0.0, tol = 1e-12, maxiter = 100)
    ks = _contract_keys(contract)
    pvs(s) = let disc = credit + ((z, t) -> z + FinanceCore.Continuous(s))
        isempty(ks) ? FinanceCore.present_value(disc, contract) :
            FinanceCore.present_value(disc, FinanceModels.Projection(contract, Dict(k => forward for k in ks), FinanceModels.CashflowProjection()))
    end
    f(s) = pvs(s) - market_price
    s = float(s0)
    converged = false
    for _ in 1:maxiter
        fs = f(s)
        (abs(fs) < tol) && (converged = true; break)
        d = ForwardDiff.derivative(f, s)
        iszero(d) && break
        s -= fs / d
    end
    converged || throw(ErrorException("zspread did not converge (last residual = $(f(s)))"))
    return (; zspread = s, zspread_dv01 = -ForwardDiff.derivative(pvs, s) / 10_000)
end

"""
    locked_floater(fl::FinanceModels.Bond.Floating, current_coupon, next_reset)

Model an in-force floater whose current coupon is LOCKED at `current_coupon` (the
per-period amount fixed at the last reset) until `next_reset`, after which it floats.
A `Composite` of a coupon-only stub at `next_reset` plus a forward-starting floater for
the remainder (principal rides the forward leg). Gives the conventional rate duration
≈ time to next reset; without it the idealized effective duration is ≈ 0 at a reset.

The remaining term `fl.maturity - next_reset` must be an integer number of coupon
periods. Otherwise the forward-starting leg would carry a stub first coupon whose
fix-in-advance reference rate looks back before time zero — a quietly mispriced
quantity on curves that extrapolate below ``t = 0`` and a `DomainError` on
`ZeroRateCurve` — so non-commensurate inputs throw an `ArgumentError`.
"""
function locked_floater(fl::FinanceModels.Bond.Floating, current_coupon, next_reset)
    freq = fl.frequency.frequency
    n_periods = (fl.maturity - next_reset) * freq
    isapprox(n_periods, round(n_periods); atol = 1.0e-8) || throw(
        ArgumentError(
            "locked_floater requires fl.maturity - next_reset ($(fl.maturity - next_reset)) to be an integer number of coupon periods (frequency $freq); a stub first coupon on the forward leg would reference a forward rate starting before time zero"
        )
    )
    stub = FinanceCore.Cashflow(current_coupon, next_reset)
    rest = FinanceModels.Forward(next_reset,
        FinanceModels.Bond.Floating(fl.coupon_rate, fl.frequency, fl.maturity - next_reset, fl.key))
    return FinanceCore.Composite(stub, rest)
end

## Hull-White convenience methods
#
# `hw.curve` can be any `AbstractYieldModel` — the AD path uses TenorShift
# bumps over the curve via the AYM-based `sensitivities` impls above.

const HW = FinanceModels.ShortRate.HullWhite

# Rebuild HW under a perturbed curve and produce its scenario set under the same dynamics.
function _hw_paths(hw::HW, curve; n_scenarios, timestep, horizon, rng)
    hw_new = FinanceModels.ShortRate.HullWhite(hw.a, hw.σ, curve)
    return FinanceModels.simulate(hw_new; n_scenarios, timestep, horizon, rng)
end

# Do-block primary forms
#
# Pathwise seeding: every AD evaluation of the inner closure must see the same
# MC sample, otherwise `KRD = -∇V/V` divides a gradient computed over one
# sample by a value computed over another (ForwardDiff calls the closure many
# times for value, gradient chunks, and Hessian chunks). Snapshot a UInt64 from
# the user's rng once per call and rebuild a fresh `Xoshiro(seed)` inside the
# closure so every AD step draws the same scenarios.
function sensitivities(kr::KeyRates, valuation_fn::Function, hw::HW;
                       n_scenarios=1000, timestep=1/12, horizon=30.0,
                       rng=Random.default_rng())
    seed = rand(rng, UInt64)
    sensitivities(kr, hw.curve) do curve
        valuation_fn(_hw_paths(hw, curve; n_scenarios, timestep, horizon, rng=Random.Xoshiro(seed)))
    end
end

function sensitivities(::DV01, kr::KeyRates, valuation_fn::Function, hw::HW;
                       n_scenarios=1000, timestep=1/12, horizon=30.0,
                       rng=Random.default_rng())
    seed = rand(rng, UInt64)
    sensitivities(DV01(), kr, hw.curve) do curve
        valuation_fn(_hw_paths(hw, curve; n_scenarios, timestep, horizon, rng=Random.Xoshiro(seed)))
    end
end

# Do-block-first forwarders (support `f(args...) do x; ...; end` syntax)
sensitivities(vf::Function, kr::KeyRates, hw::HW; kw...)         = sensitivities(kr, vf, hw; kw...)
sensitivities(vf::Function, ::DV01, kr::KeyRates, hw::HW; kw...) = sensitivities(DV01(), kr, vf, hw; kw...)

# Cashflow-form wrappers that delegate to the do-block forms above
function sensitivities(kr::KeyRates, hw::HW, cfs::AbstractVector, times;
                       n_scenarios=1000, timestep=1/12, horizon=nothing,
                       rng=Random.default_rng())
    h = horizon === nothing ? maximum(times) + 1.0 : Float64(horizon)
    sensitivities(kr, hw; n_scenarios, timestep, horizon=h, rng) do scenarios
        sum(FinanceCore.pv(sc, cfs, times) for sc in scenarios) / n_scenarios
    end
end

function sensitivities(::DV01, kr::KeyRates, hw::HW, cfs::AbstractVector, times;
                       n_scenarios=1000, timestep=1/12, horizon=nothing,
                       rng=Random.default_rng())
    h = horizon === nothing ? maximum(times) + 1.0 : Float64(horizon)
    sensitivities(DV01(), kr, hw; n_scenarios, timestep, horizon=h, rng) do scenarios
        sum(FinanceCore.pv(sc, cfs, times) for sc in scenarios) / n_scenarios
    end
end

end
