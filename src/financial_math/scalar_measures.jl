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
struct KeyRates{T <: AbstractVector{<:Real}} <: Duration
    tenors::T
    function KeyRates(tenors::T) where {T <: AbstractVector{<:Real}}
        isempty(tenors)   && throw(ArgumentError("KeyRates tenors must be non-empty"))
        issorted(tenors)  || throw(ArgumentError("KeyRates tenors must be sorted ascending"))
        allunique(tenors) || throw(ArgumentError("KeyRates tenors must be distinct"))
        all(t -> isfinite(t) && t > 0, tenors) || throw(ArgumentError("KeyRates tenors must be finite and strictly positive"))
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
    return _macaulay_ratio(yield, vec(cfs), times)
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
    # A single-knot tent is the modern `_hat_bump` kernel (defined alongside the
    # AD KRD path below) evaluated at a unit bump vector for this knot — one
    # source of truth for the key-rate hat shape across the legacy
    # bump-and-reprice path and the AD path. The flat extrapolation `_hat_bump`
    # applies beyond the first / last knot reproduces the original first-point
    # (flat left) and last-point (flat right) tent behavior. Numerically
    # identical to the prior explicit tent (max abs deviation ~7e-18 over a knot
    # / t sweep, from float associativity in the ramp).
    bumps = [k == idx ? shift : zero(shift) for k in eachindex(krd_points)]
    return (z, t) -> z + FinanceCore.Continuous(_hat_bump(krd_points, bumps, t))
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
