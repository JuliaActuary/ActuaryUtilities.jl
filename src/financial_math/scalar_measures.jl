"""
    present_values(interest, cashflows, timepoints)

Return the value of remaining cashflows before each payment period.
Entry `k` values cashflows `k:end` at `timepoints[k-1]`, or time zero for `k = 1`.

Empty collections return an empty vector. Collections whose amounts are all
exactly zero return a vector of positive zeros without evaluating the curve;
the element type comes from the amounts and timepoints.
Every cashflow requires a time; additional trailing times are ignored.

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
    _check_cashflow_times(cashflows, times)
    n = length(cashflows)
    _iszero_cashflow_stream(cashflows) && return zeros(typeof(_zero_cashflow_value(cashflows, times)), n)
    # Discount backward in one pass; derive the accumulator type from valuation.
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

Return `abs(present_value(...))`. Use `present_value` when the position sign matters.
"""
price(x1, x2) = FinanceCore.present_value(x1, x2) |> abs
price(x1, x2, x3) = FinanceCore.present_value(x1, x2, x3) |> abs

"""
    breakeven(yield, cashflows::Vector)
    breakeven(yield, cashflows::Vector,times::Vector)

Return the payment time when accumulated cashflows break even at the given yield.

Assumptions:

- cashflows occur at the end of the period
- cashflows are evenly spaced from time zero if `times` is omitted

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

    # Resolve embedded amounts and times for Cashflow inputs.
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

Signed dollar risk for a one-basis-point (0.01%) parallel rate shift:
`DV01 = -∂V/∂r / 10000`. A positive DV01 of 0.045 means a 1bp rate increase
reduces the position's value by approximately 0.045, in the cashflows' currency units.

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

Measure contract risk while reprojecting cashflows under shifted curves, so
floating coupons reset. Use `duration(Effective(), contract, curve, tenors)`;
the same marker applies to `dv01` and `convexity`. `Modified` and `Macaulay`
operate on fixed cashflows.

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

Select the tenor grid for key-rate [`duration`](@ref), [`convexity`](@ref), and
[`sensitivities`](@ref). Results contain per-tenor vectors and convexity matrices.
`tenors` must be a nonempty `AbstractVector{<:Real}` of finite, positive, strictly
increasing knot times in years.

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
        _validate_tenors(tenors)
        return new{T}(tenors)
    end
end

function _validate_tenors(tenors::AbstractVector{<:Real})
    isempty(tenors) && throw(ArgumentError("KeyRates tenors must be non-empty"))
    # Strict increase establishes sortedness and uniqueness in one pass.
    previous = zero(first(tenors))
    for t in tenors
        isfinite(t) && t > previous || throw(ArgumentError("KeyRates tenors must be finite, strictly positive, and strictly increasing"))
        previous = t
    end
    return tenors
end

abstract type KeyRateDuration <: Duration end


"""
    KeyRatePar(timepoint,shift=0.001) <: KeyRateDuration

Select a par-rate bump at `timepoint` for `duration`. The calculation refits the
curve on `krd_points` after upward and downward bumps of size `shift`, then uses
central differences.

"""
struct KeyRatePar{T, R} <: KeyRateDuration
    timepoint::T
    shift::R
    KeyRatePar(timepoint, shift = 0.001) = new{typeof(timepoint), typeof(shift)}(timepoint, shift)
end

"""
    KeyRateZero(timepoint,shift=0.001) <: KeyRateDuration

Select a triangular continuous-zero bump at `timepoint` for `duration`.
The calculation uses central differences with upward and downward bumps of size
`shift`. Neighboring `krd_points` define the bump width; endpoint bumps extend flat
beyond the grid.
"""
struct KeyRateZero{T, R} <: KeyRateDuration
    timepoint::T
    shift::R
    KeyRateZero(timepoint, shift = 0.001) = new{typeof(timepoint), typeof(shift)}(timepoint, shift)
end

"""
    KeyRate(timepoints,shift=0.001)

Alias for [`KeyRateZero`](@ref).
"""
const KeyRate = KeyRateZero

# Cashflow routes accept the yield inputs supported by present_value; this
# distinguishes them from metric-first and callable-valuation signatures.
const _YieldInput = Union{Real, FinanceCore.Rate, FinanceModels.Yield.AbstractYieldModel}
const _CashflowCollection = Union{AbstractArray, Tuple, Base.Generator}

# Indexed kernels share one representation; materialize generators before AD
# reevaluates a valuation, including generators backed by a stateful iterator.
_cashflow_vector(cfs::AbstractArray) = vec(cfs)
# Empty tuples collect to Union{}[], whose element type also matches Cashflow.
# Use the zero-stream fallback type before dispatch derives embedded times.
_cashflow_vector(::AbstractArray{Union{}}) = Float64[]
_cashflow_vector(cfs::Union{Tuple, Base.Generator}) = _cashflow_vector(collect(cfs))

"""
    duration(Macaulay(),interest_rate,cfs,times)
    duration(Modified(),interest_rate,cfs,times)
    duration(DV01(),interest_rate,cfs,times)
    duration(IR01(),base_curve,credit_spread,cfs,times)
    duration(CS01(),base_curve,credit_spread,cfs,times)
    duration(interest_rate,cfs,times)             # Modified Duration
    duration(interest_rate,valuation_function)    # Modified Duration

Calculate Macaulay or modified duration, or signed dollar DV01, IR01, or CS01.
For numeric amounts, omitted `times` default to `1:length(cfs)`.

`cfs` can be an `AbstractVector{<:Cashflow}` (from FinanceCore), in which case
`times` may be omitted. A `Cashflow` always supplies its embedded amount and time;
explicit `times` supply payment times for numeric amounts. If supplied, `times`
must still contain an entry for each cashflow; unused trailing entries are ignored.

Scalar cashflow methods accept arrays, tuples, and finite generators. Arrays are
flattened in column-major order; generators are collected once before valuation.
Use `collect` for other iterables, such as `Iterators.take` or `skipmissing`.
Relative duration is unchanged when the position sign reverses; dollar DV01,
IR01, and CS01 reverse sign with the position.

Empty collections and collections whose amounts are all exactly zero return zero
risk without evaluating the curve. Every cashflow needs a time; unused trailing
times are ignored. See [Zero cashflow streams](@ref) for the normalization convention,
numeric types, and zero-net-value portfolios. Dollar sensitivities differentiate
the signed value directly, including at zero present value; normalized duration
remains undefined there. A zero callback value alone does not identify a zero stream.

The default measure is `Modified()`.

- Modified duration: the relative change per point of yield change.
- Macaulay: the present-value-weighted average payment time.
- DV01: the signed dollar change per basis point (hundredth of a percentage point), defined as `-∂V/∂r / 10000`.
- IR01: the signed dollar change per basis point shift in the risk-free (base) curve, holding credit spread constant.
- CS01: the signed dollar change per basis point shift in the credit spread, holding the risk-free (base) curve constant.

# Periodicity convention

Modified duration depends on the shock's compounding convention:

- Scalars use annual compounding: Modified = Macaulay / (1 + y).
- `Periodic(y, m)` uses Modified = Macaulay / (1 + y/m).
- `Continuous(y)` and yield models use continuous-zero shifts: Modified = Macaulay.

Wrapping a scalar in `Yield.Constant` preserves its discount factors but changes
the shock coordinate:

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
function duration(::Macaulay, yield::_YieldInput, cfs::_CashflowCollection, times)
    return _macaulay_ratio(yield, _cashflow_vector(cfs), times)
end

duration(d::Modified, yield::_YieldInput, cfs::_CashflowCollection, times) =
    duration(d, yield, _cashflow_vector(cfs), times)

function duration(::Modified, yield::_YieldInput, cfs::AbstractVector, times)
    times = _cashflow_times(cfs, times)
    _iszero_cashflow_stream(cfs) && return _zero_cashflow_value(cfs, times)
    D(i) = price(i, cfs, times)
    return duration(yield, D)
end

## Analytic duration for flat yields
# Macaulay = Σ t·cf·d / Σ cf·d. Modified divides by 1+y/m for periodic rates
# (m=1 for scalars); continuous rates and yield models use Macaulay directly.
# Tests compare these formulas with the callback derivative of log|V|.
_macaulay_ratio(yield, cfs, times) = _weighted_ratio(yield, identity, cfs, times)

function duration(::Modified, yield::Real, cfs::AbstractVector, times)
    return _weighted_ratio(yield, identity, cfs, times; divisor = 1 + yield)
end
function duration(::Modified, yield::FinanceCore.Rate{<:Real, FinanceCore.Periodic}, cfs::AbstractVector, times)
    m = yield.compounding.frequency
    return _weighted_ratio(yield, identity, cfs, times; divisor = 1 + FinanceCore.rate(yield) / m)
end
function duration(::Modified, yield::FinanceCore.Rate{<:Real, FinanceCore.Continuous}, cfs::AbstractVector, times)
    return _macaulay_ratio(yield, cfs, times)
end
function duration(::Modified, yield::FinanceModels.Yield.Constant{<:FinanceCore.Rate}, cfs::AbstractVector, times)
    return _macaulay_ratio(yield.rate, cfs, times)
end

function duration(yield, valuation_function::T) where {T}
    # log|V| supports both asset and liability values.
    D(i) = log(abs(valuation_function(_parallel_bumped(yield, i))))
    return δV = -ForwardDiff.derivative(D, 0.0)
end

# Use Continuous(shift) for yield models. Converting an annual-rate increment
# to continuous compounding would change the second derivative.
_parallel_bumped(yield, shift) = yield + shift

function _parallel_bumped(
        yield::FinanceModels.Yield.AbstractYieldModel, shift
    )
    # Rate addition inherits the left operand's compounding convention.
    return FinanceModels.Yield.TenorShift(
        yield,
        (z, t) -> FinanceCore.Continuous(shift) + z,
    )
end


function duration(yield::_YieldInput, cfs::_CashflowCollection, times)
    return duration(Modified(), yield, _cashflow_vector(cfs), times)
end

# Use embedded Cashflow times or default numeric amounts to periods 1:n.
function duration(yield::_YieldInput, cfs::_CashflowCollection)
    cfs = _cashflow_vector(cfs)
    times = FinanceCore.timepoint.(cfs, 1:length(cfs))
    return duration(Modified(), yield, cfs, times)
end

function duration(::DV01, yield::_YieldInput, cfs::_CashflowCollection, times)
    cfs = _cashflow_vector(cfs)
    times = _cashflow_times(cfs, times)
    _iszero_cashflow_stream(cfs) && return _zero_cashflow_value(cfs, times)
    return duration(DV01(), yield, i -> FinanceCore.present_value(i, cfs, times))
end
function duration(d::Duration, yield::_YieldInput, cfs::_CashflowCollection)
    cfs = _cashflow_vector(cfs)
    times = FinanceCore.timepoint.(cfs, 1:length(cfs))
    return duration(d, yield, cfs, times)
end

# Prefer cashflow collections over the generic DV01 callback.
duration(d::DV01, yield::_YieldInput, cfs::_CashflowCollection) =
    invoke(duration, Tuple{Duration, _YieldInput, _CashflowCollection}, d, yield, cfs)

function duration(::DV01, yield, valuation_function::Y) where {Y}
    # Dollar risk is defined even when value is zero and relative duration is not.
    return -ForwardDiff.derivative(i -> valuation_function(_parallel_bumped(yield, i)), 0.0) / 10_000
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
function duration(::IR01, base_curve, credit_spread, cfs::_CashflowCollection, times)
    cfs = _cashflow_vector(cfs)
    times = _cashflow_times(cfs, times)
    _iszero_cashflow_stream(cfs) && return _zero_cashflow_value(cfs, times)
    return duration(DV01(), base_curve, i -> FinanceCore.present_value(i + credit_spread, cfs, times))
end

function duration(::IR01, base_curve, credit_spread, cfs::_CashflowCollection)
    cfs = _cashflow_vector(cfs)
    times = FinanceCore.timepoint.(cfs, 1:length(cfs))
    return duration(IR01(), base_curve, credit_spread, cfs, times)
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
function duration(::CS01, base_curve, credit_spread, cfs::_CashflowCollection, times)
    cfs = _cashflow_vector(cfs)
    times = _cashflow_times(cfs, times)
    _iszero_cashflow_stream(cfs) && return _zero_cashflow_value(cfs, times)
    return duration(DV01(), credit_spread, s -> FinanceCore.present_value(base_curve + s, cfs, times))
end

function duration(::CS01, base_curve, credit_spread, cfs::_CashflowCollection)
    cfs = _cashflow_vector(cfs)
    times = FinanceCore.timepoint.(cfs, 1:length(cfs))
    return duration(CS01(), base_curve, credit_spread, cfs, times)
end

"""
    convexity(yield,cfs,times)
    convexity(yield,valuation_function)

Calculates the normalized second derivative of value under a parallel rate shock.
`yield` may be a scalar annual yield (e.g. `0.05`), an explicit `Rate`, or an
`AbstractYieldModel`. `times` may be omitted for evenly spaced cashflows beginning
at the end of the first period. Cashflow collections may be arrays, tuples, or
finite generators; use `collect` for other iterables.

Wrapped `Cashflow` objects use their embedded payment times even when `times` is
supplied. Numeric amounts use the corresponding explicit time.

A scalar or `Rate` input is shocked in its own compounding space. An
`AbstractYieldModel` input is instead shocked additively in continuously
compounded zero-rate space, consistently across the no-tenor, tenor-aware,
and key-rate APIs.

Empty collections and collections whose amounts are all exactly zero return zero
by convention, without evaluating the curve. Every cashflow needs a time; unused
trailing times are ignored. See [Zero cashflow streams](@ref) for numeric types and
zero-net-value portfolios.

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
convexity(yield::_YieldInput, cfs::_CashflowCollection, times) =
    convexity(yield, _cashflow_vector(cfs), times)

function convexity(yield::_YieldInput, cfs::AbstractVector, times)
    times = _cashflow_times(cfs, times)
    _iszero_cashflow_stream(cfs) && return _zero_cashflow_value(cfs, times)
    return convexity(yield, i -> price(i, cfs, times))
end

function convexity(yield::_YieldInput, cfs::_CashflowCollection)
    cfs = _cashflow_vector(cfs)
    times = FinanceCore.timepoint.(cfs, 1:length(cfs))
    return convexity(yield, cfs, times)
end

# ── Analytic convexity for fixed cashflows ─────────────────────────────────
#
# Weights and divisors follow the shock coordinate. Tests compare each formula
# with the normalized second derivative from the callback API.
#
# * `Real` y: V(x) = Σ cf·(1+y+x)^(-t) → Σ cf·d·t(t+1) / V / (1+y)²
# * `Rate{Periodic(m)}`: V(x) = Σ cf·(1+(y+x)/m)^(-mt) → Σ cf·d·t(t+1/m) / V / (1+y/m)²
# * `Rate{Continuous}`: V(x) = Σ cf·e^(-(y+x)t) → Σ cf·d·t² / V
# * `AbstractYieldModel`: it is shocked in continuous-zero space,
#   so V(x) = Σ cf·d·exp(-xt) → Σ cf·d·t² / V.
#
# Signed normalization makes convexity invariant to position sign.

# Shared accumulation kernel: Σ weight(t)·cf·d / Σ cf·d. `weight = identity`
# gives the Macaulay ratio (Modified-duration fast paths above); the t(t+1)/t²
# weights below give the convexity statistics.
function _weighted_ratio(yield, weight, cfs, times; divisor = 1)
    # Check bounds before indexing times in the @inbounds loop.
    _check_cashflow_times(cfs, times)
    _iszero_cashflow_stream(cfs) && return _zero_cashflow_value(cfs, times)
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
    return _risk_ratio(Vw, V; divisor)
end

function convexity(yield::Real, cfs::AbstractVector, times)
    return _weighted_ratio(yield, t -> t * (t + 1), cfs, times; divisor = (1 + yield)^2)
end
function convexity(yield::FinanceCore.Rate{<:Real, FinanceCore.Periodic}, cfs::AbstractVector, times)
    m = yield.compounding.frequency
    return _weighted_ratio(yield, t -> t * (t + 1 / m), cfs, times; divisor = (1 + FinanceCore.rate(yield) / m)^2)
end
function convexity(yield::FinanceCore.Rate{<:Real, FinanceCore.Continuous}, cfs::AbstractVector, times)
    return _weighted_ratio(yield, t -> t * t, cfs, times)
end
function convexity(yield::FinanceModels.Yield.AbstractYieldModel, cfs::AbstractVector, times)
    # A continuous-zero shift multiplies each fixed payment's discount by
    # exp(-s*t), so the t² kernel applies to every yield model, including curves
    # that are not flat. Keep the callback form for rate-dependent cashflows.
    return _weighted_ratio(yield, t -> t * t, cfs, times)
end
function convexity(yield::FinanceModels.Yield.Constant{<:FinanceCore.Rate}, cfs::AbstractVector, times)
    return _weighted_ratio(yield.rate, t -> t * t, cfs, times)
end
# disambiguation vs `convexity(curve::AYM, tenors, cfs::AbstractVector{<:Cashflow})`:
# a Cashflow vector in the third position means (tenors, cashflows), not (cfs, times)
function convexity(yield::FinanceModels.Yield.Constant{<:FinanceCore.Rate}, tenors::AbstractVector, cfs::AbstractVector{<:FinanceCore.Cashflow})
    return convexity(yield, tenors, _extract_cfs_times(cfs)...)
end

function convexity(yield, valuation_function::T) where {T}
    v(x) = abs(valuation_function(_parallel_bumped(yield, x)))
    ∂²P = ForwardDiff.derivative(y -> ForwardDiff.derivative(v, y), 0.0)
    return ∂²P / v(0.0)
end


"""
    duration(keyrate::KeyRateDuration,curve,cashflows)
    duration(keyrate::KeyRateDuration,curve,cashflows,timepoints)
    duration(keyrate::KeyRateDuration,curve,cashflows,timepoints,krd_points)

Calculate key-rate duration by bumping and repricing. `KeyRateZero` applies
triangular continuous-zero bumps; `KeyRatePar` bumps par rates and refits the
curve. Both use the selector's `shift` for central differences.

`krd_points` defaults to annual knots from year 1 through the latest payment time.
It must contain the selected key rate. Supply a grid explicitly for payments
before year 1 or a different bucket convention. Any FinanceModels yield curve
is accepted.

!!! warning "Experimental"
    The legacy `KeyRateDuration` API may change. Use `KeyRates(tenors)` for
    derivatives under continuous-zero bumps without finite-difference error.

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

Par-rate and zero-rate bumps measure different risks. Choose the convention
used by your hedging or reporting process.

References:
- [Quant Finance Stack Exchange: To compute key rate duration, shall I use par curve or zero curve?](https://quant.stackexchange.com/questions/33891/to-compute-key-rate-duration-shall-i-use-par-curve-or-zero-curve)
- [Financial Exam Help 123](http://www.financialexamhelp123.com/key-rate-duration/)

"""
function duration(keyrate::KeyRateDuration, curve, cashflows::_CashflowCollection, timepoints, krd_points)
    cashflows = _cashflow_vector(cashflows)
    timepoints = _cashflow_times(cashflows, timepoints)
    keyrate.timepoint in krd_points || throw(ArgumentError("krd_points must contain the shifted timepoint $(keyrate.timepoint)"))
    _iszero_cashflow_stream(cashflows) && return _zero_cashflow_value(cashflows, timepoints)
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

Return a closure `(z, t) -> Continuous(bump) + z` implementing the Ho (1992)
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
    # Reuse the key-rate hat shape, including flat endpoint extrapolation.
    bumps = [k == idx ? shift : zero(shift) for k in eachindex(krd_points)]
    return (z, t) -> FinanceCore.Continuous(_hat_bump(krd_points, bumps, t)) + z
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

function _default_krd_points(cashflows, timepoints)
    mt = _maximum_cashflow_time(cashflows, timepoints)
    mt >= 1 || throw(
        ArgumentError(
            "the default krd_points grid is empty because all payment times are < 1; pass krd_points explicitly"
        )
    )
    return 1:mt
end

function duration(keyrate::KeyRateDuration, curve, cashflows::_CashflowCollection, timepoints)
    cashflows = _cashflow_vector(cashflows)
    timepoints = _cashflow_times(cashflows, timepoints)
    _iszero_cashflow_stream(cashflows) && return _zero_cashflow_value(cashflows, timepoints)
    return duration(keyrate, curve, cashflows, timepoints, _default_krd_points(cashflows, timepoints))
end

function duration(keyrate::KeyRateDuration, curve::_YieldInput, cashflows::_CashflowCollection)
    cashflows = _cashflow_vector(cashflows)
    # extract embedded Cashflow times where present; otherwise the index is the time
    timepoints = FinanceCore.timepoint.(cashflows, eachindex(cashflows))
    return duration(keyrate, curve, cashflows, timepoints)
end

"""
    spread(curve1,curve2,cashflows)

Find the constant spread to add to `curve1` so the cashflows have the same present
value as under `curve2`.

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

    # Dampen Newton steps: mixed-sign cashflows can have nearly zero price
    # derivatives, producing steps outside the valid spread domain s > -1.
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
# scalar continuous-zero-shock paths for yield models.

function duration(valuation_fn::Function, yield::FinanceModels.Yield.AbstractYieldModel)
    return duration(yield, valuation_fn)
end
function convexity(valuation_fn::Function, yield::FinanceModels.Yield.AbstractYieldModel)
    return convexity(yield, valuation_fn)
end
