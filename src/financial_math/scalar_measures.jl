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

Interest Rate 01: signed dollar risk for a one-basis-point parallel shift in the
risk-free (base) curve, holding the credit spread constant.

Requires both a base curve and a credit spread. For fixed cashflows discounted at
`base + spread`, IR01, CS01, and the DV01 of the combined rate are equal; the
measures separate in the callback, key-rate, and contract forms, where the curves
play different roles.

See also: [`CS01`](@ref), [`DV01`](@ref)
"""
struct IR01 <: Duration end

"""
    CS01 <: Duration

Credit Spread 01: signed dollar risk for a one-basis-point parallel shift in the
credit spread, holding the risk-free (base) curve constant.

Requires both a base curve and a credit spread. For fixed cashflows discounted at
`base + spread`, CS01, IR01, and the DV01 of the combined rate are equal; the
measures separate in the callback, key-rate, and contract forms, where the curves
play different roles.

See also: [`IR01`](@ref), [`DV01`](@ref)
"""
struct CS01 <: Duration end

"""
    Effective <: Duration

Measure contract risk while reprojecting cashflows under shifted curves, so
floating coupons reset. Use `duration(Effective(), contract, curve)`; the same
marker applies to `dv01` and `convexity`. `Modified` and `Macaulay` operate on
fixed cashflows.

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

# Shock coordinates

Each input is shocked in its own native form; see [Shock coordinates](@ref):

- Scalars are annual effective rates: Modified = Macaulay / (1 + y).
- `Periodic(y, m)` shocks its nominal rate: Modified = Macaulay / (1 + y/m).
- `Continuous(y)` and every yield model use continuous-zero shifts: Modified = Macaulay.

Wrapping a scalar in `Yield.Constant` preserves its discount factors but changes
the shock coordinate, so duration and DV01 change by a factor of `1 + y`:

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
# A continuous-zero shift multiplies each fixed payment's discount by exp(-s*t),
# so modified duration equals Macaulay duration for every yield model.
function duration(::Modified, yield::FinanceModels.Yield.AbstractYieldModel, cfs::AbstractVector, times)
    return _macaulay_ratio(yield, cfs, times)
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

function duration(::DV01, yield::FinanceModels.Yield.AbstractYieldModel, cfs::_CashflowCollection, times)
    cfs = _cashflow_vector(cfs)
    times = _cashflow_times(cfs, times)
    _iszero_cashflow_stream(cfs) && return _zero_cashflow_value(cfs, times)
    # -∂V/∂s under a continuous-zero shift is Σ t·cf·d; do not divide by V so
    # dollar exposure remains defined at zero present value.
    _, Vt = _weighted_sums(yield, identity, cfs, times)
    return Vt / 10_000
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

Calculate the IR01 (Interest Rate 01): the signed dollar change in value for a
1 basis point parallel shift in the risk-free (base) curve, holding the credit
spread constant.

Fixed cashflows are discounted at the combined rate `base_curve + credit_spread`,
so a one-basis-point move in either component is a one-basis-point move in the
combined rate. IR01 therefore equals [`CS01`](@ref) and the DV01 of the combined
rate, shocked in that rate's own coordinate (see [Shock coordinates](@ref)): scalars
add as annual rates, a `Rate` sum takes the left operand's compounding, and any
yield-model component makes the sum a yield model with a continuous-zero shock.

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
    # Shock the combined rate in its own coordinate so IR01 and CS01 measure the
    # same one-basis-point move whatever the component input types.
    return duration(DV01(), base_curve + credit_spread, cfs, times)
end

function duration(::IR01, base_curve, credit_spread, cfs::_CashflowCollection)
    cfs = _cashflow_vector(cfs)
    times = FinanceCore.timepoint.(cfs, 1:length(cfs))
    return duration(IR01(), base_curve, credit_spread, cfs, times)
end

"""
    duration(CS01(), base_curve, credit_spread, cfs, times)
    duration(CS01(), base_curve, credit_spread, cfs)

Calculate the CS01 (Credit Spread 01): the signed dollar change in value for a
1 basis point parallel shift in the credit spread, holding the risk-free (base)
curve constant.

Fixed cashflows are discounted at the combined rate `base_curve + credit_spread`,
so CS01 equals [`IR01`](@ref) and the DV01 of the combined rate, shocked in that
rate's own coordinate (see [Shock coordinates](@ref)). Use the callback, key-rate,
or contract forms when the base and credit curves play different roles.

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
    return duration(DV01(), base_curve + credit_spread, cfs, times)
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
compounded zero-rate space, the same coordinate as the key-rate APIs; scalar
curve convexity equals the sum of the full key-rate convexity matrix. See
[Shock coordinates](@ref).

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

# Shared accumulation kernel: V = Σ cf·d and Vw = Σ weight(t)·cf·d. The ratio
# Vw / V with `weight = identity` is the Macaulay ratio (Modified-duration fast
# paths above); the t(t+1)/t² weights below give the convexity statistics.
# Callers handle the zero-stream shortcut; the stream must be nonempty.
function _weighted_sums(yield, weight::W, cfs, times) where {W}
    # Check bounds before indexing times in the @inbounds loop.
    _check_cashflow_times(cfs, times)
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
    return V, Vw
end

# `weight` is only forwarded here, so type it to keep this method specialized.
function _weighted_ratio(yield, weight::W, cfs, times; divisor = 1) where {W}
    _check_cashflow_times(cfs, times)
    _iszero_cashflow_stream(cfs) && return _zero_cashflow_value(cfs, times)
    V, Vw = _weighted_sums(yield, weight, cfs, times)
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

function convexity(yield, valuation_function::T) where {T}
    v(x) = abs(valuation_function(_parallel_bumped(yield, x)))
    ∂²P = ForwardDiff.derivative(y -> ForwardDiff.derivative(v, y), 0.0)
    return ∂²P / v(0.0)
end


"""
    spread(curve1,curve2,cashflows)

Find the constant spread to add to `curve1` so the cashflows have the same present
value as under `curve2`.

The spread is found via a damped Newton iteration on the pricing residual. It stops once the
undamped Newton step is smaller than `tol` in rate units (not currency), so the result does not
depend on the size of the cashflows; an `ErrorException` is thrown if that does not happen within
`maxiter` iterations.

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
    max_step = 0.25
    s = 0.0
    newton = NaN
    for _ in 1:maxiter
        fs = f(s)
        iszero(fs) && return FinanceCore.Periodic(s, 1)
        newton = fs / ForwardDiff.derivative(f, s)
        # converged on the undamped step in rate units, which, unlike a price residual,
        # does not scale with the cashflows
        isfinite(newton) && abs(newton) < tol && return FinanceCore.Periodic(s - newton, 1)
        step = !isfinite(newton) || abs(newton) > max_step ? (isnan(newton) ? max_step : copysign(max_step, newton)) : newton
        s = max(s - step, -0.999)
    end
    throw(ErrorException("spread did not converge in $maxiter iterations (last Newton step = $newton)"))
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
