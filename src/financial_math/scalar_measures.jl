## Scalar duration and convexity

# Cashflow routes accept the yield inputs supported by present_value; this
# distinguishes them from metric-first and callable-valuation signatures.
const _YieldInput = Union{Real, FinanceCore.Rate, AYM}
const _CashflowCollection = Union{AbstractArray, Tuple, Base.Generator}

# Indexed kernels share one representation; materialize generators before AD
# reevaluates a valuation, including generators backed by a stateful iterator.
_cashflow_vector(cfs::AbstractArray) = vec(cfs)
_cashflow_vector(cfs::Union{Tuple, Base.Generator}) = _cashflow_vector(collect(cfs))

"""
    duration(Macaulay(),interest_rate,cfs,times)
    duration(Modified(),interest_rate,cfs,times)
    duration(DV01(),interest_rate,cfs,times)
    duration(IR01(),base,credit,cfs,times)
    duration(CS01(),base,credit,cfs,times)
    duration(interest_rate,cfs,times)             # Modified Duration
    duration(valuation_function,interest_rate)    # Modified Duration
    duration(valuation_function,DV01(),interest_rate)

Calculate Macaulay or modified duration, or signed dollar DV01, IR01, or CS01.
For numeric amounts, omitted `times` default to `1:length(cfs)`.

`cfs` can be an `AbstractVector{<:Cashflow}` (from FinanceCore), in which case
`times` may be omitted. A `Cashflow` always supplies its embedded amount and time;
explicit `times` supply payment times for numeric amounts. If supplied, `times`
must still contain an entry for each cashflow; unused trailing entries are ignored.

Scalar cashflow methods accept arrays, tuples, and finite generators. Arrays are
flattened in column-major order; generators are collected once before valuation.
Use `collect` for other iterables, such as `Iterators.take` or `skipmissing`.
Normalized duration is unchanged when the position sign reverses; dollar DV01,
IR01, and CS01 reverse sign with the position.

Empty and all-zero cashflow streams return zero risk; see [Zero cashflow streams](@ref).
Dollar sensitivities differentiate the signed value directly, so they stay defined at
zero present value, where normalized duration is not. Callback forms do not use the
zero-stream convention: a zero value from them gives undefined normalized risk.

The default measure is `Modified()`.

- Modified duration: `-∂V/∂r / V`, the relative value lost per unit increase in the shocked rate.
- Macaulay: the present-value-weighted average payment time.
- DV01: `-∂V/∂r / 10000`, the first-order value lost for a one-basis-point increase.
- IR01: the first-order value lost for a one-basis-point increase in the base curve, holding the credit curve fixed.
- CS01: the first-order value lost for a one-basis-point increase in the credit curve, holding the base curve fixed.

# Shock coordinates

Each single-rate input moves in its own shock coordinate; fixed-cashflow IR01/CS01
move the combined rate in its coordinate. See [Shock coordinates](@ref):

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
28.277877274012628

```

Using any given value function, passed first so that do-block syntax works:

```julia-repl
julia> lump_sum_value(amount,years,i) = amount / (1 + i ) ^ years
julia> my_lump_sum_value(i) = lump_sum_value(100,5,i)
julia> duration(my_lump_sum_value,0.03)
4.8543689320388355
julia> convexity(my_lump_sum_value,0.03)
28.277877274012642
julia> duration(0.03) do i
           lump_sum_value(100,5,i)
       end
4.8543689320388355

```
"""
duration(yield::_YieldInput, cfs::_CashflowCollection, times...) = duration(Modified(), yield, cfs, times...)

# Tuples and generators are collected, and arrays flattened, once before the indexed kernels.
duration(d::Union{Macaulay, Modified, DV01}, yield::_YieldInput, cfs::_CashflowCollection) =
    duration(d, yield, _cashflow_vector(cfs))
duration(d::Union{Macaulay, Modified, DV01}, yield::_YieldInput, cfs::_CashflowCollection, times) =
    duration(d, yield, _cashflow_vector(cfs), times)

## Analytic measures for fixed cashflows
# Each single-rate input moves in its own shock coordinate (docs: "Shock coordinates").
# A rate compounded m times a year gives the convexity weight t(t + 1/m), and the divisor
# 1 + y/m turns Macaulay into modified duration. Scalars have m = 1. Continuous rates and
# every yield model take a continuous-zero shift: 1/m = 0 and the divisor is 1. Tests
# compare these formulas with the callback derivatives.
_coordinate(yield::Real) = (; inv_m = 1, divisor = 1 + yield)
function _coordinate(yield::FinanceCore.Rate{<:Real, FinanceCore.Periodic})
    m = yield.compounding.frequency
    return (; inv_m = 1 / m, divisor = 1 + FinanceCore.rate(yield) / m)
end
_coordinate(::Union{FinanceCore.Rate{<:Real, FinanceCore.Continuous}, AYM}) =
    (; inv_m = false, divisor = 1)

duration(::Macaulay, yield::_YieldInput, cfs::AbstractVector, times = eachindex(cfs)) =
    _weighted_ratio(yield, identity, cfs, times)
duration(::Modified, yield::_YieldInput, cfs::AbstractVector, times = eachindex(cfs)) =
    _weighted_ratio(yield, identity, cfs, times; divisor = _coordinate(yield).divisor)

# -∂V/∂s is Σ t·cf·d divided by the coordinate's divisor; do not divide by V so dollar
# exposure remains defined at zero present value.
function duration(::DV01, yield::_YieldInput, cfs::AbstractVector, times = eachindex(cfs))
    divisor = _coordinate(yield).divisor * 10_000
    sums = _weighted_sums(yield, identity, cfs, times)
    isnothing(sums) && return _zero_weighted(yield, identity, cfs, times, divisor)
    return last(sums) / divisor
end

function duration(valuation_function::F, yield::_YieldInput) where {F}
    # log|V| supports both asset and liability values.
    D(i) = log(abs(valuation_function(_parallel_bumped(yield, i))))
    return -ForwardDiff.derivative(D, 0.0)
end

function duration(valuation_function::F, ::DV01, yield::_YieldInput) where {F}
    # Dollar risk is defined even when value is zero and normalized duration is not.
    return -ForwardDiff.derivative(i -> valuation_function(_parallel_bumped(yield, i)), 0.0) / 10_000
end

"""
    duration(IR01(), base, credit, cfs, times = eachindex(cfs))
    duration(CS01(), base, credit, cfs, times = eachindex(cfs))
    duration(IR01(), kr::KeyRates, base::AbstractYieldModel, credit::AbstractYieldModel, cfs, times = eachindex(cfs)) -> Vector
    duration(CS01(), kr::KeyRates, base::AbstractYieldModel, credit::AbstractYieldModel, cfs, times = eachindex(cfs)) -> Vector

Fixed-cashflow IR01 or CS01: `-∂V/∂s / 10000`, the first-order value lost per basis point.
The cashflows are discounted at `base + credit`. Both measures shift that combined rate by one
basis point in its own coordinate, so they equal each other and
`duration(DV01(), base + credit, cfs, times)`; the `KeyRates` forms equal
`duration(DV01(), kr, base + credit, cfs, times)`.

The combined rate's coordinate is continuous zero when either input is a yield model. Otherwise
it is annual effective for two scalars, and the `Rate`'s compounding when one input is a `Rate`
(the left one's when both are). With mixed compounding this is not a bump of either input's own
nominal rate. Credit here is an additive discount spread, not a CDS quote or a hazard rate.

When the curves play different roles, use the callback forms, or the contract measures
`Effective()` and `Spread()`. For quote risk after recalibration, see [Market Inputs](@ref).
See [Two curves: IR01 and CS01](@ref).

# Examples

```julia-repl
julia> cfs = [5, 5, 5, 105];

julia> times = 1:4;

julia> duration(IR01(), 0.03, 0.02, cfs, times)
0.0354595050416236

julia> duration(IR01(), 0.03, 0.02, cfs, times) ≈ duration(DV01(), 0.05, cfs, times)
true
```
"""
duration(::Union{IR01, CS01}, base, credit, cfs::_CashflowCollection, times...) =
    duration(DV01(), base + credit, cfs, times...)
duration(::Union{IR01, CS01}, kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector, times = eachindex(cfs)) =
    duration(DV01(), kr, base + credit, cfs, times)

"""
    convexity(yield,cfs,times)
    convexity(valuation_function,yield)

Calculates the normalized second derivative of value under a parallel rate shock.
`yield` may be a scalar annual yield (e.g. `0.05`), an explicit `Rate`, or an
`AbstractYieldModel`. `times` may be omitted for evenly spaced cashflows beginning
at the end of the first period. Cashflow collections may be arrays, tuples, or
finite generators; use `collect` for other iterables.

Wrapped `Cashflow` objects use their embedded payment times even when `times` is
supplied. Numeric amounts use the corresponding explicit time.

A scalar or `Rate` input moves in its own shock coordinate. An `AbstractYieldModel`
input takes an additive continuous-zero shift, the same coordinate as the key-rate
APIs, so its convexity is ≈ the sum of the full key-rate convexity matrix. See
[Shock coordinates](@ref).

Empty and all-zero cashflow streams return zero; see [Zero cashflow streams](@ref).
Every cashflow needs a time; unused trailing times are ignored.

# Examples

```julia-repl
julia> convexity(0.03, [0, 0, 0, 0, 100], 1:5)
28.277877274012628

julia> convexity(i -> 100 / (1 + i)^5, 0.03)
28.277877274012642
```
"""
convexity(yield::_YieldInput, cfs::_CashflowCollection) = convexity(yield, _cashflow_vector(cfs))
convexity(yield::_YieldInput, cfs::_CashflowCollection, times) = convexity(yield, _cashflow_vector(cfs), times)

# The weight t(t + 1/m) and squared divisor of the shock coordinate (see `_coordinate`):
#
# * `Real` y: V(x) = Σ cf·(1+y+x)^(-t) → Σ cf·d·t(t+1) / V / (1+y)²
# * `Rate{Periodic(m)}`: V(x) = Σ cf·(1+(y+x)/m)^(-mt) → Σ cf·d·t(t+1/m) / V / (1+y/m)²
# * `Rate{Continuous}`: V(x) = Σ cf·e^(-(y+x)t) → Σ cf·d·t² / V
# * `AbstractYieldModel`: a continuous-zero shift multiplies each fixed payment's discount
#   by exp(-xt), so V(x) = Σ cf·d·exp(-xt) → Σ cf·d·t² / V, flat curve or not. Keep the
#   callback form for rate-dependent cashflows.
#
# Signed normalization makes convexity invariant to position sign.
function convexity(yield::_YieldInput, cfs::AbstractVector, times = eachindex(cfs))
    c = _coordinate(yield)
    return _weighted_ratio(yield, t -> t * (t + c.inv_m), cfs, times; divisor = c.divisor^2)
end

# Shared accumulation kernel: V = Σ cf·d and Vw = Σ weight(t)·cf·d. Every cashflow needs a
# time, checked before the zero-stream test and the @inbounds loop. A zero stream returns
# `nothing` without valuing any payment; callers return its typed zero.
function _weighted_sums(yield, weight::W, cfs, times) where {W}
    _check_cashflow_times(cfs, times)
    _iszero_cashflow_stream(cfs) && return nothing
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

# Vw / V / divisor. `weight` is only forwarded here, so type it to keep this method specialized.
function _weighted_ratio(yield, weight::W, cfs, times; divisor = 1) where {W}
    sums = _weighted_sums(yield, weight, cfs, times)
    isnothing(sums) && return _zero_weighted(yield, weight, cfs, times, divisor)
    V, Vw = sums
    return _risk_ratio(Vw, V; divisor)
end

function convexity(valuation_function::F, yield::_YieldInput) where {F}
    v(x) = abs(valuation_function(_parallel_bumped(yield, x)))
    ∂²P = ForwardDiff.derivative(y -> ForwardDiff.derivative(v, y), 0.0)
    return ∂²P / v(0.0)
end
