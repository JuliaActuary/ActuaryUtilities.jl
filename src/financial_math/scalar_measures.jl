## Scalar duration and convexity

# Cashflow routes accept the yield inputs supported by present_value; this
# distinguishes them from metric-first and callable-valuation signatures.
const _YieldInput = Union{Real, FinanceCore.Rate, FinanceModels.Yield.AbstractYieldModel}
const _CashflowCollection = Union{AbstractArray, Tuple, Base.Generator}

# Indexed kernels share one representation; materialize generators before AD
# reevaluates a valuation, including generators backed by a stateful iterator.
_cashflow_vector(cfs::AbstractArray) = vec(cfs)
# Empty tuples collect to Union{}[], whose element type also matches Cashflow. Treat them as
# an untyped empty collection before dispatch derives embedded times.
_cashflow_vector(::AbstractArray{Union{}}) = Any[]
_cashflow_vector(cfs::Union{Tuple, Base.Generator}) = _cashflow_vector(collect(cfs))

"""
    duration(Macaulay(),interest_rate,cfs,times)
    duration(Modified(),interest_rate,cfs,times)
    duration(DV01(),interest_rate,cfs,times)
    duration(IR01(),base_curve,credit_spread,cfs,times)
    duration(CS01(),base_curve,credit_spread,cfs,times)
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
Relative duration is unchanged when the position sign reverses; dollar DV01,
IR01, and CS01 reverse sign with the position.

Empty collections and collections whose amounts are all exactly zero return zero
risk without valuing any payment. Every cashflow needs a time; unused trailing
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

Single-rate inputs are shocked in their native form; fixed-cashflow IR01/CS01 use
the combined rate's coordinate. See [Shock coordinates](@ref):

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

Using any given value function, passed first so that do-block syntax works:

```julia-repl
julia> lump_sum_value(amount,years,i) = amount / (1 + i ) ^ years
julia> my_lump_sum_value(i) = lump_sum_value(100,5,i)
julia> duration(my_lump_sum_value,0.03)
4.854368932038835
julia> convexity(my_lump_sum_value,0.03)
28.277877274012642
julia> duration(0.03) do i
           lump_sum_value(100,5,i)
       end
4.854368932038835

```
"""
function duration(::Macaulay, yield::_YieldInput, cfs::_CashflowCollection, times)
    return _macaulay_ratio(yield, _cashflow_vector(cfs), times)
end

duration(d::Modified, yield::_YieldInput, cfs::_CashflowCollection, times) =
    duration(d, yield, _cashflow_vector(cfs), times)

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

function duration(valuation_function::F, yield::_YieldInput) where {F}
    # log|V| supports both asset and liability values.
    D(i) = log(abs(valuation_function(_parallel_bumped(yield, i))))
    return -ForwardDiff.derivative(D, 0.0)
end

# A scalar or `Rate` moves in its own coordinate; yield models move every continuous
# zero rate (key_rate_sensitivities.jl).
_parallel_bumped(yield, shift) = yield + shift


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
    _iszero_cashflow_stream(cfs) && return _zero_shifted(yield, cfs, times, 10_000)
    return duration(i -> FinanceCore.present_value(i, cfs, times), DV01(), yield)
end
function duration(d::Duration, yield::_YieldInput, cfs::_CashflowCollection)
    cfs = _cashflow_vector(cfs)
    times = FinanceCore.timepoint.(cfs, 1:length(cfs))
    return duration(d, yield, cfs, times)
end

function duration(::DV01, yield::FinanceModels.Yield.AbstractYieldModel, cfs::_CashflowCollection, times)
    cfs = _cashflow_vector(cfs)
    times = _cashflow_times(cfs, times)
    _iszero_cashflow_stream(cfs) && return _zero_weighted(yield, identity, cfs, times, 10_000)
    # -∂V/∂s under a continuous-zero shift is Σ t·cf·d; do not divide by V so
    # dollar exposure remains defined at zero present value.
    _, Vt = _weighted_sums(yield, identity, cfs, times)
    return Vt / 10_000
end

function duration(valuation_function::F, ::DV01, yield::_YieldInput) where {F}
    # Dollar risk is defined even when value is zero and relative duration is not.
    return -ForwardDiff.derivative(i -> valuation_function(_parallel_bumped(yield, i)), 0.0) / 10_000
end

"""
    duration(IR01(), base_curve, credit_spread, cfs, times)
    duration(IR01(), base_curve, credit_spread, cfs)

Calculate fixed-cashflow IR01 (Interest Rate 01) using the combined-rate shock
convention: `-∂V/∂s / 10000`, the first-order value lost per basis point increase.

Fixed cashflows are discounted at `base_curve + credit_spread`. This method defines
the base-rate move as a one-basis-point shift in that combined rate's coordinate,
so it equals [`CS01`](@ref) and the DV01 of the combined rate. Scalars add as annual
rates, a `Rate` sum takes the left operand's compounding, and any yield-model
component makes the sum a yield model with a continuous-zero shock.

For mixed-compounding inputs, this is not an independent bump to the base input's
original nominal rate. With two yield models, it is a continuous-zero parallel
shift of the base curve with credit held fixed. Use the callback or contract forms
when the curves also affect projected payments. This discount-spread convention
does not imply equality with market-quote risk after recalibration. See
[Two curves: IR01 and CS01](@ref).

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

Calculate fixed-cashflow CS01 (Credit Spread 01) using the combined-rate shock
convention: `-∂V/∂s / 10000`, the first-order value lost per basis point increase.

Fixed cashflows are discounted at `base_curve + credit_spread`. This method defines
the spread move as a one-basis-point shift in that combined rate's coordinate,
so it equals [`IR01`](@ref) and the DV01 of the combined rate. The coordinate is
annual effective for scalar sums, the left operand's compounding for `Rate` sums,
and continuous zero when a yield-model component is present.

For mixed-compounding inputs, this is not an independent bump to the credit input's
original nominal rate. With two yield models, it is a continuous-zero parallel
shift of credit with the base curve held fixed. Credit here is an additive discount
spread, not a CDS par quote or hazard-rate parameter. Use the callback, key-rate,
or contract forms when the curves play different roles, and [Market Inputs](@ref)
for sensitivity to quotes used in calibration. See [Two curves: IR01 and CS01](@ref).

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
    convexity(valuation_function,yield)

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
by convention, without valuing any payment. Every cashflow needs a time; unused
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
julia> duration(my_lump_sum_value,0.03)
4.854368932038835
julia> convexity(my_lump_sum_value,0.03)
28.277877274012642

```

"""
convexity(yield::_YieldInput, cfs::_CashflowCollection, times) =
    convexity(yield, _cashflow_vector(cfs), times)

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
    _iszero_cashflow_stream(cfs) && return _zero_weighted(yield, weight, cfs, times, divisor)
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

function convexity(valuation_function::F, yield::_YieldInput) where {F}
    v(x) = abs(valuation_function(_parallel_bumped(yield, x)))
    ∂²P = ForwardDiff.derivative(y -> ForwardDiff.derivative(v, y), 0.0)
    return ∂²P / v(0.0)
end
