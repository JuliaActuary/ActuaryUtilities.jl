## Key-rate sensitivities
# Apply triangular continuous-zero bumps at the supplied tenors. Callbacks use
# ForwardDiff through Yield.TenorShift; fixed cashflows use analytic derivatives.

# Named curve roles, each bumped by triangular hats on one tenor grid.
function _ncurve_ad(valuation::F, curves::NamedTuple{roles}, tenors; order = 1) where {F, roles}
    grid = _validate_tenors(tenors)
    bumped(b) = NamedTuple{roles}(ntuple(i -> _bumped(curves[i], grid, b[i]), length(curves)))
    return _named_ad(b -> valuation(bumped(b)), map(_ -> zeros(length(grid)), curves); order)
end

## Analytic derivatives for fixed cashflows
# After locating its hat interval, each payment updates at most two gradient
# entries and a 2×2 Hessian block. The value is Σ cf(t) * discount(curve, t), over the 1-based,
# equal-length inputs of `_cashflow_inputs`.
# Inline to eliminate result tuples across derivative-order and zero-stream branches.
@inline function _keyrate_analytic(curve, tenors::AbstractVector, cfs::AbstractVector, times; order = 1)
    _validate_tenors(tenors)
    n = length(tenors)
    disc(t) = FinanceCore.discount(curve, t)
    zero_stream = _iszero_cashflow_stream(cfs)
    if zero_stream
        value = _zero_cashflow_value(disc, cfs, times)
        T = promote_type(typeof(value), eltype(tenors))
        gradient = zeros(T, n)
        return order >= 2 ? (; value, gradient, hessian = zeros(T, n, n), zero_stream) : (; value, gradient, zero_stream)
    end
    # Seed from a discounted payment to preserve curve numeric types and AD.
    # Its type must accommodate later terms, including payments at t=0.
    t0 = FinanceCore.timepoint(cfs[1], times[1])
    cfd0 = _cf_value(cfs[1]) * disc(t0)
    _, w0, _, _ = _active_hats(tenors, t0)
    T = promote_type(typeof(cfd0), typeof(t0 * cfd0 * w0), eltype(tenors))
    if order >= 2
        T = promote_type(T, typeof(t0 * t0 * cfd0 * w0 * w0))
    end
    grad_shared = zeros(T, n)
    hess_shared = order >= 2 ? zeros(T, n, n) : nothing
    V = zero(cfd0)
    @inbounds for k in eachindex(cfs)
        t = FinanceCore.timepoint(cfs[k], times[k])
        d = disc(t)
        cfd = _cf_value(cfs[k]) * d
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
        return (; value = V, gradient = grad_shared, hessian = hess_shared, zero_stream)
    else
        return (; value = V, gradient = grad_shared, zero_stream)
    end
end

## Key-rate results
# Derivatives on the key-rate grid from a valuation callback, which receives the curves in order
# (AD), keyed by curve role. One curve's result, and the analytic result for fixed cashflows, is
# unkeyed: `(; value, gradient, hessian, zero_stream)` with a vector and a matrix.
@inline _keyrate(curves::NamedTuple, tenors, valuation::F; order = 1) where {F} =
    _ncurve_ad(c -> valuation(values(c)...), curves, tenors; order)
@inline _keyrate(curve, tenors, valuation::F; order = 1) where {F} =
    _only_role(_keyrate((; curve), tenors, valuation; order))
_only_role(r) = haskey(r, :hessian) ?
    (; r.value, gradient = only(r.gradient), hessian = only(only(r.hessian)), r.zero_stream) :
    (; r.value, gradient = only(r.gradient), r.zero_stream)

# Normalized risk from part of a key-rate result `r`: a role's gradient, a block of its Hessian, or
# `only` of one on the parallel grid. Durations and convexities are relative to the value; DV01s are
# dollars per basis point. Each call makes new arrays, so public results never share storage.
_relative(r, x; negate = false) = _risk_ratio(x, r.value, r.zero_stream; negate)
_per_bp(r, x) = _risk_ratio(x, 10_000, r.zero_stream; negate = true)
# What the normalizers read from a result: closures over roles capture this, not the derivatives.
_scale(r) = (; r.value, r.zero_stream)

# The public two-curve convexity blocks (`f = only` on the parallel grid). Only the three returned
# blocks are normalized.
_base_credit_cross(r, f = identity) = (;
    base = _relative(r, f(r.hessian.base.base)),
    credit = _relative(r, f(r.hessian.credit.credit)),
    cross = _relative(r, f(r.hessian.base.credit)),
)

## Public yield-model sensitivities

"""
    duration(valuation_fn, kr::KeyRates, curve::AbstractYieldModel) -> Vector
    duration(kr::KeyRates, curve::AbstractYieldModel, cfs, times = eachindex(cfs)) -> Vector

Return normalized key-rate durations `-∂V/∂rᵢ / V` for an `AbstractYieldModel`.
Each `rᵢ` is a triangular continuous-zero bump at `kr.tenors[i]`. The base curve
is used directly, without resampling or refitting.

Empty and all-zero cashflow streams return zero key-rate durations, one per tenor;
see [Zero cashflow streams](@ref). Every cashflow needs a time; unused trailing times
are ignored. Wrapped `Cashflow` objects use their embedded amounts and payment times,
including when explicit `times` are supplied. Numeric amounts use the explicit times,
which default to `eachindex(cfs)` (periods `1:n`), and pair with the amounts by position, as
in [`duration`](@ref). The same holds for every key-rate cashflow form.

# Tenor grid

Choose `kr.tenors` independently of the curve's own knots. The grid must be
nonempty, finite, positive, and strictly increasing. It is validated both at
construction and when calculating sensitivities.

# Bump shape and endpoint extrapolation

The bump at the i-th knot is a triangular hat centered at `tenors[i]` with
support `[tenors[i-1], tenors[i+1]]` for interior knots. Endpoint bumps stay
constant beyond the grid. All sensitivity after the last tenor belongs to its
bucket; extend the grid to separate exposures at longer maturities.

The hats sum to one, so the sum of key-rate durations is ≈ the parallel duration.
These are sensitivities to the specified bumps, not to spline parameters.

# Example
```julia
duration(pv, KeyRates([0.25, 1, 5, 10, 30]), curve)

duration(KeyRates([0.25, 1, 5, 10, 30]), curve) do c
    pv(c)
end
```
"""
function duration(valuation_fn::F, kr::KeyRates, curve::AYM) where {F}
    r = _keyrate(curve, kr.tenors, valuation_fn)
    return _relative(r, r.gradient; negate = true)
end
function duration(kr::KeyRates, curve::AYM, cfs::_CashflowCollection, times...)
    r = _keyrate_analytic(curve, kr.tenors, _cashflow_inputs(cfs, times...)...)
    return _relative(r, r.gradient; negate = true)
end

"""
    duration(valuation_fn, ::DV01, kr::KeyRates, curve::AbstractYieldModel) -> Vector
    duration(::DV01, kr::KeyRates, curve::AbstractYieldModel, cfs, times) -> Vector

Per-knot signed DV01s for any `AbstractYieldModel`: the `KeyRates` variants of
`duration` in dollars per basis point. Their sum is ≈ the parallel DV01,
`duration(DV01(), curve, cfs, times)`.
"""
function duration(valuation_fn::F, ::DV01, kr::KeyRates, curve::AYM) where {F}
    r = _keyrate(curve, kr.tenors, valuation_fn)
    return _per_bp(r, r.gradient)
end
function duration(::DV01, kr::KeyRates, curve::AYM, cfs::_CashflowCollection, times...)
    r = _keyrate_analytic(curve, kr.tenors, _cashflow_inputs(cfs, times...)...)
    return _per_bp(r, r.gradient)
end

"""
    duration(valuation_fn, ::IR01, base::AbstractYieldModel, credit::AbstractYieldModel) -> scalar
    duration(valuation_fn, ::IR01, kr::KeyRates, base, credit) -> Vector
    duration(valuation_fn, ::CS01, ...) -> ...

Two-curve IR01 and CS01 of a valuation callback that receives `(base, credit)`:
`-∂V/∂s / 10000`, the first-order value lost per basis point. IR01 shifts the continuous zero
rates of the base curve and holds the credit curve fixed; CS01 shifts the credit curve and holds
the base curve fixed. Each is the callback's DV01 in its one curve. The scalar forms apply a
parallel shift and are ≈ the sums of the `KeyRates` vectors.

For fixed cashflows discounted at `base + credit`, IR01 and CS01 both equal
`duration(DV01(), base + credit, cfs, times)`; a callback can give different values when the
curves affect payments differently. These curve shifts do not
recalibrate to bumped market quotes; see [Two curves: IR01 and CS01](@ref) and
[Market Inputs](@ref).

```julia
duration(IR01(), base, credit) do b, c
    present_value(b + c, cfs, times)
end
```
"""
duration(valuation_fn::F, ::IR01, base::AYM, credit::AYM) where {F} = duration(b -> valuation_fn(b, credit), DV01(), base)
duration(valuation_fn::F, ::CS01, base::AYM, credit::AYM) where {F} = duration(c -> valuation_fn(base, c), DV01(), credit)
duration(valuation_fn::F, ::IR01, kr::KeyRates, base::AYM, credit::AYM) where {F} =
    duration(b -> valuation_fn(b, credit), DV01(), kr, base)
duration(valuation_fn::F, ::CS01, kr::KeyRates, base::AYM, credit::AYM) where {F} =
    duration(c -> valuation_fn(base, c), DV01(), kr, credit)

"""
    convexity(valuation_fn, kr::KeyRates, curve::AbstractYieldModel) -> Matrix
    convexity(kr::KeyRates, curve::AbstractYieldModel, cfs, times) -> Matrix
    convexity(valuation_fn, base::AbstractYieldModel, credit::AbstractYieldModel) -> NamedTuple
    convexity(valuation_fn, kr::KeyRates, base, credit) -> NamedTuple

Return normalized convexity for a yield model or a pair of curves. Matrix entries are
`(∂²V/∂rᵢ∂rⱼ) / V`. For a single curve's scalar parallel convexity, use
`convexity(curve, cfs, times)` or `convexity(valuation_fn, curve)`.

Empty and all-zero cashflow streams return zero convexity in the usual shape; see
[Zero cashflow streams](@ref).

The two-curve scalar forms return the parallel blocks `(; base, credit, cross)`,
each `(∂²V/∂sᵢ∂sⱼ) / V` for continuous-zero parallel shifts of the named curves.
They are ≈ the sums of the corresponding `KeyRates` blocks, including cross terms.
`cross` is the mixed derivative divided by `V`, without an extra factor of two.
For decimal shifts `u` and `v`, the second-order P&L is
`V / 2 * (base * u^2 + 2 * cross * u * v + credit * v^2)`.
For fixed cashflows discounted at `base + credit`, all three blocks equal
`convexity(base + credit, cfs, times)` when the value is nonzero. Shifting both curves by one
basis point shifts their combined continuous zero rates by two basis points. See
[Two-curve convexity blocks](@ref).
Use [`sensitivities`](@ref) to also obtain value and duration or DV01 from the same
derivatives.
"""
function convexity(valuation_fn::F, kr::KeyRates, curve::AYM) where {F}
    r = _keyrate(curve, kr.tenors, valuation_fn; order = 2)
    return _relative(r, r.hessian)
end
function convexity(kr::KeyRates, curve::AYM, cfs::_CashflowCollection, times...)
    r = _keyrate_analytic(curve, kr.tenors, _cashflow_inputs(cfs, times...)...; order = 2)
    return _relative(r, r.hessian)
end

function convexity(valuation_fn::F, base::AYM, credit::AYM) where {F}
    r = _keyrate((; base, credit), _PARALLEL_GRID, valuation_fn; order = 2)
    return _base_credit_cross(r, only)
end
function convexity(valuation_fn::F, kr::KeyRates, base::AYM, credit::AYM) where {F}
    r = _keyrate((; base, credit), kr.tenors, valuation_fn; order = 2)
    return _base_credit_cross(r)
end

"""
    sensitivities(valuation_fn, kr::KeyRates, curve::AbstractYieldModel) -> NamedTuple
    sensitivities(kr::KeyRates, curve::AbstractYieldModel, cfs, times) -> NamedTuple
    sensitivities(valuation_fn, ::DV01, kr::KeyRates, curve::AbstractYieldModel) -> NamedTuple
    sensitivities(valuation_fn, kr::KeyRates, base::AbstractYieldModel, credit::AbstractYieldModel) -> NamedTuple
    sensitivities(valuation_fn, ::DV01, kr::KeyRates, base, credit) -> NamedTuple

Calculate value, key-rate durations or DV01s, and convexity together on the
[`KeyRates`](@ref) grid. Callbacks use AD; fixed cashflows use analytic derivatives.
Normalized durations and convexities do not change when the position's sign does; dollar
DV01s change sign with the position.

Empty and all-zero cashflow streams return zero value and risk in the usual shapes; see
[Zero cashflow streams](@ref) for their numeric types. Nonzero amounts that offset to
zero present value keep their dollar risk but have undefined normalized risk
(`NaN`/`Inf`); for portfolio risk, sum values and dollar derivatives before
normalizing. Callback and contract forms do not use the zero-stream convention: a zero
value from them gives undefined normalized risk.
"""
function sensitivities(valuation_fn::F, kr::KeyRates, curve::AYM) where {F}
    return _single_curve_sensitivities(_keyrate(curve, kr.tenors, valuation_fn; order = 2))
end
function sensitivities(kr::KeyRates, curve::AYM, cfs::_CashflowCollection, times...)
    return _single_curve_sensitivities(_keyrate_analytic(curve, kr.tenors, _cashflow_inputs(cfs, times...)...; order = 2))
end
_single_curve_sensitivities(r) = (;
    value = r.value,
    durations = _relative(r, r.gradient; negate = true),
    convexities = _relative(r, r.hessian),
)

function sensitivities(valuation_fn::F, ::DV01, kr::KeyRates, curve::AYM) where {F}
    return _single_curve_dv01s(_keyrate(curve, kr.tenors, valuation_fn; order = 2))
end
function sensitivities(::DV01, kr::KeyRates, curve::AYM, cfs::_CashflowCollection, times...)
    return _single_curve_dv01s(_keyrate_analytic(curve, kr.tenors, _cashflow_inputs(cfs, times...)...; order = 2))
end
_single_curve_dv01s(r) = (;
    value = r.value,
    dv01s = _per_bp(r, r.gradient),
    convexities = _relative(r, r.hessian),
)

function sensitivities(valuation_fn::F, kr::KeyRates, base::AYM, credit::AYM) where {F}
    return _two_curve_sensitivities(_keyrate((; base, credit), kr.tenors, valuation_fn; order = 2))
end
_two_curve_sensitivities(r) = (;
    value = r.value,
    base_durations = _relative(r, r.gradient.base; negate = true),
    credit_durations = _relative(r, r.gradient.credit; negate = true),
    convexities = _base_credit_cross(r),
)

function sensitivities(valuation_fn::F, ::DV01, kr::KeyRates, base::AYM, credit::AYM) where {F}
    return _two_curve_dv01s(_keyrate((; base, credit), kr.tenors, valuation_fn; order = 2))
end
_two_curve_dv01s(r) = (;
    value = r.value,
    base_dv01s = _per_bp(r, r.gradient.base),
    credit_dv01s = _per_bp(r, r.gradient.credit),
    convexities = _base_credit_cross(r),
)

"""
    sensitivities(valuation, kr::KeyRates, curves::NamedTuple) -> (; value, duration, dv01, key_rate, key_rate_dv01)
    sensitivities(kr::KeyRates, target; discount::NamedTuple, index) -> same

Differentiate `valuation(curves)` with respect to each named curve. Return value,
per-role parallel duration and DV01, and per-role key-rate duration and DV01
vectors on the `kr.tenors` grid. The contract form sums the `discount` layers and
projects coupons using `index`. For example, `discount = (; rf, credit, ilp)`
produces separate risk-free, credit, liquidity, and index sensitivities. The result
names the projection curve `index`, so a discount layer with that name throws an
`ArgumentError`.

Every named value must be an `AbstractYieldModel`. To differentiate with respect
to market inputs that the valuation turns into curves, pass named input vectors
instead: `sensitivities(valuation, inputs::NamedTuple)`.

```julia
sensitivities(KeyRates(tenors), (; rf, credit)) do c
    present_value(c.rf + c.credit, cfs, times)
end
```
"""
function sensitivities(valuation::F, kr::KeyRates, curves::NamedTuple{roles, <:Tuple{AYM, Vararg{AYM}}}) where {F, roles}
    return _parallel_and_key_rate(_ncurve_ad(valuation, curves, kr.tenors; order = 1))
end
# Per role: parallel duration and DV01 from the summed gradient, and the per-element vectors.
function _parallel_and_key_rate(r)
    n = _scale(r)
    return (;
        value = r.value,
        duration = map(g -> _relative(n, sum(g); negate = true), r.gradient),
        dv01 = map(g -> _per_bp(n, sum(g)), r.gradient),
        key_rate = map(g -> _relative(n, g; negate = true), r.gradient),
        key_rate_dv01 = map(g -> _per_bp(n, g), r.gradient),
    )
end
