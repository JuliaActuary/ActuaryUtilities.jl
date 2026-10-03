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
# entries and a 2×2 Hessian block.

# Value is Σ cf(t) * ∏ discount(curve, t). Every curve must be a discount layer;
# an index curve that projects coupons does not belong here. All curve roles
# have identical gradients and Hessian blocks, stored once internally.
# Inline to eliminate result tuples across derivative-order and zero-stream branches.
@inline function _ncurve_analytic(
        curves::NamedTuple, tenors::AbstractVector,
        cfs::AbstractVector, times; order = 1
    )
    _validate_tenors(tenors)
    _check_cashflow_times(cfs, times)
    n = length(tenors)
    disc(t) = prod(c -> FinanceCore.discount(c, t), values(curves))
    zero_stream = _iszero_cashflow_stream(cfs)
    if zero_stream
        value = _zero_cashflow_value(disc, cfs, times)
        T = promote_type(typeof(value), eltype(tenors))
        gradient = zeros(T, n)
        return order >= 2 ? (; value, gradient, hessian = zeros(T, n, n), zero_stream) : (; value, gradient, zero_stream)
    end
    # Seed from a discounted payment to preserve curve numeric types and AD.
    # Its type must accommodate later terms, including payments at t=0.
    k0 = firstindex(cfs)
    t0 = FinanceCore.timepoint(cfs[k0], times[k0])
    cfd0 = _cf_value(cfs[k0]) * disc(t0)
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
        # Tuple reduction specializes each curve type, avoiding per-payment boxing.
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

# The analytic kernel's derivatives are the same for every curve role and stored once. Keyed by
# role (without copying), they take the shape of `_ncurve_ad`'s result.
@inline function _role_keyed(an, ::NamedTuple{roles}) where {roles}
    k = length(roles)
    gradient = NamedTuple{roles}(ntuple(_ -> an.gradient, k))
    haskey(an, :hessian) || return (; an.value, gradient, an.zero_stream)
    hessian = NamedTuple{roles}(ntuple(_ -> NamedTuple{roles}(ntuple(_ -> an.hessian, k)), k))
    return (; an.value, gradient, hessian, an.zero_stream)
end

## Key-rate results
# Derivatives on the key-rate grid keyed by curve role, in the shape of `_ncurve_ad`'s result: from a
# valuation callback, which receives the curves in order (AD), or from fixed cashflows (analytic).
@inline _keyrate(curves::NamedTuple, tenors, valuation::F; order = 1) where {F} =
    _ncurve_ad(c -> valuation(values(c)...), curves, tenors; order)
@inline _keyrate(curves::NamedTuple, tenors, cfs::AbstractVector, times; order = 1) =
    _role_keyed(_ncurve_analytic(curves, tenors, cfs, times; order), curves)

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

Empty collections and collections whose amounts are all exactly zero return zero
key-rate durations by convention, with one entry per tenor, without valuing any payment.
Every cashflow needs a time; unused trailing times are ignored.
Wrapped `Cashflow` objects use their embedded amounts and payment times, including
when explicit `times` are supplied. Numeric amounts use the explicit times, which
default to `eachindex(cfs)` (periods `1:n`). The same holds for every key-rate,
two-curve, and named-curve cashflow form.
See [Zero cashflow streams](@ref) for numeric types and zero-net-value portfolios.

# Tenor grid

Choose `kr.tenors` independently of the curve's own knots. The grid must be
nonempty, finite, positive, and strictly increasing. It is validated both at
construction and when calculating sensitivities.

# Bump shape and endpoint extrapolation

The bump at the i-th knot is a triangular hat centered at `tenors[i]` with
support `[tenors[i-1], tenors[i+1]]` for interior knots. Endpoint bumps stay
constant beyond the grid. All sensitivity after the last tenor belongs to its
bucket; extend the grid to separate exposures at longer maturities.

The hats sum to one, so the sum of key-rate durations equals parallel duration.
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
    r = _keyrate((; curve), kr.tenors, valuation_fn)
    return _relative(r, r.gradient.curve; negate = true)
end
function duration(kr::KeyRates, curve::AYM, cfs::AbstractVector, times = eachindex(cfs))
    r = _keyrate((; curve), kr.tenors, cfs, times)
    return _relative(r, r.gradient.curve; negate = true)
end

"""
    duration(valuation_fn, ::DV01, kr::KeyRates, curve::AbstractYieldModel) -> Vector
    duration(::DV01, kr::KeyRates, curve::AbstractYieldModel, cfs, times) -> Vector

Per-knot signed DV01s for any `AbstractYieldModel`: the `KeyRates` variants of
`duration` in dollars per basis point. Their sum is the parallel DV01,
`duration(DV01(), curve, cfs, times)`.
"""
function duration(valuation_fn::F, ::DV01, kr::KeyRates, curve::AYM) where {F}
    r = _keyrate((; curve), kr.tenors, valuation_fn)
    return _per_bp(r, r.gradient.curve)
end
function duration(::DV01, kr::KeyRates, curve::AYM, cfs::AbstractVector, times = eachindex(cfs))
    r = _keyrate((; curve), kr.tenors, cfs, times)
    return _per_bp(r, r.gradient.curve)
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

For fixed cashflows discounted at `base + credit`, IR01 and CS01 are equal; a callback can give
different values when the curves affect payments differently. These curve shifts do not
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
    convexity(base::AbstractYieldModel, credit::AbstractYieldModel, cfs, times) -> NamedTuple
    convexity(valuation_fn, kr::KeyRates, base, credit) -> NamedTuple
    convexity(kr::KeyRates, base, credit, cfs, times) -> NamedTuple
    convexity(kr::KeyRates, curves::NamedTuple, cfs, times) -> NamedTuple{roles}{roles}

Return normalized convexity for a yield model, a pair of curves, or named
discount layers. Matrix entries are `(∂²V/∂rᵢ∂rⱼ) / V`. For a single curve's
scalar parallel convexity, use `convexity(curve, cfs, times)` or
`convexity(valuation_fn, curve)`.

Empty collections and collections whose amounts are all exactly zero return zero
convexity by convention, retaining the usual scalar, matrix, or named-block shape
without valuing any payment. Nonzero amounts that offset to zero present value
still have undefined normalized convexity (`NaN`/`Inf`).

For the `NamedTuple` form, every named curve must be a discount-role layer
(multiplicatively composed); do not pass `:index`. Per-pair outputs have equal
values under multiplicative composition, but each matrix is independent and can
be mutated without changing another block.

The two-curve scalar forms return the parallel blocks `(; base, credit, cross)`,
each `(∂²V/∂sᵢ∂sⱼ) / V` for continuous-zero parallel shifts of the named curves.
They equal the sums of the corresponding `KeyRates` blocks, including cross terms.
`cross` is the mixed derivative divided by `V`, without an extra factor of two.
For decimal shifts `u` and `v`, the second-order P&L is
`V / 2 * (base * u^2 + 2 * cross * u * v + credit * v^2)`.
For fixed cashflows discounted at `base + credit`, all three blocks coincide
where normalization is defined; key-rate block equality also requires matching
grids and bump functions. Shifting both curves by one basis point shifts their
combined continuous zero rates by two basis points. See [Two-curve convexity blocks](@ref).
Use [`sensitivities`](@ref) to also obtain value and duration or DV01 from the same
derivatives.
"""
function convexity(valuation_fn::F, kr::KeyRates, curve::AYM) where {F}
    r = _keyrate((; curve), kr.tenors, valuation_fn; order = 2)
    return _relative(r, r.hessian.curve.curve)
end
function convexity(kr::KeyRates, curve::AYM, cfs::AbstractVector, times = eachindex(cfs))
    r = _keyrate((; curve), kr.tenors, cfs, times; order = 2)
    return _relative(r, r.hessian.curve.curve)
end

function convexity(valuation_fn::F, base::AYM, credit::AYM) where {F}
    r = _keyrate((; base, credit), _PARALLEL_GRID, valuation_fn; order = 2)
    return _base_credit_cross(r, only)
end
function convexity(base::AYM, credit::AYM, cfs::AbstractVector, times = eachindex(cfs))
    # Fixed cashflows have analytic base, credit, and cross derivatives.
    r = _keyrate((; base, credit), _PARALLEL_GRID, cfs, times; order = 2)
    return _base_credit_cross(r, only)
end

function convexity(valuation_fn::F, kr::KeyRates, base::AYM, credit::AYM) where {F}
    r = _keyrate((; base, credit), kr.tenors, valuation_fn; order = 2)
    return _base_credit_cross(r)
end
function convexity(kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector, times = eachindex(cfs))
    r = _keyrate((; base, credit), kr.tenors, cfs, times; order = 2)
    return _base_credit_cross(r)
end

# Multi-curve NamedTuple cashflow form. Per-role and per-role-pair convexities
# for static cashflows. Values coincide under multiplicative discount
# composition; each public block owns an independent matrix.
function convexity(kr::KeyRates, curves::NamedTuple, cfs::AbstractVector, times = eachindex(cfs))
    r = _keyrate(curves, kr.tenors, cfs, times; order = 2)
    n = _scale(r)
    return map(row -> map(h -> _relative(n, h), row), r.hessian)
end

"""
    sensitivities(valuation_fn, kr::KeyRates, curve::AbstractYieldModel) -> NamedTuple
    sensitivities(kr::KeyRates, curve::AbstractYieldModel, cfs, times) -> NamedTuple
    sensitivities(valuation_fn, ::DV01, kr::KeyRates, curve::AbstractYieldModel) -> NamedTuple
    sensitivities(valuation_fn, kr::KeyRates, base::AbstractYieldModel, credit::AbstractYieldModel) -> NamedTuple
    sensitivities(kr::KeyRates, base::AbstractYieldModel, credit::AbstractYieldModel, cfs, times) -> NamedTuple
    sensitivities(valuation_fn, ::DV01, kr::KeyRates, base, credit) -> NamedTuple
    sensitivities(::DV01, kr::KeyRates, base, credit, cfs, times) -> NamedTuple
    sensitivities(kr::KeyRates, curves::NamedTuple, cfs, times) -> NamedTuple

Calculate value, key-rate durations or DV01s, and convexity together on the
[`KeyRates`](@ref) grid. Callbacks use AD; fixed cashflows use analytic derivatives.

For the `NamedTuple` cashflow form, every named curve is a multiplicatively
composed discount layer. Per-role durations and per-pair convexity matrices
have equal values but independent storage, so modifying one does not affect
another. Relative durations and convexities are invariant to a change of
position sign; dollar DV01s change sign with the position.

Empty collections and collections whose amounts are all exactly zero have zero
value and dollar risk; normalized duration and convexity are zero by convention.
Every cashflow needs a time; unused trailing times are ignored.
Shapes are preserved without valuing any payment. Zero-stream results have the
numeric type a nonempty stream's would: amounts, times, tenor grid, and the curve,
which is queried once at time zero. An untyped empty collection takes its type from
the curve. The zero check includes automatic-differentiation partials.

Nonzero amounts that offset to zero present value retain dollar exposures and have
undefined normalized risk (`NaN`/`Inf`). For portfolio risk, sum values and dollar
derivatives before normalizing. A zero callback or contract value alone does not
identify an empty or all-zero cashflow stream.
See [Zero cashflow streams](@ref) for batch numeric types and simulation RNG behavior.
"""
function sensitivities(valuation_fn::F, kr::KeyRates, curve::AYM) where {F}
    return _single_curve_sensitivities(_keyrate((; curve), kr.tenors, valuation_fn; order = 2))
end
function sensitivities(kr::KeyRates, curve::AYM, cfs::AbstractVector, times = eachindex(cfs))
    return _single_curve_sensitivities(_keyrate((; curve), kr.tenors, cfs, times; order = 2))
end
_single_curve_sensitivities(r) = (;
    value = r.value,
    durations = _relative(r, r.gradient.curve; negate = true),
    convexities = _relative(r, r.hessian.curve.curve),
)

function sensitivities(valuation_fn::F, ::DV01, kr::KeyRates, curve::AYM) where {F}
    return _single_curve_dv01s(_keyrate((; curve), kr.tenors, valuation_fn; order = 2))
end
function sensitivities(::DV01, kr::KeyRates, curve::AYM, cfs::AbstractVector, times = eachindex(cfs))
    return _single_curve_dv01s(_keyrate((; curve), kr.tenors, cfs, times; order = 2))
end
_single_curve_dv01s(r) = (;
    value = r.value,
    dv01s = _per_bp(r, r.gradient.curve),
    convexities = _relative(r, r.hessian.curve.curve),
)

function sensitivities(valuation_fn::F, kr::KeyRates, base::AYM, credit::AYM) where {F}
    return _two_curve_sensitivities(_keyrate((; base, credit), kr.tenors, valuation_fn; order = 2))
end
function sensitivities(kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector, times = eachindex(cfs))
    return _two_curve_sensitivities(_keyrate((; base, credit), kr.tenors, cfs, times; order = 2))
end
_two_curve_sensitivities(r) = (;
    value = r.value,
    base_durations = _relative(r, r.gradient.base; negate = true),
    credit_durations = _relative(r, r.gradient.credit; negate = true),
    convexities = _base_credit_cross(r),
)

# Multi-curve NamedTuple cashflow form. One AD-free pass returns per-role
# durations + per-role-pair N×N convexity blocks. Values coincide under
# multiplicative discount composition; public arrays are independent.
function sensitivities(kr::KeyRates, curves::NamedTuple, cfs::AbstractVector, times = eachindex(cfs))
    r = _keyrate(curves, kr.tenors, cfs, times; order = 2)
    n = _scale(r)
    return (;
        value = r.value,
        durations = map(g -> _relative(n, g; negate = true), r.gradient),
        convexities = map(row -> map(h -> _relative(n, h), row), r.hessian),
    )
end

function sensitivities(valuation_fn::F, ::DV01, kr::KeyRates, base::AYM, credit::AYM) where {F}
    return _two_curve_dv01s(_keyrate((; base, credit), kr.tenors, valuation_fn; order = 2))
end
function sensitivities(::DV01, kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector, times = eachindex(cfs))
    return _two_curve_dv01s(_keyrate((; base, credit), kr.tenors, cfs, times; order = 2))
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
produces separate risk-free, credit, liquidity, and index sensitivities.

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
