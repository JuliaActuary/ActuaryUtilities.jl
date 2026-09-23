## Key-rate sensitivities
# Apply triangular continuous-zero bumps at the supplied tenors. Callbacks use
# ForwardDiff through Yield.TenorShift; fixed cashflows use analytic derivatives.

const AYM = FinanceModels.Yield.AbstractYieldModel

# Triangular hats with flat extrapolation outside the knot range.
# At a knot τᵢ the bump equals bᵢ; between τᵢ and τᵢ₊₁ it is linear.
function _hat_bump(tenors, bumps, t)
    t <= first(tenors) && return first(bumps)
    t >= last(tenors)  && return last(bumps)
    i = searchsortedlast(tenors, t)
    w = (t - tenors[i]) / (tenors[i + 1] - tenors[i])
    return (one(w) - w) * bumps[i] + w * bumps[i + 1]
end

# Layer a hat-function zero-rate bump over `curve` lazily.
_bumped(curve, tenors, bumps) = FinanceModels.Yield.TenorShift(
    curve,
    (z, t) -> FinanceCore.Continuous(_hat_bump(tenors, bumps, t)) + z,
)

function _ad_derivatives(f::F, z, order) where {F}
    # The valuation can return BigFloat or an outer AD Dual even when bumps are
    # Float64. Establish its type before allocating the Hessian result buffers.
    value = f(z)
    g = zeros(typeof(value), length(z))
    if order == 1
        ForwardDiff.gradient!(g, f, z)
        return (; value, gradient = g)
    end
    result = DiffResults.DiffResult(value, g, similar(g, length(z), length(z)))
    result = ForwardDiff.hessian!(result, f, z)
    return (; value = DiffResults.value(result), gradient = g, hessian = DiffResults.hessian(result))
end

# Shared derivative engine for named curve roles on one tenor grid.
function _ncurve_ad(valuation::F, curves::NamedTuple{roles}, tenors; order = 1) where {F, roles}
    order in (1, 2) || throw(ArgumentError("derivative order must be 1 or 2"))
    grid = _validate_tenors(tenors)
    isempty(curves) && throw(ArgumentError("at least one curve role is required"))
    all(c -> c isa AYM, curves) || throw(ArgumentError("every curve role must be an AbstractYieldModel"))
    n, k = length(grid), length(curves)
    indices(i) = ((i - 1) * n + 1):(i * n)
    slice(b, i) = length(curves) == 1 ? b : view(b, indices(i))
    # Use the typed tuple's length to keep the callback's return type inferable.
    f(b) = valuation(NamedTuple{roles}(ntuple(i -> _bumped(curves[i], grid, slice(b, i)), length(curves))))
    z = zeros(k * n)
    result = _ad_derivatives(f, z, order)
    value, g = result.value, result.gradient
    gradient = NamedTuple{roles}(ntuple(i -> slice(g, i), k))
    order == 1 && return (; value, gradient)
    h = result.hessian
    block(i, j) = k == 1 ? h : view(h, indices(i), indices(j))
    hessian = NamedTuple{roles}(ntuple(i -> NamedTuple{roles}(ntuple(j -> block(i, j), k)), k))
    return (; value, gradient, hessian)
end

# Adapt named derivatives to the single- and two-curve result fields.
function _keyrate_ad(curve::AYM, tenors::AbstractVector, valuation_fn::F; order = 1) where {F}
    r = _ncurve_ad(c -> valuation_fn(c.curve), (; curve), tenors; order)
    result = (; value = r.value, gradient = r.gradient.curve)
    return order == 1 ? result : merge(result, (; hessian = r.hessian.curve.curve))
end
function _keyrate_ad(base::AYM, credit::AYM, tenors::AbstractVector, valuation_fn::F; order = 1) where {F}
    r = _ncurve_ad(c -> valuation_fn(c.base, c.credit), (; base, credit), tenors; order)
    result = (; value = r.value, base_gradient = r.gradient.base, credit_gradient = r.gradient.credit)
    return order == 1 ? result : merge(
            result, (;
                base_hessian = r.hessian.base.base,
                credit_hessian = r.hessian.credit.credit,
                cross_hessian = r.hessian.base.credit,
            )
        )
end

## Analytic derivatives for fixed cashflows
# After locating its hat interval, each payment updates at most two gradient
# entries and a 2×2 Hessian block.

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
        w_left = one(w_right) - w_right
        return i, w_left, i + 1, w_right
    end
end

# Adapt the shared analytic derivatives to one or two discount curves.
# Internal arrays may alias; public normalization creates independent arrays.
_keyrate_analytic(curve::AYM, tenors::AbstractVector, cfs::AbstractVector, times; order = 1) =
    _ncurve_analytic((; curve), tenors, cfs, times; order)

function _keyrate_analytic(
        base::AYM, credit::AYM, tenors::AbstractVector,
        cfs::AbstractVector, times; order = 1
    )
    an = _ncurve_analytic((; base, credit), tenors, cfs, times; order)
    order >= 2 || return (;
        value = an.value, zero_stream = an.zero_stream,
        base_gradient = an.gradient, credit_gradient = an.gradient,
    )
    return (;
        value = an.value, zero_stream = an.zero_stream,
        base_gradient = an.gradient, credit_gradient = an.gradient,
        base_hessian = an.hessian, credit_hessian = an.hessian, cross_hessian = an.hessian,
    )
end

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
    zero_stream = _iszero_cashflow_stream(cfs)
    if zero_stream
        value = _zero_cashflow_value(cfs, times)
        T = promote_type(typeof(value), eltype(tenors))
        gradient = zeros(T, n)
        return order >= 2 ? (; value, gradient, hessian = zeros(T, n, n), zero_stream) : (; value, gradient, zero_stream)
    end
    # Seed from a discounted payment to preserve curve numeric types and AD.
    # Its type must accommodate later terms, including payments at t=0.
    disc(t) = prod(c -> FinanceCore.discount(c, t), values(curves))
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

# Normalize into independent base, credit, and cross-convexity matrices.
_conv_blocks(r, zero_stream = false) = (;
    base = _risk_ratio(r.base_hessian, r.value, zero_stream),
    credit = _risk_ratio(r.credit_hessian, r.value, zero_stream),
    cross = _risk_ratio(r.cross_hessian, r.value, zero_stream),
)

## Public yield-model sensitivities

# A one-knot grid is an exact parallel shift: its hat is flat everywhere. The
# scalar two-curve forms use it so they match the sums of the key-rate results.
const _PARALLEL_GRID = 1.0:1.0

"""
    duration(kr::KeyRates, valuation_fn, curve::AbstractYieldModel) -> Vector
    duration(kr::KeyRates, curve::AbstractYieldModel, cfs, times) -> Vector
    duration(kr::KeyRates, curve::AbstractYieldModel, cfs::AbstractVector{<:Cashflow}) -> Vector

Return normalized key-rate durations `-∂V/∂rᵢ / V` for an `AbstractYieldModel`.
Each `rᵢ` is a triangular continuous-zero bump at `kr.tenors[i]`. The base curve
is used directly, without resampling or refitting.

Empty collections and collections whose amounts are all exactly zero return zero
key-rate durations by convention, with one entry per tenor and no curve evaluation.
Every cashflow needs a time; unused trailing times are ignored.
Wrapped `Cashflow` objects use their embedded amounts and payment times, including
when explicit `times` are supplied. Numeric amounts use the explicit times.
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
duration(KeyRates([0.25, 1, 5, 10, 30]), pv, curve)

duration(KeyRates([0.25, 1, 5, 10, 30]), curve) do c
    pv(c)
end
```
"""
function duration(kr::KeyRates, valuation_fn::F, curve::AYM) where {F}
    ad = _keyrate_ad(curve, kr.tenors, valuation_fn)
    return -ad.gradient ./ ad.value
end
function duration(kr::KeyRates, curve::AYM, cfs::AbstractVector, times)
    an = _keyrate_analytic(curve, kr.tenors, cfs, times)
    return _risk_ratio(an.gradient, an.value, an.zero_stream; negate = true)
end
duration(kr::KeyRates, curve::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(kr, curve, _extract_cfs_times(cfs)...)

"""
    duration(::DV01, kr::KeyRates, valuation_fn, curve::AbstractYieldModel) -> Vector
    duration(::DV01, kr::KeyRates, curve::AbstractYieldModel, cfs, times) -> Vector

Per-knot signed DV01s for any `AbstractYieldModel`: the `KeyRates` variants of
`duration` in dollars per basis point. Their sum is the parallel DV01,
`duration(DV01(), curve, cfs, times)`.
"""
function duration(::DV01, kr::KeyRates, valuation_fn::F, curve::AYM) where {F}
    ad = _keyrate_ad(curve, kr.tenors, valuation_fn)
    return -ad.gradient ./ 10_000
end
function duration(::DV01, kr::KeyRates, curve::AYM, cfs::AbstractVector, times)
    an = _keyrate_analytic(curve, kr.tenors, cfs, times)
    return _risk_ratio(an.gradient, 10_000, an.zero_stream; negate = true)
end
duration(::DV01, kr::KeyRates, curve::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(DV01(), kr, curve, _extract_cfs_times(cfs)...)

"""
    duration(::IR01, valuation_fn, base::AbstractYieldModel, credit::AbstractYieldModel) -> scalar
    duration(::IR01, kr::KeyRates, valuation_fn, base, credit) -> Vector
    duration(::IR01, kr::KeyRates, base, credit, cfs, times) -> Vector
    duration(::CS01, ...) -> ...

Two-curve signed IR01/CS01 for any `AbstractYieldModel` pair. IR01 applies a
continuous-zero bump to the base (risk-free) curve only; CS01 bumps the credit
(spread) curve only. The callback receives `(base, credit)`, so the two curves can
play different roles. The scalar callback forms apply a parallel bump and equal the
sums of the `KeyRates` vectors. For fixed cashflows discounted at `base + credit`,
use the scalar cashflow form `duration(IR01(), base, credit, cfs, times)`.

```julia
duration(IR01(), base, credit) do b, c
    present_value(b + c, cfs, times)
end
```
"""
function duration(::IR01, valuation_fn::F, base::AYM, credit::AYM) where {F}
    ad = _keyrate_ad(base, credit, _PARALLEL_GRID, valuation_fn)
    return -only(ad.base_gradient) / 10_000
end

function duration(::IR01, kr::KeyRates, valuation_fn::F, base::AYM, credit::AYM) where {F}
    ad = _keyrate_ad(base, credit, kr.tenors, valuation_fn)
    return -ad.base_gradient ./ 10_000
end
function duration(::IR01, kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector, times)
    an = _keyrate_analytic(base, credit, kr.tenors, cfs, times)
    return _risk_ratio(an.base_gradient, 10_000, an.zero_stream; negate = true)
end
duration(::IR01, kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(IR01(), kr, base, credit, _extract_cfs_times(cfs)...)

function duration(::CS01, valuation_fn::F, base::AYM, credit::AYM) where {F}
    ad = _keyrate_ad(base, credit, _PARALLEL_GRID, valuation_fn)
    return -only(ad.credit_gradient) / 10_000
end

function duration(::CS01, kr::KeyRates, valuation_fn::F, base::AYM, credit::AYM) where {F}
    ad = _keyrate_ad(base, credit, kr.tenors, valuation_fn)
    return -ad.credit_gradient ./ 10_000
end
function duration(::CS01, kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector, times)
    an = _keyrate_analytic(base, credit, kr.tenors, cfs, times)
    return _risk_ratio(an.credit_gradient, 10_000, an.zero_stream; negate = true)
end
duration(::CS01, kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(CS01(), kr, base, credit, _extract_cfs_times(cfs)...)

# Do-block-first forwarders (support `f(args...) do x; ...; end` syntax)
duration(vf::Function, kr::KeyRates, curve::AYM) = duration(kr, vf, curve)
duration(vf::Function, ::DV01, curve::AYM) = duration(DV01(), curve, vf)
duration(vf::Function, ::DV01, kr::KeyRates, curve::AYM) = duration(DV01(), kr, vf, curve)
duration(vf::Function, ::IR01, base::AYM, credit::AYM) = duration(IR01(), vf, base, credit)
duration(vf::Function, ::IR01, kr::KeyRates, base::AYM, credit::AYM) = duration(IR01(), kr, vf, base, credit)
duration(vf::Function, ::CS01, base::AYM, credit::AYM) = duration(CS01(), vf, base, credit)
duration(vf::Function, ::CS01, kr::KeyRates, base::AYM, credit::AYM) = duration(CS01(), kr, vf, base, credit)

"""
    convexity(kr::KeyRates, valuation_fn, curve::AbstractYieldModel) -> Matrix
    convexity(kr::KeyRates, curve::AbstractYieldModel, cfs, times) -> Matrix
    convexity(valuation_fn, base::AbstractYieldModel, credit::AbstractYieldModel) -> NamedTuple
    convexity(base::AbstractYieldModel, credit::AbstractYieldModel, cfs, times) -> NamedTuple
    convexity(kr::KeyRates, base, credit, cfs, times) -> NamedTuple
    convexity(kr::KeyRates, curves::NamedTuple, cfs, times) -> NamedTuple{roles}{roles}

Return normalized convexity for a yield model, a pair of curves, or named
discount layers. Matrix entries are `(∂²V/∂rᵢ∂rⱼ) / V`. For a single curve's
scalar parallel convexity, use `convexity(curve, cfs, times)` or
`convexity(curve, valuation_fn)`.

Empty collections and collections whose amounts are all exactly zero return zero
convexity by convention, retaining the usual scalar, matrix, or named-block shape
without evaluating the curve. Nonzero amounts that offset to zero present value
still have undefined normalized convexity (`NaN`/`Inf`).

For the `NamedTuple` form, every named curve must be a discount-role layer
(multiplicatively composed); do not pass `:index`. Per-pair outputs have equal
values under multiplicative composition, but each matrix is independent and can
be mutated without changing another block.

The two-curve scalar forms return the parallel blocks `(; base, credit, cross)`,
each `(∂²V/∂sᵢ∂sⱼ) / V` for continuous-zero parallel shifts of the named curves.
They equal the sums of the corresponding `KeyRates` blocks, including cross terms.
Use [`sensitivities`](@ref) to also obtain value and duration or DV01 from the same
derivatives.
"""
function convexity(kr::KeyRates, valuation_fn::F, curve::AYM) where {F}
    ad = _keyrate_ad(curve, kr.tenors, valuation_fn; order = 2)
    return ad.hessian ./ ad.value
end
function convexity(kr::KeyRates, curve::AYM, cfs::AbstractVector, times)
    an = _keyrate_analytic(curve, kr.tenors, cfs, times; order = 2)
    return _risk_ratio(an.hessian, an.value, an.zero_stream)
end
convexity(kr::KeyRates, curve::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = convexity(kr, curve, _extract_cfs_times(cfs)...)

function convexity(valuation_fn::F, base::AYM, credit::AYM) where {F}
    ad = _keyrate_ad(base, credit, _PARALLEL_GRID, valuation_fn; order = 2)
    return (;
        base = only(ad.base_hessian) / ad.value,
        credit = only(ad.credit_hessian) / ad.value,
        cross = only(ad.cross_hessian) / ad.value,
    )
end
# A KeyRates marker in the first position selects the key-rate method rather than
# treating the marker as a two-curve valuation callback.
convexity(kr::KeyRates, valuation_fn::AYM, curve::AYM) =
    invoke(convexity, Tuple{KeyRates, Any, AYM}, kr, valuation_fn, curve)
function convexity(base::AYM, credit::AYM, cfs::AbstractVector, times)
    # Fixed cashflows have analytic base, credit, and cross derivatives.
    an = _keyrate_analytic(base, credit, _PARALLEL_GRID, cfs, times; order = 2)
    zero_stream = an.zero_stream
    return (;
        base = _risk_ratio(only(an.base_hessian), an.value, zero_stream),
        credit = _risk_ratio(only(an.credit_hessian), an.value, zero_stream),
        cross = _risk_ratio(only(an.cross_hessian), an.value, zero_stream),
    )
end
convexity(base::AYM, credit::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = convexity(base, credit, _extract_cfs_times(cfs)...)

function convexity(kr::KeyRates, valuation_fn::F, base::AYM, credit::AYM) where {F}
    ad = _keyrate_ad(base, credit, kr.tenors, valuation_fn; order = 2)
    return _conv_blocks(ad)
end
function convexity(kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector, times)
    an = _keyrate_analytic(base, credit, kr.tenors, cfs, times; order = 2)
    return _conv_blocks(an, an.zero_stream)
end
convexity(kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = convexity(kr, base, credit, _extract_cfs_times(cfs)...)

# Multi-curve NamedTuple cashflow form. Per-role and per-role-pair convexities
# for static cashflows. Values coincide under multiplicative discount
# composition; each public block owns an independent matrix.
function convexity(kr::KeyRates, curves::NamedTuple, cfs::AbstractVector, times)
    an = _ncurve_analytic(curves, kr.tenors, cfs, times; order = 2)
    roles = keys(curves)
    L = length(roles)
    normalized = _risk_ratio(an.hessian, an.value, an.zero_stream)
    return NamedTuple{roles}(ntuple(_ -> NamedTuple{roles}(ntuple(_ -> copy(normalized), L)), L))
end
convexity(kr::KeyRates, curves::NamedTuple, cfs::AbstractVector{<:FinanceCore.Cashflow}) =
    convexity(kr, curves, _extract_cfs_times(cfs)...)

# Do-block-first forwarders (support `f(args...) do x; ...; end` syntax). The
# two-curve scalar callback already takes the function first.
convexity(vf::Function, kr::KeyRates, curve::AYM) = convexity(kr, vf, curve)
convexity(vf::Function, kr::KeyRates, base::AYM, credit::AYM) = convexity(kr, vf, base, credit)

"""
    sensitivities(kr::KeyRates, valuation_fn, curve::AbstractYieldModel) -> NamedTuple
    sensitivities(kr::KeyRates, curve::AbstractYieldModel, cfs, times) -> NamedTuple
    sensitivities(::DV01, kr::KeyRates, valuation_fn, curve::AbstractYieldModel) -> NamedTuple
    sensitivities(kr::KeyRates, base::AbstractYieldModel, credit::AbstractYieldModel, cfs, times) -> NamedTuple
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
Shapes are preserved without evaluating the curve. Zero-stream result types come
from the amounts, times, and tenor grid; abstractly typed empty inputs fall back
to `Float64`. The zero check includes automatic-differentiation partials.

Nonzero amounts that offset to zero present value retain dollar exposures and have
undefined normalized risk (`NaN`/`Inf`). For portfolio risk, sum values and dollar
derivatives before normalizing. A zero callback or contract value alone does not
identify an empty or all-zero cashflow stream.
See [Zero cashflow streams](@ref) for batch numeric types and simulation RNG behavior.
"""
function sensitivities(kr::KeyRates, valuation_fn::F, curve::AYM) where {F}
    ad = _keyrate_ad(curve, kr.tenors, valuation_fn; order = 2)
    return (;
        value = ad.value,
        durations = -ad.gradient ./ ad.value,
        convexities = ad.hessian ./ ad.value,
    )
end
function sensitivities(kr::KeyRates, curve::AYM, cfs::AbstractVector, times)
    an = _keyrate_analytic(curve, kr.tenors, cfs, times; order = 2)
    zero_stream = an.zero_stream
    return (;
        value = an.value,
        durations = _risk_ratio(an.gradient, an.value, zero_stream; negate = true),
        convexities = _risk_ratio(an.hessian, an.value, zero_stream),
    )
end
sensitivities(kr::KeyRates, curve::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = sensitivities(kr, curve, _extract_cfs_times(cfs)...)

function sensitivities(::DV01, kr::KeyRates, valuation_fn::F, curve::AYM) where {F}
    ad = _keyrate_ad(curve, kr.tenors, valuation_fn; order = 2)
    return (;
        value = ad.value,
        dv01s = -ad.gradient ./ 10_000,
        convexities = ad.hessian ./ ad.value,
    )
end
function sensitivities(::DV01, kr::KeyRates, curve::AYM, cfs::AbstractVector, times)
    an = _keyrate_analytic(curve, kr.tenors, cfs, times; order = 2)
    zero_stream = an.zero_stream
    return (;
        value = an.value,
        dv01s = _risk_ratio(an.gradient, 10_000, zero_stream; negate = true),
        convexities = _risk_ratio(an.hessian, an.value, zero_stream),
    )
end
sensitivities(::DV01, kr::KeyRates, curve::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = sensitivities(DV01(), kr, curve, _extract_cfs_times(cfs)...)

function sensitivities(kr::KeyRates, valuation_fn::F, base::AYM, credit::AYM) where {F}
    ad = _keyrate_ad(base, credit, kr.tenors, valuation_fn; order = 2)
    return (;
        value = ad.value,
        base_durations = -ad.base_gradient ./ ad.value,
        credit_durations = -ad.credit_gradient ./ ad.value,
        convexities = _conv_blocks(ad),
    )
end
function sensitivities(kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector, times)
    an = _keyrate_analytic(base, credit, kr.tenors, cfs, times; order = 2)
    zero_stream = an.zero_stream
    return (;
        value = an.value,
        base_durations = _risk_ratio(an.base_gradient, an.value, zero_stream; negate = true),
        credit_durations = _risk_ratio(an.credit_gradient, an.value, zero_stream; negate = true),
        convexities = _conv_blocks(an, zero_stream),
    )
end
sensitivities(kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = sensitivities(kr, base, credit, _extract_cfs_times(cfs)...)

# Multi-curve NamedTuple cashflow form. One AD-free pass returns per-role
# durations + per-role-pair N×N convexity blocks. Values coincide under
# multiplicative discount composition; public arrays are independent.
function sensitivities(kr::KeyRates, curves::NamedTuple, cfs::AbstractVector, times)
    an = _ncurve_analytic(curves, kr.tenors, cfs, times; order = 2)
    roles = keys(curves)
    L = length(roles)
    zero_stream = an.zero_stream
    dur_normalized = _risk_ratio(an.gradient, an.value, zero_stream; negate = true)
    conv_normalized = _risk_ratio(an.hessian, an.value, zero_stream)
    durations = NamedTuple{roles}(ntuple(_ -> copy(dur_normalized), L))
    convexities = NamedTuple{roles}(ntuple(_ -> NamedTuple{roles}(ntuple(_ -> copy(conv_normalized), L)), L))
    return (; value = an.value, durations, convexities)
end
sensitivities(kr::KeyRates, curves::NamedTuple, cfs::AbstractVector{<:FinanceCore.Cashflow}) =
    sensitivities(kr, curves, _extract_cfs_times(cfs)...)

function sensitivities(::DV01, kr::KeyRates, valuation_fn::F, base::AYM, credit::AYM) where {F}
    ad = _keyrate_ad(base, credit, kr.tenors, valuation_fn; order = 2)
    return (;
        value = ad.value,
        base_dv01s = -ad.base_gradient ./ 10_000,
        credit_dv01s = -ad.credit_gradient ./ 10_000,
        convexities = _conv_blocks(ad),
    )
end
function sensitivities(::DV01, kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector, times)
    an = _keyrate_analytic(base, credit, kr.tenors, cfs, times; order = 2)
    zero_stream = an.zero_stream
    return (;
        value = an.value,
        base_dv01s = _risk_ratio(an.base_gradient, 10_000, zero_stream; negate = true),
        credit_dv01s = _risk_ratio(an.credit_gradient, 10_000, zero_stream; negate = true),
        convexities = _conv_blocks(an, zero_stream),
    )
end
sensitivities(::DV01, kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = sensitivities(DV01(), kr, base, credit, _extract_cfs_times(cfs)...)

# Do-block-first forwarders (support `f(args...) do x; ...; end` syntax)
sensitivities(vf::Function, kr::KeyRates, curve::AYM) = sensitivities(kr, vf, curve)
sensitivities(vf::Function, ::DV01, kr::KeyRates, curve::AYM) = sensitivities(DV01(), kr, vf, curve)
sensitivities(vf::Function, kr::KeyRates, base::AYM, credit::AYM) = sensitivities(kr, vf, base, credit)
sensitivities(vf::Function, ::DV01, kr::KeyRates, base::AYM, credit::AYM) = sensitivities(DV01(), kr, vf, base, credit)

"""
    sensitivities(valuation, curves::NamedTuple; tenors) -> (; value, duration, dv01, key_rate)
    sensitivities(target, tenors; discount::NamedTuple, index) -> same

Differentiate `valuation(curves)` with respect to each named curve. Return value
and per-role duration, DV01, and key-rate vectors. The contract form sums the
`discount` layers and projects coupons using `index`. For example,
`discount = (; rf, credit, ilp)` produces separate risk-free, credit, liquidity,
and index sensitivities.
"""
function sensitivities(valuation::F, curves::NamedTuple; tenors) where {F}
    r = _ncurve_ad(valuation, curves, tenors; order = 1)
    v, grads = r.value, r.gradient
    roles = keys(curves)
    return (;
        value = v,
        duration = NamedTuple{roles}(map(g -> -sum(g) / v, values(grads))),
        dv01 = NamedTuple{roles}(map(g -> -sum(g) / 10_000, values(grads))),
        key_rate = NamedTuple{roles}(map(g -> -g ./ v, values(grads))),
    )
end
sensitivities(curves::NamedTuple, valuation::Function; tenors) = sensitivities(valuation, curves; tenors)  # do-block form
