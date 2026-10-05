## Curve sensitivities
# Parallel shifts and triangular continuous-zero bumps at the supplied tenors. Callbacks use
# ForwardDiff through Yield.TenorShift; fixed cashflows use analytic derivatives.

# Derivatives of `valuation(curves)` at zero shocks, with one shock per curve role: a parallel shift
# without a grid (`grid === nothing`), or a triangular hat bump at each tenor of `grid`. Without a
# grid the derivatives are numbers, so no tenor gradient or Hessian is built and summed.
function _curve_ad(valuation::F, curves::NamedTuple{roles}, grid, order) where {F, roles}
    _validate_grid(grid)
    shocked(s) = NamedTuple{roles}(ntuple(i -> _shocked(curves[i], grid, s[i]), length(curves)))
    return _shock_numbers(_named_ad(s -> valuation(shocked(s)), map(_ -> _zero_shocks(grid), curves), order), grid)
end
# One curve's derivatives, unkeyed.
_one_curve_ad(valuation::F, curve, grid, order) where {F} = _only_role(_curve_ad(c -> valuation(c.curve), (; curve), grid, order))

_validate_grid(::Nothing) = nothing
_validate_grid(grid) = _validate_tenors(grid)
_zero_shocks(::Nothing) = zeros(1)
_zero_shocks(grid) = zeros(length(grid))
_shocked(curve, ::Nothing, s) = _parallel_bumped(curve, only(s))
_shocked(curve, grid, s) = _bumped(curve, grid, s)
_shock_numbers(r, grid) = r
_shock_numbers(r, ::Nothing) = haskey(r, :hessian) ?
    (; r.value, gradient = map(only, r.gradient), hessian = map(row -> map(only, row), r.hessian), r.zero_stream) :
    (; r.value, gradient = map(only, r.gradient), r.zero_stream)

## Analytic derivatives for fixed cashflows
# After locating its hat interval, each payment updates at most two gradient
# entries and a 2×2 Hessian block. The value is Σ cf(t) * discount(curve, t), over the 1-based,
# equal-length inputs of `_cashflow_inputs`.
# Inline to eliminate result tuples across derivative-order and zero-stream branches.
@inline function _keyrate_analytic(curve, tenors::AbstractVector, cfs::AbstractVector, times, order)
    _validate_tenors(tenors)
    n = length(tenors)
    disc(t) = FinanceCore.discount(curve, t)
    second = order isa SecondOrder
    zero_stream = _iszero_cashflow_stream(cfs)
    if zero_stream
        value = _zero_cashflow_value(disc, cfs, times)
        T = promote_type(typeof(value), eltype(tenors))
        gradient = zeros(T, n)
        return second ? (; value, gradient, hessian = zeros(T, n, n), zero_stream) : (; value, gradient, zero_stream)
    end
    # Seed from a discounted payment to preserve curve numeric types and AD.
    # Its type must accommodate later terms, including payments at t=0.
    t0 = FinanceCore.timepoint(cfs[1], times[1])
    cfd0 = _cf_value(cfs[1]) * disc(t0)
    _, w0, _, _ = _active_hats(tenors, t0)
    T = promote_type(typeof(cfd0), typeof(t0 * cfd0 * w0), eltype(tenors))
    if second
        T = promote_type(T, typeof(t0 * t0 * cfd0 * w0 * w0))
    end
    grad_shared = zeros(T, n)
    hess_shared = second ? zeros(T, n, n) : nothing
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
        if second
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
    if second
        return (; value = V, gradient = grad_shared, hessian = hess_shared, zero_stream)
    else
        return (; value = V, gradient = grad_shared, zero_stream)
    end
end
_fixed_keyrate(curve, kr::KeyRates, order, cfs, times...) =
    _keyrate_analytic(curve, kr.tenors, _cashflow_inputs(cfs, times...)..., order)

## Public yield-model sensitivities

const _NamedCurves = NamedTuple{<:Any, <:Tuple{AYM, Vararg{AYM}}}

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
    r = _one_curve_ad(valuation_fn, curve, kr.tenors, FirstOrder())
    return _relative(r, r.gradient; negate = true)
end
duration(kr::KeyRates, curve::AYM, x::_Instrument, times...; kwargs...) =
    _fixed(x) ? _keyrate_duration(curve, kr, x, times...; kwargs...) : duration(Effective(), kr, curve, x, times...; kwargs...)
function _keyrate_duration(curve, kr, cfs, times...)
    r = _fixed_keyrate(curve, kr, FirstOrder(), cfs, times...)
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
    r = _one_curve_ad(valuation_fn, curve, kr.tenors, FirstOrder())
    return _per_bp(r, r.gradient)
end
duration(::DV01, kr::KeyRates, curve::AYM, x::_Instrument, times...; kwargs...) =
    _fixed(x) ? _keyrate_dv01(curve, kr, x, times...; kwargs...) : duration(DV01(), Effective(), kr, curve, x, times...; kwargs...)
function _keyrate_dv01(curve, kr, cfs, times...)
    r = _fixed_keyrate(curve, kr, FirstOrder(), cfs, times...)
    return _per_bp(r, r.gradient)
end

"""
    duration(valuation, ::DV01, [kr::KeyRates,] base::AbstractYieldModel, credit::AbstractYieldModel) -> NamedTuple
    duration(valuation, ::DV01, [kr::KeyRates,] curves::NamedTuple) -> NamedTuple

Signed DV01 of a valuation callback for each curve role, `(; base, credit)` or keyed by the names
of `curves`: numbers for a parallel shift of each curve, or per-tenor vectors with `KeyRates`.
These are the `dv01` fields of [`sensitivities`](@ref). The valuation receives `(base, credit)` or
a `NamedTuple` of curves. For two curves, the roles are the [`IR01`](@ref) and [`CS01`](@ref).
"""
duration(valuation::F, ::DV01, base::AYM, credit::AYM) where {F} =
    duration(c -> valuation(c.base, c.credit), DV01(), (; base, credit))
duration(valuation::F, ::DV01, kr::KeyRates, base::AYM, credit::AYM) where {F} =
    duration(c -> valuation(c.base, c.credit), DV01(), kr, (; base, credit))
duration(valuation::F, ::DV01, curves::_NamedCurves) where {F} = _role_dv01(_curve_ad(valuation, curves, nothing, FirstOrder()))
duration(valuation::F, ::DV01, kr::KeyRates, curves::_NamedCurves) where {F} =
    _role_dv01(_curve_ad(valuation, curves, kr.tenors, FirstOrder()))
_role_dv01(r) = map(g -> _per_bp(r, g), r.gradient)

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

The two-curve forms return the blocks keyed by role,
`(; base = (; base, credit), credit = (; base, credit))`, the `convexity` field of
[`sensitivities`](@ref) with [`SecondOrder`](@ref): each block is `(∂²V/∂sᵢ∂sⱼ) / V` for
continuous-zero shifts of the named curves. The parallel blocks are numbers, ≈ the sums of the
`KeyRates` blocks, and `base.credit == credit.base`. The `KeyRates` blocks are matrices, and
`credit.base` is the transpose of `base.credit`, which need not be symmetric. For decimal parallel
shifts `u` and `v`, the second-order P&L is
`V / 2 * (base.base * u^2 + 2 * base.credit * u * v + credit.credit * v^2)`.
For fixed cashflows discounted at `base + credit`, all four blocks equal
`convexity(base + credit, cfs, times)` when the value is nonzero. Shifting both curves by one
basis point shifts their combined continuous zero rates by two basis points. See
[Two-curve convexity blocks](@ref).
Use [`sensitivities`](@ref) with [`SecondOrder`](@ref) to also obtain the value, durations and
DV01s from the same derivatives.
"""
function convexity(valuation_fn::F, kr::KeyRates, curve::AYM) where {F}
    r = _one_curve_ad(valuation_fn, curve, kr.tenors, SecondOrder())
    return _relative(r, r.hessian)
end
convexity(kr::KeyRates, curve::AYM, x::_Instrument, times...; kwargs...) =
    _fixed(x) ? _keyrate_convexity(curve, kr, x, times...; kwargs...) : convexity(Effective(), kr, curve, x, times...; kwargs...)
function _keyrate_convexity(curve, kr, cfs, times...)
    r = _fixed_keyrate(curve, kr, SecondOrder(), cfs, times...)
    return _relative(r, r.hessian)
end

convexity(valuation_fn::F, base::AYM, credit::AYM) where {F} =
    _convexities(_curve_ad(c -> valuation_fn(c.base, c.credit), (; base, credit), nothing, SecondOrder()))
convexity(valuation_fn::F, kr::KeyRates, base::AYM, credit::AYM) where {F} =
    _convexities(_curve_ad(c -> valuation_fn(c.base, c.credit), (; base, credit), kr.tenors, SecondOrder()))

## Value, duration, DV01 and convexity together

"""
    sensitivities([order,] [KeyRates(tenors),] curve, cfs, times = eachindex(cfs)) -> NamedTuple
    sensitivities(valuation, [order,] [KeyRates(tenors),] curve) -> NamedTuple
    sensitivities(valuation, [order,] [KeyRates(tenors),] base, credit) -> NamedTuple
    sensitivities(valuation, [order,] [KeyRates(tenors),] curves::NamedTuple) -> NamedTuple

Return value and risk from one derivative calculation. `order` is [`FirstOrder()`](@ref), the
default, or [`SecondOrder()`](@ref):

- `FirstOrder()` returns `(; value, duration, dv01)`.
- `SecondOrder()` returns `(; value, duration, dv01, convexity)`.

`value` is always a number. The derivative fields are numbers for one parallel shift of the
curve, which moves in its own coordinate as in [`duration`](@ref) (see
[Shock coordinates](@ref)). With [`KeyRates`](@ref) they are per-tenor vectors, and `convexity`
is a matrix. With several curves, the derivative fields are keyed by curve role: `dv01.base` and
`convexity.base.credit` for `(base, credit)`, and the given names for `curves`. Each role takes
its own shock, so the cross blocks of `convexity` are the mixed derivatives.

- `duration`: `-∂V/∂s / V`
- `dv01`: `-∂V/∂s / 10000`, the first-order value lost for a one-basis-point increase
- `convexity`: `∂²V/∂sᵢ∂sⱼ / V`

Without `KeyRates`, each role's shift is differentiated directly. With them, the hats sum to one,
so the parallel measures are ≈ the sums of the key-rate ones. `FirstOrder()` takes no second
derivatives, and `SecondOrder()` takes the value and both derivatives from one pass. A valuation
that has no second derivative throws its own error.

Fixed cashflows use analytic derivatives on one curve; valuation callbacks use AD. The valuation
receives the shifted curve, the shifted `(base, credit)`, or a `NamedTuple` of shifted curves with
the given names. Normalized duration and convexity do not change when the position's sign does;
DV01 does.

Empty and all-zero cashflow streams return zero value and risk; see [Zero cashflow streams](@ref).
Nonzero amounts that offset to zero present value keep their dollar risk but have undefined
normalized risk (`NaN`/`Inf`); for portfolio risk, sum values and dollar derivatives before
normalizing. Callback forms do not use the zero-stream convention: a zero value from them gives
undefined normalized risk.

```julia
s = sensitivities(KeyRates(tenors), curve, cfs, times)       # s.duration, s.dv01: vectors
s = sensitivities(SecondOrder(), curve, cfs, times)          # s.convexity: a number
sensitivities(KeyRates(tenors), (; rf, credit)) do c         # s.dv01.rf, s.dv01.credit
    present_value(c.rf + c.credit, cfs, times)
end
```

See also the contract forms, `sensitivities(discount, contract; index)`, the market-input form,
`sensitivities(valuation, inputs::NamedTuple)`, and [`Scenarios`](@ref).
"""
sensitivities(valuation::F, curve::_YieldInput) where {F} = sensitivities(valuation, FirstOrder(), curve)
sensitivities(valuation::F, kr::KeyRates, curve::AYM) where {F} = sensitivities(valuation, FirstOrder(), kr, curve)
sensitivities(valuation::F, order::_Order, curve::_YieldInput) where {F} =
    _sensitivities(_one_curve_ad(valuation, curve, nothing, order), order)
sensitivities(valuation::F, order::_Order, kr::KeyRates, curve::AYM) where {F} =
    _sensitivities(_one_curve_ad(valuation, curve, kr.tenors, order), order)

sensitivities(valuation::F, base::AYM, credit::AYM) where {F} = sensitivities(valuation, FirstOrder(), base, credit)
sensitivities(valuation::F, kr::KeyRates, base::AYM, credit::AYM) where {F} = sensitivities(valuation, FirstOrder(), kr, base, credit)
sensitivities(valuation::F, order::_Order, base::AYM, credit::AYM) where {F} =
    _sensitivities(_curve_ad(c -> valuation(c.base, c.credit), (; base, credit), nothing, order), order)
sensitivities(valuation::F, order::_Order, kr::KeyRates, base::AYM, credit::AYM) where {F} =
    _sensitivities(_curve_ad(c -> valuation(c.base, c.credit), (; base, credit), kr.tenors, order), order)

sensitivities(valuation::F, curves::_NamedCurves) where {F} = sensitivities(valuation, FirstOrder(), curves)
sensitivities(valuation::F, kr::KeyRates, curves::_NamedCurves) where {F} = sensitivities(valuation, FirstOrder(), kr, curves)
sensitivities(valuation::F, order::_Order, curves::_NamedCurves) where {F} =
    _sensitivities(_curve_ad(valuation, curves, nothing, order), order)
sensitivities(valuation::F, order::_Order, kr::KeyRates, curves::_NamedCurves) where {F} =
    _sensitivities(_curve_ad(valuation, curves, kr.tenors, order), order)

# Fixed cashflows, a contract or a portfolio after a curve (`Scenarios`, or for contracts named
# discount layers): optional order and grid markers first.
const _CurveInput = Union{_YieldInput, NamedTuple, Scenarios}
sensitivities(curve::_CurveInput, x::_Instrument, times...; kwargs...) =
    _sensitivities_of(FirstOrder(), nothing, curve, x, times...; kwargs...)
sensitivities(kr::KeyRates, curve::_CurveInput, x::_Instrument, times...; kwargs...) =
    _sensitivities_of(FirstOrder(), kr.tenors, curve, x, times...; kwargs...)
sensitivities(order::_Order, curve::_CurveInput, x::_Instrument, times...; kwargs...) =
    _sensitivities_of(order, nothing, curve, x, times...; kwargs...)
sensitivities(order::_Order, kr::KeyRates, curve::_CurveInput, x::_Instrument, times...; kwargs...) =
    _sensitivities_of(order, kr.tenors, curve, x, times...; kwargs...)

_sensitivities_of(order, grid, curve, x, times...; kwargs...) = _fixed(x) ?
    _fixed_sensitivities(order, grid, curve, x, times...; kwargs...) :
    _contract_sensitivities(order, grid, curve, x, times...; kwargs...)
function _fixed_sensitivities(order, grid, yield, cfs, times...)
    amounts, ts = _cashflow_inputs(cfs, times...)
    return _sensitivities(_fixed_derivatives(yield, grid, amounts, ts, order), order)
end
_fixed_derivatives(yield, ::Nothing, cfs, times, order) = _parallel_analytic(yield, cfs, times, order)
_fixed_derivatives(curve::AYM, grid::AbstractVector, cfs, times, order) = _keyrate_analytic(curve, grid, cfs, times, order)
