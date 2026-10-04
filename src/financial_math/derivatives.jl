## Automatic-differentiation helpers

_primal(x) = x
_primal(x::ForwardDiff.Dual) = _primal(ForwardDiff.value(x))

# `f(x)` and `f′(x)` from the one evaluation `ForwardDiff.derivative(f, x)` makes. `x` and `f` may
# already carry partials of an outer differentiation, which the value keeps.
function _value_and_derivative(f::F, x::R) where {F, R <: Real}
    T = typeof(ForwardDiff.Tag(f, R))
    y = f(ForwardDiff.Dual{T}(x, one(x)))
    return ForwardDiff.value(T, y), ForwardDiff.extract_derivative(T, y)
end

# `f″(0) / f(0)`, the normalized second derivative. Taking the absolute value first leaves the
# ratio unchanged and keeps the nested derivatives of a negative value positive.
function _second_over_value(f::F) where {F}
    v(x) = abs(f(x))
    return ForwardDiff.derivative(y -> ForwardDiff.derivative(v, y), 0.0) / v(0.0)
end

# The value of `f` at the vector `z`, with its gradient at first order, and its gradient and Hessian
# from one pass at second order. The valuation can return BigFloat or an outer AD Dual even when
# `z` is Float64, so its type is established before allocating the result buffers.
function _ad_derivatives(f::F, z, ::FirstOrder) where {F}
    value = f(z)
    gradient = zeros(typeof(value), length(z))
    # Gradients take up to 64 inputs per pass; measured faster than ForwardDiff's default.
    ForwardDiff.gradient!(gradient, f, z, ForwardDiff.GradientConfig(f, z, ForwardDiff.Chunk(z, 64)))
    return (; value, gradient)
end
function _ad_derivatives(f::F, z, ::SecondOrder) where {F}
    value = f(z)
    gradient = zeros(typeof(value), length(z))
    # Hessians keep the default chunk: their nested partials grow with its square.
    result = DiffResults.DiffResult(value, gradient, similar(gradient, length(z), length(z)))
    result = ForwardDiff.hessian!(result, f, z)
    return (; value = DiffResults.value(result), gradient, hessian = DiffResults.hessian(result))
end

# Shared derivative engine: derivatives of `f` at the named input vectors `x`. Its result,
# `(; value, gradient, hessian, zero_stream)` with derivatives keyed by role (`gradient.role`,
# `hessian.role.role`, the Hessian at second order only), is the shape `_sensitivities`
# normalizes. A callback's value can be zero without its cashflows being zero, so `zero_stream`
# is always false.
function _named_ad(f::F, x::NamedTuple{roles}, order) where {F, roles}
    stops = cumsum(map(length, values(x)))
    ranges = ntuple(i -> (i == 1 ? 1 : stops[i - 1] + 1):stops[i], length(x))
    # Closures read the typed tuple's length, which keeps the callback's return type inferable.
    part(v, i) = length(x) == 1 ? v : view(v, ranges[i])
    g(z) = f(NamedTuple{roles}(ntuple(i -> part(z, i), length(x))))
    result = _ad_derivatives(g, reduce(vcat, map(v -> float.(v), values(x))), order)
    gradient = NamedTuple{roles}(ntuple(i -> part(result.gradient, i), length(x)))
    return _with_hessian((; result.value, gradient, zero_stream = false), result, ranges, roles, order)
end
_with_hessian(r, result, ranges, roles, ::FirstOrder) = r
function _with_hessian(r, result, ranges, roles, ::SecondOrder)
    h = result.hessian
    block(i, j) = length(ranges) == 1 ? h : view(h, ranges[i], ranges[j])
    hessian = NamedTuple{roles}(ntuple(i -> NamedTuple{roles}(ntuple(j -> block(i, j), length(ranges))), length(ranges)))
    return (; r.value, r.gradient, hessian, r.zero_stream)
end

# The derivatives of the one role of a result, unkeyed.
_only_role(r) = haskey(r, :hessian) ?
    (; r.value, gradient = only(r.gradient), hessian = only(only(r.hessian)), r.zero_stream) :
    (; r.value, gradient = only(r.gradient), r.zero_stream)

## The one normalizer
# `(; value, duration, dv01)` at first order, and `convexity` too at second order, from a derivative
# result `r`. Duration and convexity divide the dollar derivatives by the value; DV01 scales the
# first derivative to one basis point. Derivatives keyed by role give measures keyed by role. The
# derivatives are of the total value, so a portfolio's measures are its total dollar risk divided by
# its total value: a zero value gives non-finite normalized measures, while a zero stream gives
# zeros by convention. Each call makes new arrays, so results never share storage.
function _sensitivities(r, order)
    duration = _per_role(g -> _risk_ratio(g, r.value, r.zero_stream; negate = true), r.gradient)
    dv01 = _per_role(g -> _risk_ratio(g, 10_000, r.zero_stream; negate = true), r.gradient)
    order isa FirstOrder && return (; r.value, duration, dv01)
    convexity = _per_role(h -> _risk_ratio(h, r.value, r.zero_stream), r.hessian)
    return (; r.value, duration, dv01, convexity)
end
_per_role(f::F, x::NamedTuple) where {F} = map(y -> _per_role(f, y), x)
_per_role(f::F, x) where {F} = f(x)

# Normalized risk from part of a derivative result `r`, for the standalone measures. Durations and
# convexities are relative to the value; DV01s are dollars per basis point.
_relative(r, x; negate = false) = _risk_ratio(x, r.value, r.zero_stream; negate)
_per_bp(r, x) = _risk_ratio(x, 10_000, r.zero_stream; negate = true)
