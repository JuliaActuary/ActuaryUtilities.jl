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

function _ad_derivatives(f::F, z, order) where {F}
    # The valuation can return BigFloat or an outer AD Dual even when bumps are
    # Float64. Establish its type before allocating the Hessian result buffers.
    value = f(z)
    g = zeros(typeof(value), length(z))
    if order == 1
        # Gradients take up to 64 inputs per pass; measured faster than ForwardDiff's default.
        ForwardDiff.gradient!(g, f, z, ForwardDiff.GradientConfig(f, z, ForwardDiff.Chunk(z, 64)))
        return (; value, gradient = g)
    end
    # Hessians keep the default chunk: their nested partials grow with its square.
    result = DiffResults.DiffResult(value, g, similar(g, length(z), length(z)))
    result = ForwardDiff.hessian!(result, f, z)
    return (; value = DiffResults.value(result), gradient = g, hessian = DiffResults.hessian(result))
end

# Shared derivative engine: derivatives of `f` at the named input vectors `x`. Its result,
# `(; value, gradient, hessian, zero_stream)` with derivatives keyed by role (`gradient.role`,
# `hessian.role.role`), is the one shape every key-rate and market-input method normalizes. A
# callback's value can be zero without its cashflows being zero, so `zero_stream` is always false.
function _named_ad(f::F, x::NamedTuple{roles}; order = 1) where {F, roles}
    stops = cumsum(map(length, values(x)))
    ranges = ntuple(i -> (i == 1 ? 1 : stops[i - 1] + 1):stops[i], length(x))
    # Closures read the typed tuple's length, which keeps the callback's return type inferable.
    part(v, i) = length(x) == 1 ? v : view(v, ranges[i])
    g(z) = f(NamedTuple{roles}(ntuple(i -> part(z, i), length(x))))
    result = _ad_derivatives(g, reduce(vcat, map(v -> float.(v), values(x))), order)
    # Closures capture the concretely typed parts, not `result`: Julia 1.10 loses the result type
    # when a closure captures a value whose type depends on `order`.
    value, grad = result.value, result.gradient
    gradient = NamedTuple{roles}(ntuple(i -> part(grad, i), length(x)))
    order == 1 && return (; value, gradient, zero_stream = false)
    h = result.hessian
    block(i, j) = length(x) == 1 ? h : view(h, ranges[i], ranges[j])
    hessian = NamedTuple{roles}(ntuple(i -> NamedTuple{roles}(ntuple(j -> block(i, j), length(x))), length(x)))
    return (; value, gradient, hessian, zero_stream = false)
end
