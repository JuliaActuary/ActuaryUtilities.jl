module RiskMeasures
import ..Distributions
import ..StatsBase
import ..QuadGK

export Expectation, VaR, ValueAtRisk, CTE, ConditionalTailExpectation, WangTransform, DualPower, ProportionalHazard

abstract type RiskMeasure end

"""
    g(rm::RiskMeasure,x)

The probability distortion function associated with the given risk measure.

See [Distortion Function g(u)](@ref)
"""
function g(rm::RiskMeasure, x) end

"""
    Expectation()::RiskMeasure
    Expectation()(risk)::T (where T is the type of values sampled in `risk`)

The expected value of the risk.

`Expectation()` returns a functor which can then be called on a risk distribution.

## Examples

```julia-repl
julia> Expectation()(rand(1000))
0.4793223308812537

julia> rm = Expectation()
ActuaryUtilities.RiskMeasures.Expectation()

julia> rm(rand(1000))
0.4941708036889741
```
"""
struct Expectation <: RiskMeasure end
g(rm::Expectation, x) = x

"""
     VaR(α)::RiskMeasure
     VaR(α)(risk)::T (where T is the type of values sampled in `risk`)

The `α`th quantile of the `risk` distribution is the Value at Risk, or αth quantile. `risk` can be a univariate distribution or an array of outcomes.
Assumes more positive values are higher risk measures, so a higher p will return a more positive number. For a discrete risk, the VaR returned is the first value above the αth percentile.

`VaR(α)` returns a functor which can then be called on a risk distribution.

## Parameters
- α: [0,1.0) 

## Examples

```julia-repl
julia> VaR(0.95)(rand(1000))
0.9561843082268024

julia> rm = VaR(0.95)
VaR{Float64}(0.95)

julia> rm(rand(1000))
0.9597070153670079
```
"""
struct VaR{T<:Real} <: RiskMeasure
    α::T

    function VaR(α::T) where {T}
        @assert 0 <= α < 1 "α of $α is not 0 ≤ α < 1"
        return new{T}(α)
    end
end
g(rm::VaR, x) = x < (1 - rm.α) ? 0 : 1

"""
[`VaR`](@ref)
"""
const ValueAtRisk = VaR

"""
    CTE(α)::RiskMeasure
    CTE(α)(risk)::T (where T is the type of values sampled in risk)

The Conditional Tail Expectation (CTE) at level α is the expected value of the risk distribution above the αth quantile. `risk` can be a univariate distribution or an array of outcomes.
Assumes more positive values are higher risk measures, so a higher p will return a more positive number.

CTE(α) returns a functor which can then be called on a risk distribution.

## Parameters

- α: [0,1.0) 

## Examples

```julia-repl
julia> CTE(0.95)(rand(1000))
0.9766218612020593

julia> rm = CTE(0.95)
CTE{Float64}(0.95)

julia> rm(rand(1000))
0.9739835010268733
```
"""
struct CTE{T<:Real} <: RiskMeasure
    α::T

    function CTE(α::T) where {T}
        @assert 0 <= α < 1 "α of $α is not 0 ≤ α < 1"
        return new{T}(α)
    end
end
g(rm::CTE, x) = x < (1 - rm.α) ? x / (1 - rm.α) : 1

"""
[`CTE`](@ref)
"""
const ConditionalTailExpectation = CTE

"""
    WangTransform(α)::RiskMeasure
    WangTransform(α)(risk)::T (where T is the type of values sampled in risk)

The Wang Transform is a distortion risk measure that transforms the cumulative distribution function (CDF) of the risk distribution using a normal distribution with mean Φ⁻¹(α) and standard deviation 1. risk can be a univariate distribution or an array of outcomes.

WangTransform(α) returns a functor which can then be called on a risk distribution.

## Parameters
- α: [0,1.0]

In the literature, sometimes λ is used where ``\\lambda = \\Phi^{-1}(\\alpha)``.


## Examples

```julia-repl
julia> WangTransform(0.95)(rand(1000))
0.8799465543360105

julia> rm = WangTransform(0.95)
WangTransform{Float64}(0.95)

julia> rm(rand(1000))
0.8892245759705852
```

## References
- "A Risk Measure That Goes Beyond Coherence", Shaun S. Wang, 2002
"""
struct WangTransform{T} <: RiskMeasure
    α::T
    function WangTransform(α::T) where {T}
        @assert 0 < α < 1 "α of $α is not 0 < α < 1"
        return new{T}(α)
    end
end
function g(rm::WangTransform, x)
    Φ_inv(x) = Distributions.quantile(Distributions.Normal(), x)
    Distributions.cdf(Distributions.Normal(), Φ_inv(x) + Φ_inv(rm.α))
end

"""
    DualPower(v)::RiskMeasure
    DualPower(v)(risk)::T (where T is the type of values sampled in risk)

The Dual Power distortion risk measure is defined as ``1 - (1 - x)^v``, where x is the cumulative distribution function (CDF) of the risk distribution and v is a positive parameter. risk can be a univariate distribution or an array of outcomes.

DualPower(v) returns a functor which can then be called on a risk distribution.

"""
struct DualPower{T} <: RiskMeasure
    v::T
end
g(rm::DualPower, x) = 1 - (1 - x)^rm.v

"""
    ProportionalHazard(y)::RiskMeasure
    ProportionalHazard(y)(risk)::T (where T is the type of values sampled in risk)

The Proportional Hazard distortion risk measure is defined as ``x^(1/y)``, where x is the cumulative distribution function (CDF) of the risk distribution and y is a positive parameter. risk can be a univariate distribution or an array of outcomes.
ProportionalHazard(y) returns a functor which can then be called on a risk distribution.

## Examples

```julia-repl
julia> ProportionalHazard(2)(rand(1000))
0.6659603556774121

julia> rm = ProportionalHazard(2)
ProportionalHazard{Int64}(2)

julia> rm(rand(1000))
0.6710587338367799
```
"""
struct ProportionalHazard{T} <: RiskMeasure
    y::T
end
g(rm::ProportionalHazard, x) = x^(1 / rm.y)

function (rm::RiskMeasure)(risk)
    # Definition 4.2 of "A Risk Measure that Goes Beyond Coherence", Wang 2002
    F(x) = cdf_func(risk)(x)
    H(x) = 1 - g(rm, 1 - x)
    integral1, _ = QuadGK.quadgk(x -> 1 - H(F(x)), 0, Inf)
    integral2, _ = QuadGK.quadgk(x -> H(F(x)), -Inf, 0)
    return integral1 - integral2
end

# ── Exact empirical specializations ─────────────────────────────────────────
#
# For a finite sample the Choquet integral above reduces to order statistics:
# computing it by adaptive quadrature over the ecdf's step function is both
# orders of magnitude slower and exposes integration tolerance at the steps.
# These methods evaluate the same functional exactly.
#
# With the empirical cdf F(x_(k)) = k/n, the "first value above the αth
# percentile" is the order statistic x_(k) with k the smallest index such that
# k/n > α.
function _first_index_above(n, α)
    k = clamp(floor(Int, n * α) + 1, 1, n)
    # floating-point n*α can land on either side of an exact boundary; nudge so
    # that k is exactly the first index with k/n > α under float comparison
    while k > 1 && (k - 1) / n > α
        k -= 1
    end
    while k < n && k / n <= α
        k += 1
    end
    return k
end

function (rm::VaR)(risk::AbstractArray{<:Real})
    n = length(risk)
    k = _first_index_above(n, rm.α)
    return partialsort(vec(risk), k)
end

# The Choquet-CTE distorts the tail by 1/(1-α): the crossing order statistic
# x_(k) receives the partial weight (k/n - α), and each of x_(k+1)…x_(n)
# receives 1/n, all normalized by (1-α). (CTE(0) is then exactly the mean.)
function (rm::CTE)(risk::AbstractArray{<:Real})
    n = length(risk)
    α = rm.α
    k = _first_index_above(n, α)
    tail = partialsort(vec(risk), k:n)
    partial = (k / n - α) * first(tail)
    rest = sum(@view tail[2:end]) / n
    return (partial + rest) / (1 - α)
end

(rm::Expectation)(risk::AbstractArray{<:Real}) = sum(risk) / length(risk)

"""
    cdf_func(risk)

Returns the appropriate cumulative distribution function depending on the type, specifically:

    cdf_func(S::AbstractArray{<:Real}) = StatsBase.ecdf(S)
    cdf_func(S::Distributions.UnivariateDistribution) = x -> Distributions.cdf(S, x)

"""
cdf_func(S::AbstractArray{<:Real}) = StatsBase.ecdf(S)
cdf_func(S::Distributions.UnivariateDistribution) = x -> Distributions.cdf(S, x)

end