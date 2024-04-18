module RiskMeasures
import ..Distributions
import ..StatsBase
import ..QuadGK

export VaR, ValueAtRisk, CTE, ConditionalTailExpectation, WangTransform, DualPower, ProportionalHazard

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
julia> RiskMeasures.Expectation(rand(1000))
0.4793223308812537

julia> rm = RiskMeasures.Expectation()
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
ValueAtRisk = VaR

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
ConditionalTailExpectation = CTE

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

"""
    cdf_function(risk)

Returns the appropriate cumulative distribution function depending on the type, specifically:

    cdf_func(S::AbstractArray{<:Real}) = StatsBase.ecdf(S)
    cdf_func(S::Distributions.UnivariateDistribution) = x -> Distributions.cdf(S, x)

"""
cdf_func(S::AbstractArray{<:Real}) = StatsBase.ecdf(S)
cdf_func(S::Distributions.UnivariateDistribution) = x -> Distributions.cdf(S, x)

######################################################################
## This section is old, work-in-progress VaR and CTE revamp applicable only for 
# AbstractArrays. Keeping this around for now in case perforamnce needs dicatate a specialized
# version of the two, but the above implementation has proved more flexible and general
# than below.


# """
#     VaR(v::AbstractArray,p::Real;rev::Bool=false)

# The `p`th quantile of the vector `v` is the Value at Risk. Assumes more positive values are higher risk measures, so a higher p will return a more positive number., but this can be reversed if `rev` is `true`.

# Also can be called with `ValueAtRisk(...)`.
# """
# function VaR_empirical(v::T, p; sorted=false) where {T<:AbstractArray}
#     if sorted
#         _VaR_sorted(v, p)
#     else
#         _VaR_sorted(sort(v), p)
#     end
# end

# # Core VaR assumes v is sorted
# function _VaR_sorted(v, p)
#     i = 1
#     n = length(v)
#     q_prior = 0.0
#     x_prior = first(v)
#     for (i, x) in enumerate(v)
#         q = i / n
#         if q >= p
#             # return weighted between two points
#             return x * (p - q_prior) / (q - q_prior) + x_prior * (q - p) / (q - q_prior)

#         end
#         x_prior = x
#         q_prior = q
#     end

#     return last(v)
# end




# """
#     CTE(v::AbstractArray,p::Real;rev::Bool=false)

# The average of the values ≥ the `p`th percentile of the vector `v` is the Conditiona Tail Expectation. Assumes more positive values are higher risk measures, so a higher p will return a more positive number, but this can be reversed if `rev` is `true`.

# May also be called with `ConditionalTailExpectation(...)`.

# Also known as Tail Value at Risk (TVaR), or Tail Conditional Expectation (TCE)
# """
# function CTE(v::T, p; sorted=false) where {T<:AbstractArray}
#     if sorted
#         _CTE_sorted(v, p)
#     else
#         _CTE_sorted(sort(v), p)
#     end
# end

# # Core CTE assumes v is sorted
# function _CTE_sorted(v, p)
#     i = 1
#     n = length(v)
#     q_prior = 0.0
#     x_prior = first(v)
#     sub_total = zero(eltype(v))
#     in_range = false
#     for (i, x) in enumerate(v)
#         q = i / n
#         if in_range || q >= p
#             # return weighted between two points
#             # return x * (p - q_prior) / (q - q_prior) + x_prior * (q - p) / (q - q_prior)
#             in_range = true

#         end
#         x_prior = x
#         q_prior = q
#     end

#     return last(v)
# end

######################
end