abstract type DistortionFunction end


struct Expectation <: DistortionFunction end
(g::Expectation)(x) = x

struct VaR{T} <: DistortionFunction
    α::T
end
(g::VaR)(x) = x < (1 - g.α) ? 0 : 1

struct CTE{T} <: DistortionFunction
    α::T
end
(g::CTE)(x) = x < (1 - g.α) ? x / (1 - g.α) : 1

struct WangTransform{T} <: DistortionFunction
    α::T
end
function (g::WangTransform)(x)
    Φ_inv(x) = Distributions.quantile(Distributions.Normal(), x)
    Distributions.cdf(Distributions.Normal(), Φ_inv(x) + g.α)
end

struct DualPower{T} <: DistortionFunction
    v::T
end
(g::DualPower)(x) = 1 - (1 - x)^g.v

struct ProportionalHazard{T} <: DistortionFunction
    y::T
end
(g::ProportionalHazard)(x) = x^(1 / g.y)


function ρ(g::DistortionFunction, risk)

    # integral from 0 to infinity of g(S(x))dx
    # where S(x) is 1-cdf(risk,x)
    F(x) = cdf_func(risk)(x)
    S(x) = 1 - F(x)
    H(x) = 1 - g(1 - x)
    integral1, _ = quadgk(x -> 1 - H(F(x)), 0, Inf)
    integral2, _ = quadgk(x -> H(F(x)), -Inf, 0)
    return integral1 - integral2
end


cdf_func(S::AbstractArray{<:Real}) = StatsBase.ecdf(S)
cdf_func(S::Distributions.UnivariateDistribution) = x -> Distributions.cdf(S, x)

# """
#     VaR(v::AbstractArray,p::Real;rev::Bool=false)

# The `p`th quantile of the vector `v` is the Value at Risk. Assumes more positive values are higher risk measures, so a higher p will return a more positive number, but this can be reversed if `rev` is `true`.

# Also can be called with `ValueAtRisk(...)`.
# """
# function VaR(v::T, p; sorted=false) where {T<:AbstractArray}
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
# [`VaR`](@ref)
# """
# ValueAtRisk = VaR

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

# """
# [`CTE`](@ref)
# """
# ConditionalTailExpectation = CTE
