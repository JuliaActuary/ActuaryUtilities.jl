"""
    VaR(v::AbstractArray,p::Real;rev::Bool=false)

The `p`th quantile of the vector `v` is the Value at Risk. Assumes more positive values are higher risk measures, so a higher p will return a more positive number, but this can be reversed if `rev` is `true`.

Also can be called with `ValueAtRisk(...)`.
"""
function VaR(v,p;rev=false)
    if rev
        return StatsBase.quantile(v,1-p)
    else
        return StatsBase.quantile(v,p)
    end
end
"""
[VaR](@ref)
"""
ValueAtRisk = VaR

"""
    CTE(v::AbstractArray,p::Real;rev::Bool=false)

The average of the values ≥ the `p`th percentile of the vector `v` is the Conditiona Tail Expectation. Assumes more positive values are higher risk measures, so a higher p will return a more positive number, but this can be reversed if `rev` is `true`.

May also be called with `ConditionalTailExpectation(...)`.

Also known as Tail Value at Risk (TVaR), or Tail Conditional Expectation (TCE)
"""
function CTE(v,p;rev=false)
    # filter has the "or approximately equalt to quantile" because
    # of floating point path might make the quantile slightly off from the right indexing 
    # e.g. if values should capture <= q, where q should be 10 but is calculated to be 
    # 9.99999...
    if rev
        q = StatsBase.quantile(v,1-p)
        filter = (v .<= q) .| (v .≈ q) 
    else
        q = StatsBase.quantile(v,p)
        filter = (v .>= q) .| (v .≈ q) 
    end    

    return sum(v[filter]) / sum(filter)

end

"""
[CTE](@ref)
"""
ConditionalTailExpectation = CTE
"""
Expected Shortfall
"""
function ES(v,p,rev=false)
end