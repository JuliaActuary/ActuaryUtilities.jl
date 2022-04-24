"""
    parbond(yield,term; par = 100)
The `times` and `cfs` in a named tuple for a bond paying semi-annual coupons with the given `yield` (assume annual effective if not specified) and `par` over the `term` (in integer years).
"""
function parbond(yield::Yields.Rate, term; par=100.)
    r = convert(Yields.Periodic(2), yield)
    coupon = Yields.rate(r) / 2
    times = 0.5:0.5:term
    cfs = map(times) do t
        if t == term
            return par + par * coupon
        else
            return par * coupon
        end
    end

    (; times, cfs)
end

function parbond(yield::T,term; par=100.) where {T<:Real}
    y = Yields.Periodic(yield,1)
    return parbond(y,term;par)
end