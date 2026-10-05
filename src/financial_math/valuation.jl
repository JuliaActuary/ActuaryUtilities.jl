## Cashflow valuation utilities

"""
    present_values(interest, cashflows, timepoints)

Return the value of remaining cashflows before each payment period.
Entry `k` values payments `k:end` at the time payment `k-1` is made, or time zero for `k = 1`.
A `Cashflow` is paid at its own time, whatever its paired time; a number is paid at its paired time.
The entries are numbers.

Empty collections return an empty vector. Collections whose amounts are all
exactly zero return a vector of positive zeros without valuing any payment; the
element type is the one a nonempty stream's result would have (see
[Zero cashflow streams](@ref)).
Every cashflow requires a time; additional trailing times are ignored. Amounts and times pair
by position, so offset vectors pair too.

# Examples
```julia-repl
julia> present_values(0.00, [1,1,1])
3-element Vector{Float64}:
 3.0
 2.0
 1.0

julia> present_values(0.05, [10,10,110], [1,2,3])
3-element Vector{Float64}:
 113.61624014685238
 109.297052154195
 104.76190476190476
```

"""
present_values(interest, cashflows, times...) = _present_values(interest, _cashflow_inputs(cashflows, times...)...)
function _present_values(interest, cashflows, times)
    # The entries have the type of a nonempty stream's value, from the amounts and the payment times,
    # so a zero stream returns these zeros without valuing any payment.
    acc = _zero_stream_value(interest, cashflows, times)
    pvs = fill(acc, length(cashflows))
    _iszero_cashflow_stream(cashflows) && return pvs
    # Discount backward in one pass, each payment at its own time.
    paid(k) = FinanceCore.timepoint(cashflows[k], times[k])
    @inbounds for k in reverse(eachindex(cashflows))
        from = k == 1 ? zero(paid(k)) : paid(k - 1)
        acc = FinanceCore.discount(interest, from, paid(k)) * (acc + _cf_value(cashflows[k]))
        pvs[k] = acc
    end
    return pvs
end


"""
    breakeven(yield, cashflows::Vector)
    breakeven(yield, cashflows::Vector,times::Vector)

Return the payment time from which the balance accumulated at `yield` stays
nonnegative, or `nothing` if it never does. Without `times`, cashflow `k` is paid at
time `k - 1`.

```julia-repl
julia> breakeven(0.10, [-10,1,2,3,4,8])
5

julia> breakeven(0.10, [-10,15,2,3,4,8])
1

julia> breakeven(0.10, [-10,-15,2,3,4,8]) # returns the `nothing` value


```
"""
breakeven(y, cashflows) = (cfs = _vector(cashflows); breakeven(y, cfs, eachindex(cfs) .- 1))
breakeven(y, cashflows, timepoints) = _breakeven(y, _cashflow_inputs(cashflows, timepoints)...)
@inline function _breakeven(y, cashflows, timepoints)
    accum = 0.0
    last_neg = nothing

    # Resolve embedded amounts and times for Cashflow inputs.
    accum += FinanceCore.amount(cashflows[1])
    if accum >= 0 && isnothing(last_neg)
        last_neg = FinanceCore.timepoint(cashflows[1], timepoints[1])
    end

    # The inputs of `_cashflow_inputs` are 1-based and of equal length.
    @inbounds for i in 2:length(cashflows)
        # accumulate the flow from each timepoint to the next
        a = FinanceCore.timepoint(cashflows[i - 1], timepoints[i - 1])
        b = FinanceCore.timepoint(cashflows[i], timepoints[i])
        accum *= FinanceCore.accumulation(y, a, b)
        accum += FinanceCore.amount(cashflows[i])

        if accum >= 0 && isnothing(last_neg)
            last_neg = b
        elseif accum < 0
            last_neg = nothing
        end
    end

    return last_neg

end


"""
    moic(cashflows<:AbstractArray)

The multiple on invested capital ("moic") is the un-discounted sum of distributions divided by the sum of the contributions. The function assumes that negative numbers in the array represent contributions and positive numbers represent distributions.

A total loss (contributions only) has a moic of `0.0`. With no contributions the ratio divides by zero: `Inf` when distributions are positive, and `NaN` for an empty or all-zero stream. Neither is a meaningful multiple.

# Examples

```julia-repl
julia> moic([-10,20,30])
5.0

julia> moic([-10,-20])
0.0
```

"""
function moic(cfs::T) where {T <: AbstractArray}
    # An empty sum is zero, of the amounts' type when the element type says it (so the sums are
    # type-stable). The contributions are negated after `sum` widens small integers (negating
    # typemin(Int8) first would wrap), and subtracted from +0 so that none gives +0, a ratio of +Inf.
    z = _zero_amount(cfs)
    returned = sum((FinanceCore.amount(cf) for cf in cfs if FinanceCore.amount(cf) > 0); init = z)
    invested = z - sum((FinanceCore.amount(cf) for cf in cfs if FinanceCore.amount(cf) < 0); init = z)
    return returned / invested
end
