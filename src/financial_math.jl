"""
    internal_rate_of_return(cashflows::vector)
    internal_rate_of_return(cashflows::Vector, timepoints::Vector)
    
Calculate the internal_rate_of_return with given timepoints. If no timepoints given, will assume that a series of equally spaced cashflows, assuming the first cashflow occurring at time zero. 

First tries to find a rate in the interval `[-0.1,0.25]`. If none is found, will triple the search range until the range is [-1.5,1.65]. If none is still found, will return `nothing`.

# Example
```julia-repl
julia> internal_rate_of_return([-100,110],[0,1]) # e.g. cashflows at time 0 and 1
0.10000000001652906
```

"""
function internal_rate_of_return(cashflows)
    

    return internal_rate_of_return(cashflows,[t for t in 0:(length(cashflows)-1)])
    
end

function internal_rate_of_return(cashflows,times)
    f(i) =  sum(@views cashflows .* [1/(1+i[1])^t for t in times])
    loss_func = x -> f(x)^2
    result = irr_root(loss_func)
end

function irr_root(f,low=-.1,high=0.25)
    range = high - low
    
    # short circuit if the range has gotten too wide
    range > 3.2 && return nothing

    result = optimize(f, low,high)

    if abs(f(result.minimizer)) < 1.0e-3 # arbitrary that seems to work
        return result.minimizer
    else
        return irr_root(f,low - range, high + range)
    end
end

"""
    irr()

    An alias for `internal_rate_of_return`.
"""
irr = internal_rate_of_return

"""
    present_value(interest, cashflows::Vector, timepoints)
    present_value(interest, cashflows::Vector)

Discount the `cashflows` vector at the given `interest_interestrate`,  with the cashflows occurring
at the times specified in `timepoints`. If no `timepoints` given, assumes that cashflows happen at times 1,2,...,n.

The `interest` can be an `InterestCurve`, a single scalar, or a vector wrapped in an `InterestCurve`. 

# Examples
```julia-repl
julia> present_value(0.1, [10,20],[0,1])
28.18181818181818
julia> present_value(InterestVector([0.1,0.2]), [10,20],[0,1])
```

Example on how to use real dates using the [DayCounts.jl](https://github.com/JuliaFinance/DayCounts.jl) package
```jldoctest

using DayCounts 
dates = Date(2012,12,31):Year(1):Date(2013,12,31)
times = map(d -> yearfrac(dates[1], d, DayCounts.Actual365Fixed()),dates) # [0.0,1.0]
present_value(0.1, [10,20],times)

# output
28.18181818181818

```

"""
function present_value(yc::T,cashflows,timepoints) where {T <: Yields.AbstractYield}
    sum(discount.(yc,timepoints) .* cashflows)
end

function present_value(yc::T,cashflows) where {T <: Yields.AbstractYield}
    sum(discount.(yc,1:length(cashflows)) .* cashflows)
end

function present_value(i,v)
    yc = Yields.Constant(i)
    return sum(discount(yc,t) * v[t] for t in 1:length(v))
end

function present_value(i,v,times)
    return present_value(Yields.Constant(i),v,times)
end

# Interest Given is an array, assume forwards.
function present_value(i::AbstractArray,v)
    yc = Yields.Forward(i)
    return sum(discount(yc,t) * v[t] for t in 1:length(v))
end

# Interest Given is an array, assume forwards.
function present_value(i::AbstractArray,v,times)
    yc = Yields.Forward(i,times)
    return sum(discount(yc,t) * v[i] for (i,t) in enumerate(times))
end

"""
    pv()

    An alias for `present_value`.
"""
pv = present_value


"""
    breakeven(accumulation_rate, cashflows::Vector)

Calculate the time when the accumulated cashflows breakeven.
Assumes that:
- cashflows evenly spaced with the first one occuring at time zero 
- cashflows occur at the end of the period
- that the accumulation rate correponds to the periodicity of the cashflows.

Returns `nothing` if cashflow stream never breaks even.

```jldoctest
julia> breakeven(0.10, [-10,1,2,3,4,8])
5

julia> breakeven(0.10, [-10,15,2,3,4,8])
1

julia> breakeven(0.10, [-10,-15,2,3,4,8]) # returns the `nothing` value


```
"""
function breakeven(i,cashflows::Vector)
    return breakeven(i,cashflows,[t for t in 0:length(cashflows)-1])
end

"""
    breakeven(accumulation_rate, cashflows::Vector,timepoints::Vector)

Calculate the time when the accumulated cashflows breakeven.
Assumes that:
- cashflows occur at the timepoint indicated at the corresponding `timepoints` position
- cashflows occur at the end of the period
- that the accumulation rate corresponds to the periodicity of the cashflows. 
- If given a vector of interest rates, the first rate is effectively never used, as it's treated as the accumulation 
rate between time zero and the first cashflow.

Returns `nothing` if cashflow stream never breaks even.

```jldoctest; setup = :(times = [0,1,2,3,4,5])
julia> times = [0,1,2,3,4,5];

julia> breakeven(0.10, [-10,1,2,3,4,8],times)
5

julia> breakeven(0.10, [-10,15,2,3,4,8],times)
1

julia> breakeven(0.10, [-10,-15,2,3,4,8],times) # returns the `nothing` value
```

"""
function breakeven(i::Vector,cashflows::Vector, timepoints::Vector)
    accum = cashflows[1]
    last_neg = nothing


    for t in 2:length(cashflows)
        timespan = timepoints[t] - timepoints[t-1]
        accum *= (1+i[t]) ^ timespan
        accum += cashflows[t]
        
        # keep last negative timepoint, but if 
        # we go negative then go back to `nothing`
        if accum >= 0.0 && isnothing(last_neg)
            last_neg = timepoints[t]
        elseif accum < 0.0
            last_neg = nothing
        end
    end

    return last_neg

end
function breakeven(i,cashflows::Vector,timepoints::Vector)
    accum = cashflows[1]
    last_neg = nothing


    for t in 2:length(cashflows)
        timespan = timepoints[t] - timepoints[t-1]
        accum *= (1+i) ^ timespan
        accum += cashflows[t]
        
        # keep last negative timepoint, but if 
        # we go negative then go back to `nothing`
        if accum >= 0.0 && isnothing(last_neg)
            last_neg = timepoints[t]
        elseif accum < 0.0
            last_neg = nothing
        end
    end
    return last_neg

end

abstract type Duration end

struct Macaulay <: Duration end
struct Modified <: Duration end
struct DV01 <: Duration end

""" 
    duration(Macaulay(),interest_rate,cfs,times)
    duration(Modified(),interest_rate,cfs,times)
    duration(::DV01,interest_rate,cfs,times)
    duration(interest_rate,cfs,times)             # Modified Duration
    duration(interest_rate,valuation_function)    # modified Duration

Calculates the Macaulay, Modified, or DV01 duration.
- `interest_rate` should be a fixed effective yield (e.g. `0.05`).

When not given `Modified()` or `Macaulay()` as an argument, will default to `Modified()`.

# Examples

Using vectors of cashflows and times
```julia-repl
julia> times = 1:5
julia> cfs = [0,0,0,0,100]
julia> duration(0.03,cfs,times)
4.854368932038834
julia> duration(Macaulay(),0.03,cfs,times)
5.0
julia> duration(Modified(),0.03,cfs,times)
4.854368932038835
julia> convexity(0.03,cfs,times)
28.277877274012614

```

Using any given value function: 

```julia-repl
julia> lump_sum_value(amount,years,i) = amount / (1 + i ) ^ years
julia> my_lump_sum_value(i) = lump_sum_value(100,5,i)
julia> duration(0.03,my_lump_sum_value)
4.854368932038835
julia> convexity(0.03,my_lump_sum_value)
28.277877274012617

```
"""
function duration(::Macaulay,interest_rate,cfs,times)
    return sum(times .* present_value.(interest_rate,cfs,times) / present_value(interest_rate,cfs,times))
end
function duration(::Modified,interest_rate,cfs,times)
    return duration(interest_rate,i -> present_value(i,cfs,times))
end

function duration(interest_rate,valuation_function)
    δV =  - ForwardDiff.derivative(i -> log(valuation_function(i)),interest_rate)
end

function duration(interest_rate,cfs,times)
    return duration(Modified(),interest_rate,cfs,times)
end

function duration(::DV01,interest_rate,cfs,times)
    return duration(DV01(),interest_rate,i->present_value(i,cfs,times))
end

function duration(::DV01,interest_rate,valuation_function)
    return duration(interest_rate,valuation_function) * valuation_function(interest_rate) / 100
end

""" 
    convexity(interest_rate,cfs,times)
    convexity(interest_rate,valuation_function)

Calculates the convexity.
    - `interest_rate` should be a fixed effective yield (e.g. `0.05`).

# Examples

Using vectors of cashflows and times
```julia-repl
julia> times = 1:5
julia> cfs = [0,0,0,0,100]
julia> duration(0.03,cfs,times)
4.854368932038834
julia> duration(Macaulay(),0.03,cfs,times)
5.0
julia> duration(Modified(),0.03,cfs,times)
4.854368932038835
julia> convexity(0.03,cfs,times)
28.277877274012614

```

Using any given value function: 

```julia-repl
julia> lump_sum_value(amount,years,i) = amount / (1 + i ) ^ years
julia> my_lump_sum_value(i) = lump_sum_value(100,5,i)
julia> duration(0.03,my_lump_sum_value)
4.854368932038835
julia> convexity(0.03,my_lump_sum_value)
28.277877274012617

```

"""
function convexity(interest_rate,cfs,times)
    return convexity(interest_rate, i -> present_value(i,cfs,times))
end

function convexity(interest_rate,valuation_function)
    D(i) = duration(i,valuation_function)

    return D(interest_rate) ^ 2 - ForwardDiff.derivative(D,interest_rate)
end
