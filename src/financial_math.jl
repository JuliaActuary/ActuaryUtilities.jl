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
    res = Optim.optimize(loss_func, [0.0], Optim.Newton())


    if Optim.converged(res) 
        min = Optim.minimizer(res)[1]

        # check if function does change signs at location
        if f(min+.001) * f(min - .001) <= 0 
            return min
        else
            return nothing
        end

    else
        return nothing
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
    present_value(interest, cashflows::Vector, timepoints)
    present_value(interest, cashflows::Vector)

Efficiently calculate a vector representing the present value of the given cashflows at each period prior to the given timepoint.

!!! note

    If your source directory is not accessible through Julia's LOAD_PATH, you might wish to
    add the following line at the top of make.jl

    ```julia
    push!(LOAD_PATH,"../src/")
    ```

# Examples
```julia-repl
julia> present_values(0.00, [1,1,1])
28.18181818181818
julia> present_value(InterestVector([0.1,0.2]), [10,20],[0,1])
```

"""
function present_values(interest,cashflows)
    pvs = similar(cashflows)
    pvs[end] = Yields.discount(interest,lastindex(cashflows)-1,lastindex(cashflows)) * cashflows[end]
    for (t,cf) in Iterators.reverse(enumerate(cashflows[1:end-1]))
        pvs[t] = Yields.discount(interest,t-1,t) * (cf+pvs[t+1])
end

return pvs
end

"""
    price(...)

The absolute value of the `present_value(...)`. 

# Extended help

Using `price` can be helpful if the directionality of the value doesn't matter. For example, in the common usage, duration is more interested in the change in price than present value, so `price` is used there.
"""
price(x1,x2) = present_value(x1,x2) |> abs
price(x1,x2,x3) = present_value(x1,x2,x3) |> abs

"""
    breakeven(yield, cashflows::Vector)
    breakeven(yield, cashflows::Vector,times::Vector)

Calculate the time when the accumulated cashflows breakeven given the yield.

Assumptions:

- cashflows occur at the end of the period
- cashflows evenly spaced with the first one occuring at time zero if `times` not given

Returns `nothing` if cashflow stream never breaks even.

```jldoctest
julia> breakeven(0.10, [-10,1,2,3,4,8])
5

julia> breakeven(0.10, [-10,15,2,3,4,8])
1

julia> breakeven(0.10, [-10,-15,2,3,4,8]) # returns the `nothing` value


```
"""
function breakeven(y::T,cashflows::Vector,timepoints::Vector) where {T<:Yields.AbstractYield}
    accum = zero(eltype(cashflows))
    last_neg = nothing

    accum += cashflows[1]
    if accum >= 0 && isnothing(last_neg)
        last_neg = timepoints[1]
    end

    for i in 2:length(cashflows)
        # accumulate the flow from each timepoint to the next
        accum *= accumulate(y,timepoints[i-1],timepoints[i])
        accum += cashflows[i]

        if accum >= 0 && isnothing(last_neg)
            last_neg = timepoints[i]
        elseif accum < 0
            last_neg = nothing
        end
    end

    return last_neg

end

function breakeven(y::T,cfs,times) where {T<:Real}
    return breakeven(Yields.Constant(y),cfs,times)
end

function breakeven(y::Vector{T},cfs,times) where {T<:Real}
    return breakeven(Yields.Forward(y),cfs,times)
end

function breakeven(i,cashflows::Vector)
    return breakeven(i,cashflows,[t for t in 0:length(cashflows)-1])
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

Calculates the Macaulay, Modified, or DV01 duration. `times` may be ommitted and the valuation will assume evenly spaced cashflows starting at the end of the first period.
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
function duration(::Macaulay,yield,cfs,times)
    return sum(times .* price.(yield,vec(cfs),times) / price(yield,vec(cfs),times))
end

function duration(::Modified,yield,cfs,times)
    D(i) = price(i,vec(cfs),times)
    return duration(yield,D)
end

function duration(yield,valuation_function)
    D(i) = log(valuation_function(i+yield))
    δV =  - ForwardDiff.derivative(D,0.0)
end

function duration(yield::Y,valuation_function) where {Y <: Yields.AbstractYield}
    D(i) = log(valuation_function(i+yield))
    δV =  - ForwardDiff.derivative(D,0.0)
end

function duration(yield,cfs,times)
    return duration(Modified(),yield,vec(cfs),times)
end
function duration(yield::Y,cfs::A) where {Y <: Yields.AbstractYield,A <: AbstractArray}
    times = 1:length(cfs)
    return duration(Modified(),yield,vec(cfs),times)
end

function duration(yield::R,cfs) where {R <: Real}
    return duration(Yields.Constant(yield),cfs)
end

function duration(::DV01,yield,cfs,times)
    return duration(DV01(),yield,i->price(i,vec(cfs),times))
end
function duration(d::Duration,yield,cfs)
    times = 1:length(cfs)
    return duration(d,yield,vec(cfs),times)
end

function duration(::DV01,yield,valuation_function)
    return duration(yield,valuation_function) * valuation_function(yield) / 100
end

""" 
    convexity(yield,cfs,times)
    convexity(yield,valuation_function)

Calculates the convexity.
    - `yield` should be a fixed effective yield (e.g. `0.05`).
    - `times` may be omitted and it will assume `cfs` are evenly spaced beginning at the end of the first period.

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
function convexity(yield,cfs,times)
    return convexity(yield, i -> price(i,vec(cfs),times))
end

function convexity(yield,cfs::A) where {A <: AbstractArray}
    times = 1:length(cfs)
    return convexity(yield, i -> price(i,vec(cfs),times))
end

function convexity(yield,valuation_function)
    v(x) = abs(valuation_function(yield + x[1]))
    ∂²P = ForwardDiff.hessian(v,[0.0])
    return ∂²P[1] / v([0.0])  
end