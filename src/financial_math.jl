struct InterestCurve
    rates
    times
    interpolation_method
    interp_func
end

struct StepwiseInterp end
struct LinearInterp end

"""
    InterestCurve(rates,times,interpolation_method)

Creates an InterestCurve object:
- `rates` are annual effective forward interest rates
- `times` are the end of the period for which the 
- `interpolation_method` is either option below (with `StepwiseInterp()` as the default)
    - `StepwiseInterp()` which is an *ActuaryUtilities.jl* provided type which will use the given `rate` for the period up to the given `time`.
        - For example, with `rates= [0.05,0.1]` and `times=[1,2]` then `0.05` will be used for the period `(0,1]` and `0.1` will be used for the period `(1,2]`.
    - `LinearInterp()`, which linearly interpolates between `times` and is flat outside the boundaries of `times`.

"""
function InterestCurve(rates,times,interpolation_method::LinearInterp)
    f(time) = LinearInterpolation(times,rates,extrapolation_bc = Interpolations.Flat())(time)
    return InterestCurve(rates,times,interpolation_method,f)

end

function InterestCurve(rates)
    return InterestCurve(rates,1:length(rates),StepwiseInterp())
end
function InterestCurve(rates,times)
    return InterestCurve(rates,times,StepwiseInterp())
end

function InterestCurve(rates,times,interpolation_method::StepwiseInterp)
    f(time) = time > last(times) ? last(rates) : rates[findfirst(t -> t >= time,times)]
    return InterestCurve(rates,times,interpolation_method,f)

end


# make interest curve broadcastable so that you can broadcast over multiple`time`s in `interest_rate`
Base.Broadcast.broadcastable(ic::InterestCurve) = Ref(ic) 


"""
interest_rate(interest,time)

Return the interest rate at `time`. 

`interest` can be:
    - an `InterestCurve`
    - a vector wrapped in an interest curve
    - a scalar rate

# Examples

## InterestCurve
```julia-repl
julia> ic = InterestCurve([0.01,0.05,0.05,0.1],[1,2,3,4])
julia> ic = InterestCurve([0.01,0.05,0.05,0.1])  # this is equivalent to the line above

julia> interest_rate(ic,1)
0.01

julia> interest_rate.(ic,0:4) # function is broadcastable
5-element Array{Float64,1}:
 0.01
 0.01
 0.05
 0.05
 0.1
```

## Scalar
```julia-repl
julia> interest_rate(0.05,1)
0.05

julia> interest_rate.(0.05,1:3) # function can be broadcasted
3-element Array{Float64,1}:
0.05
0.05
0.05
```
"""
function interest_rate(ic::InterestCurve,time)
    return ic.interp_func(time)
end

function interest_rate(i,time)
    return i
end

"""
    discount_rate(interest,time)

Return the discount rate at `time`. 

`interest` can be:
    - a `InterestCurve`
    - a vector wrapped in an interest curve
    - a scalar rate

Internally, if not a scalar argument, this method will use the interpolated interest rate to use an integral approxmation to the accumulated force of interest. This generalizes well. For more performance on repeated calls, use an `InterestCurve` instead of a vector.

# Examples

## InterestCurve

```julia-repl
julia> ic = InterestCurve([0.01,0.05,0.05,0.1],[1,2,3,4])
julia> ic = InterestCurve([0.01,0.05,0.05,0.1])  # this is equivalent to the line above

julia> discount_rate(ic,1)
0.9900990099009901

julia> discount_rate.(ic,0:4) # function is broadcastable
5-element Array{Float64,1}:
 1.0
 0.9900990099009901
 0.9613215887833819
 0.9155443705701612
 0.8517459660161829
```

## Scalar
```julia-repl
julia> discount_rate(0.05,1)
0.9523809523809523

julia> discount_rate.(0.05,1:3) # function can be broadcasted
3-element Array{Float64,1}:
 0.9523809523809523
 0.9070294784580498
 0.863837598531476

```
"""
function discount_rate(ic::InterestCurve,time)
    # as a general approach, convert Effective Annual Rates
    # to continuously compounded rates for integration
    # i = exp(δ) - 1
    # δ = ln(1+i)
    integral, err = quadgk(t -> log(1+interest_rate(ic,t)),0,time)
    return 1/exp(integral)
end

function discount_rate(i, time)
    return  1 / ((1 + i) ^ time)
end

"""
    internal_rate_of_return(cashflows::vector)
    internal_rate_of_return(cashflows::Vector, timepoints::Vector)
    
Calculate the internal_rate_of_return with given timepoints. If no timepoints given, will assume that a series of equally spaced cashflows, assuming the first 
cashflow occurring at time zero. 

First tries to find a positive rate in the interval `[0.0,1.0]`. If none is found,
will extend search to [-1.0,1.0]. If still not found, will return `nothing`.

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
    # Optim requires the optimizing variable to be an array, thus the i[1]
    f(i) = sum(cashflows .* [1/(1+i[1])^t for t in times])
    result = optimize(x -> f(x)^2, 0.0,1.0)
    if abs(f(result.minimizer)) < 1.0e-3 # arbitrary that seems to work
        return result.minimizer
    else
        # try finding a negative irr
        result = optimize(x -> f(x)^2, -1.0,1.0)
        if abs(f(result.minimizer)) < 1.0e-3 # arbitrary that seems to work
            return result.minimizer
        else
            return nothing
        end
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
function present_value(ic::InterestCurve,cashflows,timepoints)
    sum(discount_rate.(ic,timepoints) .* cashflows)
end

function present_value(interest,cashflows,timepoints)
    sum(discount_rate.(interest,timepoints) .* cashflows)
end

function present_value(i,v)
    return present_value(i,v,[t for t in 1:length(v)])
end

"""
    pv()

    An alias for `present_value`.
"""
pv = present_value


"""
    breakeven(cashflows::Vector,accumulation_rate::Real)

Calculate the time when the accumulated cashflows breakeven.
Assumes that :
- cashflows evenly spaced with the first one occuring at time zero 
- cashflows occur at the end of the period
- that the accumulation rate correponds to the periodicity of the cashflows.

Returns `nothing` if cashflow stream never breaks even.

```jldoctest
julia> breakeven([-10,1,2,3,4,8],0.10)
5

julia> breakeven([-10,15,2,3,4,8],0.10)
1

julia> breakeven([-10,-15,2,3,4,8],0.10) # returns the `nothing` value


```
"""
function breakeven(cashflows::Vector,i)
    return breakeven(cashflows,[t for t in 0:length(cashflows)-1],i)
end

"""
    breakeven(cashflows::Vector,timepoints::Vector, accumulation_rate)

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

julia> breakeven([-10,1,2,3,4,8],times,0.10)
5

julia> breakeven([-10,15,2,3,4,8],times,0.10)
1

julia> breakeven([-10,-15,2,3,4,8],times,0.10) # returns the `nothing` value
```

"""
function breakeven(cashflows::Vector,timepoints::Vector, i::Vector)
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
function breakeven(cashflows::Vector,timepoints::Vector, i)
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

