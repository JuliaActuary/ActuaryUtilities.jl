"""
    internal_rate_of_return(cashflows::vector)::Yields.Rate
    internal_rate_of_return(cashflows::Vector, timepoints::Vector)::Yields.Rate
    
Calculate the internal_rate_of_return with given timepoints. If no timepoints given, will assume that a series of equally spaced cashflows, assuming the first cashflow occurring at time zero and subsequent elements at time 1, 2, 3, ..., n. 

Returns a Yields.Rate type with periodic compounding once per period (e.g. annual effective if the `timepoints` given represent years). Get the scalar rate by calling `Yields.rate()` on the result.

# Example
```julia-repl
julia> internal_rate_of_return([-100,110],[0,1]) # e.g. cashflows at time 0 and 1
0.10000000001652906
julia> internal_rate_of_return([-100,110]) # implied the same as above
0.10000000001652906
```

# Solver notes
Will try to return a root within the range [-2,2]. If the fast solver does not find one matching this condition, then a more robust search will be performed over the [.99,2] range.

The solution returned will be in the range [-2,2], but may not be the one nearest zero. For a slightly slower, but more robust version, call `ActuaryUtilities.irr_robust(cashflows,timepoints)` directly.
"""
function internal_rate_of_return(cashflows)
    

    return internal_rate_of_return(cashflows, 0:length(cashflows)-1)
    
end

function internal_rate_of_return(cashflows,times)
    # first try to quickly solve with newton's method, otherwise 
    # revert to a more robust method
    lower,upper = -2.,2.
    
    v = try 
        return irr_newton(cashflows,times)
    catch e
        if isa(e,Roots.ConvergenceFailed) || sprint(showerror, e) =="No convergence"
            return irr_robust(cashflows,times)
        else
            throw(e)
        end
    end
    
    if v <= upper && v >= lower
        return v
    else
        return irr_robust(cashflows,times)
    end
end

irr_robust(cashflows) = irr_robust(cashflows,0:length(cashflows)-1)

function irr_robust(cashflows, times)
    f(i) =  sum(cf / (1+i)^t for (cf,t) in zip(cashflows,times))
    # lower bound at -.99 because otherwise we can start taking the root of a negative number
    # when a time is fractional. 
    roots = Roots.find_zeros(f, -0.99, 2)
    
    # short circuit and return nothing if no roots found
    isempty(roots) && return nothing
    # find and return the one nearest zero
    min_i = argmin(roots)
    return Yields.Periodic(roots[min_i],1)

end

irr_newton(cashflows) = irr_newton(cashflows,0:length(cashflows)-1)

function irr_newton(cashflows, times)
    # use newton's method with hand-coded derivative
    f(r) =  sum(cf * exp(-r*t) for (cf,t) in zip(cashflows,times))
    f′(r) = sum(-t*cf * exp(-r*t) for (cf,t) in zip(cashflows,times) if t > 0)
    # r = Roots.solve(Roots.ZeroProblem((f,f′), 0.0), Roots.Newton())
    r = Roots.newton(x->(f(x),f(x)/f′(x)),0.0)
    return Yields.Periodic(exp(r)-1,1)

end

"""
    irr(cashflows::vector)
    irr(cashflows::Vector, timepoints::Vector)

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
julia> present_value(Yields.Forward([0.1,0.2]), [10,20],[0,1])
28.18181818181818 # same as above, because first cashflow is at time zero
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
function present_value(yc::T, cashflows, timepoints) where {T <: Yields.AbstractYield}
    sum(discount(yc,t) * cf for (t,cf) in zip(timepoints, cashflows))
end

function present_value(yc::T, cashflows) where {T <: Yields.AbstractYield}
    present_value(yc,cashflows,1:length(cashflows))
end

function present_value(i, x)
    
    v = 1.0
    v_factor = discount(i,0,1)
    pv = 0.0

    for (t,cf) in zip(1:length(x),x)
        v *= v_factor
        pv += v * cf
    end
    return pv 
end

function present_value(i, v, times)
    return present_value(Yields.Constant(i), v, times)
end

# Interest Given is an array, assume forwards.
function present_value(i::AbstractArray, v)
    yc = Yields.Forward(i)
    return sum(discount(yc, t) * cf for (t,cf) in zip(1:length(v),v))
end

# Interest Given is an array, assume forwards.
function present_value(i::AbstractArray, v, times)
    yc = Yields.Forward(i, times)
    return sum(discount(yc, t) * cf for (cf, t) in zip(v,times))
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

# Examples
```julia-repl
julia> present_values(0.00, [1,1,1])
[3,2,1]

julia> present_values(Yields.Forward([0.1,0.2]), [10,20],[0,1])
2-element Vector{Float64}:
 28.18181818181818
 18.18181818181818
```

"""
function present_values(interest, cashflows)
    pvs = Vector{Float64}(undef,length(cashflows))
    pvs[end] = Yields.discount(interest, lastindex(cashflows) - 1, lastindex(cashflows)) * cashflows[end]
    for (t, cf) in Iterators.reverse(enumerate(cashflows[1:end - 1]))
        pvs[t] = Yields.discount(interest, t - 1, t) * (cf + pvs[t + 1])
    end

    return pvs
end


function present_values(interest,cashflows,times)
    present_values_accumulator(interest,cashflows,times)
end

function present_values_accumulator(interest,cashflows,times,pvs=[0.0])
    from_time = length(times) == 1 ? 0. : times[end-1]
    pv = discount(interest,from_time,last(times)) *(first(pvs) + last(cashflows))
    pvs = pushfirst!(pvs,pv)

    if length(cashflows) > 1

        new_cfs = @view cashflows[1:end-1]
        new_times = @view times[1:end-1]
        return present_values_accumulator(interest,new_cfs,new_times,pvs)
    else
        # last discount and return
        return pvs[1:end-1] # end-1 get rid of trailing 0.0
    end
end

# if given a vector of rates, assume that it should be a forward discount yield
function present_values(y::Vector{T}, cfs, times) where {T <: Real}
    return present_values(Yields.Forward(y), cfs, times)
end


"""
    price(...)

The absolute value of the `present_value(...)`. 

# Extended help

Using `price` can be helpful if the directionality of the value doesn't matter. For example, in the common usage, duration is more interested in the change in price than present value, so `price` is used there.
"""
price(x1,x2) = present_value(x1, x2) |> abs
price(x1,x2,x3) = present_value(x1, x2, x3) |> abs

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
function breakeven(y::T, cashflows::Vector, timepoints::Vector) where {T <: Yields.AbstractYield}
    accum = zero(eltype(cashflows))
    last_neg = nothing

    accum += cashflows[1]
    if accum >= 0 && isnothing(last_neg)
        last_neg = timepoints[1]
    end

    for i in 2:length(cashflows)
        # accumulate the flow from each timepoint to the next
        accum *= Yields.accumulation(y, timepoints[i - 1], timepoints[i])
        accum += cashflows[i]

        if accum >= 0 && isnothing(last_neg)
            last_neg = timepoints[i]
        elseif accum < 0
            last_neg = nothing
        end
    end

    return last_neg

end

function breakeven(y::T, cfs, times) where {T <: Real}
    return breakeven(Yields.Constant(y), cfs, times)
end

function breakeven(y::Vector{T}, cfs, times) where {T <: Real}
    return breakeven(Yields.Forward(y), cfs, times)
end

function breakeven(i, cashflows::Vector)
    return breakeven(i, cashflows, [t for t in 0:length(cashflows) - 1])
end

abstract type Duration end

struct Macaulay <: Duration end
struct Modified <: Duration end
struct DV01 <: Duration end

abstract type KeyRateDuration <: Duration end
struct KeyRatePar{T} <: KeyRateDuration 
    timepoint::T
end
struct KeyRateZero{T} <: KeyRateDuration 
    timepoint::T
end

""" 
    duration(Macaulay(),interest_rate,cfs,times)
    duration(Modified(),interest_rate,cfs,times)
    duration(DV01(),interest_rate,cfs,times)
    duration(interest_rate,cfs,times)             # Modified Duration
    duration(interest_rate,valuation_function)    # Modified Duration

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
function duration(::Macaulay, yield, cfs, times)
    return sum(times .* price.(yield, vec(cfs), times) / price(yield, vec(cfs), times))
end

function duration(::Modified, yield, cfs, times)
    D(i) = price(i, vec(cfs), times)
    return duration(yield, D)
end

function duration(yield, valuation_function)
    D(i) = log(valuation_function(i + yield))
    δV =  - ForwardDiff.derivative(D, 0.0)
end

function duration(yield::Y, valuation_function) where {Y <: Yields.AbstractYield}
    D(i) = log(valuation_function(i + yield))
    δV =  - ForwardDiff.derivative(D, 0.0)
end

function duration(yield, cfs, times)
    return duration(Modified(), yield, vec(cfs), times)
end
function duration(yield::Y, cfs::A) where {Y <: Yields.AbstractYield,A <: AbstractArray}
    times = 1:length(cfs)
    return duration(Modified(), yield, vec(cfs), times)
end

function duration(yield::R, cfs) where {R <: Real}
    return duration(Yields.Constant(yield), cfs)
end

function duration(::DV01, yield, cfs, times)
    return duration(DV01(), yield, i -> price(i, vec(cfs), times))
end
function duration(d::Duration, yield, cfs)
    times = 1:length(cfs)
    return duration(d, yield, vec(cfs), times)
end

function duration(::DV01, yield, valuation_function)
    return duration(yield, valuation_function) * valuation_function(yield) / 100
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
function convexity(yield, cfs, times)
    return convexity(yield, i -> price(i, vec(cfs), times))
end

function convexity(yield, cfs::A) where {A <: AbstractArray}
    times = 1:length(cfs)
    return convexity(yield, i -> price(i, vec(cfs), times))
end

function convexity(yield, valuation_function)
    v(x) = abs(valuation_function(yield + x[1]))
    ∂²P = ForwardDiff.hessian(v, [0.0])
    return ∂²P[1] / v([0.0])  
end


"""
    duration(keyrate::KeyRate,curve,cashflows; shift=0.01)    
    duration(keyrate::KeyRate,curve,cashflows,timepoints; shift=0.01)
    duration(keyrate::KeyRate,curve,cashflows,timepoints,krd_points; shift=0.01)

Calculate the key rate duration by shifting the **zero** (not par) curve by the kwarg `shift` at the timepoint specified by a KeyRate(time).

The approach is to carve up the curve into `krd_points` (default is the unit steps between `1` and  the last timepoint of the casfhlows). The 
zero rate corresponding to the timepoint within the `KeyRate` is shifted by `shift` and a new curve is created from the new spot rates. This means that the 
"width" of the shifted section is ± 1 time period, unless specific points are specified via `krd_points`.

The `curve` may be any Yields.jl curve (e.g. does not have to be a curve constructed via `Yields.Zero(...)`).

!!! Experimental: Due to the paucity of examples in the literature, this feature does not have unit tests like the rest of JuliaActuary functionality. Additionally, the API may change in a future major/minor version update.

# Examples


```julia-repl
julia> riskfree_maturities = [0.5, 1.0, 1.5, 2.0];

julia> riskfree    = [0.05, 0.058, 0.064,0.068];

julia> rf_curve = Yields.Zero(riskfree,riskfree_maturities);

julia> cfs = [10,10,10,10,10];

julia> duration(KeyRate(1),rf_curve,cfs)
8.932800152336995

```

# Extended Help

Key Rate Duration is not a well specified topic in the literature and in practice. The reference below suggest that shocking the par curve is more common 
in practice, but that the zero curve produces more consistent results. Future versions may support shifting the par curve.

References: 
- [Quant Finance Stack Exchange: To compute key rate duration, shall I use par curve or zero curve?](https://quant.stackexchange.com/questions/33891/to-compute-key-rate-duration-shall-i-use-par-curve-or-zero-curve)


"""
function duration(keyrate::KeyRateDuration, curve, cashflows, timepoints, krd_points; shift = 0.0001)
    new_curve = _krd_new_curve(keyrate,curve,krd_points)
    price = pv(curve, cashflows, timepoints)
    price_shock = pv(new_curve, cashflows, timepoints)

    return -(price_shock - price) / (shift * price)

end

function _krd_new_curve(keyrate::KeyRateZero,curve,krd_points;shift)
    curve_times = krd_points
    zeros = Yields.zero.(curve, curve_times)

    zero_index = findfirst(==(keyrate.timepoint), curve_times)

    target_rate = pars[zero_index]

    zeros[zero_index] += Yields.Rate(shift,target_rate.compounding)

    new_curve = Yields.Zero(zeros, curve_times)

    return new_curve
end

function _krd_new_curve(keyrate::KeyRatePar,curve,krd_points;shift)
    curve_times = krd_points
    pars = Yields.par.(curve, curve_times)

    zero_index = findfirst(==(keyrate.timepoint), curve_times)

    target_rate = pars[zero_index]
    pars[zero_index] += Yields.Rate(shift,target_rate.compounding)

    new_curve = Yields.Par(pars, curve_times)

    return new_curve
end

function duration(keyrate::KeyRateDuration, curve, cashflows, timepoints; shift = 0.0001)
    krd_points = 1:maximum(timepoints)
    return duration(keyrate, curve, cashflows, timepoints, krd_points; shift)

end

function duration(keyrate::KeyRateDuration, curve, cashflows; shift = 0.0001)
    timepoints = eachindex(cashflows)
    krd_points = 1:maximum(timepoints)
    return duration(keyrate, curve, cashflows, timepoints, krd_points; shift)

end

"""
    moic(cashflows<:AbstractArray)

The multiple on invested capital ("moic") is the un-discounted sum of distributions divided by the sum of the contributions. The function assumes that negative numbers in the array represent contributions and positive numbers represent distributions.

# Examples

```julia-repl
julia> moic([-10,20,30])
5.0
```

"""
function moic(cfs::T) where {T<:AbstractArray}
    invested = zero(eltype(cfs))
    returned = zero(eltype(cfs))
    for i = 1:length(cfs)
        @inbounds cf = cfs[i]
        if cf > 0
            returned += cf
        else
            invested += -cf
        end
    end

    return returned / invested
end