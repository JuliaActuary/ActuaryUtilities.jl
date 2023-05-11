"""
    present_value(interest, cashflows::Vector, timepoints)
    present_value(interest, cashflows::Vector)

Discount the `cashflows` vector at the given `interest_interestrate`,  with the cashflows occurring
at the times specified in `timepoints`. If no `timepoints` given, assumes that cashflows happen at times 1,2,...,n.

The `interest` can be an any yield curve from Yields.jl or a scalar annual effective interest rate.

# Examples
```julia-repl
julia> present_value(0.1, [10,20],[0,1])
28.18181818181818
julia> present_value(Yields.Forward([0.1,0.2]), [10,20],[0,1])
28.18181818181818 # same as above, because first cashflow is at time zero
```

Example on how to use real dates using the [DayCounts.jl](https://github.com/JuliaFinance/DayCounts.jl) package

```julia
using DayCounts 
dates = Date(2012,12,31):Year(1):Date(2013,12,31)
times = map(d -> yearfrac(dates[1], d, DayCounts.Actual365Fixed()),dates) # [0.0,1.0]
present_value(0.1, [10,20],times)

# output
28.18181818181818
```

# Extended help

Under the hood, this function uses LoopVectorization to compile specialized code, at the expense of not being auto-differentiable. A differentiable version is available as [`present_value_differentiable`](@ref).


"""
function present_value(yc, cashflows, timepoints)
    s = zero(first(cashflows)*.1)
    for i ∈ eachindex(cashflows) 
        v = discount(yc,0,timepoints[i])
        s += v * cashflows[i]
    end
    s
end

function present_value(yc, cashflows::G, timepoints) where {G<:Base.Generator}
    present_value_differentiable(yc,cashflows,timepoints)
end


"""
    present_value_differentiable(yc, cashflows[, timepoints])

An auto-diffable version of [`present_value`](@ref). This function is not as efficient as `present_value`, but is useful when you want to use the function in a differentiable context.
"""
function present_value_differentiable(yc, cashflows, timepoints)
    s = 0.0
     for (cf,t) ∈ zip(cashflows,timepoints) 
        v = discount(yc,t)
        s += v * cf
    end
    s
end

# dispatch on a scalar value to avoid repeated discount computations
present_value(y::Y, c,t) where {Y<:Real} = _present_value_scalar(y,c,t)
present_value(y::Y, c,t) where {Y<:Yields.Constant} = _present_value_scalar(y,c,t)
present_value(y::Y, c,t) where {Y<:FinanceCore.Rate} = _present_value_scalar(y,c,t)

function _present_value_scalar(y, cashflows,times)
    s = zero(first(cashflows) * 0.1)
    v = discount(y,1)
    k = -log(v)
    @turbo for i ∈ eachindex(cashflows) 
        v = exp(-k*times[i])
        s += v * cashflows[i]
    end
    s
end

function _present_value_scalar_one_to_n(y, cashflows)
    s = zero(eltype(cashflows))
    v = 1.0
    v_factor = discount(y,1)
    for i ∈ eachindex(cashflows) 
        v *= v_factor
        @muladd  s = s + v * cashflows[i]
    end
    s
end

function present_value(yc, cashflows)
    present_value(yc,cashflows,eachindex(cashflows))
end

present_value(y, c::G) where {G<:Base.Generator} = present_value_differentiable(y,c,eachindex(c))
present_value(y::Y, c) where {Y<:Real} = _present_value_scalar_one_to_n(y,c)
present_value(y::Y, c) where {Y<:Yields.Constant} = _present_value_scalar_one_to_n(y,c)
present_value(y::Y, c) where {Y<:FinanceCore.Rate} = _present_value_scalar_one_to_n(y,c)

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

price_differentiable(x1,x2) = present_value_differentiable(x1, x2) |> abs
price_differentiable(x1,x2,x3) = present_value_differentiable(x1, x2, x3) |> abs


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


"""
    KeyRatePar(timepoint,shift=0.001) <: KeyRateDuration

Shift the par curve by the given amount at the given timepoint. Use in conjunction with `duration` to calculate the key rate duration. 

Unlike other duration statistics which are computed using analytic derivatives, `KeyRateDuration`s are computed via a shift-and-compute the yield curve approach.

`KeyRatePar` is more commonly reported (than [`KayRateZero`](@ref)) in the fixed income markets, even though the latter has more analytically attractive properties. See the discussion of KeyRateDuration in the Yields.jl docs.

"""
struct KeyRatePar{T,R} <: KeyRateDuration 
    timepoint::T
    shift::R
    KeyRatePar(timepoint, shift=.001) = new{typeof(timepoint),typeof(shift)}(timepoint,shift)
end

"""
    KeyRateZero(timepoint,shift=0.001) <: KeyRateDuration

Shift the par curve by the given amount at the given timepoint. Use in conjunction with `duration` to calculate the key rate duration.

Unlike other duration statistics which are computed using analytic derivatives, `KeyRateDuration` is computed via a shift-and-compute the yield curve approach.

`KeyRateZero` is less commonly reported (than [`KayRatePar`](@ref)) in the fixed income markets, even though the latter has more analytically attractive properties. See the discussion of KeyRateDuration in the Yields.jl docs.
"""
struct KeyRateZero{T,R} <: KeyRateDuration 
    timepoint::T
    shift::R
    KeyRateZero(timepoint, shift=.001) = new{typeof(timepoint),typeof(shift)}(timepoint,shift)
end

"""
    KeyRate(timepoints,shift=0.001)

A convenience constructor for [`KeyRateZero`](@ref). 

## Extended Help
[`KeyRateZero`](@ref) is chosen as the default constructor because it has more attractive properties than [`KeyRatePar`](@ref):

- rates after the key `timepoint` remain unaffected by the `shift`
  - e.g. this causes a 6-year zero coupon bond would have a negative duration if the 5-year par rate was used


"""
KeyRate = KeyRateZero

""" 
    duration(Macaulay(),interest_rate,cfs,times)
    duration(Modified(),interest_rate,cfs,times)
    duration(DV01(),interest_rate,cfs,times)
    duration(interest_rate,cfs,times)             # Modified Duration
    duration(interest_rate,valuation_function)    # Modified Duration

Calculates the Macaulay, Modified, or DV01 duration. `times` may be ommitted and the valuation will assume evenly spaced cashflows starting at the end of the first period.
- `interest_rate` should be a fixed effective yield (e.g. `0.05`).


When not given `Modified()` or `Macaulay()` as an argument, will default to `Modified()`.

- Modified duration: the relative change per point of yield change.
- Macaulay: the cashflow-weighted average time.
- DV01: the absolute change per basis point (hundredth of a percentage point).

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
    return sum(times .* price_differentiable.(yield, cfs, times) / price_differentiable(yield, cfs, times))
end

function duration(::Modified, yield, cfs, times)
    D(i) = price_differentiable(i, cfs, times)
    return duration(yield, D)
end

function duration(yield::Y, valuation_function::T) where {Y<:Yields.AbstractYield,T<:Function}
    D(i) = log(valuation_function(i + yield))
    δV =  - ForwardDiff.derivative(D, 0.0)
end

function duration(yield, cfs, times)
    return duration(Modified(), yield, vec(cfs), times)
end
function duration(yield::Y, cfs) where {Y <: Yields.AbstractYield}
    times = 1:length(cfs)
    return duration(Modified(), yield, cfs, times)
end

function duration(yield::R, cfs) where {R <: Real}
    return duration(Yields.Constant(yield), cfs)
end

function duration(::DV01, yield, cfs, times)
    return duration(DV01(), yield, i -> price_differentiable(i, vec(cfs), times))
end
function duration(d::Duration, yield, cfs)
    times = 1:length(cfs)
    return duration(d, yield, vec(cfs), times)
end

function duration(::DV01, yield, valuation_function::Y) where {Y<:Function}
    return duration(yield, valuation_function) * valuation_function(yield) / 10000
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
    return convexity(yield, i -> price_differentiable(i, cfs, times))
end

function convexity(yield,cfs)
    times = 1:length(cfs)
    return convexity(yield, i -> price_differentiable(i, cfs, times))
end

function convexity(yield, valuation_function::T) where {T<:Function}
    v(x) = abs(valuation_function(yield + x[1]))
    ∂²P = ForwardDiff.hessian(v, [0.0])
    return ∂²P[1] / v([0.0])  
end


"""
    duration(keyrate::KeyRateDuration,curve,cashflows)    
    duration(keyrate::KeyRateDuration,curve,cashflows,timepoints)
    duration(keyrate::KeyRateDuration,curve,cashflows,timepoints,krd_points)

Calculate the key rate duration by shifting the **zero** (not par) curve by the kwarg `shift` at the timepoint specified by a KeyRateDuration(time).

The approach is to carve up the curve into `krd_points` (default is the unit steps between `1` and  the last timepoint of the casfhlows). The 
zero rate corresponding to the timepoint within the `KeyRateDuration` is shifted by `shift` (specified by the `KeyRateZero` or `KeyRatePar` constructors. A new curve is created from the shifted rates. This means that the 
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
- (Financial Exam Help 123](http://www.financialexamhelp123.com/key-rate-duration/)

"""
function duration(keyrate::KeyRateDuration, curve, cashflows, timepoints, krd_points)
    shift = keyrate.shift
    curve_up = _krd_new_curve(keyrate,curve,krd_points)
    curve_down = _krd_new_curve(opposite(keyrate),curve,krd_points)
    price = present_value_differentiable(curve, cashflows, timepoints)
    price_up = present_value_differentiable(curve_up, cashflows, timepoints)
    price_down = present_value_differentiable(curve_down, cashflows, timepoints)
    

    return (price_down - price_up) / (2*shift*price)

end

opposite(kr::KeyRateZero) = KeyRateZero(kr.timepoint,-kr.shift)
opposite(kr::KeyRatePar) = KeyRatePar(kr.timepoint,-kr.shift)

function _krd_new_curve(keyrate::KeyRateZero,curve,krd_points)
    curve_times = krd_points
    shift = keyrate.shift

    zeros = Yields.zero.(curve, curve_times)

    zero_index = findfirst(==(keyrate.timepoint), curve_times)

    target_rate = zeros[zero_index]

    zeros[zero_index] += Yields.Rate(shift,target_rate.compounding)

    new_curve = Yields.Zero(zeros, curve_times)

    return new_curve
end

function _krd_new_curve(keyrate::KeyRatePar,curve,krd_points)
    curve_times = krd_points
    shift = keyrate.shift

    pars = Yields.par.(curve, curve_times)

    zero_index = findfirst(==(keyrate.timepoint), curve_times)

    target_rate = pars[zero_index]
    pars[zero_index] += Yields.Rate(shift,target_rate.compounding)

    new_curve = Yields.Par(pars, curve_times)

    return new_curve
end

function duration(keyrate::KeyRateDuration, curve, cashflows, timepoints)
    krd_points = 1:maximum(timepoints)
    return duration(keyrate, curve, cashflows, timepoints, krd_points)

end

function duration(keyrate::KeyRateDuration, curve, cashflows)
    timepoints = eachindex(cashflows)
    krd_points = 1:maximum(timepoints)
    return duration(keyrate, curve, cashflows, timepoints, krd_points)

end

""" 
    spread(curve1,curve2,cashflows)

Return the solved-for constant spread to add to `curve1` in order to equate the discounted `cashflows` with `curve2`
"""
function spread(curve1,curve2,cashflows,times=eachindex(cashflows))
    pv1 = present_value_differentiable(curve1,cashflows,times)
    pv2 = present_value_differentiable(curve2,cashflows,times)
    irr1 = irr([-pv1;cashflows], [0.;times])
    irr2 = irr([-pv2;cashflows], [0.;times])

    return irr2 - irr1

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
    returned = sum(cf for cf in cfs if cf > 0)
    invested = -sum(cf for cf in cfs if cf < 0)
    return returned / invested
end