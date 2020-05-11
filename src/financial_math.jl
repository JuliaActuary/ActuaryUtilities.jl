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
function InterestCurve(rates,times)
    return InterestCurve(rates,times,LinearInterp())
end

function InterestCurve(rates,times,interpolation_method::StepwiseInterp)
    f(time) = rates[findfirst(t -> t >= time,times)]
    return InterestCurve(rates,times,interpolation_method,f)

end


# make interest curve broadcastable so that you can broadcast over multiple`time`s in `interest_rate`
Base.Broadcast.broadcastable(ic::InterestCurve) = Ref(ic) 


"""
    interest_rate(InterestCurve,time)

Return the interest rate at the instant `time`.

# Examples
```julia-repl
julia> ic = InterestCurve([0.01,0.05,0.05,0.1],[1,2,3,4])

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
"""
function interest_rate(ic::InterestCurve,time)
    return ic.interp_func(time)
end

"""
    interest_rate(i,time)

Returns `i` at all times.

# Examples
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
function interest_rate(i,time)
    return i
end

"""
    interest_rate(InterestCurve,time)

Return the discount rate at `time`. 

Internally, this method is general in that it will use the interpolated interest rate to use an integral approxmation to the accumulated force of interest.

# Examples
```julia-repl
julia> ic = InterestCurve([0.01,0.05,0.05,0.1],[1,2,3,4])

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
"""
function discount_rate(ic::InterestCurve,time)
    # as a general approach, convert Effective Annual Rates
    # to continuously compounded rates for integration
    # i = exp(δ) - 1
    # δ = ln(1+i)
    integral, err = quadgk(t -> log(1+interest_rate(ic,t)),0,time)
    return 1/exp(integral)
end


"""
discount_rate(rate,times)

Turn a rate into a given discount vector matching the times given.

# Examples
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
function discount_rate(i, time)
    return  1 / ((1 + i) ^ time)
end