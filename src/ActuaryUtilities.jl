module ActuaryUtilities

using Dates
using Roots

"""
    Years_Between(d1::Date, d2::Date)
    
Compute the number of integer years between two dates, with the 
first date typically before the second. Will return negative number if
first date is after the second. Use third argument to indicate if calendar 
annivesary should count as a full year.

# Examples
```jldoctest
julia> d1 = Date(2018,09,30);

julia> d2 = Date(2019,09,30);

julia> d3 = Date(2019,10,01);

julia> years_between(d1,d3) 
1
julia> years_between(d1,d2,false) # same month/day but `false` overlap
0 
julia> years_between(d1,d2) # same month/day but `true` overlap
1 
julia> years_between(d1,d2) # using default `true` overlap
1 
```
"""
function years_between(d1::Date,d2::Date,overlap=true)
    iy,im,id = Dates.year(d1), Dates.month(d1), Dates.day(d1)
    vy,vm,vd = Dates.year(d2), Dates.month(d2), Dates.day(d2)
    dur = vy - iy
    if vm == im
        if overlap
            if vd >= id
                 dur += 1
            end
        else
            if vd > id
                 dur += 1
            end
        end
    elseif vm > im
        dur += 1
    end

    return dur - 1
end


"""
    duration(d1::Date, d2::Date)

Compute the duration given two dates, which is the number of years
since the first date. The interval `[0,1)` is defined as having 
duration `1`. Can return negative durations if second argument is before the first.


```jldoctest
julia> issue_date  = Date(2018,9,30);

julia> duration(issue_date , Date(2019,9,30) ) 
2
julia> duration(issue_date , issue_date) 
1
julia> duration(issue_date , Date(2018,10,1) ) 
1
julia> duration(issue_date , Date(2019,10,1) ) 
2
julia> duration(issue_date , Date(2018,6,30) ) 
0
julia> duration(Date(2018,9,30),Date(2017,6,30)) 
-1
```

"""
function duration(issue_date::Date, proj_date::Date)
    return years_between(issue_date,proj_date,true) + 1
end

"""
    internal_rate_of_return(cashflows::vector; search_interval::Tuple{Real,Real})
    
Calculate the internal_rate_of_return of a series of equally spaced cashflows, assuming the first 
element occurs at time zero. By default searches the `search_interval`in the numeric range `[-1,1]`.

"""
function internal_rate_of_return(cashflows;search_interval::Tuple{Real,Real}=(-1.0,1.0))

    f(i) = pv(i,cashflows[2:end]) + cashflows[1]

    return find_zero(f,search_interval)
end

"""
    internal_rate_of_return(cashflows::Vector, timepoints::Vector; search_interval::Tuple{Real,Real})

Calculate the internal_rate_of_return with given timepoints. 
By default searches the `search_interval`in the numeric range `[-1,1]`.

```jldoctest
julia> internal_rate_of_return([-100,110],[0,1]) # e.g. cashflows at time 0 and 1
0.10000000000000005
```
"""
function internal_rate_of_return(cashflows,times;search_interval::Tuple{Real,Real}=(-1.0,1.0))

    f(i) = sum(cashflows[1:end] .* [1/(1+i)^t for t in times])

    return find_zero(f,search_interval)

end

"""
    irr()

    A shorthand for `internal_rate_of_return`.
"""
irr = internal_rate_of_return


"""
    pv(interest_rate::Real, cashflows::Vector)

Discount the `cashflows` vector at the given decimal `interest_rate`. It is assumed that the cashflows are 
periodic commisurate with the period of the interest rate (ie use an annual rate for 
annual values in the vector, quarterly interest rate for quarterly cashflows). The first
value of the `cashflows` vector is assumed to occur at the end of period 1.

"""
function present_value(i,v)
    return sum(v .* [1/(1+i)^t for t in 1:length(v)])
end

"""
    present_value(interest_rate::Real, cashflows::Vector, timepoints)

Discount the `cashflows` vector at the given decimal `interest_rate`,  with the cashflows occuring
at the times specified in `timepoints`. 

```jldoctest
julia> present_value(0.1, [10,20],[0,1])
28.18181818181818
```

Example on how to use real dates using the [DayCounts.jl](https://github.com/JuliaFinance/DayCounts.jl) package
```jldoctest

using DayCounts 
dates = Date(2012,12,31):Year(1):Date(2013,12,31)
times = yearfrac.(dates[1], dates, Actual365) # [0.0,1.0]
present_value(0.1, [10,20],times)

# output
28.18181818181818


```

"""
function present_value(i,v,timepoints;)
    return sum(v .* [1/(1+i)^t for t in timepoints])
end


"""
    pv()

    A shorthand for `present_value`.
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
function breakeven(cashflows::Vector,i::Real)
    λ = (accum,next) -> accum * (1+i) + next
    accum = accumulate(λ,cashflows )
    last_neg = findlast(x -> x < 0.0, accum)
    if last_neg == length(cashflows)
        return nothing
    else
        return last_neg
    end
end

"""
    breakeven(cashflows::Vector,timepoints::Vector, accumulation_rate::Real)

Calculate the time when the accumulated cashflows breakeven.
Assumes that:
- cashflows occur at the timepoint indicated at the corresponding `timepoints` position
- cashflows occur at the end of the period
- that the accumulation rate correponds to the periodicity of the cashflows.

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
function breakeven(cashflows::Vector,timepoints::Vector, i::Real)
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




export years_between, duration,
    irr, internal_rate_of_return, 
    pv, present_value,
    breakeven

end # module
