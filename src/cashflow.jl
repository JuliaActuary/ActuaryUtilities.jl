struct Cashflow{A,T}
    amount::A
    time::T
end

#TODO Define Cashflow on a vector/iterable?

@inline function present_value(yield,cf::Cashflow)
    return discount(yield,cf.time) * cf.amount
end

# this method ignores the time argument and instead uses 
# the time within the cashflow. This is to allow for `present_value(yield,cfs)`
# to dispatch on the values of `cfs`. Otherwise, if `cfs` is a generator or otherwise
# an iterable of ambiguous types, we lose the ability to dispatch on the right method
# for Cashflow or just a real type where we infer the timepoint
function present_value(yield,cf::Cashflow,time::T) where {T<:Real}
    return discount(yield,cf.time) * cf.amount
end


# There are places where we want to infer a 1:n range in a lazy way if not Cashflows; if Cashflows, we often want to ignore 
# the times given in the function and stay true to the times embedded in the Cashflows, and these utility functions accomplish this
__times(cfs;start=1) = (__time(cf,t) for (t,cf) in zip(Lazy.range(start),cfs))
__times(cfs,times) = (__time(cf,t) for (t,cf) in zip(times,cfs))
__time(cf::Cashflow,t) = cf.time
__time(cf,t) = t

#look through to the cashflow and grab the amount
__cashflows(cfs) = (__cashflow(cf) for cf in cfs)
__cashflow(cf::Cashflow) = cf.amount
__cashflow(cf) = cf