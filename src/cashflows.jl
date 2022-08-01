struct Cashflow{A,T}
    amount::A
    time::T
end

#TODO Define Cashflow on a vector/iterable?

function present_value(yield,cf::Cashflow)
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
