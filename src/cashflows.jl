struct Cashflow{A,T}
    amount::A
    time::T
end

function present_value(yield,cf::Cashflow)
    return discount(yield,cf.time) * cf.amount
end