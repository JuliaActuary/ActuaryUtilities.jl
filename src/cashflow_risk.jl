# A zero stream is an empty collection or one whose amounts are all exactly zero.
# `iszero` checks Dual partials too: a zero primal with a nonzero derivative must
# continue through valuation. Never infer a zero stream from its net present value.
_iszero_cashflow_stream(cfs) = all(cf -> iszero(_cf_value(cf)), cfs)

function _check_cashflow_times(cfs, times; equal_length = false)
    if equal_length && length(cfs) != length(times)
        throw(DimensionMismatch("cfs and times must have equal length"))
    end
    checkbounds(Bool, times, eachindex(cfs)) || throw(
        DimensionMismatch("times must contain at least one entry for each cashflow")
    )
    return nothing
end

# Zero streams do not require a curve query. Concrete input types determine the
# value type; abstractly typed empty collections have no values to promote.
_cashflow_amount_type(::Type{T}) where {T} = T
_cashflow_amount_type(::Type{<:FinanceCore.Cashflow{T}}) where {T} = T
function _zero_cashflow_value(cfs, times)
    C = if isempty(cfs)
        _cashflow_amount_type(eltype(cfs))
    else
        mapreduce(cf -> typeof(float(_cf_value(cf))), promote_type, cfs)
    end
    T = promote_type(C, eltype(times))
    return zero(isconcretetype(T) && T <: Real ? float(T) : Float64)
end

# The same normalization handles scalars and arrays. Signs stay inside the
# broadcast, and zero streams return positive typed zeros before any division.
@inline function _risk_ratio(numerator, value, zero_stream = false; negate = false, divisor = 1)
    zero_stream && return zero.(numerator)
    return negate ? .-numerator ./ value ./ divisor : numerator ./ value ./ divisor
end
