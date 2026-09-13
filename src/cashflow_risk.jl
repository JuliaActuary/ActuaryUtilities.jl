@inline _cf_value(c::FinanceCore.Cashflow) = FinanceCore.amount(c)
@inline _cf_value(c) = c

# A zero stream is an empty collection or one whose amounts are all exactly zero.
# `iszero` checks Dual partials too: a zero primal with a nonzero derivative must
# continue through valuation. Never infer a zero stream from its net present value.
_iszero_cashflow_stream(cfs) = all(cf -> iszero(_cf_value(cf)), cfs)

# A shared projection grid may extend beyond a stream. Every cashflow needs an
# indexable time; unused trailing times do not enter valuation or derived defaults.
function _check_cashflow_times(cfs, times)
    checkbounds(Bool, times, eachindex(cfs)) || throw(
        DimensionMismatch("times must contain at least one entry for each cashflow")
    )
    return nothing
end

# Trim only when delegating to code that consumes all times (including default
# key-rate grids and simulation horizons). Indexed accumulation loops need only
# the bounds check above. The ordinary equal-length path retains the input.
@inline function _cashflow_times(cfs, times)
    _check_cashflow_times(cfs, times)
    return length(times) == length(cfs) ? times : view(times, eachindex(cfs))
end

# Zero streams do not require a curve query. Concrete input types determine the
# value type; abstractly typed empty collections have no values to promote.
_cashflow_amount_type(::Type{T}) where {T} = T
_cashflow_amount_type(::Type{<:FinanceCore.Cashflow{T}}) where {T} = T
function _zero_cashflow_value(cfs, times)
    C = _cashflow_amount_type(eltype(cfs))
    if !isconcretetype(C) && !isempty(cfs)
        C = mapreduce(cf -> typeof(float(_cf_value(cf))), promote_type, cfs)
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
