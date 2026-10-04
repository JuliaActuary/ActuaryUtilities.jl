## Cashflow inputs
# Every fixed-cashflow kernel takes its amounts and times from `_cashflow_inputs`, as 1-based vectors
# of equal length that pair by position:
#
# - Omitted times are the amounts' indices, taken before any reindexing, so an offset vector's
#   amounts are paid at its indices. A `Cashflow` is paid at its own time whatever its paired time.
# - Another array is flattened in column-major order, and a tuple or generator is collected once,
#   so a single-pass iterator is read once. A 1-based vector is used as it is; an offset vector is
#   viewed from 1.
# - Times pair from their first entry, so offset times pair too. There must be one for each amount;
#   a view leaves out the unused trailing ones.
_cashflow_inputs(cfs) = (amounts = _vector(cfs); _paired(amounts, eachindex(amounts)))
_cashflow_inputs(cfs, times) = _paired(_vector(cfs), _vector(times))

function _paired(amounts::AbstractVector, times::AbstractVector)
    n = length(amounts)
    length(times) >= n || throw(DimensionMismatch("times must contain at least one entry for each cashflow"))
    first_time = firstindex(times)
    return _one_based(amounts, axes(amounts, 1)), view(times, first_time:(first_time + n - 1))
end

_vector(x::AbstractVector) = x
_vector(x::AbstractArray) = vec(x)
_vector(x::Union{Tuple, Base.Generator}) = collect(x)

_one_based(x, ::Base.OneTo) = x
_one_based(x, _) = view(x, firstindex(x):lastindex(x))

@inline _cf_value(c::FinanceCore.Cashflow) = FinanceCore.amount(c)
@inline _cf_value(c) = c

# A zero stream is an empty collection or one whose amounts are all exactly zero.
# `iszero` checks Dual partials too: a zero primal with a nonzero derivative must
# continue through valuation. Never infer a zero stream from its net present value.
_iszero_cashflow_stream(cfs) = all(cf -> iszero(_cf_value(cf)), cfs)

# Call after the zero-stream return. Simulation horizons must use the same embedded payment times
# as valuation.
_maximum_cashflow_time(cfs, times) = maximum(k -> FinanceCore.timepoint(cfs[k], times[k]), eachindex(cfs))

# Zero streams return exact zeros without valuing each payment; linearity forces the value.
# Its type is a convention: the type a nonempty stream's result would have, the promotion of
# the amount, time and discount types. The discount type comes from one query at time zero
# (`disc(t)` is the discount factor), so the rate or curve sets the type when the element
# type says nothing, as for `Any[]` or `Cashflow[]`. Taking `zero` of that product gives a
# positive zero of its type even when a factor is not finite, and zero partials under AD.
function _zero_cashflow_value(disc, cfs, times)
    t = _zero_stream_time(cfs, times)
    return zero(disc(t) * _zero_amount(cfs) * one(t))
end
# The time type follows valuation: a numeric amount is paid at its supplied time, but a
# `Cashflow` carries its own, so the supplied time does not count for it. An abstractly typed
# stream, such as `Cashflow[...]` or `Any[...]`, uses the times its payments are made at, as
# `_zero_amount` uses the amounts present.
function _zero_stream_time(cfs, times)
    isconcretetype(eltype(cfs)) || isempty(cfs) ||
        return _scanned_zero(k -> FinanceCore.timepoint(cfs[k], times[k]), cfs)
    return _zero_stream_time(eltype(cfs), times)
end
_zero_stream_time(::Type, times) = _zero_time(eltype(times))
_zero_stream_time(::Type{Union{}}, times) = _zero_time(eltype(times))
_zero_stream_time(::Type{<:FinanceCore.Cashflow}, times) = false
# Bound both parameters as `Cashflow` does: with an unbounded amount type, Julia ranks the
# `Type{<:Cashflow}` method above as more specific, and every `Cashflow` would give `false`.
_zero_stream_time(::Type{FinanceCore.Cashflow{A, T}}, times) where {A <: Real, T <: Real} = zero(T)

# The zero-stream value for a single rate, number or curve.
_zero_stream_value(yield, cfs, times) = _zero_cashflow_value(t -> FinanceCore.discount(yield, t), cfs, times)
# The same for a statistic Σ weight(t)⋅cf⋅d / divisor, so it has the nonempty formula's type.
_zero_weighted(yield, weight, cfs, times, divisor = 1) =
    zero(weight(_zero_stream_time(cfs, times)) * _zero_stream_value(yield, cfs, times) / divisor)

# An all-zero stream in an abstractly typed collection uses the amounts present.
function _zero_amount(cfs)
    isconcretetype(eltype(cfs)) || isempty(cfs) || return _scanned_zero(k -> _cf_value(cfs[k]), cfs)
    return _zero_amount_of(eltype(cfs))
end
# The zero of the promoted type that `f` gives each payment `k`.
_scanned_zero(f::F, cfs) where {F} = zero(mapreduce(k -> typeof(f(k)), promote_type, eachindex(cfs)))
# An element type that says nothing about amounts contributes `false`, which every
# numeric type absorbs under promotion.
_zero_amount_of(::Type) = false
_zero_amount_of(::Type{Union{}}) = false
_zero_amount_of(::Type{T}) where {T <: Real} = zero(T)
_zero_amount_of(::Type{<:FinanceCore.Cashflow{A}}) where {A <: Real} = zero(A)
_zero_time(::Type) = false
_zero_time(::Type{Union{}}) = false
_zero_time(::Type{T}) where {T <: Real} = zero(T)

# The same normalization handles scalars and arrays. Signs stay inside the
# broadcast, and zero streams return positive typed zeros before any division.
@inline function _risk_ratio(numerator, value, zero_stream = false; negate = false, divisor = 1)
    zero_stream && return zero.(numerator)
    return negate ? .-numerator ./ value ./ divisor : numerator ./ value ./ divisor
end
