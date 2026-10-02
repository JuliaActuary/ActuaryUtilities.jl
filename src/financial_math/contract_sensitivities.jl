## Contract and portfolio sensitivities
# Value contracts under bumped curves in a FinanceModels valuation context, so
# ActuaryUtilities keeps no list of contract types.

const _Contractish = Union{FinanceCore.AbstractContract, AbstractVector{<:FinanceCore.AbstractContract}}

# `discount` discounts and prices; every model a contract reads by key is `index`, so floating
# coupons reset on the bumped index curve. Closed forms, composites and portfolios value linearly.
_value(target, index, discount) = FinanceCore.present_value(FinanceModels.Models(discount; index), target)

"""
    sensitivities(kr::KeyRates, target, curve) -> NamedTuple
    sensitivities(kr::KeyRates, target, forward, credit) -> NamedTuple

Calculate sensitivities for a contract or portfolio, reprojecting cashflows under
bumped curves. Project coupons on `forward` and discount on `credit`, or pass one
`curve` for both. Return these results on the `kr.tenors` grid:

  - `value`
  - `effective_duration` / `effective_dv01` / `effective_key_rate` — bump both curves
    (coupons re-fix): the interest-rate duration (≈ next reset for a floater).
  - `spread_duration` / `spread_dv01` / `spread_key_rate` — bump the discount only:
    the discount-margin / credit duration (≈ maturity for a floater).
  - `forward_duration` / `forward_dv01` / `forward_key_rate` — bump the index only;
    `effective = forward + spread` (first order).

Durations are in years; DV01s are in dollars per basis point. Dollar risk uses
signed value derivatives and remains defined at zero value, where normalized
duration is undefined. For a fixed bond, effective and spread duration equal its
continuous-zero duration; forward duration is zero. See [`duration`](@ref) with [`Effective`](@ref)/
[`Spread`](@ref), [`dv01`](@ref), [`zspread`](@ref), [`locked_floater`](@ref).
"""
function sensitivities(kr::KeyRates, target::_Contractish, forward::AYM, credit::AYM)
    return _contract_bundle(_ncurve_ad(c -> _value(target, c.forward, c.credit), (; forward, credit), kr.tenors; order = 1))
end
# A function barrier: some projected contracts (floaters) have uninferred values.
function _contract_bundle(r)
    # Use signed derivatives to retain dollar exposure at zero present value.
    forward_dv01 = _per_bp(r, sum(r.gradient.forward))
    spread_dv01 = _per_bp(r, sum(r.gradient.credit))
    effective_dv01 = forward_dv01 + spread_dv01
    fwd = _relative(r, r.gradient.forward; negate = true)
    spr = _relative(r, r.gradient.credit; negate = true)
    eff = fwd .+ spr
    return (;
        value = r.value,
        effective_duration = sum(eff), effective_dv01, effective_key_rate = eff,
        spread_duration = sum(spr), spread_dv01, spread_key_rate = spr,
        forward_duration = sum(fwd), forward_dv01, forward_key_rate = fwd,
    )
end
sensitivities(kr::KeyRates, target::_Contractish, curve::AYM) = sensitivities(kr, target, curve, curve)

# The value under a continuous-zero parallel shift `s` of the curves the metric moves.
_contract_parallel_value(::Effective, target, forward, credit, s) =
    _value(target, _parallel_bumped(forward, s), _parallel_bumped(credit, s))
_contract_parallel_value(::Spread, target, forward, credit, s) =
    _value(target, forward, _parallel_bumped(credit, s))

"""
    duration(Effective(), target, curve)                   # rate duration, yrs
    duration(Spread(),    target, curve)                   # spread duration, yrs
    duration(Effective(), target, forward, credit)         # two-curve forms
    duration(Effective(), KeyRates(tenors), target, curve) # key-rate vector
    dv01(Effective()/Spread(), target, curve)              # the dollar versions
    duration(target, curve)                                # defaults to Effective()
    dv01(target, curve)                                    # defaults to Effective()
    convexity(target, curve)                               # defaults to Effective()

Effective (rate) and spread (credit) duration / DV01 for a contract or portfolio,
re-projecting cashflows under continuous-zero parallel shifts. Two-curve forms
project coupons on `forward` and discount on `credit`. Unmarked contract and
portfolio calls use `Effective()` for duration, DV01, and convexity; spread risk
requires an explicit `Spread()` marker. Parallel measures take no tenor grid; use
`KeyRates(tenors)` or [`sensitivities`](@ref) for key-rate decompositions.
"""
function duration(metric::Union{Effective, Spread}, target::_Contractish, forward::AYM, credit::AYM)
    value, derivative = _value_and_derivative(s -> _contract_parallel_value(metric, target, forward, credit, s), 0.0)
    return -derivative / value
end
duration(metric::Union{Effective, Spread}, target::_Contractish, curve::AYM) = duration(metric, target, curve, curve)
duration(::Effective, kr::KeyRates, target::_Contractish, curve::AYM) = sensitivities(kr, target, curve).effective_key_rate
duration(::Spread, kr::KeyRates, target::_Contractish, curve::AYM) = sensitivities(kr, target, curve).spread_key_rate
# Unmarked contract and portfolio calls use Effective().
duration(target::_Contractish, curve::AYM) = duration(Effective(), target, curve)
duration(kr::KeyRates, target::_Contractish, curve::AYM) = duration(Effective(), kr, target, curve)
duration(::DV01, target::_Contractish, curve::AYM) = dv01(Effective(), target, curve)
convexity(target::_Contractish, curve::AYM) = convexity(Effective(), target, curve)

# Parallel convexity equals the full key-rate matrix sum. The scalar callback
# computes it directly while reprojecting coupons under each curve shock.
convexity(::Effective, target::_Contractish, curve::AYM) = convexity(curve, c -> _value(target, c, c))

"""
    dv01(args...)

Return signed dollar risk `-∂V/∂r / 10000`. Cashflow and callback forms alias
`duration(DV01(), args...)`. Contract forms accept `Effective()` or `Spread()`;
unmarked contract and portfolio calls default to `Effective()`.
"""
function dv01(metric::Union{Effective, Spread}, target::_Contractish, forward::AYM, credit::AYM)
    return -ForwardDiff.derivative(s -> _contract_parallel_value(metric, target, forward, credit, s), 0.0) / 10_000
end
dv01(metric::Union{Effective, Spread}, target::_Contractish, curve::AYM) = dv01(metric, target, curve, curve)
dv01(args...; kwargs...) = duration(DV01(), args...; kwargs...)

function sensitivities(kr::KeyRates, target::_Contractish; discount::NamedTuple, index)
    layers = keys(discount)
    return sensitivities(kr, merge(discount, (; index = index))) do c
        _value(target, c.index, reduce(+, getfield(c, r) for r in layers))
    end
end

"""
    zspread(contract, credit, market_price; forward=credit) -> (; zspread, zspread_dv01)

Constant continuously-compounded spread `s` on the `credit` (discount) curve such that
the model price equals `market_price`, with coupons estimated on `forward` (held fixed).
Returns the spread and its sensitivity (\\\$/1bp parallel move of `credit + s`). Newton + AD.

The solve stops once a Newton step is smaller than `tol` in rate units (not currency), so the
result does not depend on the contract's notional and is defined for a zero `market_price`.
An `ErrorException` is thrown if that does not happen within `maxiter` steps.
"""
function zspread(contract::FinanceCore.AbstractContract, credit::AYM, market_price; forward::AYM = credit, s0 = 0.0, tol = 1.0e-12, maxiter = 100)
    pvs(s) = _contract_parallel_value(Spread(), contract, forward, credit, s)
    f(s) = pvs(s) - market_price
    result(s) = (; zspread = s, zspread_dv01 = -ForwardDiff.derivative(pvs, s) / 10_000)
    s = float(s0)
    step = oftype(s, NaN)
    for _ in 1:maxiter
        fs, dfs = _value_and_derivative(f, s)
        iszero(fs) && return result(s)
        step = fs / dfs
        isfinite(step) || break
        s -= step
        # A Newton step in rate units, unlike a price residual, does not scale with the notional.
        # In continuous coordinates |f′| ≤ t_max ⋅ Σ|terms|, so a small step bounds the residual
        # locally: |f|/Σ|terms| ≤ |step| ⋅ t_max at this iterate. That assumes positive discount
        # factors and finite terms; it is not a global guarantee about other roots.
        abs(step) < tol && return result(s)
    end
    throw(ErrorException("zspread did not converge (last Newton step = $step, residual = $(f(s)))"))
end

"""
    locked_floater(fl::FinanceModels.Bond.Floating, current_coupon, next_reset)

Pay the fixed coupon amount `current_coupon` at `next_reset`, then resume floating
coupons. Return a `Composite` of that coupon and a forward-starting floater carrying
the principal. Effective duration is approximately the time to the next reset.

The remaining term `fl.maturity - next_reset` must contain a whole number of coupon
periods. Otherwise throw `ArgumentError`: a stub coupon would require a reference
rate before time zero.
"""
function locked_floater(fl::FinanceModels.Bond.Floating, current_coupon, next_reset)
    freq = fl.frequency.frequency
    n_periods = (fl.maturity - next_reset) * freq
    isapprox(n_periods, round(n_periods); atol = 1.0e-8) || throw(
        ArgumentError(
            "locked_floater requires fl.maturity - next_reset ($(fl.maturity - next_reset)) to be an integer number of coupon periods (frequency $freq); a stub first coupon on the forward leg would reference a forward rate starting before time zero"
        )
    )
    stub = FinanceCore.Cashflow(current_coupon, next_reset)
    rest = FinanceModels.Forward(
        next_reset,
        FinanceModels.Bond.Floating(fl.coupon_rate, fl.frequency, fl.maturity - next_reset, fl.key)
    )
    return FinanceCore.Composite(stub, rest)
end
