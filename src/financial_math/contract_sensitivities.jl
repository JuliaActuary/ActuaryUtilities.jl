## Contract and portfolio sensitivities
# Reproject under each bumped curve so floating coupons reset. `_contract_keys`
# identifies which contracts need a projection curve.

_contract_keys(c::FinanceModels.Bond.Floating) = (c.key,)
_contract_keys(c::FinanceCore.Composite) = (_contract_keys(c.a)..., _contract_keys(c.b)...)
_contract_keys(c::FinanceModels.Forward) = _contract_keys(c.instrument)
_contract_keys(::FinanceCore.AbstractContract) = ()

const _Contractish = Union{FinanceCore.AbstractContract, AbstractVector{<:FinanceCore.AbstractContract}}

"""
    reproject(contract, index_curve)

Project coupons using `index_curve`. Return the contract unchanged if its
cashflows are fixed; otherwise map its model keys to `index_curve` in a `Projection`.
"""
reproject(c::FinanceCore.AbstractContract, index) =
    isempty(_contract_keys(c)) ? c :
    FinanceModels.Projection(c, Dict(k => index for k in _contract_keys(c)), FinanceModels.CashflowProjection())

# Use one curve for coupon projection and discounting.
_cvalue(c::FinanceCore.AbstractContract, curve) = FinanceCore.present_value(curve, reproject(c, curve))
_cvalue(cs::AbstractVector{<:FinanceCore.AbstractContract}, curve) = sum(_cvalue(c, curve) for c in cs)
# Project coupons on `fwd` and discount on `credit`.
_cvalue2(c::FinanceCore.AbstractContract, fwd, credit) = FinanceCore.present_value(credit, reproject(c, fwd))
_cvalue2(cs::AbstractVector{<:FinanceCore.AbstractContract}, fwd, credit) = sum(_cvalue2(c, fwd, credit) for c in cs)

"""
    sensitivities(target, curve, tenors) -> NamedTuple
    sensitivities(target, forward, credit, tenors) -> NamedTuple

Calculate sensitivities for a contract or portfolio, reprojecting cashflows under
bumped curves. Project coupons on `forward` and discount on `credit`, or pass one
`curve` for both. Return these results on the `tenors` grid:

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
function sensitivities(target::_Contractish, forward::AYM, credit::AYM, tenors)
    r = _ncurve_ad(c -> _cvalue2(target, c.forward, c.credit), (; forward, credit), tenors; order = 1)
    v = r.value
    gf, gc = r.gradient.forward, r.gradient.credit
    # Use signed derivatives to retain dollar exposure at zero present value.
    forward_dv01 = -sum(gf) / 10_000
    spread_dv01 = -sum(gc) / 10_000
    effective_dv01 = forward_dv01 + spread_dv01
    fwd = -gf ./ v
    spr = -gc ./ v
    eff = fwd .+ spr
    return (;
        value = v,
        effective_duration = sum(eff), effective_dv01, effective_key_rate = eff,
        spread_duration = sum(spr), spread_dv01, spread_key_rate = spr,
        forward_duration = sum(fwd), forward_dv01, forward_key_rate = fwd,
    )
end
sensitivities(target::_Contractish, curve::AYM, tenors) = sensitivities(target, curve, curve, tenors)

function _contract_parallel(metric, target, forward, credit, tenors)
    _validate_tenors(tenors)
    f(s) = _contract_parallel_value(metric, target, forward, credit, s)
    return (; value = f(0.0), derivative = ForwardDiff.derivative(f, 0.0))
end
_contract_parallel_value(::Effective, target, forward, credit, s) =
    _cvalue2(target, _parallel_bumped(forward, s), _parallel_bumped(credit, s))
_contract_parallel_value(::Spread, target, forward, credit, s) =
    _cvalue2(target, forward, _parallel_bumped(credit, s))

"""
    duration(Effective(), target, curve, tenors)          # rate duration, yrs
    duration(Spread(),    target, curve, tenors)          # spread duration, yrs
    duration(Effective(), KeyRates(tenors), target, curve) # key-rate vector
    dv01(Effective()/Spread(), target, curve, tenors)     # the dollar versions
    duration(target, curve, tenors)                     # defaults to Effective()
    dv01(target, curve, tenors)                         # defaults to Effective()
    convexity(target, curve, tenors)                    # defaults to Effective()

Effective (rate) and spread (credit) duration / DV01 for a contract or portfolio,
re-projecting cashflows under bumped curves. Two-curve forms take `(forward, credit)`.
Unmarked single-curve contract and portfolio calls use `Effective()` for duration,
DV01, and convexity; spread risk requires an explicit `Spread()` marker.
See [`sensitivities`](@ref) for the full one-pass bundle.
"""
function duration(metric::Effective, target::_Contractish, forward::AYM, credit::AYM, tenors)
    r = _contract_parallel(metric, target, forward, credit, tenors)
    return -r.derivative / r.value
end
duration(::Effective, target::_Contractish, curve::AYM, tenors) = duration(Effective(), target, curve, curve, tenors)
function duration(metric::Spread, target::_Contractish, forward::AYM, credit::AYM, tenors)
    r = _contract_parallel(metric, target, forward, credit, tenors)
    return -r.derivative / r.value
end
duration(::Spread, target::_Contractish, curve::AYM, tenors) = duration(Spread(), target, curve, curve, tenors)
duration(::Effective, kr::KeyRates, target::_Contractish, curve::AYM) = sensitivities(target, curve, kr.tenors).effective_key_rate
duration(::Spread, kr::KeyRates, target::_Contractish, curve::AYM) = sensitivities(target, curve, kr.tenors).spread_key_rate
# Unmarked contract and portfolio calls use Effective().
duration(target::_Contractish, curve::AYM, tenors::AbstractVector) = duration(Effective(), target, curve, tenors)
duration(kr::KeyRates, target::_Contractish, curve::AYM) = duration(Effective(), kr, target, curve)
duration(::DV01, target::_Contractish, curve::AYM, tenors::AbstractVector) = dv01(Effective(), target, curve, tenors)
convexity(target::_Contractish, curve::AYM, tenors::AbstractVector) = convexity(Effective(), target, curve, tenors)

# Parallel convexity equals the full key-rate matrix sum. The scalar callback
# computes it directly while reprojecting coupons under each curve shock.
function convexity(::Effective, target::_Contractish, curve::AYM, tenors)
    _validate_tenors(tenors)
    return convexity(curve, c -> _cvalue(target, c))
end

"""
    dv01(args...)

Return signed dollar risk `-∂V/∂r / 10000`. Cashflow and callback forms alias
`duration(DV01(), args...)`. Contract forms accept `Effective()` or `Spread()`;
unmarked single-curve contract and portfolio calls default to `Effective()`.
"""
function dv01(metric::Effective, target::_Contractish, forward::AYM, credit::AYM, tenors)
    return -_contract_parallel(metric, target, forward, credit, tenors).derivative / 10_000
end
dv01(::Effective, target::_Contractish, curve::AYM, tenors) = dv01(Effective(), target, curve, curve, tenors)
function dv01(metric::Spread, target::_Contractish, forward::AYM, credit::AYM, tenors)
    return -_contract_parallel(metric, target, forward, credit, tenors).derivative / 10_000
end
dv01(::Spread, target::_Contractish, curve::AYM, tenors) = dv01(Spread(), target, curve, curve, tenors)
dv01(args...; kwargs...) = duration(DV01(), args...; kwargs...)

function sensitivities(target::_Contractish, tenors::AbstractVector; discount::NamedTuple, index)
    layers = keys(discount)
    return sensitivities(merge(discount, (; index = index)); tenors) do c
        _cvalue2(target, c.index, reduce(+, getfield(c, r) for r in layers))
    end
end

"""
    zspread(contract, credit, market_price; forward=credit) -> (; zspread, zspread_dv01)

Constant continuously-compounded spread `s` on the `credit` (discount) curve such that
the model price equals `market_price`, with coupons estimated on `forward` (held fixed).
Returns the spread and its sensitivity (\\\$/1bp parallel move of `credit + s`). Newton + AD.
"""
function zspread(contract::FinanceCore.AbstractContract, credit::AYM, market_price; forward::AYM = credit, s0 = 0.0, tol = 1.0e-12, maxiter = 100)
    ks = _contract_keys(contract)
    pvs(s) = let disc = credit + ((z, t) -> FinanceCore.Continuous(s) + z)
        isempty(ks) ? FinanceCore.present_value(disc, contract) :
            FinanceCore.present_value(disc, FinanceModels.Projection(contract, Dict(k => forward for k in ks), FinanceModels.CashflowProjection()))
    end
    f(s) = pvs(s) - market_price
    s = float(s0)
    converged = false
    for _ in 1:maxiter
        fs = f(s)
        (abs(fs) < tol) && (converged = true; break)
        d = ForwardDiff.derivative(f, s)
        iszero(d) && break
        s -= fs / d
    end
    converged || throw(ErrorException("zspread did not converge (last residual = $(f(s)))"))
    return (; zspread = s, zspread_dv01 = -ForwardDiff.derivative(pvs, s) / 10_000)
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
