## ── Contract / portfolio-aware, re-projecting duration & sensitivities ────────
#
# A contract (or a vector of contracts = a portfolio) is a "target": it is valued
# by RE-PROJECTING under a bumped curve, so a floater's coupons re-fix automatically
# (its projection reads the model) while a fixed bond's do not. `_contract_keys`
# (empty vs not) is the only discriminator — no `ValuationStyle` trait. The risk
# factor stays the familiar vocabulary: `Effective` (rate) / `Spread` (credit) /
# `KeyRates`; units are the verb (`duration` yrs / `dv01` $ / `convexity`).

_contract_keys(c::FinanceModels.Bond.Floating) = (c.key,)
_contract_keys(c::FinanceCore.Composite) = (_contract_keys(c.a)..., _contract_keys(c.b)...)
_contract_keys(c::FinanceModels.Forward) = _contract_keys(c.instrument)
_contract_keys(::FinanceCore.AbstractContract) = ()

const _Contractish = Union{FinanceCore.AbstractContract, AbstractVector{<:FinanceCore.AbstractContract}}

"""
    reproject(contract, index_curve)

Wrap `contract` so its coupons are estimated off `index_curve`: returns the contract
itself if it reads no model, else a `Projection` mapping the contract's keys to
`index_curve`. Lets a multi-curve valuation avoid hand-writing the model `Dict`.
"""
reproject(c::FinanceCore.AbstractContract, index) =
    isempty(_contract_keys(c)) ? c :
    FinanceModels.Projection(c, Dict(k => index for k in _contract_keys(c)), FinanceModels.CashflowProjection())

# value of a contract/portfolio under a single curve (coupons + discount = curve) …
_cvalue(c::FinanceCore.AbstractContract, curve) = FinanceCore.present_value(curve, reproject(c, curve))
_cvalue(cs::AbstractVector{<:FinanceCore.AbstractContract}, curve) = sum(_cvalue(c, curve) for c in cs)
# … and two curves: estimate coupons on `fwd`, discount on `credit`.
_cvalue2(c::FinanceCore.AbstractContract, fwd, credit) = FinanceCore.present_value(credit, reproject(c, fwd))
_cvalue2(cs::AbstractVector{<:FinanceCore.AbstractContract}, fwd, credit) = sum(_cvalue2(c, fwd, credit) for c in cs)

"""
    sensitivities(target, curve, tenors) -> NamedTuple
    sensitivities(target, forward, credit, tenors) -> NamedTuple

One-AD-pass bundle for a (possibly curve-dependent) `target` — a `FinanceModels`
contract or a vector of contracts (portfolio) — re-projecting cashflows under bumped
curves. Coupons are estimated on `forward`, discounted on `credit` (pass a single
`curve` for both). Returns, over the `tenors` key-rate grid:

  - `value`
  - `effective_duration` / `effective_dv01` / `effective_key_rate` — bump both curves
    (coupons re-fix): the interest-rate duration (≈ next reset for a floater).
  - `spread_duration` / `spread_dv01` / `spread_key_rate` — bump the discount only:
    the discount-margin / credit duration (≈ maturity for a floater).
  - `forward_duration` / `forward_dv01` / `forward_key_rate` — bump the index only;
    `effective = forward + spread` (first order).

Durations in years; DV01s in dollars per 1bp. For a fixed bond `effective == spread ==`
the modified duration and `forward == 0`. See [`duration`](@ref) with [`Effective`](@ref)/
[`Spread`](@ref), [`dv01`](@ref), [`zspread`](@ref), [`locked_floater`](@ref).
"""
function sensitivities(target::_Contractish, forward::AYM, credit::AYM, tenors)
    r = _ncurve_ad(c -> _cvalue2(target, c.forward, c.credit), (; forward, credit), tenors; order = 1)
    v = r.value
    fwd = -r.gradient.forward ./ v
    spr = -r.gradient.credit ./ v
    eff = fwd .+ spr
    return (;
        value = v,
        effective_duration = sum(eff), effective_dv01 = sum(eff) * v / 10_000, effective_key_rate = eff,
        spread_duration = sum(spr), spread_dv01 = sum(spr) * v / 10_000, spread_key_rate = spr,
        forward_duration = sum(fwd), forward_dv01 = sum(fwd) * v / 10_000, forward_key_rate = fwd,
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

Effective (rate) and spread (credit) duration / DV01 for a contract or portfolio,
re-projecting cashflows under bumped curves. Two-curve forms take `(forward, credit)`.
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
# default (no marker) on a contract/portfolio = effective
duration(target::_Contractish, curve::AYM, tenors::AbstractVector) = duration(Effective(), target, curve, tenors)
duration(kr::KeyRates, target::_Contractish, curve::AYM) = duration(Effective(), kr, target, curve)

# Effective convexity: parallel-shift second derivative of the contract's
# present value under a continuous-rate shock. Routes through the O(1) scalar
# callback path, which is numerically
# equivalent to the prior `sum(convexity(KeyRates(tenors), …))` matrix-sum form
# under partition of unity but avoids the O(N²) Hessian.
convexity(::Effective, target::_Contractish, curve::AYM, _tenors) =
    convexity(curve, c -> _cvalue(target, c))

"""
    dv01(args...)

Dollar value of a 1bp move. `dv01(args...)` ≡ `duration(DV01(), args...)` for the
cashflow/curve forms, with `dv01(Effective()/Spread(), target, [forward, credit,] tenors)`
giving the floating-rate dollar durations (years × value ÷ 10⁴).
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

Model an in-force floater whose current coupon is LOCKED at `current_coupon` (the
per-period amount fixed at the last reset) until `next_reset`, after which it floats.
A `Composite` of a coupon-only stub at `next_reset` plus a forward-starting floater for
the remainder (principal rides the forward leg). Gives the conventional rate duration
≈ time to next reset; without it the idealized effective duration is ≈ 0 at a reset.

The remaining term `fl.maturity - next_reset` must be an integer number of coupon
periods. Otherwise the forward-starting leg would carry a stub first coupon whose
fix-in-advance reference rate looks back before time zero — a quietly mispriced
quantity on curves that extrapolate below ``t = 0`` and a `DomainError` on
`ZeroRateCurve` — so non-commensurate inputs throw an `ArgumentError`.
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
