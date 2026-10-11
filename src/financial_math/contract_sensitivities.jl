## Contract and portfolio sensitivities
# Value contracts under bumped curves in a FinanceModels valuation context, so
# ActuaryUtilities keeps no list of contract types.

const _Contractish = Union{FinanceCore.AbstractContract, AbstractVector{<:FinanceCore.AbstractContract}}
const _ContractMetric = Union{Effective, Spread}

# `discount` discounts and prices; `index` serves every model a contract reads by key, so floating
# coupons reset on the bumped index curve. A closed-form contract uses its own
# `present_value(ctx, c)`; composites and portfolios are valued as the sums of their parts.
_value(target, index, discount) = FinanceCore.present_value(FinanceModels.Models(discount; index), target)

# The value under one shock, `shock(curve)`, to the curves the metric moves: the index and discount
# curves for Effective, the discount curve for Spread.
_contract_value(::Effective, target, discount, index, shock::S) where {S} = _value(target, shock(index), shock(discount))
_contract_value(::Spread, target, discount, index, shock::S) where {S} = _value(target, index, shock(discount))
_parallel_value(metric, target, discount, index, s) =
    _contract_value(metric, target, discount, index, c -> _parallel_bumped(c, s))
# Derivatives on the key-rate grid of one shock vector applied to the curves the metric moves.
function _contract_keyrate(metric, grid, discount, target, index, order)
    _validate_tenors(grid)
    v(b) = _contract_value(metric, target, discount, index, c -> _bumped(c, grid, b.shift))
    return _only_role(_named_ad(v, (; shift = zeros(length(grid))), order))
end

"""
    duration(Effective(), discount, contract; index = discount)     # rate duration, yrs
    duration(Spread(),    discount, contract; index = discount)     # spread duration, yrs
    duration(Effective(), KeyRates(tenors), discount, contract; index = discount)   # vector
    duration(DV01(), Effective(), discount, contract; index = discount)   # the dollar versions
    convexity(Effective(), discount, contract; index = discount)    # and the convexities
    duration(discount, contract; index = discount)                  # Effective() by default
    duration(DV01(), [KeyRates(tenors),] discount, contract; index = discount)
    convexity(discount, contract; index = discount)

Effective (rate) and spread (credit) duration, DV01 and convexity for a contract or a portfolio
(a vector of contracts), reprojecting cashflows under continuous-zero shifts. Coupons project on
the `index` curve, which defaults to `discount`, and every payment is discounted on `discount`:

- `Effective()` shifts both curves, so floating coupons reset.
- `Spread()` shifts the discount curve only, so projected coupons stay fixed.

Markers come in the order `DV01()`, then `Effective()` or `Spread()`, then `KeyRates(tenors)`.
Without `Effective()` or `Spread()`, contract and portfolio measures use `Effective()`. The parallel forms take no
tenor grid; `KeyRates(tenors)` gives per-tenor vectors and convexity matrices. Effective
convexity includes the cross terms between the curves: for the parallel second derivatives `Cᵢᵢ`,
`Cᵢd`, `Cdᵢ` and `Cdd` of the index and discount roles, it is their sum, and spread convexity is
`Cdd`. A portfolio's value and dollar derivatives are summed before normalizing. See
[`sensitivities`](@ref) for the roles separately, [`zspread`](@ref) and [`locked_floater`](@ref).
"""
function duration(metric::_ContractMetric, discount::AYM, target::_Contractish; index::AYM = discount)
    value, derivative = _value_and_derivative(s -> _parallel_value(metric, target, discount, index, s), 0.0)
    return -derivative / value
end
function duration(metric::_ContractMetric, kr::KeyRates, discount::AYM, target::_Contractish; index::AYM = discount)
    r = _contract_keyrate(metric, kr.tenors, discount, target, index, FirstOrder())
    return _relative(r, r.gradient; negate = true)
end

function duration(::DV01, metric::_ContractMetric, discount::AYM, target::_Contractish; index::AYM = discount)
    return -ForwardDiff.derivative(s -> _parallel_value(metric, target, discount, index, s), 0.0) / 10_000
end
function duration(::DV01, metric::_ContractMetric, kr::KeyRates, discount::AYM, target::_Contractish; index::AYM = discount)
    r = _contract_keyrate(metric, kr.tenors, discount, target, index, FirstOrder())
    return _per_bp(r, r.gradient)
end

function convexity(metric::_ContractMetric, discount::AYM, target::_Contractish; index::AYM = discount)
    value, second = _value_and_second(s -> _parallel_value(metric, target, discount, index, s))
    return second / value
end
function convexity(metric::_ContractMetric, kr::KeyRates, discount::AYM, target::_Contractish; index::AYM = discount)
    r = _contract_keyrate(metric, kr.tenors, discount, target, index, SecondOrder())
    return _relative(r, r.hessian)
end
convexity(::DollarConvexity, metric::_ContractMetric, discount::AYM, target::_Contractish; index::AYM = discount) =
    _second(s -> _parallel_value(metric, target, discount, index, s))
convexity(::DollarConvexity, metric::_ContractMetric, kr::KeyRates, discount::AYM, target::_Contractish; index::AYM = discount) =
    _contract_keyrate(metric, kr.tenors, discount, target, index, SecondOrder()).hessian

"""
    sensitivities([order,] [KeyRates(tenors),] discount, contract; index = discount) -> NamedTuple
    sensitivities([order,] [KeyRates(tenors),] discount::NamedTuple, contract; index) -> NamedTuple

Calculate value and risk for a contract or a portfolio (a vector of contracts), reprojecting
cashflows under shifted curves. Coupons project on `index`, which defaults to `discount`, and every
payment is discounted on `discount`. The result has the shape of [`sensitivities`](@ref) for
curves, with the derivative fields keyed by role:

- `discount`: shift the discount curve only, so coupons stay fixed. This is spread (credit) risk,
  close to that of a fixed-rate bond with the same maturity.
- `index`: shift the index curve only, so coupons reset but are discounted as before.

Effective (rate) risk shifts both: its DV01 is `dv01.discount + dv01.index`, and its convexity is
the sum of all four blocks of `convexity` (see [`duration`](@ref) with [`Effective`](@ref)). Each
role is shifted separately even when `index` is `discount`, so the index exposure is explicit.
For a fixed bond, `index` risk is zero.

With a `NamedTuple` of discount layers, such as `(; rf, credit, ilp)`, every payment is discounted
on their sum, and each layer is its own role next to `index`, which is then required. A layer
named `index` throws an `ArgumentError`.

Durations are in years; DV01s are in dollars per basis point. Dollar risk uses signed value
derivatives and remains defined at zero value, where normalized duration is undefined.

```julia
s = sensitivities(KeyRates(tenors), curve, floater)   # s.dv01.discount, s.dv01.index
s = sensitivities(SecondOrder(), (; rf, credit), floater; index = sofr)
```
"""
sensitivities(::Union{AYM, NamedTuple}, ::_Contractish)

_contract_sensitivities(order, grid, discount::AYM, target::_Contractish; index::AYM = discount) =
    _contract_sensitivities(order, grid, (; discount), target; index)
function _contract_sensitivities(order, grid, discount::NamedTuple{layers}, target::_Contractish; index::AYM) where {layers}
    # `merge` would replace a layer named `index` with the projection curve.
    haskey(discount, :index) && throw(ArgumentError("a discount layer cannot be named :index"))
    value(c) = _value(target, c.index, reduce(+, values(NamedTuple{layers}(c))))
    return _sensitivities(_curve_ad(value, merge(discount, (; index)), grid, order), order)
end

"""
    zspread(discount, contract, market_price; index = discount, s0 = 0.0, tol = 1e-12, maxiter = 100) -> (; zspread, zspread_dv01)

Constant continuously-compounded spread `s` on the `discount` curve such that the model price
equals `market_price`, with coupons estimated on `index` (held fixed). `zspread` is `s` as a
`Continuous` rate, so `discount + result.zspread` is the spread curve;
`FinanceCore.rate(result.zspread)` is the number. `zspread_dv01` is a number,
`-∂V/∂s / 10000` at the solved spread: the value lost per basis point of `s`. The solve takes
Newton steps from `s0`, a number read as continuously compounded or a `Rate`, with ForwardDiff
derivatives.

The solve stops once a Newton step is smaller than `tol` in rate units (not currency), so the
result does not depend on the contract's notional and is defined for a zero `market_price`.
An `ErrorException` is thrown if that does not happen within `maxiter` steps.
"""
function zspread(discount::AYM, contract::FinanceCore.AbstractContract, market_price; index::AYM = discount, s0 = 0.0, tol = 1.0e-12, maxiter = 100)
    pvs(s) = _parallel_value(Spread(), contract, discount, index, s)
    f(s) = pvs(s) - market_price
    failed(step, s) = ErrorException("zspread did not converge (last Newton step = $step, residual = $(f(s)))")
    converged, s, step = _newton(f, float(FinanceCore.rate(FinanceCore.Continuous(s0))), maxiter) do s, step, _
        isfinite(step) || throw(failed(step, s))
        # A Newton step in rate units, unlike a price residual, does not scale with the notional.
        # In continuous coordinates |f′| ≤ t_max ⋅ Σ|terms|, so a small step bounds the residual
        # locally: |f|/Σ|terms| ≤ |step| ⋅ t_max at this iterate. That assumes positive discount
        # factors and finite terms; it is not a global guarantee about other roots.
        return abs(step) < tol, s - step
    end
    converged || throw(failed(step, s))
    return (; zspread = FinanceCore.Continuous(s), zspread_dv01 = -ForwardDiff.derivative(pvs, s) / 10_000)
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
