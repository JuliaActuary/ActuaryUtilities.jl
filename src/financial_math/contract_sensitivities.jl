## Contract and portfolio sensitivities
# Value contracts under bumped curves in a FinanceModels valuation context, so
# ActuaryUtilities keeps no list of contract types.

const _Contractish = Union{FinanceCore.AbstractContract, AbstractVector{<:FinanceCore.AbstractContract}}
const _ContractMetric = Union{Effective, Spread}

# `discount` discounts and prices; `index` serves every model a contract reads by key, so floating
# coupons reset on the bumped index curve. A closed-form contract uses its own
# `present_value(ctx, c)`; composites and portfolios are valued as the sums of their parts.
_value(target, index, discount) = FinanceCore.present_value(FinanceModels.Models(discount; index), target)

# A valuation context: its model discounts and prices, and its store holds the models contracts read
# by key (index curves, credit and prepayment assumptions). AU's two-curve forms are the context
# `Models(discount; index)`, whose store serves `index` under every key.
const _Context = FinanceModels.Models
_context(discount, index) = FinanceModels.Models(discount; index)

# The value under one shock, `shock(curve)`, to the curves the metric moves. Effective moves the
# model and every yield model in the store, so floating coupons reset; Spread moves the model only.
# Other store entries, such as credit and prepayment assumptions, never move.
_context_value(::Effective, target, ctx, shock::S) where {S} =
    FinanceCore.present_value(FinanceModels.Models(shock(ctx.model), _shift_store(ctx.store, shock)), target)
_context_value(::Spread, target, ctx, shock::S) where {S} =
    FinanceCore.present_value(FinanceModels.Models(shock(ctx.model), ctx.store), target)
_shift_store(store::FinanceModels.__EveryKey, shock::S) where {S} = FinanceModels.__EveryKey(shock(store.index))
_shift_store(store::NamedTuple, shock::S) where {S} = map(v -> _shift_entry(v, shock), store)
_shift_store(store::AbstractDict, shock::S) where {S} = Dict(k => _shift_entry(v, shock) for (k, v) in store)
_shift_entry(v::AYM, shock::S) where {S} = shock(v)
_shift_entry(v, shock) = v

_contract_value(metric, target, discount, index, shock::S) where {S} = _context_value(metric, target, _context(discount, index), shock)
_parallel_value(metric, target, ctx, s) = _context_value(metric, target, ctx, c -> _parallel_bumped(c, s))
_parallel_value(metric, target, discount, index, s) = _parallel_value(metric, target, _context(discount, index), s)
# Derivatives on the key-rate grid of one shock vector applied to the curves the metric moves.
function _contract_keyrate(metric, grid, ctx, target, order)
    _validate_tenors(grid)
    v(b) = _context_value(metric, target, ctx, c -> _bumped(c, grid, b.shift))
    return _only_role(_named_ad(v, (; shift = zeros(length(grid))), order))
end
_contract_keyrate(metric, grid, discount, target, index, order) = _contract_keyrate(metric, grid, _context(discount, index), target, order)

# The value of the parallel shock `f(s)` and its first and second derivatives at s = 0: by ForwardDiff,
# or, with a `bump`, by central differences, revaluing at ±bump. Revaluation measures what the
# derivatives along a valuation's decisions leave out, such as an exercise boundary that moves with
# the curve (FinanceModels throws on second derivatives through exercise decisions).
_parallel_first(f::F, ::Nothing) where {F} = _value_and_derivative(f, 0.0)
_parallel_second(f::F, ::Nothing) where {F} = _value_and_second(f)
function _parallel_first(f::F, bump::Real) where {F}
    h = _check_bump(bump)
    return f(0.0), (f(h) - f(-h)) / 2h
end
function _parallel_second(f::F, bump::Real) where {F}
    h = _check_bump(bump)
    v = f(0.0)
    return v, (f(h) - 2v + f(-h)) / h^2
end
function _check_bump(bump)
    (isfinite(bump) && bump > 0) || throw(ArgumentError("a bump is a finite positive shift; got $bump"))
    return float(bump)
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

**Valuation contexts.** Every form also takes a FinanceModels `Models` context in place of
`discount` (and without `index`): `duration(Effective(), ctx, contract)`. Its model discounts and
prices, and its store holds what contracts read by key. `Effective()` shifts the model and every
yield model in the store, such as index curves; `Spread()` shifts the model only. Other entries,
such as `Credit.Assumptions` and prepayment assumptions, are never shifted, so a defaultable bond's
hazard and recovery stay fixed. A `ShortRate.HullWhiteTree` model shifts the curve its lattice is
calibrated to, so callable and putable assets are revalued on the recalibrated lattice.

**Revaluation.** The parallel forms take `bump`, a shift in continuous-zero units such as
`1e-4`: the duration and DV01 are then central differences and the convexity a second difference
of values at `±bump`, instead of ForwardDiff derivatives. FinanceModels throws on second derivatives
through an exercise decision, because they leave out the moving exercise boundary; the effective
convexity of a callable or putable asset is `convexity(ctx, asset; bump)`.

```julia
ctx = Models(ShortRate.HullWhiteTree(hw), Dict("ACME" => Credit.Assumptions(hazard; recovery = 0.4)))
callable = Credit.Defaultable(Bond.Callable(Bond.Fixed(0.05, Periodic(2), 10), 3.0:0.5:9.5, 1.0), "ACME")
duration(ctx, callable)                    # effective duration, along the lattice's decisions
convexity(ctx, callable; bump = 1.0e-3)    # effective convexity by revaluation
```
"""
# The contract measures on a context; AU's two-curve forms build `Models(discount; index)`.
function duration(metric::_ContractMetric, ctx::_Context, target::_Contractish; bump = nothing)
    value, derivative = _parallel_first(s -> _parallel_value(metric, target, ctx, s), bump)
    return -derivative / value
end
function duration(metric::_ContractMetric, kr::KeyRates, ctx::_Context, target::_Contractish)
    r = _contract_keyrate(metric, kr.tenors, ctx, target, FirstOrder())
    return _relative(r, r.gradient; negate = true)
end
duration(::DV01, metric::_ContractMetric, ctx::_Context, target::_Contractish; bump = nothing) =
    -last(_parallel_first(s -> _parallel_value(metric, target, ctx, s), bump)) / 10_000
function duration(::DV01, metric::_ContractMetric, kr::KeyRates, ctx::_Context, target::_Contractish)
    r = _contract_keyrate(metric, kr.tenors, ctx, target, FirstOrder())
    return _per_bp(r, r.gradient)
end
function convexity(metric::_ContractMetric, ctx::_Context, target::_Contractish; bump = nothing)
    value, second = _parallel_second(s -> _parallel_value(metric, target, ctx, s), bump)
    return second / value
end
function convexity(metric::_ContractMetric, kr::KeyRates, ctx::_Context, target::_Contractish)
    r = _contract_keyrate(metric, kr.tenors, ctx, target, SecondOrder())
    return _relative(r, r.hessian)
end
convexity(::DollarConvexity, metric::_ContractMetric, ctx::_Context, target::_Contractish; bump = nothing) =
    last(_parallel_second(s -> _parallel_value(metric, target, ctx, s), bump))
convexity(::DollarConvexity, metric::_ContractMetric, kr::KeyRates, ctx::_Context, target::_Contractish) =
    _contract_keyrate(metric, kr.tenors, ctx, target, SecondOrder()).hessian

# Without a metric, contracts take Effective().
duration(ctx::_Context, target::_Contractish; kwargs...) = duration(Effective(), ctx, target; kwargs...)
duration(kr::KeyRates, ctx::_Context, target::_Contractish) = duration(Effective(), kr, ctx, target)
duration(::DV01, ctx::_Context, target::_Contractish; kwargs...) = duration(DV01(), Effective(), ctx, target; kwargs...)
duration(::DV01, kr::KeyRates, ctx::_Context, target::_Contractish) = duration(DV01(), Effective(), kr, ctx, target)
convexity(ctx::_Context, target::_Contractish; kwargs...) = convexity(Effective(), ctx, target; kwargs...)
convexity(kr::KeyRates, ctx::_Context, target::_Contractish) = convexity(Effective(), kr, ctx, target)
convexity(::DollarConvexity, ctx::_Context, target::_Contractish; kwargs...) = convexity(DollarConvexity(), Effective(), ctx, target; kwargs...)
convexity(::DollarConvexity, kr::KeyRates, ctx::_Context, target::_Contractish) = convexity(DollarConvexity(), Effective(), kr, ctx, target)

duration(metric::_ContractMetric, discount::AYM, target::_Contractish; index::AYM = discount, bump = nothing) =
    duration(metric, _context(discount, index), target; bump)
duration(metric::_ContractMetric, kr::KeyRates, discount::AYM, target::_Contractish; index::AYM = discount) =
    duration(metric, kr, _context(discount, index), target)
duration(::DV01, metric::_ContractMetric, discount::AYM, target::_Contractish; index::AYM = discount, bump = nothing) =
    duration(DV01(), metric, _context(discount, index), target; bump)
duration(::DV01, metric::_ContractMetric, kr::KeyRates, discount::AYM, target::_Contractish; index::AYM = discount) =
    duration(DV01(), metric, kr, _context(discount, index), target)
convexity(metric::_ContractMetric, discount::AYM, target::_Contractish; index::AYM = discount, bump = nothing) =
    convexity(metric, _context(discount, index), target; bump)
convexity(metric::_ContractMetric, kr::KeyRates, discount::AYM, target::_Contractish; index::AYM = discount) =
    convexity(metric, kr, _context(discount, index), target)
convexity(::DollarConvexity, metric::_ContractMetric, discount::AYM, target::_Contractish; index::AYM = discount, bump = nothing) =
    convexity(DollarConvexity(), metric, _context(discount, index), target; bump)
convexity(::DollarConvexity, metric::_ContractMetric, kr::KeyRates, discount::AYM, target::_Contractish; index::AYM = discount) =
    convexity(DollarConvexity(), metric, kr, _context(discount, index), target)

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

    zspread(ctx::Models, contract, market_price; s0 = 0.0, tol = 1e-12, maxiter = 100)

The spread on the model of a FinanceModels valuation context; the store's entries, such as index
curves and credit assumptions, stay fixed. With a `ShortRate.HullWhiteTree` model it is the
option-adjusted spread of a callable or putable asset: the spread moves the curve the lattice is
calibrated to, which for Hull–White is the same constant spread on every node's rate, and each
trial spread redoes the exercise decisions. For a `Credit.Defaultable` asset it is the residual
spread beyond the hazard and recovery assumptions, which stay fixed.

```julia
oas = zspread(Models(ShortRate.HullWhiteTree(hw), store), callable, 0.97).zspread
```
"""
zspread(discount::AYM, contract::FinanceCore.AbstractContract, market_price; index::AYM = discount, kwargs...) =
    zspread(_context(discount, index), contract, market_price; kwargs...)
function zspread(ctx::_Context, contract::FinanceCore.AbstractContract, market_price; s0 = 0.0, tol = 1.0e-12, maxiter = 100)
    pvs(s) = _parallel_value(Spread(), contract, ctx, s)
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
