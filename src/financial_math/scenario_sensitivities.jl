## Hull–White scenario sensitivities
#
# A bare `HullWhite` is a curve everywhere: a bump moves `hw.curve` and keeps the model
# (curve_shifts.jl). `Scenarios` asks for simulation: the shared continuous-zero shocks
# move `hw.curve`, and the valuation sees the paths simulated from each shocked model.

"""
    Scenarios(hw::HullWhite; n_scenarios = 1000, timestep = 1/12, horizon = nothing, rng = Random.default_rng())

Simulated Hull–White short-rate paths for [`sensitivities`](@ref). Key-rate shocks move
`hw.curve`, the model recalibrates its drift to each shocked curve, and the valuation sees the
paths simulated from it; mean reversion and volatility stay fixed. Without `Scenarios`, `hw`
is a curve like any other: a bump moves `hw.curve`, and caps and swaptions keep their closed forms.

```julia
s = Scenarios(hw; n_scenarios = 500, rng = Xoshiro(42))
sensitivities(KeyRates(tenors), s, cfs, times)   # mean present value across paths
sensitivities(KeyRates(tenors), s) do paths      # a valuation of the vector of paths
    sum(pv(p, cfs, times) for p in paths) / length(paths)
end
```

`horizon = nothing` simulates to 30 years for a valuation callback, and to one year past the
last payment for fixed cashflows. Each call draws one seed from `rng` and reuses it in every
automatic-differentiation evaluation, so value and derivatives share the same random draws.
Empty and all-zero cashflow streams are valued on `hw.curve` without simulating and leave `rng`
untouched. The `DV01()` forms return dollar risk, as for a curve.
"""
struct Scenarios{M <: HW, N, T, H, R}
    model::M
    n_scenarios::N
    timestep::T
    horizon::H
    rng::R
end
Scenarios(model::HW; n_scenarios = 1000, timestep = 1 / 12, horizon = nothing, rng = Random.default_rng()) =
    Scenarios(model, n_scenarios, timestep, horizon, rng)

# Draw one seed per call; every AD evaluation rebuilds `Xoshiro(seed)`, so value and derivatives
# share the same draws. The engine bumps `s.model` itself: `_bumped` moves a Hull–White curve.
function _simulated(valuation_fn::F, s::Scenarios, horizon = something(s.horizon, 30.0)) where {F}
    seed = rand(s.rng, UInt64)
    return model -> valuation_fn(FinanceModels.simulate(model; s.n_scenarios, s.timestep, horizon, rng = Random.Xoshiro(seed)))
end

sensitivities(valuation_fn::F, kr::KeyRates, s::Scenarios) where {F} = sensitivities(_simulated(valuation_fn, s), kr, s.model)
sensitivities(valuation_fn::F, ::DV01, kr::KeyRates, s::Scenarios) where {F} = sensitivities(_simulated(valuation_fn, s), DV01(), kr, s.model)

# Zero streams are valued on `s.model` without simulating: its discount factors are its curve's.
function sensitivities(kr::KeyRates, s::Scenarios, cfs::AbstractVector, times = eachindex(cfs))
    _check_cashflow_times(cfs, times)
    _iszero_cashflow_stream(cfs) && return sensitivities(kr, s.model, cfs, times)
    return sensitivities(_simulated_pv(s, cfs, times), kr, s.model)
end
function sensitivities(::DV01, kr::KeyRates, s::Scenarios, cfs::AbstractVector, times = eachindex(cfs))
    _check_cashflow_times(cfs, times)
    _iszero_cashflow_stream(cfs) && return sensitivities(DV01(), kr, s.model, cfs, times)
    return sensitivities(_simulated_pv(s, cfs, times), DV01(), kr, s.model)
end

# The mean present value of fixed cashflows across the paths, simulated to one year past the
# last payment unless `s` sets a horizon.
function _simulated_pv(s::Scenarios, cfs, times)
    horizon = something(s.horizon, _maximum_cashflow_time(cfs, times) + 1.0)
    return _simulated(s, horizon) do paths
        sum(FinanceCore.pv(p, cfs, times) for p in paths) / s.n_scenarios
    end
end
