## Hull–White scenario sensitivities
#
# A bare `HullWhite` is a curve everywhere: a bump moves `hw.curve` and keeps the model
# (curve_shifts.jl). `Scenarios` asks for simulation: the shared continuous-zero shocks
# move `hw.curve`, and the valuation sees the paths simulated from each shocked model.

"""
    Scenarios(hw::HullWhite; horizon, n_scenarios = 1000, timestep = 1/12, rng = Random.default_rng())

Simulated Hull–White short-rate paths for [`sensitivities`](@ref). Curve shocks move
`hw.curve`, the model recalibrates its drift to each shocked curve, and the valuation sees the
paths simulated from it; mean reversion and volatility stay fixed. Without `Scenarios`, `hw`
is a curve like any other: a bump moves `hw.curve`, and caps and swaptions keep their closed forms.

```julia
s = Scenarios(hw; horizon = 10.0, n_scenarios = 500, rng = Xoshiro(42))
sensitivities(KeyRates(tenors), s, cfs, times)   # mean present value across paths
sensitivities(KeyRates(tenors), s) do paths      # a valuation of the vector of paths
    sum(pv(p, cfs, times) for p in paths) / length(paths)
end
```

The forms and results are those of a curve: an optional [`FirstOrder`](@ref) or
[`SecondOrder`](@ref) marker, then an optional [`KeyRates`](@ref) grid.

`horizon` is required. It must be a whole number of `timestep`s, up to floating-point rounding
(`FinanceModels.simulation_steps` decides), or the constructor throws `ArgumentError`; the paths
end there, and a payment after `horizon` throws. The constructor fixes the time grid and draws
one seed from `rng`, and nothing else uses `rng`. Every valuation with the same `Scenarios`
simulates the same paths from that seed, in every automatic-differentiation evaluation too, so
value and derivatives share the same draws, and results add across calls: values and dollar
derivatives add, and normalized duration and convexity add when weighted by value. Empty and
all-zero cashflow streams are valued on `hw.curve` without simulating.
"""
struct Scenarios{M <: HW, T, H}
    model::M
    n_scenarios::Int
    timestep::T
    horizon::H
    nsteps::Int
    seed::UInt64
    function Scenarios(model::HW; horizon, n_scenarios = 1000, timestep = 1 / 12, rng = Random.default_rng())
        # Validate the grid before drawing the seed, with the function `simulate` counts steps with.
        (; nsteps, aligned) = FinanceModels.simulation_steps(horizon, timestep)
        aligned || throw(ArgumentError("horizon $horizon is not a whole number of timesteps $timestep"))
        return new{typeof(model), typeof(timestep), typeof(horizon)}(model, n_scenarios, timestep, horizon, nsteps, rand(rng, UInt64))
    end
end

# Every AD evaluation rebuilds `Xoshiro(s.seed)`, so value and derivatives share the same draws. The
# engine bumps `s.model` itself: `_bumped` moves a Hull–White curve.
_simulated(valuation::F, s::Scenarios) where {F} =
    model -> valuation(FinanceModels.simulate(model; s.n_scenarios, s.timestep, s.horizon, rng = Random.Xoshiro(s.seed)))

sensitivities(valuation::F, s::Scenarios) where {F} = sensitivities(valuation, FirstOrder(), s)
sensitivities(valuation::F, kr::KeyRates, s::Scenarios) where {F} = sensitivities(valuation, FirstOrder(), kr, s)
sensitivities(valuation::F, order::_Order, s::Scenarios) where {F} = sensitivities(_simulated(valuation, s), order, s.model)
sensitivities(valuation::F, order::_Order, kr::KeyRates, s::Scenarios) where {F} =
    sensitivities(_simulated(valuation, s), order, kr, s.model)

# Fixed cashflows. Zero streams are valued on `s.model` without simulating: its discount factors are
# its curve's.
function _sensitivities_of(order, grid, s::Scenarios, cfs::_CashflowCollection, times...)
    amounts, ts = _cashflow_inputs(cfs, times...)
    _iszero_cashflow_stream(amounts) && return _fixed_sensitivities(order, grid, s.model, amounts, ts)
    return _sensitivities(_one_curve_ad(_simulated_pv(s, amounts, ts), s.model, grid, order), order)
end

# The mean present value of fixed cashflows across the paths. The inputs are those of
# `_cashflow_inputs`, so their lengths are equal, as FinanceCore requires.
_simulated_pv(s::Scenarios, cfs, times) = _simulated(s) do paths
    sum(FinanceCore.pv(p, cfs, times) for p in paths) / s.n_scenarios
end
