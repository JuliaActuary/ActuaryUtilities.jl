## Hull-White convenience methods
#
# Apply the shared continuous-zero shocks to `hw.curve`.

const HW = FinanceModels.ShortRate.HullWhite

# Rebuild and simulate with the bumped curve, holding model parameters fixed.
function _hw_paths(hw::HW, curve; n_scenarios, timestep, horizon, rng)
    hw_new = FinanceModels.ShortRate.HullWhite(hw.a, hw.σ, curve)
    return FinanceModels.simulate(hw_new; n_scenarios, timestep, horizon, rng)
end

# Do-block primary forms
#
# Draw one seed per call and reset Xoshiro inside the valuation. Every AD
# evaluation must use the same random draws for value and derivatives to agree.
function sensitivities(
        kr::KeyRates, valuation_fn::F, hw::HW;
        n_scenarios = 1000, timestep = 1 / 12, horizon = 30.0,
        rng = Random.default_rng()
    ) where {F}
    seed = rand(rng, UInt64)
    return sensitivities(kr, hw.curve) do curve
        valuation_fn(_hw_paths(hw, curve; n_scenarios, timestep, horizon, rng = Random.Xoshiro(seed)))
    end
end

function sensitivities(
        ::DV01, kr::KeyRates, valuation_fn::F, hw::HW;
        n_scenarios = 1000, timestep = 1 / 12, horizon = 30.0,
        rng = Random.default_rng()
    ) where {F}
    seed = rand(rng, UInt64)
    return sensitivities(DV01(), kr, hw.curve) do curve
        valuation_fn(_hw_paths(hw, curve; n_scenarios, timestep, horizon, rng = Random.Xoshiro(seed)))
    end
end

# Do-block-first forwarders (support `f(args...) do x; ...; end` syntax)
sensitivities(vf::Function, kr::KeyRates, hw::HW; kw...) = sensitivities(kr, vf, hw; kw...)
sensitivities(vf::Function, ::DV01, kr::KeyRates, hw::HW; kw...) = sensitivities(DV01(), kr, vf, hw; kw...)

# Cashflow-form wrappers that delegate to the do-block forms above
function sensitivities(
        kr::KeyRates, hw::HW, cfs::AbstractVector, times;
        n_scenarios = 1000, timestep = 1 / 12, horizon = nothing,
        rng = Random.default_rng()
    )
    times = _cashflow_times(cfs, times)
    _iszero_cashflow_stream(cfs) && return sensitivities(kr, hw.curve, cfs, times)
    h = horizon === nothing ? _maximum_cashflow_time(cfs, times) + 1.0 : Float64(horizon)
    return sensitivities(kr, hw; n_scenarios, timestep, horizon = h, rng) do scenarios
        sum(FinanceCore.pv(sc, cfs, times) for sc in scenarios) / n_scenarios
    end
end

function sensitivities(
        ::DV01, kr::KeyRates, hw::HW, cfs::AbstractVector, times;
        n_scenarios = 1000, timestep = 1 / 12, horizon = nothing,
        rng = Random.default_rng()
    )
    times = _cashflow_times(cfs, times)
    _iszero_cashflow_stream(cfs) && return sensitivities(DV01(), kr, hw.curve, cfs, times)
    h = horizon === nothing ? _maximum_cashflow_time(cfs, times) + 1.0 : Float64(horizon)
    return sensitivities(DV01(), kr, hw; n_scenarios, timestep, horizon = h, rng) do scenarios
        sum(FinanceCore.pv(sc, cfs, times) for sc in scenarios) / n_scenarios
    end
end
