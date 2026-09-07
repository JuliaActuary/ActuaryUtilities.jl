## Hull-White convenience methods
#
# `hw.curve` can be any `AbstractYieldModel` — the AD path uses TenorShift
# bumps over the curve via the AYM-based `sensitivities` impls above.

const HW = FinanceModels.ShortRate.HullWhite

# Rebuild HW under a perturbed curve and produce its scenario set under the same dynamics.
function _hw_paths(hw::HW, curve; n_scenarios, timestep, horizon, rng)
    hw_new = FinanceModels.ShortRate.HullWhite(hw.a, hw.σ, curve)
    return FinanceModels.simulate(hw_new; n_scenarios, timestep, horizon, rng)
end

# Do-block primary forms
#
# Pathwise seeding: every AD evaluation of the inner closure must see the same
# MC sample, otherwise `KRD = -∇V/V` divides a gradient computed over one
# sample by a value computed over another (ForwardDiff calls the closure many
# times for value, gradient chunks, and Hessian chunks). Snapshot a UInt64 from
# the user's rng once per call and rebuild a fresh `Xoshiro(seed)` inside the
# closure so every AD step draws the same scenarios.
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
    h = horizon === nothing ? maximum(times) + 1.0 : Float64(horizon)
    return sensitivities(kr, hw; n_scenarios, timestep, horizon = h, rng) do scenarios
        sum(FinanceCore.pv(sc, cfs, times) for sc in scenarios) / n_scenarios
    end
end

function sensitivities(
        ::DV01, kr::KeyRates, hw::HW, cfs::AbstractVector, times;
        n_scenarios = 1000, timestep = 1 / 12, horizon = nothing,
        rng = Random.default_rng()
    )
    h = horizon === nothing ? maximum(times) + 1.0 : Float64(horizon)
    return sensitivities(DV01(), kr, hw; n_scenarios, timestep, horizon = h, rng) do scenarios
        sum(FinanceCore.pv(sc, cfs, times) for sc in scenarios) / n_scenarios
    end
end
