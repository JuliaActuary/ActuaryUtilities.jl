using ActuaryUtilities, BenchmarkTools, Random, Serialization

const FM = ActuaryUtilities.FinanceModels
const FC = ActuaryUtilities.FinanceCore

function cases()
    base = FM.Yield.Spline(FM.Spline.Linear(), [0.5, 2.0, 5.0, 10.0, 30.0], [0.02, 0.025, 0.03, 0.035, 0.04])
    credit = FM.Yield.Constant(FC.Continuous(0.01))
    times = collect(0.5:0.5:30.0)
    cashflows = fill(2.0, length(times))
    cashflows[end] += 100
    value(c) = FC.pv(c, cashflows, times)
    value2(b, c) = value(b + c)
    fixed = FM.Bond.Fixed(0.04, FC.Periodic(2), 30.0)
    floating = FM.Bond.Floating(0.005, FC.Periodic(2), 30.0, :index)
    portfolio = [fixed, floating]
    suite = Pair{String, Function}[]
    for n in (3, 12, 30)
        tenors = collect(range(0.5, 30.0; length = n))
        kr = KeyRates(tenors)
        push!(
            suite,
            "analytic_duration_$n" => () -> duration(kr, base, cashflows, times),
            "analytic_bundle_$n" => () -> sensitivities(kr, base, cashflows, times),
            "callback_duration_$n" => () -> duration(kr, value, base),
            "callback_bundle_$n" => () -> sensitivities(kr, value, base),
            "two_curve_bundle_$n" => () -> sensitivities(kr, value2, base, credit),
            "named_three_curve_$n" => () -> sensitivities(c -> value(c.base + c.credit + c.liquidity), (; base, credit, liquidity = credit); tenors),
            "fixed_contract_$n" => () -> sensitivities(fixed, base, tenors),
            "floating_contract_$n" => () -> sensitivities(floating, base, base + credit, tenors),
            "portfolio_$n" => () -> sensitivities(portfolio, base, base + credit, tenors),
            "effective_duration_$n" => () -> duration(Effective(), floating, base, base + credit, tenors),
            "spread_dv01_$n" => () -> dv01(Spread(), floating, base, base + credit, tenors),
        )
    end
    hw_kr = KeyRates([1.0, 3.0, 5.0])
    hw = FM.ShortRate.HullWhite(0.1, 0.01, base)
    push!(suite, "hull_white_bundle" => () -> sensitivities(hw_kr, hw, [5.0, 105.0], [1.0, 5.0]; n_scenarios = 50, timestep = 0.5, horizon = 5.0, rng = Random.Xoshiro(1234)))
    return suite
end

function main(output; seconds = parse(Float64, get(ENV, "AU_BENCH_SECONDS", "0.3")))
    suite = cases()
    pattern = get(ENV, "AU_BENCH_FILTER", "")
    filter!(case -> occursin(pattern, first(case)), suite)
    results = Dict{String, Any}()
    # Warm every case before measuring, so compilation of a later case does not
    # contaminate measurements of an earlier case. Run revisions sequentially.
    for (name, f) in suite
        results[name] = f()
        if startswith(name, "analytic_duration_") || startswith(name, "callback_duration_")
            @assert length(results[name]) == parse(Int, last(split(name, '_')))
        end
        f()
    end
    serialize(output * ".values", results)
    return open(output, "w") do io
        println(io, "case\tmedian_ns\tminimum_ns\tmemory_bytes\tallocations\tsamples")
        for (name, f) in suite
            trial = @benchmark $f() seconds = seconds samples = 10000 evals = 1
            med = median(trial)
            println(io, join((name, med.time, minimum(trial).time, med.memory, med.allocs, length(trial)), '\t'))
            flush(io)
        end
    end
end

main(only(ARGS))
