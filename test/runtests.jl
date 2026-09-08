using ActuaryUtilities

using Dates
using Test
using Distributions
using StatsBase
using Random
import ForwardDiff
import QuadGK

const FM = ActuaryUtilities.FinanceModels
const FC = ActuaryUtilities.FinanceCore


include("risk_measures.jl")
include("optimal_transport.jl")
include("audit.jl")

@testset "Temporal functions" begin
    @testset "years_between" begin
        @test years_between(Date(2018, 9, 30), Date(2018, 9, 30)) == 0
        @test years_between(Date(2022, 10, 30), Date(2022, 10, 30)) == 0
        @test years_between(Date(2021, 10, 30), Date(2021, 10, 30)) == 0
        @test years_between(Date(2021, 11, 30), Date(2021, 10, 30)) == -1
        @test years_between(Date(2022, 10, 30), Date(2021, 10, 30)) == -1
        @test years_between(Date(2018, 9, 30), Date(2018, 9, 30), true) == 0
        @test years_between(Date(2018, 9, 30), Date(2019, 9, 30), false) == 0
        @test years_between(Date(2018, 9, 30), Date(2019, 9, 30), true) == 1
        @test years_between(Date(2018, 9, 30), Date(2019, 10, 1), true) == 1
        @test years_between(Date(2018, 9, 30), Date(2019, 10, 1), false) == 1
    end

    @testset "duration tests" begin
        @test duration(Date(2018, 9, 30), Date(2019, 9, 30)) == 2
        @test duration(Date(2018, 9, 30), Date(2018, 9, 30)) == 1
        @test duration(Date(2018, 9, 30), Date(2018, 10, 1)) == 1
        @test duration(Date(2018, 9, 30), Date(2019, 10, 1)) == 2
        @test duration(Date(2018, 9, 30), Date(2018, 6, 30)) == 0
        @test duration(Date(2018, 9, 30), Date(2017, 6, 30)) == -1
        @test duration(Date(2018, 10, 15), Date(2019, 9, 30)) == 1
        @test duration(Date(2018, 10, 15), Date(2019, 10, 30)) == 2
        @test duration(Date(2018, 10, 15), Date(2019, 10, 15)) == 2
        @test duration(Date(2018, 10, 15), Date(2019, 10, 14)) == 1
    end
end


@testset "accum_offset" begin
    @test all(accum_offset([0.9, 0.8, 0.7]) .== [1.0, 0.9, 1.0 * 0.9 * 0.8])
    @test all(accum_offset([0.9, 0.8, 0.7], op = +) .== [1.0, 1.9, 2.7])
    @test all(accum_offset([0.9, 0.8, 0.7], op = +, init = 2) .== [2.0, 2.9, 3.7])

    @test all(accum_offset(1:5, op = +) .== [1, 2, 4, 7, 11])
    @test all(accum_offset(1:5) .== [1, 1, 2, 6, 24])
    @test all(accum_offset([1, 2, 3]) .== [1, 1, 2])
end

@testset "financial calcs" begin

    @testset "price and present_value" begin
        cf = [100, 100]

        @test price(0.05, cf) ≈ pv(0.05, cf)


        cfs = ones(3)
        @test present_values(FM.Yield.Constant(0.0), cfs) == [3, 2, 1]
        pvs = present_values(FM.Yield.Constant(0.1), cfs)
        @test pvs[3] ≈ 1 / 1.1
        @test pvs[2] ≈ (1 / 1.1 + 1) / 1.1


        @test all(present_values(0.0, [1, 1, 1]) .≈ [3, 2, 1])
        @test all(present_values(0.0, [1, 1, 1], [0, 1, 2]) .≈ [3, 2, 1])
        @test all(present_values(0.0, [1, 1, 1], [1, 2, 3]) .≈ [3, 2, 1])
        @test all(present_values(0.0, [1, 1, 1], [1, 2, 3]) .≈ [3, 2, 1])
        @test all(present_values(0.01, [1, 2, 3]) .≈ [5.862461552497766, 4.921086168022744, 2.9702970297029707])

        cf = [100, 100]

        ts = [0.5, 1]

        @test pv(0.05, cf, ts) ≈ 100 / 1.05^0.5 + 100 / 1.05^1

        @test price(0.05, cf, ts) ≈ pv(0.05, cf, ts)
        @test price(0.05, -1 .* cf, ts) ≈ abs(pv(0.05, cf, ts))

        @test pv(0.05, FC.Cashflow.(cf, ts)) ≈ pv(0.05, cf, ts)
        @test price(0.05, FC.Cashflow.(cf, ts)) ≈ price(0.05, cf, ts)


    end


end

@testset "Breakeven time" begin

    @testset "basic" begin
        @test breakeven(0.1, [-10, 1, 2, 3, 4, 8]) == 5
        @test breakeven(0.1, [-10, 15, 2, 3, 4, 8]) == 1
        @test breakeven(0.1, [-10, 15, 2, 3, 4, 8]) == 1
        @test breakeven(0.1, [10, 15, 2, 3, 4, 8]) == 0
        @test isnothing(breakeven(0.1, [-10, -15, 2, 3, 4, 8]))
        @test breakeven(0.1, FC.Cashflow.([-10, 1, 2, 3, 4, 8], 0:5)) == 5
    end

    @testset "timepoints" begin
        times = [t for t in 0:5]
        @test breakeven(0.1, [-10, 1, 2, 3, 4, 8], times) == 5
        @test breakeven(0.1, [-10, 15, 2, 3, 4, 8], times) == 1
        @test breakeven(0.1, [-10, 15, 2, 3, 4, 8], times) == 1
        @test isnothing(breakeven(0.1, [-10, -15, 2, 3, 4, 8], times))
    end
end

@testset "moic" begin

    # https://bankingprep.com/multiple-on-invested-capital/
    ex1 = [-100; [t == 200 ? 100 * 1.067^t : 0 for t in 1:200]]
    @test moic(ex1) ≈ 429421.59914697794
    @test moic(FC.Cashflow.(ex1, 0:200)) ≈ 429421.59914697794


    ex2 = ex1[end] *= 0.5
    @test moic(ex1) ≈ 429421.59914697794 * 0.5


end

@testset "spread" begin
    cfs = fill(10, 10)
    cfo = FC.Cashflow.(cfs, 1:10)

    @test spread(0.04, 0.05, cfs) ≈ FC.Periodic(0.01, 1) atol = 1.0e-6
    @test spread(0.04, 0.05, cfo) ≈ FC.Periodic(0.01, 1) atol = 1.0e-6

    @test spread(FC.Continuous(0.04), FC.Continuous(0.05), cfs) ≈ FC.Periodic(1)(FC.Continuous(0.05) - FC.Continuous(0.04)) atol = 1.0e-6

    # 2021-03-31 rates from Treasury.gov
    rates = [0.01, 0.01, 0.03, 0.05, 0.07, 0.16, 0.35, 0.92, 1.4, 1.74, 2.31, 2.41] ./ 100
    mats = [1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30]

    y = FM.fit(FM.Spline.Linear(), FM.CMTYield.(rates, mats), FM.Fit.Bootstrap())

    y2 = y + FC.Periodic(0.01, 1)

    s = spread(y, y2, cfs)

    @test s ≈ FC.Periodic(0.01, 1) atol = 1.0e-6
end

include("duration.jl")
include("key_rate_durations.jl")
include("sensitivities.jl")
include("analytic_types.jl")
include("zero_cashflows.jl")
include("stochastic_sensitivities.jl")
include("contract_sensitivities.jl")

using Aqua
@testset "Aqua.jl" begin
    Aqua.test_all(
        ActuaryUtilities;
        # The persistent_tasks probe spawns a subprocess that precompiles the
        # package; with a heavy dep tree (FinanceModels, Distributions) it
        # flakily fails within the CI runner's limits ("done.log was not
        # created"). ActuaryUtilities spawns no background tasks, so the check
        # is disabled rather than left flaky — same rationale as FinanceModels.
        persistent_tasks = false,
    )
end
