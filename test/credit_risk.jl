# Sensitivities of defaultable and exercisable contracts in FinanceModels valuation contexts.

@testset "valuation contexts, credit and optionality" begin
    curve = FM.ZeroRateCurve([0.03, 0.035, 0.04, 0.042], [1.0, 3.0, 7.0, 12.0])
    sofr = FM.ZeroRateCurve([0.032, 0.036, 0.041, 0.043], [1.0, 3.0, 7.0, 12.0])
    bond = FM.Bond.Fixed(0.05, FC.Periodic(2), 10)
    acme = FM.Credit.Assumptions(FM.Credit.ConstantHazard(0.02); recovery = 0.4)

    @testset "contexts agree with the two-curve forms" begin
        floater = FM.Bond.Floating(0.01, FC.Periodic(4), 5.0, "SOFR")
        two = FM.Models(curve; index = sofr)
        for metric in (Effective(), Spread())
            @test duration(metric, two, floater) == duration(metric, curve, floater; index = sofr)
            @test convexity(metric, two, floater) == convexity(metric, curve, floater; index = sofr)
            @test duration(DV01(), metric, KeyRates([1.0, 5.0]), two, floater) == duration(DV01(), metric, KeyRates([1.0, 5.0]), curve, floater; index = sofr)
        end
        # a keyed store: Effective moves the index curve under its key, Spread does not
        store = FM.Models(curve, Dict("SOFR" => sofr, "ACME" => acme))
        @test duration(Effective(), store, floater) ≈ duration(Effective(), curve, floater; index = sofr) rtol = 1.0e-12
        @test duration(Spread(), store, floater) ≈ duration(Spread(), curve, floater; index = sofr) rtol = 1.0e-12
        @test duration(store, floater) == duration(Effective(), store, floater)
        @test convexity(DollarConvexity(), store, floater) == convexity(DollarConvexity(), Effective(), store, floater)
    end

    @testset "defaultable bonds keep their credit assumptions" begin
        risky = FM.Credit.Defaultable(bond, "ACME")
        ctx = FM.Models(curve, Dict("ACME" => acme))
        # the parallel shift moves the discount curve, not the hazard
        v(s) = FC.present_value(FM.Models(curve + FC.Continuous(s), Dict("ACME" => acme)), risky)
        @test duration(ctx, risky) ≈ -ForwardDiff.derivative(v, 0.0) / v(0.0) rtol = 1.0e-10
        @test convexity(ctx, risky) ≈ ForwardDiff.derivative(s -> ForwardDiff.derivative(v, s), 0.0) / v(0.0) rtol = 1.0e-10
        @test sum(duration(KeyRates([1.0, 3.0, 7.0, 12.0]), ctx, risky)) ≈ duration(ctx, risky) rtol = 1.0e-10
        # a z-spread beyond the hazard and recovery
        price = v(0.005)
        @test FC.rate(zspread(ctx, risky, price).zspread) ≈ 0.005 atol = 1.0e-10
        # hazard and recovery sensitivities through named inputs
        r = sensitivities((; hazard = [0.02], recovery = [0.4])) do m
            FC.present_value(FM.Models(curve, Dict("ACME" => FM.Credit.Assumptions(FM.Credit.ConstantHazard(m.hazard[1]); recovery = m.recovery[1]))), risky)
        end
        @test r.dv01.hazard[1] > 0 && r.dv01.recovery[1] < 0
    end

    @testset "callable bonds on a Hull–White lattice" begin
        hw = FM.ShortRate.HullWhite(0.1, 0.01, curve)
        tree = FM.ShortRate.HullWhiteTree(hw; max_step = 1 / 48)
        callable = FM.Bond.Callable(bond, 3.0:0.5:9.5, 1.0)
        ctx = FM.Models(tree)
        # the shift moves the curve the lattice is calibrated to
        v(s) = FC.present_value(FM.ShortRate.HullWhiteTree(FM.ShortRate.HullWhite(0.1, 0.01, curve + FC.Continuous(s)); max_step = 1 / 48), callable)
        @test duration(ctx, callable) ≈ -ForwardDiff.derivative(v, 0.0) / v(0.0) rtol = 1.0e-10
        @test duration(tree, callable) == duration(ctx, callable)
        # second derivatives through exercise throw in FinanceModels; revaluation measures them
        @test_throws "second derivatives through an exercise decision" convexity(ctx, callable)
        h = 1.0e-3
        # (AU shifts the curve with a TenorShift, the reference with curve arithmetic: equal to rounding,
        # which a second difference at h = 1e-3 amplifies to about 1e-11)
        @test convexity(ctx, callable; bump = h) ≈ (v(h) - 2v(0.0) + v(-h)) / (h^2 * v(0.0)) rtol = 1.0e-8
        @test convexity(ctx, callable; bump = h) < 0 < convexity(FM.Models(curve), bond; bump = h)
        @test duration(ctx, callable; bump = 1.0e-5) ≈ duration(ctx, callable) rtol = 1.0e-3
        @test convexity(DollarConvexity(), ctx, callable; bump = h) ≈ (v(h) - 2v(0.0) + v(-h)) / h^2 rtol = 1.0e-8
        @test duration(DV01(), ctx, callable; bump = h) ≈ -(v(h) - v(-h)) / (2h) / 10_000 rtol = 1.0e-10
        @test_throws ArgumentError convexity(ctx, callable; bump = 0.0)
        # key rates on the lattice add up to the parallel duration
        @test sum(duration(KeyRates([1.0, 3.0, 7.0, 12.0]), ctx, callable)) ≈ duration(ctx, callable) rtol = 1.0e-8

        # the option-adjusted spread
        price = v(0.01)
        @test FC.rate(zspread(tree, callable, price).zspread) ≈ 0.01 atol = 1.0e-9
        @test FC.rate(zspread(ctx, callable, price).zspread) ≈ 0.01 atol = 1.0e-9
        # with credit, the spread beyond the hazard and recovery
        store = Dict("ACME" => acme)
        risky = FM.Credit.Defaultable(callable, "ACME")
        rprice = FC.present_value(FM.Models(FM.ShortRate.HullWhiteTree(FM.ShortRate.HullWhite(0.1, 0.01, curve + FC.Continuous(0.004)); max_step = 1 / 48), store), risky)
        @test FC.rate(zspread(FM.Models(tree, store), risky, rprice).zspread) ≈ 0.004 atol = 1.0e-9
        # volatility sensitivity through a named input
        r = sensitivities((; σ = [0.01])) do m
            FC.present_value(FM.ShortRate.HullWhiteTree(FM.ShortRate.HullWhite(0.1, m.σ[1], curve); max_step = 1 / 48), callable)
        end
        @test r.dv01.σ[1] > 0   # more volatility makes the call worth more to the issuer
    end
end
