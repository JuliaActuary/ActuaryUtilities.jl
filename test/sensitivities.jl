@testset "ZeroRateCurve sensitivities" begin
    @testset "ZCB analytical" begin
        rates = [0.03, 0.03, 0.03]
        tenors = [1.0, 2.0, 5.0]
        # Use Linear for perfect locality assertions
        zrc = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
        face = 100.0

        result = sensitivities(KeyRates(tenors), zrc, [0.0, 0.0, face], tenors)

        @test result.value ≈ face * exp(-0.03 * 5.0) atol = 1.0e-6
        @test result.durations[3] ≈ 5.0 atol = 1.0e-6
        @test result.durations[1] ≈ 0.0 atol = 1.0e-6
        @test result.convexities[3, 3] ≈ 25.0 atol = 1.0e-6
    end

    @testset "coupon bond" begin
        rates = [0.04, 0.04, 0.04, 0.04, 0.04]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        # Use Linear for positive-KRD guarantee
        zrc = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
        cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

        result = sensitivities(KeyRates(tenors), zrc, cfs, tenors)

        @test result.value > 0
        @test all(result.durations .> 0)

        # sensitivities returns same durations as calling duration(KeyRates(tenors), ...) separately
        @test result.durations ≈ duration(KeyRates(tenors), zrc, cfs, tenors) atol = 1.0e-12

        # DV01 dispatch
        dv01_result = sensitivities(DV01(), KeyRates(tenors), zrc, cfs, tenors)
        @test all(dv01_result.dv01s .> 0)
        @test dv01_result.dv01s ≈ duration(DV01(), KeyRates(tenors), zrc, cfs, tenors) atol = 1.0e-12
        @test dv01_result.value ≈ result.value atol = 1.0e-12
    end

    @testset "do-block" begin
        rates = [0.03, 0.03, 0.03]
        tenors = [1.0, 2.0, 3.0]
        # Use Linear for positive-KRD guarantee
        zrc = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
        cfs = [5.0, 5.0, 105.0]

        result = sensitivities(KeyRates(tenors), zrc) do curve
            sum(cf * curve(t) for (cf, t) in zip(cfs, tenors))
        end

        @test result.value > 0
        @test all(result.durations .> 0)
    end

    @testset "two-curve additive: IR01 ≈ CS01" begin
        base_rates = [0.03, 0.03, 0.03, 0.03, 0.03]
        credit_rates = [0.02, 0.02, 0.02, 0.02, 0.02]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        base = FM.ZeroRateCurve(base_rates, tenors)
        credit = FM.ZeroRateCurve(credit_rates, tenors)
        cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

        result = sensitivities(KeyRates(tenors), base, credit, cfs, tenors)

        @test result.base_durations ≈ result.credit_durations atol = 1.0e-10

        # DV01 dispatch
        dv01_result = sensitivities(DV01(), KeyRates(tenors), base, credit, cfs, tenors)
        @test dv01_result.base_dv01s ≈ dv01_result.credit_dv01s atol = 1.0e-12
        @test dv01_result.value ≈ result.value atol = 1.0e-12

        # Macaulay duration for flat continuous rate 0.05
        total_rate = 0.05
        dfs = [exp(-total_rate * t) for t in tenors]
        mac_dur = sum(t * cf * df for (t, cf, df) in zip(tenors, cfs, dfs)) / sum(cf * df for (cf, df) in zip(cfs, dfs))
        @test sum(result.base_durations) ≈ mac_dur atol = 1.0e-6
    end

    @testset "two-curve non-additive: base ≠ credit" begin
        base_rates = [0.03, 0.03, 0.03]
        credit_rates = [0.02, 0.02, 0.02]
        tenors = [1.0, 2.0, 5.0]
        base = FM.ZeroRateCurve(base_rates, tenors)
        credit = FM.ZeroRateCurve(credit_rates, tenors)
        face = 100.0

        result = sensitivities(KeyRates(tenors), base, credit) do base_curve, credit_curve
            face * (2.0 * base_curve(5.0) + 0.5 * credit_curve(5.0))
        end

        @test !isapprox(result.base_durations, result.credit_durations, atol = 1.0e-6)
    end

    @testset "two-curve with mismatched ZRC storage tenors" begin
        # Under the unified API the KRD knot grid is supplied explicitly, so
        # base and credit no longer need matching `tenors` fields — they can
        # be evaluated against any common knot grid.
        base = FM.ZeroRateCurve([0.03, 0.03, 0.03], [1.0, 2.0, 5.0])
        credit = FM.ZeroRateCurve([0.02, 0.02], [1.0, 2.0])
        knots = [1.0, 2.0, 5.0]
        result = sensitivities(KeyRates(knots), base, credit, [5.0, 5.0, 105.0], [1.0, 2.0, 5.0])
        @test result.value > 0
        @test length(result.base_durations) == length(knots)
        @test length(result.credit_durations) == length(knots)
    end

    @testset "chapter VGH test case" begin
        zero_rates = [0.01, 0.02, 0.02, 0.03, 0.05, 0.055]
        times = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
        zrc = FM.ZeroRateCurve(zero_rates, times, FM.Spline.Cubic())

        # 10-year fixed bond, 9% coupon, semi-annual, par=1.0
        coupon = 0.09
        cfs_times = collect(0.5:0.5:10.0)
        cfs = [coupon / 2 + (t == 10.0 ? 1.0 : 0.0) for t in cfs_times]

        result = sensitivities(KeyRates(times), zrc, cfs, cfs_times)

        @test result.value > 0
        # durations at tenors within the bond's maturity are positive
        @test all(result.durations[1:5] .> 0)
        @test sum(result.durations) > 0  # total duration is positive

        # convexity matrix is symmetric
        @test result.convexities ≈ result.convexities' atol = 1.0e-10
    end

    @testset "portfolio linearity" begin
        zero_rates = [0.03, 0.03, 0.03, 0.03, 0.03]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        zrc = FM.ZeroRateCurve(zero_rates, tenors)

        bond1_cfs = [0.05, 0.05, 1.05, 0.0, 0.0]
        bond1_times = [1.0, 2.0, 3.0, 4.0, 5.0]
        bond2_cfs = [0.03, 0.03, 0.03, 1.03, 0.0]
        bond2_times = [1.0, 2.0, 3.0, 5.0, 5.0]

        # Portfolio valuation — single AD pass over sum
        portfolio_valuation = curve -> begin
            sum(cf * curve(t) for (cf, t) in zip(bond1_cfs, bond1_times)) +
                sum(cf * curve(t) for (cf, t) in zip(bond2_cfs, bond2_times))
        end
        portfolio_dv01 = duration(DV01(), KeyRates(tenors), portfolio_valuation, zrc)

        # Individual DV01s
        dv01_1 = duration(DV01(), KeyRates(tenors), zrc, bond1_cfs, bond1_times)
        dv01_2 = duration(DV01(), KeyRates(tenors), zrc, bond2_cfs, bond2_times)

        # DV01 is additive (not value-weighted like modified duration)
        @test portfolio_dv01 ≈ dv01_1 .+ dv01_2 atol = 1.0e-10
    end
end

# Custom AbstractYieldModel: a composite of two flat curves, multiplicative in
# discount space. Has no `.rates`/`.tenors`/`.spline` field, so the only way
# AU can compute KRDs is via TenorShift bumps over the curve's own `discount`.
struct CompositeTwoFlatYield{A, B} <: FM.Yield.AbstractYieldModel
    base::A
    spread::B
end
FC.discount(c::CompositeTwoFlatYield, t) = FC.discount(c.base, t) * FC.discount(c.spread, t)

@testset "Custom AbstractYieldModel: KRD/IR01/CS01 on a non-ZRC curve" begin
    base = FM.Yield.Constant(FC.Continuous(0.04))
    spread = FM.Yield.Constant(FC.Continuous(0.012))
    curve = CompositeTwoFlatYield(base, spread)

    tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
    cfs = vcat(fill(5.0, 29), [105.0])
    times = collect(1.0:30.0)
    pv(c) = sum(cf * FC.discount(c, t) for (cf, t) in zip(cfs, times))

    @testset "scalar duration matches sum of KRDs" begin
        sd = duration(pv, curve, tenors)
        krds = duration(KeyRates(tenors), pv, curve)
        @test sd ≈ sum(krds) atol = 1.0e-10
    end

    @testset "do-block and cashflow forms agree" begin
        krds_db = duration(KeyRates(tenors), curve) do c
            pv(c)
        end
        krds_cf = duration(KeyRates(tenors), curve, cfs, times)
        @test krds_db ≈ krds_cf atol = 1.0e-10
    end

    @testset "scalar matches parallel-shift modified duration" begin
        # Total rate is 5.2% continuous; on annual coupon CFs, modified ≈ Macaulay.
        rate = 0.04 + 0.012
        dfs = [exp(-rate * t) for t in times]
        V = sum(cf * df for (cf, df) in zip(cfs, dfs))
        mac = sum(t * cf * df for (t, cf, df) in zip(times, cfs, dfs)) / V
        @test duration(pv, curve, tenors) ≈ mac atol = 1.0e-6
    end

    @testset "DV01" begin
        dv01 = duration(DV01(), pv, curve, tenors)
        krd_dv01 = duration(DV01(), KeyRates(tenors), pv, curve)
        @test dv01 ≈ sum(krd_dv01) atol = 1.0e-10
        @test all(krd_dv01 .≥ 0)
    end

    @testset "IR01 ≈ CS01 ≈ DV01 (flat additive case)" begin
        # Per the docstring: in a flat additive decomposition, bumping base
        # alone, credit alone, or the composite all shift the total zero rate
        # by 1bp — so IR01 ≈ CS01 ≈ DV01 individually.
        pv2c(b, c) = sum(cf * FC.discount(b, t) * FC.discount(c, t) for (cf, t) in zip(cfs, times))
        ir01 = duration(IR01(), pv2c, base, spread, tenors)
        cs01 = duration(CS01(), pv2c, base, spread, tenors)
        dv01 = duration(DV01(), pv, curve, tenors)
        @test ir01 ≈ cs01 atol = 1.0e-10
        @test ir01 ≈ dv01 atol = 1.0e-10
    end

    @testset "convexity matrix symmetric, scalar = sum" begin
        cmat = convexity(KeyRates(tenors), pv, curve)
        @test cmat ≈ cmat' atol = 1.0e-10
        @test convexity(pv, curve, tenors) ≈ sum(cmat) atol = 1.0e-10
    end

    @testset "sensitivities bundle" begin
        r = sensitivities(KeyRates(tenors), curve, cfs, times)
        @test r.value ≈ pv(curve) atol = 1.0e-10        # exact baseline; no resampling
        @test r.durations ≈ duration(KeyRates(tenors), pv, curve) atol = 1.0e-10
        @test sum(r.durations) ≈ duration(pv, curve, tenors) atol = 1.0e-10
        @test r.convexities ≈ r.convexities' atol = 1.0e-10

        r_dv01 = sensitivities(DV01(), KeyRates(tenors), curve, cfs, times)
        @test r_dv01.dv01s ≈ duration(DV01(), KeyRates(tenors), pv, curve) atol = 1.0e-10
    end

    @testset "two-curve sensitivities" begin
        pv2c(b, c) = sum(cf * FC.discount(b, t) * FC.discount(c, t) for (cf, t) in zip(cfs, times))
        r = sensitivities(KeyRates(tenors), pv2c, base, spread)
        @test r.value ≈ pv2c(base, spread) atol = 1.0e-10
        @test r.base_durations ≈ r.credit_durations atol = 1.0e-10   # additive ⇒ symmetric
    end

    @testset "ZRC promotion equivalence (Linear spline)" begin
        # With Spline.Linear, ZRC's KRDs match TenorShift+hat exactly because
        # linear interpolation in zero-rate space ≡ triangular-hat bumps.
        zrc = FM.Yield.ZeroRateCurve(curve, tenors, spline = FM.Spline.Linear())
        krds_zrc = duration(KeyRates(tenors), pv, zrc)
        krds_custom = duration(KeyRates(tenors), pv, curve)
        @test krds_custom ≈ krds_zrc atol = 1.0e-10
    end

    @testset "AD through non-flat base (NelsonSiegel + Constant)" begin
        # Stresses the AD chain on a curve whose `Base.zero(base, t)` is
        # non-linear in `t`, unlike the `Constant + Constant` flat composite
        # used in the other testsets here.
        ns_base = FM.Yield.NelsonSiegel(1.0, 0.04, -0.02, 0.01)
        flat_spr = FM.Yield.Constant(FC.Continuous(0.012))
        curve_nf = CompositeTwoFlatYield(ns_base, flat_spr)
        krds_nf = duration(KeyRates(tenors), pv, curve_nf)
        @test sum(krds_nf) ≈ duration(pv, curve_nf, tenors) atol = 1.0e-10
        @test argmax(krds_nf) == lastindex(tenors)   # sensitivity peaks at the long end
    end
end

@testset "AD vs analytic KRD: byte-equivalence across curve types and arities" begin
    # The analytic helpers `_keyrate_analytic` (single/two-curve) and
    # `_ncurve_analytic` (NamedTuple) must produce the same value, gradient,
    # and Hessian as the AD path (`_keyrate_ad`, `_ncurve_ad`) for the vanilla
    # cashflow case. Regression guard against future drift between the two
    # implementations of the same math.
    KRA = ActuaryUtilities.FinancialMath._keyrate_analytic
    KRA_N = ActuaryUtilities.FinancialMath._ncurve_analytic
    KRAD = ActuaryUtilities.FinancialMath._keyrate_ad
    NCAD = ActuaryUtilities.FinancialMath._ncurve_ad

    tenors = collect(1.0:30.0)
    rates = fill(0.03, 30)
    rates2 = rates .+ 0.005
    curves = [
        FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear()),
        FM.ZeroRateCurve(rates, tenors, FM.Spline.MonotoneConvex()),
        FM.Yield.Constant(FC.Continuous(0.03)),
    ]

    cfs_full = collect(FM.Projection(FM.Bond.Fixed(0.04, FC.Periodic(2), 5), curves[1], FM.CashflowProjection()))
    amts = FC.amount.(cfs_full)
    times = FC.timepoint.(cfs_full)

    @testset "single-curve [$(typeof(c).name.name)]" for c in curves
        ad = KRAD(
            c, tenors,
            i -> sum(amts[k] * FC.discount(i, times[k]) for k in eachindex(amts));
            order = 2
        )
        an = KRA(c, tenors, amts, times; order = 2)
        @test ad.value ≈ an.value rtol = 1.0e-12
        @test maximum(abs.(ad.gradient .- an.gradient)) < 1.0e-12
        @test maximum(abs.(ad.hessian .- an.hessian)) < 1.0e-12
    end

    @testset "two-curve" begin
        base = curves[1]
        credit = FM.ZeroRateCurve(rates2, tenors, FM.Spline.Linear())
        ad = KRAD(
            base, credit, tenors,
            (b, c) -> sum(
                amts[k] * FC.discount(b, times[k]) * FC.discount(c, times[k])
                    for k in eachindex(amts)
            );
            order = 2
        )
        an = KRA(base, credit, tenors, amts, times; order = 2)
        @test ad.value ≈ an.value rtol = 1.0e-12
        @test maximum(abs.(ad.base_gradient .- an.base_gradient)) < 1.0e-12
        @test maximum(abs.(ad.credit_gradient .- an.credit_gradient)) < 1.0e-12
        @test maximum(abs.(ad.base_hessian .- an.base_hessian)) < 1.0e-12
        @test maximum(abs.(ad.credit_hessian .- an.credit_hessian)) < 1.0e-12
        @test maximum(abs.(ad.cross_hessian .- an.cross_hessian)) < 1.0e-12
    end

    @testset "NamedTuple (3 curves)" begin
        c1 = curves[1]
        c2 = FM.ZeroRateCurve(rates2, tenors, FM.Spline.Linear())
        c3 = FM.ZeroRateCurve(rates .+ 0.002, tenors, FM.Spline.Linear())
        nt = (; rf = c1, credit = c2, ilp = c3)
        ad_v, ad_g = NCAD(
            c -> sum(
                amts[k] * FC.discount(c.rf, times[k]) *
                    FC.discount(c.credit, times[k]) *
                    FC.discount(c.ilp, times[k])
                    for k in eachindex(amts)
            ),
            nt, tenors
        )
        an = KRA_N(nt, tenors, amts, times; order = 2)
        @test ad_v ≈ an.value rtol = 1.0e-12
        # Per-role gradients from the AD path all agree with the single shared
        # gradient returned by the analytic helper.
        for r in (:rf, :credit, :ilp)
            @test maximum(abs.(ad_g[r] .- an.gradient)) < 1.0e-12
        end
    end
end
