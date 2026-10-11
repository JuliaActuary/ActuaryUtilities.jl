@testset "Contract/portfolio duration & sensitivities (unified)" begin
    mats = [1.0, 2.0, 3.0, 5.0, 7.0]
    curve = FM.Yield.Spline(FM.Spline.Linear(), mats, [0.02, 0.025, 0.03, 0.035, 0.04])
    tenors = mats
    fl0 = FM.Bond.Floating(0.0, FC.Periodic(1), 5.0, "IDX")
    flm = FM.Bond.Floating(0.02, FC.Periodic(1), 5.0, "IDX")
    fb = FM.Bond.Fixed(0.04, FC.Periodic(1), 5.0)

    @testset "par floater: effective ≈ 0, spread ≈ maturity" begin
        @test duration(Effective(), curve, fl0) ≈ 0.0 atol = 1.0e-8
        @test duration(Spread(), curve, fl0) > 4.0
        @test convexity(Effective(), curve, fl0) ≈ 0.0 atol = 1.0e-6
    end

    @testset "bundle: effective = index + discount; sums; dollar <-> year" begin
        s = sensitivities(KeyRates(tenors), curve, flm)
        p = sensitivities(curve, flm)
        @test s.value == p.value
        @test p.duration.discount + p.duration.index ≈ duration(Effective(), curve, flm) atol = 1.0e-12
        @test p.dv01.discount + p.dv01.index ≈ (p.duration.discount + p.duration.index) * p.value / 10_000 atol = 1.0e-12
        @test sum(s.duration.discount .+ s.duration.index) ≈ duration(Effective(), curve, flm) atol = 1.0e-10
        @test sum(s.duration.discount) ≈ p.duration.discount atol = 1.0e-10
        @test sum(s.dv01.index) ≈ p.dv01.index atol = 1.0e-12
    end

    @testset "fixed bond: effective == spread == modified, index == 0" begin
        s = sensitivities(curve, fb)
        modified = duration(curve, collect(FM.Projection(fb, curve, FM.CashflowProjection())))
        @test s.duration.discount + s.duration.index ≈ modified atol = 1.0e-8
        @test s.duration.discount ≈ modified atol = 1.0e-8
        @test s.duration.index ≈ 0.0 atol = 1.0e-8
    end

    @testset "floater: effective convexity (coupons reset under each shock)" begin
        # Reproject coupons under each shock; scalar and matrix-sum risk must agree.
        flm_value(c) = FC.present_value(FM.Models(c; index = c), flm)
        @test convexity(Effective(), curve, flm) ≈
            sum(convexity(flm_value, KeyRates(tenors), curve)) atol = 1.0e-10
    end

    @testset "fixed bond: effective convexity equals the key-rate matrix sum" begin
        # The hats sum to one, so contract, scalar, and full matrix risk agree.
        cfs = collect(FM.Projection(fb, curve, FM.CashflowProjection()))
        amts = FC.amount.(cfs); times = FC.timepoint.(cfs)
        @test convexity(Effective(), curve, fb) ≈
            sum(convexity(KeyRates(tenors), curve, amts, times)) atol = 1.0e-8
        @test convexity(curve, cfs) ≈
            convexity(Effective(), curve, fb) atol = 1.0e-8
    end

    @testset "effective convexity includes the cross terms" begin
        # With distinct index and discount curves, a parallel shift of both moves value through
        # the coupons, the discounting and their interaction.
        credit = curve + FM.Yield.Constant(FC.Continuous(0.01))
        for target in (flm, [flm, fb])
            s = sensitivities(SecondOrder(), credit, target; index = curve)
            C = s.convexity
            @test C.index.discount ≈ C.discount.index
            @test abs(C.index.discount) > 0.1
            @test convexity(Effective(), credit, target; index = curve) ≈
                C.index.index + C.index.discount + C.discount.index + C.discount.discount rtol = 1.0e-10
            @test !(convexity(Effective(), credit, target; index = curve) ≈ C.index.index + C.discount.discount)
            @test convexity(Spread(), credit, target; index = curve) ≈ C.discount.discount rtol = 1.0e-10
            k = sensitivities(SecondOrder(), KeyRates(tenors), credit, target; index = curve)
            K = k.convexity
            @test convexity(Effective(), KeyRates(tenors), credit, target; index = curve) ≈
                K.index.index .+ K.index.discount .+ K.discount.index .+ K.discount.discount rtol = 1.0e-10
            @test convexity(Spread(), KeyRates(tenors), credit, target; index = curve) ≈ K.discount.discount rtol = 1.0e-10
            @test sum(K.index.discount) ≈ C.index.discount rtol = 1.0e-10
            # The second-order fields extend the first-order ones from the same derivatives.
            f = sensitivities(credit, target; index = curve)
            @test _same_sensitivity(f, (; s.value, s.duration, s.dv01))
            @test duration(DV01(), Effective(), credit, target; index = curve) ≈ f.dv01.discount + f.dv01.index
        end
    end

    @testset "default duration and DV01" begin
        for target in (fb, flm, [fb, flm])
            @test duration(curve, target) ≈ duration(Effective(), curve, target)
            @test duration(DV01(), curve, target) ≈ duration(DV01(), Effective(), curve, target)
            @test convexity(curve, target) ≈ convexity(Effective(), curve, target)
            # The parallel measures equal the roles of the bundle and the sums of the key-rate one.
            p = sensitivities(curve, target)
            @test duration(curve, target) ≈ p.duration.discount + p.duration.index atol = 1.0e-12
            @test duration(Spread(), curve, target) ≈ p.duration.discount atol = 1.0e-12
            @test duration(DV01(), Spread(), curve, target) ≈ p.dv01.discount atol = 1.0e-12
            s = sensitivities(KeyRates(tenors), curve, target)
            # Key-rate durations match the bundle; unmarked key-rate risk is effective risk.
            kr = KeyRates(tenors)
            @test duration(Effective(), kr, curve, target) ≈ s.duration.discount .+ s.duration.index atol = 1.0e-12
            @test duration(kr, curve, target) == duration(Effective(), kr, curve, target)
            @test duration(Spread(), kr, curve, target) ≈ s.duration.discount atol = 1.0e-12
            @test duration(DV01(), kr, curve, target) ≈ s.dv01.discount .+ s.dv01.index atol = 1.0e-12
            @test duration(DV01(), Spread(), kr, curve, target) ≈ s.dv01.discount atol = 1.0e-12
            @test sum(convexity(kr, curve, target)) ≈ convexity(curve, target) atol = 1.0e-10
        end
        @test (@inferred duration(DV01(), curve, fb)) isa Float64
        @test (@inferred convexity(curve, fb)) isa Float64
        # A contract under Hull-White is valued on the model's discount function, as its duration is.
        hw = FM.ShortRate.HullWhite(0.1, 0.01, curve)
        s = sensitivities(KeyRates(tenors), hw, flm)
        @test s.duration.discount .+ s.duration.index ≈ duration(KeyRates(tenors), hw, flm) atol = 1.0e-12
        # A vector of `Cashflow`s is fixed cashflows, and a vector of other contracts a portfolio.
        cfs = collect(FM.Projection(fb, curve, FM.CashflowProjection()))
        @test duration(curve, cfs) == duration(Modified(), curve, cfs)
        @test convexity(curve, cfs) == convexity(curve, FC.amount.(cfs), FC.timepoint.(cfs))
        @test duration(DV01(), KeyRates(tenors), curve, cfs) == duration(DV01(), KeyRates(tenors), curve, FC.amount.(cfs), FC.timepoint.(cfs))
        @test sensitivities(curve, cfs) == sensitivities(curve, FC.amount.(cfs), FC.timepoint.(cfs))
        @test duration(curve, cfs) ≈ duration(curve, FC.AbstractContract[cfs...])
    end

    @testset "portfolio: one-pass == value-weighted" begin
        port = [flm, fb]
        dport = duration(curve, port)
        vfl = FC.present_value(FM.Models(curve; index = curve), flm); vfb = FC.present_value(curve, fb)
        dfl = duration(curve, flm); dfb = duration(curve, fb)
        @test dport ≈ (vfl * dfl + vfb * dfb) / (vfl + vfb) atol = 1.0e-8
        # The bundle aggregates dollar derivatives first, then normalizes by the total value.
        s, sfl, sfb = sensitivities(SecondOrder(), curve, port), sensitivities(SecondOrder(), curve, flm), sensitivities(SecondOrder(), curve, fb)
        @test s.value ≈ sfl.value + sfb.value
        @test s.dv01.discount ≈ sfl.dv01.discount + sfb.dv01.discount
        @test s.duration.index ≈ (sfl.value * sfl.duration.index + sfb.value * sfb.duration.index) / s.value
        @test s.convexity.index.discount ≈ (sfl.value * sfl.convexity.index.discount + sfb.value * sfb.convexity.index.discount) / s.value
    end

    @testset "multi-curve: structured == do-block; additive layers" begin
        credit = FM.Yield.Constant(FC.Continuous(0.01))
        ilp = FM.Yield.Constant(FC.Continuous(0.004))
        for order in (FirstOrder(), SecondOrder()), grid in ((), (KeyRates(tenors),))
            rs = sensitivities(order, grid..., (; rf = curve, credit = credit, ilp = ilp), flm; index = curve)
            rd = sensitivities(order, grid..., (; rf = curve, credit = credit, ilp = ilp, index = curve)) do c
                FC.present_value(FM.Models(c.rf + c.credit + c.ilp; index = c.index), flm)
            end
            @test _same_sensitivity(rs, rd)
            @test keys(rs.duration) == (:rf, :credit, :ilp, :index)
            @test rs.duration.rf ≈ rs.duration.credit atol = 1.0e-8       # additive layers ⇒ equal discount sensitivity
            @test rs.duration.credit ≈ rs.duration.ilp atol = 1.0e-8
            @test sum(rs.duration.index) < 0.0                           # bumping the index raises coupons → raises value
        end
        # One discount curve is the single `discount` role.
        @test _same_sensitivity(sensitivities(KeyRates(tenors), curve, flm; index = credit), sensitivities(KeyRates(tenors), (; discount = curve), flm; index = credit))
        # The projection curve is the `index` role, so no discount layer may take that name.
        @test_throws ArgumentError sensitivities(KeyRates(tenors), (; index = curve, credit), flm; index = curve)
    end

    @testset "z-spread round-trips; locked ≈ next reset" begin
        pvm = FC.present_value(FM.Models(curve; index = curve), flm)
        @test FC.rate(zspread(curve, flm, pvm).zspread) ≈ 0.0 atol = 1.0e-8
        z = zspread(curve, flm, pvm - 0.03)
        @test FC.rate(z.zspread) > 0.0
        reprice = FC.present_value(FM.Models(curve + ((zz, t) -> zz + FC.Continuous(z.zspread)); index = curve), flm)
        @test reprice ≈ pvm - 0.03 atol = 1.0e-10
        @test duration(Effective(), curve, locked_floater(fl0, 0.05, 1.0)) ≈ 1.0 atol = 0.1
    end

    @testset "z-spread is a Continuous rate; s0 may be a Rate" begin
        base = FM.Yield.Constant(FC.Continuous(0.03))
        price = FC.pv(base + FC.Continuous(0.012), fb)
        z = zspread(base, fb, price)
        @test z.zspread isa FC.Rate{Float64, FC.Continuous} && z.zspread_dv01 isa Float64
        @test FC.rate(z.zspread) ≈ 0.012 atol = 1.0e-12
        # the typed spread adds to the curve in its own convention (a number would be annual)
        @test FC.pv(base + z.zspread, fb) ≈ price rtol = 1.0e-14
        # a typed start is its continuous value
        @test zspread(base, fb, price; s0 = FC.Continuous(0.01)) == zspread(base, fb, price; s0 = 0.01)
        @test zspread(base, fb, price; s0 = FC.Periodic(0.01, 1)) == zspread(base, fb, price; s0 = log1p(0.01))
    end

    @testset "effective: AD == central finite difference (re-projecting)" begin
        Δ = 1.0e-4
        up = curve + ((z, t) -> z + FC.Continuous(+Δ)); dn = curve + ((z, t) -> z + FC.Continuous(-Δ))
        rj(crv) = FC.present_value(FM.Models(crv; index = crv), flm)
        eff_fd = (rj(dn) - rj(up)) / (2Δ * rj(curve))
        @test duration(Effective(), curve, flm) ≈ eff_fd atol = 1.0e-4
    end
end

# Reference: OpenGamma "Bond Pricing" (M. Henrard, Quantitative Research, 2011),
# §5.2 Floating rate note (FRN). That note fixes the multi-curve convention this
# package implements: coupons are ESTIMATED on the forward/index curve I (eq. 3)
# and coupons + notional are DISCOUNTED on the issuer/credit curve C (eq. 4):
#
#   F_i = (1/δ_i)(P^I(s_i)/P^I(e_i) − 1)                            (3)
#   PV  = Σ_i δ_i N_i (F_i + s_i) P^C(t_i)  +  N P^C(t_N)           (4)   (settle S = 0)
#
# Market risk of a credit FRN under this convention maps onto the package's
# floating-rate decomposition as:
#   IR01 — 1bp parallel shift of the risk-free curve, which drives BOTH the index
#          I (so coupons re-fix) AND the issuer discount C = rf + spread  ⇒ Effective
#   CS01 — 1bp parallel shift of the issuer credit spread only (C moves; the coupons
#          estimated on I are held fixed)                                  ⇒ Spread
# A floater therefore shows |IR01| ≈ 0 (≈ next reset) and CS01 ≈ maturity — the
# textbook FRN signature. The reference price / IR01 / CS01 are rebuilt below from
# eq. (3)–(4) directly (independent of the package's projection machinery), then the
# package is checked against them.
@testset "OpenGamma §5.2 FRN reference: IR01 / CS01" begin
    tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
    rf_zeros = [0.02, 0.024, 0.027, 0.029, 0.03]   # continuous-comp risk-free zeros
    cs = 0.01                                  # 100bp issuer credit spread
    margin = 0.005                                  # 50bp FRN quoted margin (coupon_rate)
    m = 1                                       # annual coupons ⇒ accrual δ = 1/m
    δ = 1 / m

    rf = FM.Yield.Spline(FM.Spline.Linear(), tenors, rf_zeros)  # index / Ibor forward, I
    credit = rf + ((z, t) -> z + FC.Continuous(cs))                 # issuer discount, C = rf + spread
    fl = FM.Bond.Floating(margin, FC.Periodic(m), 5.0, "IDX")

    # Independent OpenGamma eq. (3)+(4) PV with distinct curves I and C (N = 1, S = 0).
    function og_pv(Icrv, Ccrv)
        Pprev = 1.0                                          # P^I(t_0) = P^I(0) = 1
        pv = 0.0
        for t in tenors
            Fi = m * (Pprev / FC.discount(Icrv, t) - 1)   # eq. (3): forward over [t-1/m, t]
            pv += δ * (Fi + margin) * FC.discount(Ccrv, t) # eq. (4): coupon δ(F+s) on C
            Pprev = FC.discount(Icrv, t)
        end
        return pv + FC.discount(Ccrv, tenors[end])           # notional, discounted on C
    end

    bump(crv, d) = crv + ((z, t) -> z + FC.Continuous(d))
    bp = 1.0e-4
    pv_ir(d) = og_pv(bump(rf, d), bump(credit, d))   # IR01: rf moves ⇒ both I and C shift
    pv_cs(d) = og_pv(rf, bump(credit, d))            # CS01: only C shifts; coupons held on I

    pv_ref = og_pv(rf, credit)
    ir01_ref = (pv_ir(-bp) - pv_ir(bp)) / 2          # dv01 ≈ (V(−1bp) − V(+1bp)) / 2
    cs01_ref = (pv_cs(-bp) - pv_cs(bp)) / 2

    s = sensitivities(KeyRates(tenors), credit, fl; index = rf)
    p = sensitivities(credit, fl; index = rf)

    @testset "reproduces OpenGamma eq.(3)+(4) price" begin
        @test s.value ≈ pv_ref atol = 1.0e-12
        @test FC.present_value(FM.Models(credit; index = rf), fl) ≈ pv_ref atol = 1.0e-12
        @test pv_ref ≈ 0.9760496203 atol = 1.0e-9                       # regression anchor
    end

    @testset "IR01 ⇒ Effective, CS01 ⇒ Spread (vs eq.(3)+(4) 1bp bump)" begin
        effective_dv01 = p.dv01.discount + p.dv01.index
        @test effective_dv01 ≈ ir01_ref rtol = 1.0e-4
        @test p.dv01.discount ≈ cs01_ref rtol = 1.0e-4
        @test duration(DV01(), Effective(), credit, fl; index = rf) ≈ ir01_ref rtol = 1.0e-4   # public verbs
        @test duration(DV01(), Spread(), credit, fl; index = rf) ≈ cs01_ref rtol = 1.0e-4
        @test duration(DV01(), Effective(), credit, fl; index = rf) ≈ effective_dv01 atol = 1.0e-12  # eff = index + discount
        @test sum(s.dv01.discount .+ s.dv01.index) ≈ effective_dv01 atol = 1.0e-12
        @test effective_dv01 ≈ -2.381601e-6 rtol = 1.0e-5            # regression anchors
        @test p.dv01.discount ≈ 4.585068e-4 rtol = 1.0e-5
    end

    @testset "FRN signature: |IR01| ≈ 0 (next reset) ≪ CS01 ≈ maturity" begin
        @test abs(p.dv01.discount + p.dv01.index) < abs(p.dv01.discount) / 50   # rate risk ≈ killed by re-fixing
        @test 4.5 < p.duration.discount < 5.0                     # ≈ time to maturity
        @test abs(p.duration.discount + p.duration.index) < 0.1   # ≈ time to next reset (≈ 0)
        cs01_kr = s.dv01.discount                                 # CS01 risk concentrates at...
        @test argmax(cs01_kr) == length(tenors)                   # ...the notional repayment (maturity)
        @test cs01_kr[end] > 0.9 * cs01_ref
    end
end
