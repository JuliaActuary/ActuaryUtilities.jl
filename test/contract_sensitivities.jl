@testset "Contract/portfolio duration & sensitivities (unified)" begin
    mats = [1.0, 2.0, 3.0, 5.0, 7.0]
    curve = FM.Yield.Spline(FM.Spline.Linear(), mats, [0.02, 0.025, 0.03, 0.035, 0.04])
    tenors = mats
    fl0 = FM.Bond.Floating(0.0, FC.Periodic(1), 5.0, "IDX")
    flm = FM.Bond.Floating(0.02, FC.Periodic(1), 5.0, "IDX")
    fb = FM.Bond.Fixed(0.04, FC.Periodic(1), 5.0)

    @testset "par floater: effective ≈ 0, spread ≈ maturity" begin
        @test duration(Effective(), fl0, curve, tenors) ≈ 0.0 atol = 1.0e-8
        @test duration(Spread(), fl0, curve, tenors) > 4.0
        @test convexity(Effective(), fl0, curve, tenors) ≈ 0.0 atol = 1.0e-6
    end

    @testset "bundle: effective = forward + spread; sums; dollar <-> year" begin
        s = sensitivities(flm, curve, tenors)
        @test s.effective_duration ≈ s.forward_duration + s.spread_duration atol = 1.0e-10
        @test s.effective_dv01 ≈ s.effective_duration * s.value / 10_000 atol = 1.0e-12
        @test sum(s.effective_key_rate) ≈ s.effective_duration atol = 1.0e-10
        @test sum(s.spread_key_rate) ≈ s.spread_duration atol = 1.0e-10
    end

    @testset "fixed bond: effective == spread == modified, forward == 0" begin
        s = sensitivities(fb, curve, tenors)
        modified = duration(curve, tenors, collect(FM.Projection(fb, curve, FM.CashflowProjection())))
        @test s.effective_duration ≈ modified atol = 1.0e-8
        @test s.spread_duration ≈ modified atol = 1.0e-8
        @test s.forward_duration ≈ 0.0 atol = 1.0e-8
    end

    @testset "floater: effective convexity (dynamic cashflows under reproject)" begin
        # The new `convexity(::Effective)` routes through TenorShift +
        # ForwardDiff on a closure that calls `reproject(target, c)` — i.e.
        # cashflows are themselves curve-dependent (the coupon resets follow
        # the bumped curve). The result must still equal the matrix-sum form
        # (same AD chain, just unrolled). Locks the dynamic-cashflow path.
        _cvalue_flm(c) = FC.present_value(c, ActuaryUtilities.reproject(flm, c))
        @test convexity(Effective(), flm, curve, tenors) ≈
            sum(convexity(KeyRates(tenors), _cvalue_flm, curve)) atol = 1.0e-10
    end

    @testset "fixed bond: effective convexity matches matrix-sum (POU equivalence regression guard)" begin
        # Under partition of unity of the KRD hat functions, sum(N×N key-rate
        # Hessian) = continuous-shock parallel-shift second derivative by the
        # chain rule. The optimized `convexity(::Effective, …)` computes that
        # scalar directly via TenorShift, in O(1) rather than O(N²) AD work.
        # Locks the numerical equivalence in for future refactors of either
        # path. The no-tenor curve form uses the same continuous-zero shock.
        cfs = collect(FM.Projection(fb, curve, FM.CashflowProjection()))
        amts = FC.amount.(cfs); times = FC.timepoint.(cfs)
        @test convexity(Effective(), fb, curve, tenors) ≈
            sum(convexity(KeyRates(tenors), curve, amts, times)) atol = 1.0e-8
        @test convexity(curve, cfs) ≈
            convexity(Effective(), fb, curve, tenors) atol = 1.0e-8
    end

    @testset "default duration & dv01 verb" begin
        @test duration(flm, curve, tenors) ≈ duration(Effective(), flm, curve, tenors)
        @test dv01(Effective(), flm, curve, tenors) ≈ sensitivities(flm, curve, tenors).effective_dv01
        @test dv01(0.05, [5.0, 5.0, 105.0]) ≈ duration(DV01(), 0.05, [5.0, 5.0, 105.0])   # cashflow fallback
    end

    @testset "portfolio: one-pass == value-weighted" begin
        port = [flm, fb]
        dport = duration(port, curve, tenors)
        vfl = FC.present_value(curve, reproject(flm, curve)); vfb = FC.present_value(curve, fb)
        dfl = duration(flm, curve, tenors); dfb = duration(fb, curve, tenors)
        @test dport ≈ (vfl * dfl + vfb * dfb) / (vfl + vfb) atol = 1.0e-8
    end

    @testset "multi-curve: structured == do-block; additive layers" begin
        credit = FM.Yield.Constant(FC.Continuous(0.01))
        ilp = FM.Yield.Constant(FC.Continuous(0.004))
        rs = sensitivities(flm, tenors; discount = (; rf = curve, credit = credit, ilp = ilp), index = curve)
        rd = sensitivities((; rf = curve, credit = credit, ilp = ilp, index = curve); tenors) do c
            FC.present_value(c.rf + c.credit + c.ilp, reproject(flm, c.index))
        end
        @test rs.duration.rf ≈ rd.duration.rf atol = 1.0e-10
        @test rs.duration.index ≈ rd.duration.index atol = 1.0e-10
        @test rs.duration.rf ≈ rs.duration.credit atol = 1.0e-8       # additive layers ⇒ equal discount sensitivity
        @test rs.duration.credit ≈ rs.duration.ilp atol = 1.0e-8
        @test rs.duration.index < 0.0                                # bumping the index raises coupons → raises value
    end

    @testset "z-spread round-trips; locked ≈ next reset" begin
        pvm = FC.present_value(curve, reproject(flm, curve))
        @test zspread(flm, curve, pvm).zspread ≈ 0.0 atol = 1.0e-8
        z = zspread(flm, curve, pvm - 0.03)
        @test z.zspread > 0.0
        reprice = FC.present_value(curve + ((zz, t) -> zz + FC.Continuous(z.zspread)), reproject(flm, curve))
        @test reprice ≈ pvm - 0.03 atol = 1.0e-10
        @test duration(Effective(), locked_floater(fl0, 0.05, 1.0), curve, tenors) ≈ 1.0 atol = 0.1
    end

    @testset "effective: AD == central finite difference (re-projecting)" begin
        Δ = 1.0e-4
        up = curve + ((z, t) -> z + FC.Continuous(+Δ)); dn = curve + ((z, t) -> z + FC.Continuous(-Δ))
        rj(crv) = FC.present_value(crv, reproject(flm, crv))
        eff_fd = (rj(dn) - rj(up)) / (2Δ * rj(curve))
        @test duration(Effective(), flm, curve, tenors) ≈ eff_fd atol = 1.0e-4
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

    s = sensitivities(fl, rf, credit, tenors)

    @testset "reproduces OpenGamma eq.(3)+(4) price" begin
        @test s.value ≈ pv_ref atol = 1.0e-12
        @test FC.present_value(credit, reproject(fl, rf)) ≈ pv_ref atol = 1.0e-12
        @test pv_ref ≈ 0.9760496203 atol = 1.0e-9                       # regression anchor
    end

    @testset "IR01 ⇒ Effective, CS01 ⇒ Spread (vs eq.(3)+(4) 1bp bump)" begin
        @test s.effective_dv01 ≈ ir01_ref rtol = 1.0e-4
        @test s.spread_dv01 ≈ cs01_ref rtol = 1.0e-4
        @test dv01(Effective(), fl, rf, credit, tenors) ≈ ir01_ref rtol = 1.0e-4   # public verbs
        @test dv01(Spread(), fl, rf, credit, tenors) ≈ cs01_ref rtol = 1.0e-4
        @test s.effective_dv01 ≈ s.forward_dv01 + s.spread_dv01 atol = 1.0e-12      # eff = fwd + spr
        @test s.effective_dv01 ≈ -2.381601e-6 rtol = 1.0e-5            # regression anchors
        @test s.spread_dv01 ≈ 4.585068e-4 rtol = 1.0e-5
    end

    @testset "FRN signature: |IR01| ≈ 0 (next reset) ≪ CS01 ≈ maturity" begin
        @test abs(s.effective_dv01) < abs(s.spread_dv01) / 50   # rate risk ≈ killed by re-fixing
        @test 4.5 < s.spread_duration < 5.0                     # ≈ time to maturity
        @test abs(s.effective_duration) < 0.1                   # ≈ time to next reset (≈ 0)
        cs01_kr = s.spread_key_rate .* s.value ./ 1.0e4           # CS01 risk concentrates at...
        @test argmax(cs01_kr) == length(tenors)                 # ...the notional repayment (maturity)
        @test cs01_kr[end] > 0.9 * cs01_ref
    end
end
