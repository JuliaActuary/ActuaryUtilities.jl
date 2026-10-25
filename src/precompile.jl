# created with the help of SnoopCompile.jl
@setup_workload begin
    # Putting some things in `setup` can reduce the size of the
    # precompile file and potentially make loading faster.
    cfs = Float64[10 for i in 1:10]

    # 2021-03-31 rates from Treasury.gov
    rates = [0.01, 0.01, 0.03, 0.05, 0.07, 0.16, 0.35, 0.92, 1.4, 1.74, 2.31, 2.41] ./ 100
    mats = [1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30]

    y = FinanceModels.fit(FinanceModels.Spline.Linear(), FinanceModels.CMTYield.(rates, mats), FinanceModels.Fit.Bootstrap())
    r = 0.05
    rates = [r, y]
    # v6 curve risk: the default ZeroRateCurve (MonotoneConvex) and a linear one
    kr_tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
    kr_cfs = [5.0, 5.0, 5.0, 5.0, 105.0]
    risk_samples = collect(range(-3.0, 3.0; length = 101))
    @compile_workload begin
        # all calls in this block will be precompiled, regardless of whether
        # they belong to your package or not (on Julia 1.8 and higher)
        moic([-80; cfs])

        for v in rates
            pv(v, cfs)
            duration(v, cfs)
            convexity(v, cfs)
            duration(Macaulay(), v, cfs)
            duration(DV01(), v, cfs)
            duration(Modified(), v, cfs)

            breakeven(v, [-10, 1, 2, 3, 4, 8])

            spread(v - 0.01, v, cfs)

        end
        spread(y, y + 0.01, cfs)


        for c in (FinanceModels.ZeroRateCurve(fill(0.03, 5), kr_tenors), FinanceModels.ZeroRateCurve(fill(0.03, 5), kr_tenors, FinanceModels.Spline.Linear()))
            duration(c, kr_cfs, kr_tenors); duration(Macaulay(), c, kr_cfs, kr_tenors)
            duration(DV01(), c, kr_cfs, kr_tenors); convexity(c, kr_cfs, kr_tenors)
            duration(KeyRates(kr_tenors), c, kr_cfs, kr_tenors); duration(DV01(), KeyRates(kr_tenors), c, kr_cfs, kr_tenors)
            convexity(KeyRates(kr_tenors), c, kr_cfs, kr_tenors)
            sensitivities(c, FinanceCore.Cashflow.(kr_cfs, kr_tenors))
            sensitivities(SecondOrder(), KeyRates(kr_tenors), c, FinanceCore.Cashflow.(kr_cfs, kr_tenors))
        end
        duration(Effective(), FinanceModels.Yield.Constant(0.03), FinanceModels.Bond.Fixed(0.05, FinanceCore.Periodic(2), 5.0))
        VaR(0.95)(risk_samples); CTE(0.95)(risk_samples)
        irr([-100.0, 10.0, 10.0, 110.0])

        years_between(Dates.Date(2018, 9, 30), Dates.Date(2018, 9, 30))
        policy_duration(Dates.Date(2018, 9, 30), Dates.Date(2019, 9, 30))

    end
end
