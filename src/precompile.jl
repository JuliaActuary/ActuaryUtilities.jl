
# created with the help of SnoopCompile.jl
@precompile_setup begin
    # Putting some things in `setup` can reduce the size of the
    # precompile file and potentially make loading faster.
    cfs = [10 for i in 1:10]
    
    # 2021-03-31 rates from Treasury.gov
    rates =[0.01, 0.01, 0.03, 0.05, 0.07, 0.16, 0.35, 0.92, 1.40, 1.74, 2.31, 2.41] ./ 100
    mats = [1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]

    y = Yields.CMT(rates,mats)
    r = 0.05
    rates = [r,y]
    @precompile_all_calls begin
        # all calls in this block will be precompiled, regardless of whether
        # they belong to your package or not (on Julia 1.8 and higher)
        irr([-80;cfs])
        moic([-80;cfs])

        for v in rates
            pv(v,cfs)
            duration(v,cfs)
            convexity(v,cfs)
            duration(Macaulay(),v,cfs)
            duration(DV01(),v,cfs)
            duration(Modified(),v,cfs)
            # duration(KeyRate(5),v,cfs)

            breakeven(v, [-10,1,2,3,4,8])

            spread(v - .01,v,cfs)

        end
        spread(Yields.Continuous(0.04),Yields.Continuous(0.05),cfs)



        years_between(Date(2018, 9, 30), Date(2018, 9, 30))
        duration(Date(2018, 9, 30), Date(2019, 9, 30))

    end
end