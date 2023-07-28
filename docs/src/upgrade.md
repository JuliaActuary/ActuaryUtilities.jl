# Version Upgrade Guide

## v3 to v4

### Overview 

The shape and API of the package is mostly unchanged. The changes that have made fall into a few categores:

- Accommodating FinanceModels.jl, the next-generation version of Yields.jl.
- Simplifying the API, generally making function calls require more specific arguments to avoid ambiguity
- Accommodating the new `Cashflow` type which makes modeling heterogeneous assets and liabilities simpler.

### API Changes

- Breaking: The functions `europut` and `eurocall` have been moved to `FinanceModels`
- Breaking: Previously, the first argument to `present_value` or `present_values` would be interpreted as a set of `Periodic(1)` one-period forward rates if a vector of real values was passed. Users should explicitly create the yield model first, instead of relying on the implicit conversion:

```julia
# old 
pv([0.05,0.1], cfs)  

# new
using FinanceModels
y = fit(Spline.Linear(),ForwardYields([0.05,0.1]),Bootstrap())
pv(y,cfs)

``` 
