# ActuaryUtilities

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaActuary.github.io/ActuaryUtilities.jl/stable/) 
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaActuary.github.io/ActuaryUtilities.jl/dev/)
![CI](https://github.com/JuliaActuary/ActuaryUtilities.jl/workflows/CI/badge.svg)
[![Codecov](https://codecov.io/gh/JuliaActuary/ActuaryUtilities.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaActuary/ActuaryUtilities.jl)


A collection of common functions/manipulations used in Actuarial Calculations.

Some of the functions included:

- `duration`:
    - Calculate the duration given an issue date and date (a.k.a. policy duration)
    - Calculate the `Macaulay`, `Modified`, or `DV01` durations for a set of cashflows
- `convexity` for price sensitivity
- `present_value` or `pv` to calculate the present value of a set of cashflows
- `discount_rate` for a given fixed rate or `InterestCurve`
- `internal_rate_of_return` or `irr` to calculate the IRR given cashflows (including at timepoints like Excel's `XIRR`)
- `breakeven` to calculate the breakeven time for a set of cashflows

### Documentation
Click the docs badges above for more details and examples.

### Useful tips

Functions often use a mix of interest_rates, cashflows, and timepoints. When calling functions, the general order of the arguments is 1) interest rates, 2) cashflows, and 3) timepoints. 
