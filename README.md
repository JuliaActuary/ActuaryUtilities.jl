# ActuaryUtilities

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaActuary.github.io/ActuaryUtilities.jl/stable/) 
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaActuary.github.io/ActuaryUtilities.jl/dev/)
![CI](https://github.com/JuliaActuary/ActuaryUtilities.jl/workflows/CI/badge.svg)
[![Codecov](https://codecov.io/gh/JuliaActuary/ActuaryUtilities.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaActuary/ActuaryUtilities.jl)

## Features

A collection of common functions/manipulations used in Actuarial Calculations.

### Financial Maths
- `duration`:
  - Calculate the `Macaulay`, `Modified`, or `DV01` durations for a set of cashflows
- `convexity` for price sensitivity
- Flexible interest rate options via the [`Yields.jl`](https://github.com/JuliaActuary/Yields.jl) package.
- `internal_rate_of_return` or `irr` to calculate the IRR given cashflows (including at timepoints like Excel's `XIRR`)
- `breakeven` to calculate the breakeven time for a set of cashflows
- `accum_offset` to calculate accumulations like survivorship from a mortality vector

### Insurance mechanics

- `duration`:
  - Calculate the duration given an issue date and date (a.k.a. policy duration)
  
### Excel Utilities

You can also copy/paste to/from Excel:

![Copying to and from Excel](https://user-images.githubusercontent.com/711879/107859226-9e9a5500-6dfd-11eb-999f-5da5c7e5b8a6.gif)

- `xlcopy()` copies and parses Excel content on the clipboard
- `xlcopy(data)` will copy Julia data into your clipboard for pasting into Excel.

Also note related packages, such as [XLSX.jl](https://felipenoris.github.io/XLSX.jl/stable/) for working with Excel files.

## Documentation

Full documentation is [available here](https://JuliaActuary.github.io/ActuaryUtilities.jl/stable/).

## Examples

### Quickstart 

```julia
bond_cfs = [5, 5, 105]
times    = [1, 2, 3]

discount_rate = 0.03

present_value(discount_rate, cfs, times)           # 105.65
duration(Macaulay(), discount_rate, cfs, times)    #   2.86
duration(discount_rate, cfs, times)                #   2.78
convexity(discount_rate, cfs, times)               #  10.62
```

### Interactive, basic cashflow analysis

See [JuliaActuary.org for instructions](https://juliaactuary.org/tutorials/cashflowanalysis/) on running this example.

[![Simple cashflow analysis with ActuaryUtilities.jl](https://user-images.githubusercontent.com/711879/95857181-d646a280-0d20-11eb-8300-a4c226021334.gif)](https://juliaactuary.org/tutorials/cashflowanalysis/)



## Useful tips

Functions often use a mix of interest_rates, cashflows, and timepoints. When calling functions, the general order of the arguments is 1) interest rates, 2) cashflows, and 3) timepoints. 
