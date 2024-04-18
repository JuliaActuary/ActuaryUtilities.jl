# Financial Math Submodule

Provides a set of common routines in financial maths.

## Quickstart

```julia
cfs = [5, 5, 105]
times    = [1, 2, 3]

discount_rate = 0.03

present_value(discount_rate, cfs, times)           # 105.65
duration(Macaulay(), discount_rate, cfs, times)    #   2.86
duration(discount_rate, cfs, times)                #   2.78
convexity(discount_rate, cfs, times)               #  10.62
```


## API

### Exported API
```@autodocs
Modules = [ActuaryUtilities.FinancialMath]
Private = false
```

### Unexported API
```@autodocs
Modules = [ActuaryUtilities.FinancialMath]
Public = false
```