## Measure markers
# The shared vocabulary every sensitivity method dispatches on, including the `KeyRates` grid.

struct Macaulay end
struct Modified end
"""
    DV01

Signed dollar risk for a one-basis-point (0.01%) parallel rate shift:
`DV01 = -∂V/∂r / 10000`. A positive DV01 of 0.045 means a 1bp rate increase
reduces the position's value by approximately 0.045, in the cashflows' currency units.

See also: [`IR01`](@ref), [`CS01`](@ref)
"""
struct DV01 end

"""
    IR01

Interest Rate 01: signed dollar risk for a one-basis-point parallel shift in the
risk-free (base) curve, holding the credit curve fixed, of a valuation callback that
receives `(base, credit)`: `duration(valuation, IR01(), base, credit)`.

For fixed cashflows discounted at `base + credit`, IR01, CS01 and the combined curve's
DV01 are equal, so use `duration(DV01(), base + credit, cfs, times)`. For contracts, use
`Effective()` and `Spread()`.

See also: [`CS01`](@ref), [`DV01`](@ref)
"""
struct IR01 end

"""
    CS01

Credit Spread 01: signed dollar risk for a one-basis-point parallel shift in the
credit curve, holding the risk-free (base) curve fixed, of a valuation callback that
receives `(base, credit)`: `duration(valuation, CS01(), base, credit)`.

For fixed cashflows discounted at `base + credit`, CS01, IR01 and the combined curve's
DV01 are equal, so use `duration(DV01(), base + credit, cfs, times)`. For contracts, use
`Effective()` and `Spread()`.

See also: [`IR01`](@ref), [`DV01`](@ref)
"""
struct CS01 end

"""
    Effective

Measure contract risk while reprojecting cashflows under shifted curves, so
floating coupons reset. Use `duration(Effective(), contract, curve)`; the same
marker applies to `dv01` and `convexity`. `Modified` and `Macaulay` operate on
fixed cashflows.

See also: [`Spread`](@ref), [`sensitivities`](@ref), [`locked_floater`](@ref).
"""
struct Effective end

"""
    Spread

Spread (credit) duration: bumps the discount curve only; cashflows projected on the
index curve stay fixed. For a floating-rate bond it is close to the duration of a
fixed-rate bond with the same maturity.

See also: [`Effective`](@ref), [`sensitivities`](@ref).
"""
struct Spread end

"""
    KeyRates(tenors)

Select the tenor grid for key-rate [`duration`](@ref), [`convexity`](@ref), and
[`sensitivities`](@ref). Results contain per-tenor vectors and convexity matrices.
`tenors` must be a nonempty `AbstractVector{<:Real}` of finite, positive, strictly
increasing knot times in years. `KeyRates` keeps its own 1-based copy of them.

```julia
tenors = [1.0, 2.0, 5.0, 10.0, 30.0]
duration(KeyRates(tenors), curve, cfs, times)            # vector of key-rate durations
duration(DV01(), KeyRates(tenors), curve, cfs, times)    # vector of key-rate DV01s
convexity(KeyRates(tenors), curve, cfs, times)           # matrix of key-rate convexities
sensitivities(KeyRates(tenors), curve, cfs, times)       # value + durations + convexities
```

See also: [`DV01`](@ref), [`IR01`](@ref), [`CS01`](@ref)
"""
struct KeyRates{T <: Real}
    tenors::Vector{T}
    function KeyRates(tenors::AbstractVector{<:Real})
        grid = collect(tenors)
        _validate_tenors(grid)
        return new{eltype(grid)}(grid)
    end
end

function _validate_tenors(tenors::AbstractVector{<:Real})
    isempty(tenors) && throw(ArgumentError("KeyRates tenors must be non-empty"))
    # Strict increase establishes sortedness and uniqueness in one pass.
    previous = zero(first(tenors))
    for t in tenors
        isfinite(t) && t > previous || throw(ArgumentError("KeyRates tenors must be finite, strictly positive, and strictly increasing"))
        previous = t
    end
    return tenors
end
