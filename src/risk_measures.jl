module RiskMeasures
import ..Distributions
import ..StatsBase
import ..QuadGK

export Expectation, VaR, ValueAtRisk, CTE, ConditionalTailExpectation, WangTransform, DualPower, ProportionalHazard

abstract type RiskMeasure end

"""
    g(rm::RiskMeasure, x)

The probability distortion function associated with the given risk measure.
The argument `x` is a survival probability in `[0, 1]`. The result is the
distorted survival probability.

See [Distortion Function g(u)](@ref)
"""
function g(rm::RiskMeasure, x) end

"""
    gbar(rm::RiskMeasure, F)

The complementary distortion function. It satisfies `gbar(rm, F) == 1 - g(rm, 1 - F)`.
The argument `F` is a cumulative probability in `[0, 1]`.

The generic fallback computes `1 - g(rm, 1 - F)` directly. That expression can
lose all precision when `F` is small, because `1 - F` rounds to `1`. Specialize
`gbar` for a custom risk measure when a stable form exists. This package
specializes `gbar` for every built-in measure.
"""
gbar(rm::RiskMeasure, F) = 1 - g(rm, 1 - F)

"""
    Expectation()::RiskMeasure
    Expectation()(risk)::T (where T is the type of values sampled in `risk`)

The expected value of the risk.

`Expectation()` returns a functor which can then be called on a risk distribution.

For a `Distributions.UnivariateDistribution`, the result is the analytic
`Distributions.mean` when the distribution defines one. The mean's conventions
pass through: the result is `NaN` when the mean does not exist (for example
`Cauchy`), and `±Inf` when the mean diverges (for example `Pareto` with tail
index at or below 1). A finite-support discrete distribution is evaluated as an
exact weighted sum. A distribution without its own `mean` method (for example a
generic `truncated` wrapper) is evaluated by checked numerical quadrature; that
path throws an error when the quadrature error cannot be verified.

## Examples

```julia-repl
julia> Expectation()(rand(1000))
0.4793223308812537

julia> rm = Expectation()
ActuaryUtilities.RiskMeasures.Expectation()

julia> rm(rand(1000))
0.4941708036889741
```
"""
struct Expectation <: RiskMeasure end
g(rm::Expectation, x) = x
gbar(rm::Expectation, F) = F

"""
     VaR(α)::RiskMeasure
     VaR(α)(risk)::T (where T is the type of values sampled in `risk`)

The Value at Risk at level `α`: the lower generalized inverse of the
distribution function,

``\\mathrm{VaR}_\\alpha(X) = \\inf\\{x : F_X(x) \\ge \\alpha\\}.``

`risk` can be a univariate distribution or an array of outcomes.
Assumes more positive values are higher risk measures, so a higher `α` will
return a more positive number.

- For a distribution, the result is `Distributions.quantile(risk, α)`. The
  result keeps `quantile`'s numeric type (for example an integer for a count
  distribution).
- For an array of `n` outcomes, the result is the first order statistic whose
  rank `k` satisfies `k/n ≥ α`.
- At an atom of the distribution, the lower quantile applies. Example:
  `VaR(0.5)(Bernoulli(0.5)) == 0`.
- `VaR(0)` is the essential infimum: the sample minimum for an array, and
  `quantile(risk, 0)` for a distribution (`-Inf` when the support is unbounded
  below).

`VaR(α)` returns a functor which can then be called on a risk distribution.

## Parameters
- α: [0,1.0)

## Examples

```julia-repl
julia> VaR(0.95)(rand(1000))
0.9561843082268024

julia> rm = VaR(0.95)
VaR{Float64}(0.95)

julia> rm(rand(1000))
0.9597070153670079
```
"""
struct VaR{T<:Real} <: RiskMeasure
    α::T

    function VaR(α::T) where {T}
        @assert 0 <= α < 1 "α of $α is not 0 ≤ α < 1"
        return new{T}(α)
    end
end
# The boundary uses `<=` so that the Choquet form of this distortion selects the
# lower quantile at an atom, matching `Distributions.quantile`.
g(rm::VaR, x) = x <= (1 - rm.α) ? 0 : 1
gbar(rm::VaR, F) = F < rm.α ? 0 : 1

"""
[`VaR`](@ref)
"""
const ValueAtRisk = VaR

"""
    CTE(α)::RiskMeasure
    CTE(α)(risk)::T (where T is the type of values sampled in risk)

The Conditional Tail Expectation (also called expected shortfall or average
Value at Risk) at level `α`: the mean of exactly the worst `1 - α` of the
probability mass. When the `α` boundary falls inside an atom, that atom
contributes fractional weight, so the averaged mass is always exactly `1 - α`.
CTE's atom handling does not depend on the `VaR` boundary convention.

`risk` can be a univariate distribution or an array of outcomes.
Assumes more positive values are higher risk measures, so a higher `α` will
return a more positive number.

- `CTE(0)` is the mean, with the same conventions as [`Expectation`](@ref).
- The result is `+Inf` when the upper tail expectation diverges. The
  implementation reads the distribution's `mean` to decide this: `mean == +Inf`
  means the upper tail diverges; `mean` of `NaN` means both tails diverge.
  These are the `Distributions.jl` conventions. A `mean` of `-Inf` does not
  trigger this shortcut, because the upper tail can still be finite.
- Arrays and finite-support discrete distributions are evaluated as exact
  weighted sums. Continuous distributions use a checked quadrature of the
  quantile function. The checked path throws an error when the quadrature
  error cannot be verified against the requested tolerance.

CTE(α) returns a functor which can then be called on a risk distribution.

## Parameters

- α: [0,1.0)

## Examples

```julia-repl
julia> CTE(0.95)(rand(1000))
0.9766218612020593

julia> rm = CTE(0.95)
CTE{Float64}(0.95)

julia> rm(rand(1000))
0.9739835010268733
```
"""
struct CTE{T<:Real} <: RiskMeasure
    α::T

    function CTE(α::T) where {T}
        @assert 0 <= α < 1 "α of $α is not 0 ≤ α < 1"
        return new{T}(α)
    end
end
g(rm::CTE, x) = x < (1 - rm.α) ? x / (1 - rm.α) : 1
gbar(rm::CTE, F) = F <= rm.α ? zero(F / (1 - rm.α)) : (F - rm.α) / (1 - rm.α)

"""
[`CTE`](@ref)
"""
const ConditionalTailExpectation = CTE

"""
    WangTransform(α)::RiskMeasure
    WangTransform(α)(risk)::T (where T is the type of values sampled in risk)

The Wang Transform is a distortion risk measure that transforms the cumulative distribution function (CDF) of the risk distribution using a normal distribution with mean Φ⁻¹(α) and standard deviation 1. risk can be a univariate distribution or an array of outcomes.

WangTransform(α) returns a functor which can then be called on a risk distribution.

For a distribution that is not discrete, the value is computed by checked
numerical quadrature. That path throws an error when the defining integral
cannot be verified to converge, for example for heavy-tailed risks.

## Parameters
- α: [0,1.0]

In the literature, sometimes λ is used where ``\\lambda = \\Phi^{-1}(\\alpha)``.


## Examples

```julia-repl
julia> WangTransform(0.95)(rand(1000))
0.8799465543360105

julia> rm = WangTransform(0.95)
WangTransform{Float64}(0.95)

julia> rm(rand(1000))
0.8892245759705852
```

## References
- "A Risk Measure That Goes Beyond Coherence", Shaun S. Wang, 2002
"""
struct WangTransform{T} <: RiskMeasure
    α::T
    function WangTransform(α::T) where {T}
        @assert 0 < α < 1 "α of $α is not 0 < α < 1"
        return new{T}(α)
    end
end
function g(rm::WangTransform, x)
    Φ_inv(x) = Distributions.quantile(Distributions.Normal(), x)
    Distributions.cdf(Distributions.Normal(), Φ_inv(x) + Φ_inv(rm.α))
end
function gbar(rm::WangTransform, F)
    Φ_inv(x) = Distributions.quantile(Distributions.Normal(), x)
    Distributions.cdf(Distributions.Normal(), Φ_inv(F) - Φ_inv(rm.α))
end

"""
    DualPower(v)::RiskMeasure
    DualPower(v)(risk)::T (where T is the type of values sampled in risk)

The Dual Power distortion risk measure is defined as ``1 - (1 - x)^v``, where x is the cumulative distribution function (CDF) of the risk distribution and v is a positive parameter. risk can be a univariate distribution or an array of outcomes.

DualPower(v) returns a functor which can then be called on a risk distribution.

For a distribution that is not discrete, the value is computed by checked
numerical quadrature. That path throws an error when the defining integral
cannot be verified to converge, for example for heavy-tailed risks.
"""
struct DualPower{T} <: RiskMeasure
    v::T
    function DualPower(v::T) where {T}
        (isfinite(v) && v > 0) || throw(ArgumentError("v must be finite and strictly positive, got $v"))
        return new{T}(v)
    end
end
DualPower{T}(v) where {T} = DualPower(convert(T, v))
# `-expm1(v * log1p(-x))` equals `1 - (1 - x)^v` but stays accurate for tiny x,
# where the direct form rounds to a hard zero and silently invalidates the
# quadrature error estimate.
g(rm::DualPower, x) = -expm1(rm.v * log1p(-x))
gbar(rm::DualPower, F) = F^rm.v

"""
    ProportionalHazard(y)::RiskMeasure
    ProportionalHazard(y)(risk)::T (where T is the type of values sampled in risk)

The Proportional Hazard distortion risk measure is defined as ``x^(1/y)``, where x is the cumulative distribution function (CDF) of the risk distribution and y is a positive parameter. risk can be a univariate distribution or an array of outcomes.
ProportionalHazard(y) returns a functor which can then be called on a risk distribution.

For a distribution that is not discrete, the value is computed by checked
numerical quadrature. That path throws an error when the defining integral
cannot be verified to converge, for example for heavy-tailed risks.

## Examples

```julia-repl
julia> ProportionalHazard(2)(rand(1000))
0.6659603556774121

julia> rm = ProportionalHazard(2)
ProportionalHazard{Int64}(2)

julia> rm(rand(1000))
0.6710587338367799
```
"""
struct ProportionalHazard{T} <: RiskMeasure
    y::T
    function ProportionalHazard(y::T) where {T}
        (isfinite(y) && y > 0) || throw(ArgumentError("y must be finite and strictly positive, got $y"))
        return new{T}(y)
    end
end
ProportionalHazard{T}(y) where {T} = ProportionalHazard(convert(T, y))
g(rm::ProportionalHazard, x) = x^(1 / rm.y)
gbar(rm::ProportionalHazard, F) = -expm1(log1p(-F) / rm.y)

# ── Distortion-measure evaluation ───────────────────────────────────────────
#
# Definition 4.2 of "A Risk Measure that Goes Beyond Coherence", Wang 2002:
# the Choquet integral
#
#   ρ(X) = ∫₀^∞ g(S(x)) dx − ∫_{−∞}^0 ḡ(F(x)) dx,   ḡ(F) = 1 − g(1 − F).
#
# Evaluation tiers, most exact first:
#   1. finite-support discrete laws → exact distortion-weighted sums
#   2. integer-valued laws bounded below (Poisson, Geometric, …) → checked
#      tail summation
#   3. everything else → checked quadrature whose error estimate is enforced.
# A risk whose distorted expectation cannot be verified throws instead of
# returning a silently wrong number.
function (rm::RiskMeasure)(risk)
    if risk isa Distributions.DiscreteUnivariateDistribution
        _finite_atoms(risk) && return _distorted_sum(rm, _atoms(risk)...)
        lo = minimum(risk)
        (eltype(risk) <: Integer && isfinite(lo)) && return _distorted_tail_sum(rm, risk, lo)
    end
    return _choquet(rm, risk)
end

# `hasfinitesupport` reports whether the support VALUES are bounded, so it is
# false for a DiscreteNonParametric with an atom at ±Inf even though the atom
# COUNT is finite. The exact sum only needs finitely many atoms.
_finite_atoms(d::Distributions.DiscreteUnivariateDistribution) =
    d isa Distributions.DiscreteNonParametric || Distributions.hasfinitesupport(d)

function _choquet(rm::RiskMeasure, risk; rtol=sqrt(eps(Float64)), atol=0.0, maxevals=10^7)
    (isfinite(rtol) && rtol >= 0) || throw(ArgumentError("rtol must be finite and nonnegative"))
    (isfinite(atol) && atol >= 0) || throw(ArgumentError("atol must be finite and nonnegative"))
    maxevals > 0 || throw(ArgumentError("maxevals must be positive"))
    G = ccdf_func(risk)   # hoisted: each closure is built once, not once per node
    F = cdf_func(risk)
    # Each side runs at half the requested tolerance. The two certificates then
    # add up to at most the requested tolerance at the components' scale, so the
    # final acceptance check below cannot fail on converged components.
    upper, uerr = QuadGK.quadgk(x -> g(rm, G(x)), 0, Inf; rtol=rtol / 2, atol=atol / 2, maxevals)
    lower, lerr = QuadGK.quadgk(x -> gbar(rm, F(x)), -Inf, 0; rtol=rtol / 2, atol=atol / 2, maxevals)
    _check_component(upper, uerr, rtol / 2, atol / 2)
    _check_component(lower, lerr, rtol / 2, atol / 2)
    result = upper - lower
    # Cancellation-aware final check. The result's absolute error is at most
    # uerr + lerr; certify it at the scale of the components, not only the
    # (possibly near-zero) difference. This certifies that the quadrature error
    # met the requested tolerance; it does not prove the integral exists.
    err_total = uerr + lerr
    tolerance = max(atol, rtol * abs(result), rtol * max(upper, lower))
    err_total <= tolerance || throw(ErrorException(
        "distortion risk measure quadrature could not verify the result: combined error estimate $err_total exceeds tolerance $tolerance"))
    return result
end

# Reject every unverifiable quadrature output: a NaN or infinite integral, a
# NaN or negative error estimate, or an error above tolerance. NaN comparisons
# are false, so `isfinite(err)` must be checked explicitly.
function _check_component(I, err, rtol, atol)
    (isfinite(I) && isfinite(err) && err >= 0 && err <= max(atol, rtol * abs(I))) ||
        throw(ErrorException(
            "distortion risk measure quadrature did not converge (integral ≈ $I, estimated error $err): " *
            "the distorted expectation may not exist for this risk (e.g. heavy tails such as Cauchy, or Pareto with tail index ≤ 1)"))
    return nothing
end

# ── Exact atomic evaluation ─────────────────────────────────────────────────
#
# For a purely atomic law {(xᵢ, pᵢ)} with sorted atoms, the Choquet integral is
# a finite distortion-weighted sum: with S(xᵢ⁻) = P(X ≥ xᵢ),
#
#   ρ = Σᵢ xᵢ · (g(S(xᵢ⁻)) − g(S(xᵢ))).
#
# No quadrature crosses the cdf's jump discontinuities, so the sum is exact.

function _atoms(d::Distributions.DiscreteUnivariateDistribution)
    xs = collect(Distributions.support(d))   # sorted; `collect` also handles Dirac's tuple
    return xs, Distributions.pdf.(d, xs)
end

function _distorted_sum(rm::RiskMeasure, xs, ps)
    total = sum(ps)
    (isfinite(total) && total > 0) || throw(ArgumentError("atom probabilities must have a positive finite sum, got $total"))
    n = length(xs)
    T = float(promote_type(eltype(xs), eltype(ps)))
    # Survival values are normalized by the actual total and clamped into
    # [0, 1]: even a pristine Binomial pdf sums to 1 + 7e-16, which would push
    # a reverse cumsum above 1 and break distortions such as Φ⁻¹.
    tailp = reverse!(cumsum(reverse(ps)))
    S(i) = i == 1 ? one(T) : i > n ? zero(T) : clamp(T(tailp[i]) / total, zero(T), one(T))
    return sum(eachindex(xs)) do i
        w = g(rm, S(i)) - g(rm, S(i + 1))
        # Skip zero weights before multiplying: an atom at ±Inf with zero
        # distorted weight must not turn the sum into NaN through Inf * 0.
        iszero(w) ? zero(T) : T(xs[i] * w)
    end
end

# Checked tail summation for integer-valued laws that are bounded below but not
# above (Poisson, Geometric, …). Integers outside the support carry zero
# probability, so their distortion weight is zero and they are skipped; the sum
# is therefore correct for any integer-valued law. The truncation point doubles
# until (a) the un-scanned distorted tail is provably negligible and (b) two
# successive doublings agree within `rtol`. A distorted tail whose sum does not
# stabilize within the atom budget throws.
function _distorted_tail_sum(rm::RiskMeasure, d, lo::Integer; rtol=sqrt(eps(Float64)), maxatoms=2^24)
    acc = 0.0
    consumed = 0
    N = 32
    previous = NaN   # NaN comparisons are false: the first pass never counts as stable
    stable = 0
    while N <= maxatoms
        for i in (consumed + 1):N
            x = lo + (i - 1)
            Sminus = i == 1 ? 1.0 : clamp(Distributions.ccdf(d, x - 1), 0.0, 1.0)
            S = clamp(Distributions.ccdf(d, x), 0.0, 1.0)
            w = g(rm, Sminus) - g(rm, S)
            iszero(w) || (acc += x * w)
        end
        consumed = N
        isfinite(acc) || break
        # The un-scanned atoms carry total distorted weight g(S(x_N)) at values
        # at or beyond x_N, so `tailguard` bounds their smallest possible
        # contribution. Stability alone is not enough: an all-zero prefix
        # (Poisson(1000) under Wang has ccdf == 1.0 for its first hundreds of
        # atoms) produces identical truncations long before any mass is seen.
        # A divergent distorted tail needs g(S(x)) ≳ 1/x infinitely often,
        # which keeps x·g(S(x)) away from zero — so divergence can never pass
        # this guard, and the budget-exhaustion throw below fires instead.
        xN = lo + (N - 1)
        wrem = abs(g(rm, clamp(Distributions.ccdf(d, xN), 0.0, 1.0)))
        tolerance = rtol * max(1.0, abs(acc))
        if wrem * max(1.0, abs(xN)) <= tolerance && abs(acc - previous) <= tolerance
            stable += 1
            stable >= 2 && return acc
        else
            stable = 0
        end
        previous = acc
        N *= 2
    end
    throw(ErrorException(
        "distortion risk measure tail sum did not converge within $maxatoms atoms: the distorted expectation may not exist for this risk"))
end

# ── Exact empirical specializations ─────────────────────────────────────────
#
# For a finite sample the Choquet integral reduces to order statistics:
# computing it by adaptive quadrature over the ecdf's step function is both
# orders of magnitude slower and exposes integration tolerance at the steps.
# These methods evaluate the same functional exactly.
#
# Two boundary rules serve different purposes and both are needed:
#   * `_first_index_above`      — smallest k with k/n > α. CTE uses it: the
#     first rank with strictly positive tail weight.
#   * `_first_index_at_or_above` — smallest k with k/n ≥ α. VaR uses it: the
#     lower quantile of the empirical distribution.
function _first_index_above(n, α)
    k = clamp(floor(Int, n * α) + 1, 1, n)
    # floating-point n*α can land on either side of an exact boundary; nudge so
    # that k is exactly the first index with k/n > α under float comparison
    while k > 1 && (k - 1) / n > α
        k -= 1
    end
    while k < n && k / n <= α
        k += 1
    end
    return k
end

function _first_index_at_or_above(n, α)
    k = clamp(ceil(Int, n * α), 1, n)
    # same float-boundary nudging, for the ≥ rule
    while k > 1 && (k - 1) / n >= α
        k -= 1
    end
    while k < n && k / n < α
        k += 1
    end
    return k
end

# The generic exact sum for an equally weighted sample serves the distortion
# measures without a dedicated fast path (WangTransform, DualPower,
# ProportionalHazard, user-defined). The VaR/CTE/Expectation methods below are
# more specific and intercept first.
function (rm::RiskMeasure)(risk::AbstractArray{<:Real})
    isempty(risk) && throw(ArgumentError("the risk sample must contain at least one outcome"))
    xs = sort(vec(risk))
    n = length(xs)
    T = float(eltype(xs))
    return sum(eachindex(xs)) do i
        w = g(rm, T(n - i + 1) / n) - g(rm, T(n - i) / n)
        iszero(w) ? zero(T) : T(xs[i] * w)
    end
end

function (rm::VaR)(risk::AbstractArray{<:Real})
    isempty(risk) && throw(ArgumentError("the risk sample must contain at least one outcome"))
    n = length(risk)
    k = _first_index_at_or_above(n, rm.α)
    return partialsort(vec(risk), k)
end

# The Choquet-CTE distorts the tail by 1/(1-α): the crossing order statistic
# x_(k) receives the partial weight (k/n - α), and each of x_(k+1)…x_(n)
# receives 1/n, all normalized by (1-α). (CTE(0) is then exactly the mean.)
# Elements are divided by n before summing so that large samples cannot
# overflow the accumulator.
function (rm::CTE)(risk::AbstractArray{<:Real})
    isempty(risk) && throw(ArgumentError("the risk sample must contain at least one outcome"))
    n = length(risk)
    α = rm.α
    k = _first_index_above(n, α)
    tail = partialsort(vec(risk), k:n)
    T = float(promote_type(eltype(risk), typeof(α)))
    partial = (T(k) / n - α) * first(tail)
    rest = sum(x -> T(x) / n, @view(tail[2:end]); init=zero(T))
    return (partial + rest) / (1 - α)
end

function (rm::Expectation)(risk::AbstractArray{<:Real})
    isempty(risk) && throw(ArgumentError("the risk sample must contain at least one outcome"))
    n = length(risk)
    return sum(x -> x / n, risk)
end

# ── Exact distributional specializations ────────────────────────────────────

# A distribution "has a mean" exactly when its `which` match is not Statistics'
# generic `mean(itr)` fallback (which a distribution cannot satisfy — it throws
# a MethodError at `iterate`). Errors raised inside a real `mean` method still
# propagate; only the absence of a specific method routes to quadrature.
_mean_available(d) =
    which(Distributions.mean, Tuple{typeof(d)}) !== which(Distributions.mean, Tuple{Any})

function (rm::Expectation)(risk::Distributions.UnivariateDistribution)
    if risk isa Distributions.DiscreteUnivariateDistribution && _finite_atoms(risk)
        # exact weighted sum; the identity distortion makes this Σ xᵢpᵢ with
        # zero-probability atoms skipped (an Inf atom with p = 0 must not
        # produce NaN, as `mean` would)
        return _distorted_sum(rm, _atoms(risk)...)
    end
    _mean_available(risk) && return Distributions.mean(risk)   # NaN / ±Inf pass through
    return _choquet(rm, risk)   # e.g. truncated wrappers without a `mean` method
end

# The VaR distortion collapses the Choquet integral to the lower α-quantile,
# which `Distributions.quantile` already implements for every distribution.
(rm::VaR)(risk::Distributions.UnivariateDistribution) = Distributions.quantile(risk, rm.α)

function (rm::CTE)(risk::Distributions.UnivariateDistribution)
    α = rm.α
    if risk isa Distributions.DiscreteUnivariateDistribution && _finite_atoms(risk)
        return _distorted_sum(rm, _atoms(risk)...)   # exact fractional-atom tail sum
    end
    m = _mean_available(risk) ? Distributions.mean(risk) : nothing
    if iszero(α)
        # CTE(0) is the mean. This runs before any divergence shortcut so that
        # CTE(0)(Cauchy()) is NaN, matching Expectation.
        return isnothing(m) ? _choquet(rm, risk) : m
    end
    if !isnothing(m) && (m == Inf || isnan(m))
        # Distributions.jl conventions: mean == +Inf means the upper tail
        # diverges; mean == NaN means both tails diverge. Either way
        # E[X⁺] = ∞, so CTE_α = +∞ for α > 0. A mean of -Inf is NOT a
        # shortcut: only the lower tail diverges and the CTE stays finite.
        return Inf
    end
    if risk isa Distributions.DiscreteUnivariateDistribution
        lo = minimum(risk)
        (eltype(risk) <: Integer && isfinite(lo)) && return _distorted_tail_sum(rm, risk, lo)
    end
    # Expected shortfall as a single one-sided quantile integral:
    #   CTE_α = (1/(1-α)) ∫_α^1 quantile(u) du.
    # There is no subtraction, so no cancellation: the certified error below
    # controls the error of the final result after dividing by (1-α).
    rtol, atol, maxevals = sqrt(eps(Float64)), 0.0, 10^7
    I, err = QuadGK.quadgk(u -> Distributions.quantile(risk, u), α, 1; rtol, atol, maxevals)
    (isfinite(I) && isfinite(err) && err >= 0 && err <= max(atol * (1 - α), rtol * abs(I))) ||
        throw(ErrorException(
            "CTE quantile integral did not converge (integral ≈ $I, estimated error $err): E[X⁺] may be infinite for this risk"))
    return I / (1 - α)
end

"""
    cdf_func(risk)

Returns the appropriate cumulative distribution function depending on the type, specifically:

    cdf_func(S::AbstractArray{<:Real}) = StatsBase.ecdf(S)
    cdf_func(S::Distributions.UnivariateDistribution) = x -> Distributions.cdf(S, x)

"""
cdf_func(S::AbstractArray{<:Real}) = StatsBase.ecdf(S)
cdf_func(S::Distributions.UnivariateDistribution) = x -> Distributions.cdf(S, x)

"""
    ccdf_func(risk)

Survival-function counterpart of [`cdf_func`](@ref):

    ccdf_func(S::Distributions.UnivariateDistribution) = x -> Distributions.ccdf(S, x)

Other risk types fall back to `x -> 1 - F(x)` with `F = cdf_func(S)` built
once. The distribution method uses `ccdf` directly because `1 - cdf(x)` loses
all precision in the far tail, where `cdf(x)` rounds to `1`.
"""
ccdf_func(S::Distributions.UnivariateDistribution) = x -> Distributions.ccdf(S, x)
function ccdf_func(S)
    F = cdf_func(S)
    return x -> 1 - F(x)
end

end
