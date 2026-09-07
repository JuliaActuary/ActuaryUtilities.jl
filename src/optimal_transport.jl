module OptimalTransport

import ..Distributions
import ..StatsBase
import ..RiskMeasures
import ..RiskMeasures: RiskMeasure, CTE, VaR
import ..QuadGK
import Statistics

export wasserstein, transportmap, pushforward, robustvalue

# ── One-dimensional optimal transport is closed form ────────────────────────
#
# Everything in this module rests on a single fact: in one dimension the optimal
# transport between two laws is rank matching, so the p-Wasserstein distance is
# the L^p gap between quantile functions and needs no solver. For two equally
# sized samples that reduces to sorting both and comparing point-by-point.

# Build a quantile function `u -> Q(u)` for either a sample or a distribution,
# doing any sorting ONCE up front. For a sample this is the inverse ECDF
# Q(u) = x_(⌈u·n⌉) — a step function, NOT `Statistics.quantile`: its default
# interpolation describes a different, continuous distribution, which breaks the
# exact OT identities at atoms (e.g. it would push [1,2] onto [12.5,17.5] instead
# of [10,20] for target [10,20]).
_quantile_fn(d::Distributions.UnivariateDistribution) = u -> Distributions.quantile(d, u)
function _quantile_fn(x::AbstractVector{<:Real})
    xs = sort(collect(x))
    n = length(xs)
    return u -> xs[clamp(ceil(Int, u * n), 1, n)]
end

_pnorm(diffs, p) = isinf(p) ? maximum(abs, diffs) :
    (Statistics.mean(abs.(diffs) .^ p))^(1 / p)

"""
    wasserstein(a, b; p=1, rtol=1e-6, atol=0, maxevals=nothing)

The `p`-Wasserstein (optimal transport) distance between two one-dimensional
risks. Each of `a`, `b` may be a sample (`AbstractVector{<:Real}`) or a
`Distributions.UnivariateDistribution`.

In one dimension optimal transport is closed form: the distance is the ``L^p``
norm of the gap between the two quantile functions,

```math
W_p(a,b) = \\left( \\int_0^1 |Q_a(u) - Q_b(u)|^p \\, du \\right)^{1/p} .
```

For two equally sized samples this is exactly the sorted, point-by-point matching
`mean(abs.(sort(a) .- sort(b)).^p)^(1/p)`; for unequally sized samples the two
step quantile functions are integrated exactly over their merged probability
breakpoints — in both cases no solver is required and no approximation is made.
Unlike
divergence-based distances (KL, ``\\chi^2``) the Wasserstein distance is
expressed in the units of the risk itself and is aware of the geometry of the
outcome space: a \\\$10k loss is closer to \\\$11k than to \\\$1M.

## Examples

```julia-repl
julia> wasserstein([1, 2, 3], [4, 5, 6])            # a rigid \$3 shift
3.0

julia> wasserstein(Normal(0, 1), Normal(3, 1))      # translation ⇒ W_p = 3
3.0

julia> wasserstein(Normal(0, 1), Normal(0, 2); p=2) # same mean, |σ₁-σ₂|
1.0
```

See also [`transportmap`](@ref), [`robustvalue`](@ref).

For risks involving a distribution, the quantile integral is evaluated adaptively.
`rtol`, `atol`, and `maxevals` control that calculation. By default the
evaluation budget scales with the number of empirical quantile segments; an
explicit `maxevals` is a hard cap. If the requested tolerance cannot be verified,
`wasserstein` throws rather than returning an unconverged approximation.
For distributional `p=Inf`, this includes empirical-versus-bounded-distribution
cases whose supremum is not certified within the evaluation budget. Infinite
distance is returned only when proved by support bounds or a supported analytic
family; unresolved same-side-unbounded pairs throw rather than relying on tail
growth heuristics.

## References
- "Optimal Transport for Actuarial Science", Arthur Charpentier, 2026.
  [⟨hal-05684645⟩](https://hal.science/hal-05684645)
"""
function wasserstein(
        a, b; p::Real = 1, rtol::Real = 1.0e-6,
        atol::Real = 0, maxevals::Union{Nothing, Integer} = nothing
    )
    (p >= 1 && (isfinite(p) || p == Inf)) ||
        throw(ArgumentError("p must be finite and ≥ 1, or Inf; got $p"))
    (isfinite(rtol) && rtol >= 0) || throw(ArgumentError("rtol must be finite and nonnegative"))
    (isfinite(atol) && atol >= 0) || throw(ArgumentError("atol must be finite and nonnegative"))
    (isnothing(maxevals) || maxevals > 0) || throw(ArgumentError("maxevals must be positive"))
    return _wasserstein(a, b, p; rtol, atol, maxevals)
end

# both samples: exact in both cases. Equal sizes reduce to sorted point-by-point
# matching; unequal sizes integrate |Qa - Qb|^p exactly by merging the breakpoints
# {i/na} ∪ {j/nb} of the two inverse-ECDF step functions. A fixed evaluation grid
# (or an interpolated quantile) computes a different number — e.g. the true
# W₂([0], [0,2]) is √2, while a size-2 midpoint grid through `Statistics.quantile`
# gives √1.25.
function _wasserstein(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}, p; kwargs...)
    na, nb = length(a), length(b)
    as, bs = sort(collect(a)), sort(collect(b))
    na == nb && return _pnorm(as .- bs, p)
    i = j = 1
    prev = 0.0
    acc = 0.0
    while i <= na && j <= nb
        u = min(i / na, j / nb)
        gap = abs(as[i] - bs[j])
        acc = isinf(p) ? max(acc, gap) : acc + (u - prev) * gap^p
        prev = u
        i / na <= u && (i += 1)
        j / nb <= u && (j += 1)
    end
    return isinf(p) ? acc : acc^(1 / p)
end

# A divergent endpoint integral has tail-shell masses that do not tend to zero.
# This deliberately conservative detector only reports divergence when the last
# five dyadic shells are essentially flat; ambiguous non-convergence remains an
# error rather than being converted to a finite number or to Inf.
function _divergent_quantile_tail(f)
    shells = map(20:30) do k
        ε = exp2(-k)
        width = ε / 2
        width * (f(3ε / 4) + f(1 - 3ε / 4))
    end
    tail = @view shells[(end - 4):end]
    return all(isfinite, tail) && minimum(tail) > 0 && minimum(tail) / maximum(tail) > 0.98
end

_quantile_breaks(::Distributions.UnivariateDistribution) = Float64[]
_quantile_breaks(x::AbstractVector) = collect((1:(length(x) - 1)) ./ length(x))

function _wasserstein_finite(a, b, p; rtol, atol, maxevals)
    Qa, Qb = _quantile_fn(a), _quantile_fn(b)
    f(u) = abs(Qa(u) - Qb(u))^p
    # Empirical quantiles jump at i/n. Supplying those known discontinuities as
    # interval boundaries lets QuadGK spend its error budget on the continuous
    # distribution quantile rather than repeatedly rediscovering every rank.
    points = sort!(unique!([0.0; 0.5; _quantile_breaks(a); _quantile_breaks(b); 1.0]))
    budget = isnothing(maxevals) ? max(100_000, 30 * (length(points) - 1)) : maxevals
    integral, err = QuadGK.quadgk(f, points; rtol, atol, maxevals = budget)
    tolerance = max(atol, rtol * abs(integral))
    if !isfinite(integral)
        return Inf
    elseif err <= tolerance
        return integral^(1 / p)
    elseif _divergent_quantile_tail(f)
        return Inf
    end
    throw(ErrorException("wasserstein quantile integration did not converge: estimated error $err exceeds tolerance $tolerance"))
end

# Distributional W∞ is the supremum quantile gap. The k/n grids are nested under
# doubling, so their maxima are monotone lower bounds. A finite answer is returned
# only after several refinements agree; unresolved tails fail loudly.
_support_bounds(d::Distributions.UnivariateDistribution) = (minimum(d), maximum(d))
_support_bounds(x::AbstractVector) = extrema(x)

function _support_proves_infinite(a, b)
    alo, ahi = _support_bounds(a)
    blo, bhi = _support_bounds(b)
    return xor(isfinite(alo), isfinite(blo)) || xor(isfinite(ahi), isfinite(bhi))
end

function _wasserstein_infinity(a::Distributions.Normal, b::Distributions.Normal; kwargs...)
    return a.σ == b.σ ? abs(a.μ - b.μ) : Inf
end

function _wasserstein_infinity(a::Distributions.LogNormal, b::Distributions.LogNormal; kwargs...)
    return Distributions.params(a) == Distributions.params(b) ? 0.0 : Inf
end

function _wasserstein_infinity(a::Distributions.Cauchy, b::Distributions.Cauchy; kwargs...)
    return a.σ == b.σ ? abs(a.μ - b.μ) : Inf
end

function _wasserstein_infinity(a, b; rtol, atol, maxevals)
    _support_proves_infinite(a, b) && return Inf
    alo, ahi = _support_bounds(a)
    blo, bhi = _support_bounds(b)
    endpoint_gap = zero(float(promote_type(typeof(alo), typeof(ahi), typeof(blo), typeof(bhi))))
    isfinite(alo) && isfinite(blo) && (endpoint_gap = max(endpoint_gap, abs(alo - blo)))
    isfinite(ahi) && isfinite(bhi) && (endpoint_gap = max(endpoint_gap, abs(ahi - bhi)))
    Qa, Qb = _quantile_fn(a), _quantile_fn(b)
    previous = -Inf
    stable = 0
    evaluations = 0
    n = 64
    budget = isnothing(maxevals) ? 100_000 : maxevals
    while evaluations + n - 1 <= budget
        interior = maximum(1:(n - 1)) do k
            u = k / n
            abs(Qa(u) - Qb(u))
        end
        current = max(endpoint_gap, interior)
        evaluations += n - 1
        !isfinite(current) && return Inf
        tolerance = max(atol, rtol * abs(current))
        if current - previous <= tolerance
            stable += 1
            stable >= 3 && return current
        else
            stable = 0
        end
        previous = current
        n *= 2
    end
    throw(ErrorException("wasserstein W∞ supremum search did not converge within maxevals=$budget"))
end

# At least one argument is a distribution: use verified adaptive computation.
function _wasserstein(a, b, p; rtol, atol, maxevals)
    return isinf(p) ? _wasserstein_infinity(a, b; rtol, atol, maxevals) :
        _wasserstein_finite(a, b, p; rtol, atol, maxevals)
end

"""
    transportmap(source, target)

Return the rank-preserving optimal transport map `T`, where each of `source`,
`target` may be a sample or a `Distributions.UnivariateDistribution`. `T` carries
a value at rank ``u = F_{source}(x)`` to the corresponding quantile of `target`:

```math
T(x) = Q_{target}(F_{source}(x)) .
```

In one dimension this monotone map is the (Brenier) optimal transport map. Pushing
a `source` sample through `T` — see [`pushforward`](@ref) — yields a sample
distributed as `target` while preserving each observation's rank, which makes `T`
a natural, auditable **stress / scenario transform**: one map revalues every
quantile consistently rather than reshuffling which outcomes are risky.

When both `source` and `target` are samples, order statistics are matched through
the inverse empirical cdf, so for equally sized, tie-free samples
`pushforward(source, T)` reproduces `sort(target)` exactly. Two qualifications:
for unequally sized samples the result is the target's quantiles evaluated at the
source's ranks rather than a permutation of `target`, and tied `source` values
cannot be split by any deterministic map — every copy of a tied value transports
to the same target quantile.

## Example

```julia-repl
julia> T = transportmap(Normal(0, 1), Normal(3, 1));  # a rigid +3 shift

julia> T(0.0), T(1.0)
(3.0, 4.0)
```
"""
function transportmap(source, target)
    F = RiskMeasures.cdf_func(source)
    Qt = _quantile_fn(target)                  # sorts a sample `target` once, not per call
    return x -> Qt(F(x))
end

# When `source` is an empirical sample, the plain ecdf reaches exactly 1.0 at the
# largest observation, so `Q_target(1)` would be `+Inf` for an unbounded `target`.
# Use midpoint plotting positions instead: the k-th of n order statistics maps to
# rank (k - 0.5)/n, keeping every transported value finite.
function transportmap(source::AbstractVector{<:Real}, target)
    xs = sort(collect(source))
    n = length(xs)
    Qt = _quantile_fn(target)                  # sorts a sample `target` once, not per call
    return function (x)
        k = searchsortedlast(xs, x)            # number of sample points ≤ x (0…n)
        u = clamp((k - 0.5) / n, 0.5 / n, 1 - 0.5 / n)
        return Qt(u)
    end
end

"""
    pushforward(sample, T)

Apply a transport map `T` (e.g. from [`transportmap`](@ref)) to every element of
`sample`, returning the transported (stressed) sample. Equivalent to `T.(sample)`.
"""
pushforward(sample, T) = map(T, sample)

# the natural tail level of a risk measure, where one exists
_taillevel(rm::CTE) = rm.α
_taillevel(rm::VaR) = rm.α
_taillevel(rm::RiskMeasure) =
    throw(ArgumentError("$(typeof(rm)) has no natural tail level; pass `tail`"))

"""
    robustvalue(rm::RiskMeasure, sample; radius, p=2, tail=<rm's α>)

A distributionally robust (Wasserstein-DRO) value of risk measure `rm` over the
`p`-Wasserstein ball of the given `radius` around the empirical distribution of
`sample`. `radius` answers "how bad could this number be if my book is off by up
to `radius` of transport cost?" and is a governance dial in the units of the
loss. What is returned depends on the measure:

- For `rm = CTE(α)` with `tail = α` (the default) the result is the **exact
  worst case** `CTE(α)(sample) + radius * (1-α)^(-1/p)`, attaining the sharp
  stability bound

  ```math
  |\\mathrm{CTE}_\\alpha(\\mu) - \\mathrm{CTE}_\\alpha(\\nu)|
      \\le (1-\\alpha)^{-1/p}\\, W_p(\\mu,\\nu) .
  ```

  The maximizing distribution moves exactly the worst `1-α` of probability mass
  outward by `radius * (1-α)^(-1/p)`, splitting the atom the tail boundary cuts
  through, and sits on the boundary of the ball.

- For any other risk measure the result is `rm` evaluated on a concrete adverse
  scenario *inside* the ball: the tail order statistics ``x_{(k)}, \\dots,
  x_{(n)}`` are shifted outward by `radius * (m/n)^(-1/p)`, where `m/n` is the
  fraction of observations shifted, so the scenario costs exactly `radius` in
  ``W_p``. The starting rank ``k`` depends on the measure. For [`VaR`](@ref),
  ``k`` is the rank VaR itself selects: the smallest ``k`` with ``k/n \\ge``
  `tail`. For every other measure, ``k`` is the first rank with strictly
  positive [`CTE`](@ref) tail weight: the smallest ``k`` with ``k/n >`` `tail`.
  The two rules differ exactly at atom boundaries. This is a **lower bound** on
  the true worst case over the ball, not the supremum itself — treat it as a
  principled adverse scenario, not a proven maximum.

Because the exact CTE branch applies only when `tail == rm.α`, changing `tail`
across that equality switches between the atom-splitting exact bound and the
whole-atom adverse scenario; the returned value need not be continuous there.

The factor `(1 - tail)^(-1/p)` is the price of tail focus: deeper-tail capital is
intrinsically more fragile to model error, so the same `radius` buys a larger
loading at `CTE(0.995)` than at `CTE(0.95)`.

## Example

```julia-repl
julia> s = rand(LogNormal(log(1000) - 0.18, 0.6), 200_000);

julia> CTE(0.95)(s), robustvalue(CTE(0.95), s; radius=250)   # ≈ (2980, 4100)
```

`robustvalue` takes the risk measure as an argument, so it works unchanged for
`VaR`, `WangTransform`, or any custom `RiskMeasure` (supply `tail` for measures
without a natural tail level).
"""
function robustvalue(
        rm::RiskMeasure, sample::AbstractVector{<:Real};
        radius::Real, p::Real = 2, tail::Real = _taillevel(rm)
    )
    radius >= 0 || throw(ArgumentError("radius must be ≥ 0, got $radius"))
    0 <= tail < 1 || throw(ArgumentError("tail must be in [0, 1), got $tail"))
    p >= 1 || throw(ArgumentError("p must be ≥ 1, got $p"))
    # CTE at its own tail level has a closed-form worst case. Its maximizer moves
    # exactly mass 1-α, splitting the atom the tail boundary cuts through — a
    # fractional weight an equally weighted sample cannot represent: any whole-atom
    # shift either overspends the W_p budget or under-attains the bound whenever
    # n*α is not an integer.
    rm isa CTE && tail == rm.α && return rm(sample) + radius * (1 - tail)^(-1 / p)
    # Otherwise evaluate `rm` on a budget-exact adverse scenario: shift the tail
    # order statistics x_(k)…x_(n) outward by Δ sized to the mass m/n actually
    # moved, so the scenario costs exactly `radius` in W_p and stays inside the
    # ball. VaR's scenario must move the rank VaR itself selects (first k with
    # k/n ≥ tail, the lower quantile); other measures shift the strict tail
    # (first rank with positive CTE tail weight, k/n > tail). The two rules
    # differ exactly at atom boundaries: with n=10 and tail=0.5, shifting only
    # ranks 6:10 would leave VaR at rank 5 unmoved and buy zero loading.
    xs = sort!(float.(sample))                 # promote: Int samples cannot hold x + Δ
    n = length(xs)
    k = rm isa VaR ? RiskMeasures._first_index_at_or_above(n, tail) :
        RiskMeasures._first_index_above(n, tail)
    m = n - k + 1
    Δ = radius * (m / n)^(-1 / p)
    @views xs[k:n] .+= Δ
    return rm(xs)
end

end
