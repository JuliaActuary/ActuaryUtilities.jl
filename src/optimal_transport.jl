module OptimalTransport

import ..Distributions
import ..StatsBase
import ..RiskMeasures
import ..RiskMeasures: RiskMeasure, CTE, VaR
import Random
import Statistics

export wasserstein, transportmap, pushforward, driftsignificance, worstcase

# ── One-dimensional optimal transport is closed form ────────────────────────
#
# Everything in this module rests on a single fact: in one dimension the optimal
# transport between two laws is rank matching, so the p-Wasserstein distance is
# the L^p gap between quantile functions and needs no solver. For two equally
# sized samples that reduces to sorting both and comparing point-by-point.

# Build a quantile function `u -> Q(u)` for either a sample or a distribution,
# doing any sorting ONCE up front. Callers evaluate it on a whole grid of `u`s, so
# a per-call `Statistics.quantile` (which re-sorts the vector every time) would be
# quadratic; sorting once here keeps the grid loop linear.
_quantile_fn(d::Distributions.UnivariateDistribution) = u -> Distributions.quantile(d, u)
function _quantile_fn(x::AbstractVector{<:Real})
    isempty(x) && throw(ArgumentError("empty sample: a Wasserstein/quantile needs at least one observation"))
    xs = sort(collect(x))
    return u -> Statistics.quantile(xs, u; sorted=true)
end

_pnorm(diffs, p) = (Statistics.mean(abs.(diffs) .^ p))^(1 / p)

"""
    wasserstein(a, b; p=1)

The `p`-Wasserstein (optimal transport) distance between two one-dimensional
risks. Each of `a`, `b` may be a sample (`AbstractVector{<:Real}`) or a
`Distributions.UnivariateDistribution`.

In one dimension optimal transport is closed form: the distance is the ``L^p``
norm of the gap between the two quantile functions,

```math
W_p(a,b) = \\left( \\int_0^1 |Q_a(u) - Q_b(u)|^p \\, du \\right)^{1/p} .
```

For two equally sized samples this is exactly the sorted, point-by-point matching
`mean(abs.(sort(a) .- sort(b)).^p)^(1/p)` — no solver is required. Unlike
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

See also [`transportmap`](@ref), [`driftsignificance`](@ref), [`worstcase`](@ref).
"""
function wasserstein(a, b; p::Real=1)
    p >= 1 || throw(ArgumentError("p must be ≥ 1, got $p"))
    _wasserstein(a, b, p)
end

# both samples: exact sorted matching when equally sized, else compare the two
# empirical quantile functions on a shared midpoint grid.
function _wasserstein(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}, p)
    (isempty(a) || isempty(b)) &&
        throw(ArgumentError("empty sample: wasserstein needs at least one observation in each of `a`, `b`"))
    if length(a) == length(b)
        return _pnorm(sort(collect(a)) .- sort(collect(b)), p)
    end
    n = max(length(a), length(b))
    us = ((1:n) .- 0.5) ./ n
    Qa, Qb = _quantile_fn(a), _quantile_fn(b)   # each sorts once, not per-`u`
    return _pnorm([Qa(u) - Qb(u) for u in us], p)
end

# at least one argument is a distribution: integrate the quantile gap on a grid
function _wasserstein(a, b, p)
    n = 4096
    us = ((1:n) .- 0.5) ./ n
    Qa, Qb = _quantile_fn(a), _quantile_fn(b)
    _pnorm([Qa(u) - Qb(u) for u in us], p)
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
    isempty(source) && throw(ArgumentError("empty `source` sample: transportmap needs at least one observation"))
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

"""
    driftsignificance(a, b; p=1, nperm=1000, level=0.9, rng=Random.default_rng())

Test whether the Wasserstein distance between samples `a` and `b` is larger than
sampling noise alone would produce. The two samples are pooled and repeatedly
re-split at random; the observed [`wasserstein`](@ref)`(a, b; p)` is then compared
against this permutation distribution. A raw distance is meaningless without such
a reference — otherwise ordinary sampling wiggle is mistaken for real drift.

Returns a `NamedTuple` `(; distance, threshold, pvalue, significant)`:

- `distance`  — the observed `wasserstein(a, b; p)`
- `threshold` — the `level` quantile of the permuted distances (the noise floor)
- `pvalue`    — fraction of permuted distances ≥ `distance` (add-one smoothed)
- `significant` — `distance > threshold` (equivalently, `pvalue < 1 - level` up to
  add-one smoothing and ties)

!!! warning "Reproducibility for decisions of record"
    The permutation floor is stochastic. With the default
    `rng = Random.default_rng()` the `pvalue`, `threshold`, and — for a borderline
    move — the `significant` flag will differ from run to run on identical data.
    For any governance/reporting decision, pass an explicitly seeded `rng` (e.g.
    `rng = MersenneTwister(seed)`) so the result is auditable and reproducible.

## Example

```julia-repl
julia> using Random

julia> a = randn(MersenneTwister(1), 2_000); b = randn(MersenneTwister(2), 2_000) .+ 1;

julia> ds = driftsignificance(a, b; nperm=500, rng=MersenneTwister(42));

julia> ds.significant   # a full-σ shift clears the noise floor
true
```
"""
function driftsignificance(a::AbstractVector{<:Real}, b::AbstractVector{<:Real};
    p::Real=1, nperm::Integer=1000, level::Real=0.9, rng=Random.default_rng())
    obs = wasserstein(a, b; p)
    pool = vcat(collect(a), collect(b))
    na = length(a)
    idx = collect(eachindex(pool))
    perm = map(1:nperm) do _
        Random.shuffle!(rng, idx)
        wasserstein(pool[idx[1:na]], pool[idx[na+1:end]]; p)
    end
    threshold = Statistics.quantile(perm, level)
    pvalue = (count(>=(obs), perm) + 1) / (nperm + 1)
    return (; distance=obs, threshold, pvalue, significant=obs > threshold)
end

# the natural tail level of a risk measure, where one exists
_taillevel(rm::CTE) = rm.α
_taillevel(rm::VaR) = rm.α
_taillevel(rm::RiskMeasure) =
    throw(ArgumentError("$(typeof(rm)) has no natural tail level; pass `tail`"))

"""
    worstcase(rm::RiskMeasure, sample; radius, p=2, tail=<rm's α>)

The worst value of risk measure `rm` over a `p`-Wasserstein ball of the given
`radius` around the empirical distribution of `sample` — a distributionally
robust (Wasserstein-DRO) version of `rm`. `radius` answers "how bad could this
number be if my book is off by up to `radius` of transport cost?" and is a
governance dial in the units of the loss.

The budget-optimal adverse distribution shifts the worst `1 - tail` fraction of
outcomes outward by `Δ = radius * (1 - tail)^(-1/p)`, which costs exactly `radius`
in ``W_p`` and maximizes tail-focused measures. Concretely it shifts the tail
order statistics ``x_{(k)}, \\dots, x_{(n)}`` — using the *same* tail boundary
``k`` that [`VaR`](@ref) and [`CTE`](@ref) use on a sample, so the two agree
exactly even under ties. For `rm = CTE(α)` with `tail = α` this attains the sharp
stability bound

```math
|\\mathrm{CTE}_\\alpha(\\mu) - \\mathrm{CTE}_\\alpha(\\nu)|
    \\le (1-\\alpha)^{-1/p}\\, W_p(\\mu,\\nu) ,
```

exactly (up to floating point): `worstcase(CTE(α), s; radius=r) ≈ CTE(α)(s) +
r*(1-α)^(-1/p)`. For other risk measures it evaluates `rm` on the same budget-`r`
adverse scenario (a concrete distribution inside the ball), i.e. a lower bound on
the true worst case.

The factor `(1 - tail)^(-1/p)` is the price of tail focus: deeper-tail capital is
intrinsically more fragile to model error, so the same `radius` buys a larger
loading at `CTE(0.995)` than at `CTE(0.95)`.

## Example

```julia-repl
julia> s = rand(LogNormal(log(1000) - 0.18, 0.6), 200_000);

julia> CTE(0.95)(s), worstcase(CTE(0.95), s; radius=250)   # ≈ (2980, 4100)
```

`worstcase` takes the risk measure as an argument, so it works unchanged for
`VaR`, `WangTransform`, or any custom `RiskMeasure` (supply `tail` for measures
without a natural tail level).
"""
function worstcase(rm::RiskMeasure, sample::AbstractVector{<:Real};
    radius::Real, p::Real=2, tail::Real=_taillevel(rm))
    radius >= 0 || throw(ArgumentError("radius must be ≥ 0, got $radius"))
    0 <= tail < 1 || throw(ArgumentError("tail must be in [0, 1), got $tail"))
    p >= 1 || throw(ArgumentError("p must be ≥ 1, got $p"))
    isempty(sample) && throw(ArgumentError("empty `sample`: worstcase needs at least one observation"))
    Δ = radius * (1 - tail)^(-1 / p)
    # Shift the tail order statistics x_(k)…x_(n) by Δ. `k` is taken from the SAME
    # boundary VaR/CTE use on a sample (`_first_index_above`), rather than from a
    # `Statistics.quantile` *value* + `x ≥ thr` test. The value-threshold form
    # silently disagrees with CTE's own tail by O(1/n) when the quantile
    # conventions differ, and over-/under-shifts under ties; the rank form makes
    # the CTE bound exact by construction and unambiguous at atoms.
    xs = sort(vec(sample))
    n = length(xs)
    k = RiskMeasures._first_index_above(n, tail)
    @views xs[k:n] .+= Δ
    return rm(xs)
end

end
