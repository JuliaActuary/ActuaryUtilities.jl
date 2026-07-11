# Risk Measures

## Quickstart

```julia
outcomes = rand(100)

# direct usage
VaR(0.90)(outcomes) # ≈ 0.90  
CTE(0.90)(outcomes) # ≈ 0.95  
WangTransform(0.90)(outcomes) # ≈ 0.81

# construct a reusable object (functor)
rm = VaR(0.90)

rm(outcomes) # ≈ 0.90
```

## Introduction

Risk measures encompass the set of functions that map a set of outcomes to an output value characterizing the associated riskiness of those outcomes. As is usual when attempting to compress information (e.g. condensing information into a single value), there are multiple ways we can characterize this riskiness.


## Coherence & Other Desirable Properties

Further, it is desirable that a risk measure has certain properties, and risk measures that meet the first four criteria are called "Coherent" in the literature.  From "An Introduction to Risk Measures for Actuarial Applications" (Hardy), she describes as follows:

Using $H$ as a risk measure and $X$ as the associated risk distribution:

### 1. Translation Invariance

For any non-random $c$

$$H(X + c) = H(X) + c$$

This means that adding a constant amount (positive or negative) to a risk adds the same amount to the risk measure. It also implies that the risk measure for a non-random loss, with known value c, say, is just the amount of the loss c.

### 2. Positive Homogeneity

For any non-random $λ > 0$:

$$H(λX) = λH(X)$$

This axiom implies that changing the units of loss does not change the risk measure.

### 3. Subadditivity

For any two random losses $X$ and $Y$,

$$H(X + Y) ≤ H(X) + H(Y)$$

It should not be possible to reduce the economic capital required (or the appropriate premium) for a risk by splitting it into constituent parts. Or, in other words, diversification (ie consolidating risks) cannot make the risk greater, but it might make the risk smaller if the risks are less than perfectly correlated.

### 4. Monotonicity

If $Pr(X ≤ Y) = 1$ then $H(X) ≤ H(Y)$.

If one risk is always bigger then another, the risk measures should be similarly ordered.

### Other Properties

In "Properties of Distortion Risk Measures" (Balbás, Garrido, Mayoral) also note other properties of interest:

#### Complete

Completeness is the property that the distortion function associated with the risk measure produces a unique mapping between the original risk's survival function $S(x)$ and the distorted  $S*(x)$ for each $x$. See [Distortion Risk Measures](@ref) for more detail on this.

In practice, this means that a non-complete risk measure ignores some part of the risk distribution (e.g. `CTE` and `VaR` do not use the full distribution, so two risks that differ only outside the measured tail can produce the same value of the risk measure).

#### Exhaustive

A risk measure is "exhaustive" if it is coherent and complete.

#### Adaptable

A risk measure is "adapted" or "adaptable" if its distortion function (see [Distortion Risk Measures](@ref)) $g$ satisfies:

1. $g$ is strictly concave, that is, $g^\prime$ is strictly decreasing.
2. $\lim_{u\to 0^+} g^\prime(u) = \infty$ and $\lim_{u\to 1^-} g^\prime(u) = 0$.

Adaptive risk measures are exhaustive but the converse is not true.

### Summary of Risk Measure Properties

| Measure      | Coherent | Complete | Exhaustive | Adaptable | Condition 2 |
|--------------|----------|----------|------------|-----------|-------------|
| [VaR](@ref)        | No       | No       | No         | No        | No          |
| [CTE](@ref)       | Yes      | No       | No         | No        | No          |
| [DualPower](@ref) $(y > 1)$   | Yes      | Yes      | Yes        | No        | Yes         |
| [ProportionalHazard](@ref) $(γ > 1)$   | Yes      | Yes      | Yes        | No        | Yes         |
| [WangTransform](@ref)           | Yes      | Yes      | Yes        | Yes       | Yes         |

## Distortion Risk Measures

Distortion Risk Measures ([Wikipedia Link](https://en.wikipedia.org/wiki/Distortion_risk_measure)) are a way of remapping the probabilities of a risk distribution in order to compute a risk measure $H$ on the risk distribution $X$.

Adapting Wang (2002), there are two key components:

### Distortion Function $g(u)$

This remaps values in the [0,1] range to another value in the [0,1] range, and in $H$ below, operates on the survival function $S$ and $F=1-S$.

Let $g:[0,1]\to[0,1]$ be an increasing function with $g(0)=0$ and $g(1)=1$. The transform $F^*(x)=g(F(x))$ defines a distorted probability distribution, where "$g$" is called a distortion function.

Note that $F^*$ and $F$ are equivalent probability measures if and only if $g:[0,1]\to[0,1]$ is continuous and one-to-one.
Definition 4.2. We define a family of distortion risk-measures using the mean-value under the distorted probability $F^*(x)=g(F(x))$:

### Risk Measure Integration

To calculate a risk measure $H$, we integrate the distorted $F$ across all possible values in the risk distribution (i.e. $x \in X$):

$$H(X) = E^*(X) = - \int_{-\infty}^0 g(F(x))dx + \int_0^{+\infty}[1-g(F(x))]dx$$

That is, the risk measure ($H$) is equal to the expected value of the distortion of the risk distribution ($E^*(X)$).

!!! note "How the computation is performed"
    When `risk` is a continuous `Distributions.jl` distribution, this integral is evaluated by numerical quadrature of the distorted distribution function. When `risk` is an array of outcomes, the same Choquet integral reduces to a finite weighted sum of the sample's order statistics, and `VaR`, `CTE`, and `Expectation` evaluate that sum exactly (no quadrature or approximation error); the other distortion measures integrate the distorted empirical CDF numerically.

## Examples

### Basic Usage

```julia
outcomes = rand(100)

# direct usage
VaR(0.90)(outcomes) # ≈ 0.90  
CTE(0.90)(outcomes) # ≈ 0.95  
WangTransform(0.90)(outcomes) # ≈ 0.81

# construct a reusable object (functor)
rm = VaR(0.90)

rm(outcomes) # ≈ 0.90
```

### Comparison

We will generate a random outcome and show how the risk measures behave:

```@example
using Distributions
using ActuaryUtilities
using CairoMakie

outcomes = Weibull(1,5)
# or this could be discrete outcomes as in the next line
#outcomes = rand(LogNormal(2,10)*100,2000) 

αs= range(0.00,0.99;length=100)

let 
    f = Figure()
    ax = Axis(f[1,1],
        xlabel="α",
        ylabel="Loss",
        title = "Comparison of Risk Measures",
        xgridvisible=false,
        ygridvisible=false,
    )

    lines!(ax,
        αs,
        [quantile(outcomes, α) for α in αs],
        label = "Quantile α of Outcome",
        color = :grey10,
        linewidth = 3,
        )
    
    lines!(ax,
        αs,
        [VaR(α)(outcomes) for α in αs],
        label = "VaR(α)",
        linestyle=:dash
        )
    lines!(ax,
        αs,
        [CTE(α)(outcomes) for α in αs],
        label = "CTE(α)",
        )
    lines!(ax,
        αs[2:end],
        [WangTransform(α)(outcomes) for α in αs[2:end]],
        label = "WangTransform(α)",
        )
    lines!(ax,
        αs,
        [ProportionalHazard(2)(outcomes) for α in αs],
        label = "ProportionalHazard(2)",
        )
    
    lines!(ax,
        αs,
        [DualPower(2)(outcomes) for α in αs],
        label = "DualPower(2)",
        )
    lines!(ax,
        αs,
        [Expectation()(outcomes) for α in αs],
        label = "Expectation",
        )
    axislegend(ax,position=:lt)

        f
end
```

## Optimal Transport & Robustness

A risk measure collapses a whole loss distribution to a single number. Optimal
transport (OT) supplies the complementary half — the **geometry *between*
distributions**: how far apart two risks are, how a rank-preserving stress moves
*every* measure at once, whether a quarter-over-quarter change is real, and how
much a risk measure can move under model error. In one dimension OT is closed
form (transport is just rank matching), so these tools need no solver — they are
sorting and quantile arithmetic layered on the same `rm(risk)` interface.

### Distance between two risks — [`wasserstein`](@ref)

Where a risk measure says *where* a book sits, the Wasserstein distance says *how
far apart* two books are, in the units of the loss and aware of the whole shape:

```julia
a = rand(LogNormal(log(1000) - 0.18, 0.6), 100_000)
b = rand(LogNormal(log(1150) - 0.32, 0.8), 100_000)   # higher mean + fatter tail

wasserstein(a, b)          # ≈ 210   average claim displacement (W₁, in \$)
wasserstein(a, b; p=2)     # ≈ 480   W₂ penalizes the tail move more
```

It also accepts `Distributions.UnivariateDistribution`s directly, e.g.
`wasserstein(Normal(0, 1), Normal(3, 1)) == 3`.

### One stress drives the whole panel — [`transportmap`](@ref) / [`pushforward`](@ref)

Rather than stressing each measure separately, define **one** rank-preserving
transport map, push the book through it, and recompute every measure consistently:

```julia
base_law = LogNormal(log(1000) - 0.18, 0.6)                    # the assumed base
T        = transportmap(base_law, LogNormal(log(1300) - 0.32, 0.8))  # → target
sample   = rand(base_law, 100_000)
stress   = pushforward(sample, T)

for rm in (Expectation(), VaR(0.95), CTE(0.95), VaR(0.995), CTE(0.995))
    println(rm, ": ", round(rm(sample)), " → ", round(rm(stress)))
end
```

The map is auditable ("we revalued each percentile") rather than a reshuffling of
who is risky.

### Distributionally robust risk measures — [`worstcase`](@ref)

`worstcase(rm, sample; radius=r)` returns the worst value of `rm` over a
Wasserstein ball of radius `r` — a governance dial in the units of the loss for
"how bad could this number be if my book is off by up to `r` of transport cost?"
For `CTE(α)` it attains the sharp stability bound
``|\mathrm{CTE}_\alpha(\mu)-\mathrm{CTE}_\alpha(\nu)| \le (1-\alpha)^{-1/p}\,W_p``:

```julia
base = rand(LogNormal(log(1000) - 0.18, 0.6), 200_000)

CTE(0.95)(base),  worstcase(CTE(0.95),  base; radius=250)   # ≈ (2980, 4100)
CTE(0.995)(base), worstcase(CTE(0.995), base; radius=250)   # ≈ (4890, 8430)
```

Same \$250 radius, a larger loading deeper in the tail — deep-tail capital is
intrinsically more fragile to model error. Because `worstcase` takes the risk
measure as an argument it works for `VaR`, `WangTransform`, or any custom
`RiskMeasure` (pass `tail` for measures without a natural tail level).

!!! note "VaR is fragile, CTE is not"
    These tools also expose a *structural* fact about the measures. Where the loss
    density is thin near the quantile, a tiny transport move (small `wasserstein`)
    can swing `VaR` by a large amount, while `CTE` — a tail *average* — obeys the
    Lipschitz bound above. For capital that must be stable under model or portfolio
    perturbation, prefer `CTE`; if you must report `VaR`, check the density near the
    quantile.

### Is a change real? — [`driftsignificance`](@ref)

A risk number that moved quarter-to-quarter may just be sampling noise.
`driftsignificance` compares the observed `wasserstein` against the distances
produced by *random* re-splits of the pooled data, so only moves that clear the
noise floor are flagged:

```julia
q1 = rand(LogNormal(log(1000) - 0.18, 0.60), 4_000)   # this quarter
q2 = rand(LogNormal(log(1030) - 0.19, 0.62), 4_000)   # next quarter

ds = driftsignificance(q1, q2)
ds.distance, ds.threshold, ds.significant             # e.g. (≈50, ≈28, true)
```

!!! warning "Three cautions"
    (1) A distance or robustness number is only meaningful *with* its ground cost —
    absolute \$, log/relative, or tail-weighted — so report the cost alongside.
    (2) These are distributional statements, not per-policyholder causal ones.
    (3) With atoms/ties (discrete losses, curtate lifetimes) fix the quantile
    convention so the OT layer and the risk measures agree at the ties.

## API

### Exported API
```@autodocs
Modules = [ActuaryUtilities.RiskMeasures, ActuaryUtilities.OptimalTransport]
Private = false
```

### Unexported API
```@autodocs
Modules = [ActuaryUtilities.RiskMeasures]
Public = false
```
