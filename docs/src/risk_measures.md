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

Risk measures encompass the set of functions that map a set of outcomes to an output value characterizing the associated riskiness of those outcomes. As is usual when attempting to compress information (e.g. condensing information into a single value), there are multiple ways we can charactize this riskiness.


## Coherence & Other Desirable Properties

Further, it is desireable that a risk measure has certain properties, and risk measures that meet the first four criteria are called "Coherent" in the literature.  From "An Introduction to Risk Measures for Actuarial Applications" (Hardy), she describes as follows:

Using $H$ as a risk measure and $X$ as the associated risk distribution:

### 1. Translation Invariance

For any non-random $c$

%$H(X + c) = H(X) + c$%
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

Completeness is the property that the distortion function associated with the risk measure produces a unique mapping between the original risk's survial function $S(x)$ and the distorted  $S*(x)$ for each $x$. See [Distortion Risk Measures](@ref) for more detail on this.

In practice, this means that a non-complete risk measure ignores some part of the risk distribution (e.g. CTE and VaR don't use the full distribution and have the same)

#### Exhaustive

A risk measure is "exhaustive" if it is coherent and complete.

#### Adaptable

A risk measure is "adapted" or "adaptable" if its distortion function (ee [Distortion Risk Measures](@ref)).$g$:

    1. $g$ is strictly concave, that is $g$ is strictly decreasing. 
    2. $lim_{u\to0+} g\prime(u) = \inf and lim_{u\to1-} g\prime(u) = 0.

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

That is, the risk measure ($H$) is equal to the expected value of the distortion of the risk ditribution ($E^*(X)$).

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
        [RiskMeasures.Expectation()(outcomes) for α in αs],
        label = "Expectation",
        )
    axislegend(ax,position=:lt)

        f
end
```

## API

### Exported API
```@autodocs
Modules = [ActuaryUtilities.RiskMeasures]
Private = false
```

### Unexported API
```@autodocs
Modules = [ActuaryUtilities.RiskMeasures]
Public = false
```