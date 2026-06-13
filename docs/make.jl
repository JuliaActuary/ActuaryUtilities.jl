using Documenter, ActuaryUtilities, FinanceCore

# Setup for doctests embedded in docstrings.
# DayCounts comes from docs/Project.toml (a runtime Pkg.add here was fragile).
DocMeta.setdocmeta!(ActuaryUtilities, :DocTestSetup, :(using ActuaryUtilities, Dates, DayCounts), recursive=true)

makedocs(;
    modules=[ActuaryUtilities, FinanceCore],
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://docs.juliaactuary.org/ActuaryUtilities",
    ),
    pages=[
        "Overview" => "index.md",
        "Financial Math" => "financial_math.md",
        "Key Rate Sensitivities" => "sensitivities.md",
        "Risk Measures" => "risk_measures.md",
        "Other Utilities" => "utilities.md",
        "API" => [
            "ActuaryUtilities" => "API/ActuaryUtilities.md",
            "FinanceCore (re-exported)" => "API/FinanceCore.md",
        ],
        "Upgrade from Prior Versions" => "upgrade.md",
    ],
    repo=Remotes.GitHub("JuliaActuary", "ActuaryUtilities.jl"),
    sitename="ActuaryUtilities.jl",
    authors="Alec Loudenback"
)

deploydocs(;
    repo="github.com/JuliaActuary/ActuaryUtilities.jl"
)
