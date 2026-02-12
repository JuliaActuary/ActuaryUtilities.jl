using Documenter, ActuaryUtilities, FinanceCore

# Setup for doctests embedded in docstrings.
DocMeta.setdocmeta!(ActuaryUtilities, :DocTestSetup, :(import Pkg; Pkg.add("DayCounts"); using ActuaryUtilities, Dates, DayCounts), recursive=true)

makedocs(;
    modules=[ActuaryUtilities, FinanceCore],
    format=Documenter.HTML(),
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
    repo="https://github.com/JuliaActuary/ActuaryUtilities.jl/blob/{commit}{path}#L{line}",
    sitename="ActuaryUtilities.jl",
    authors="Alec Loudenback"
)

deploydocs(;
    repo="github.com/JuliaActuary/ActuaryUtilities.jl"
)
