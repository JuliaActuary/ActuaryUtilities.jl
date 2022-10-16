using Documenter, ActuaryUtilities

# Setup for doctests embedded in docstrings.
DocMeta.setdocmeta!(ActuaryUtilities, :DocTestSetup, :(import Pkg; Pkg.add("DayCounts");using ActuaryUtilities, Dates,DayCounts),recursive=true)

makedocs(;
    modules=[ActuaryUtilities],
    format=Documenter.HTML(),
    pages=[
        "Introduction" => "index.md",
        "API" => "api.md",
    ],
    repo="https://github.com/JuliaActuary/ActuaryUtilities.jl/blob/{commit}{path}#L{line}",
    sitename="ActuaryUtilities.jl",
    authors="Alec Loudenback",
)

deploydocs(;
    repo="github.com/JuliaActuary/ActuaryUtilities.jl",
)
