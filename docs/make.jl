using Documenter, ActuaryUtilities

# Setup for doctests embedded in docstrings.
DocMeta.setdocmeta!(ActuaryUtilities, :DocTestSetup, :(using ActuaryUtilities, Dates))

makedocs(;
    modules=[ActuaryUtilities],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/JuliaActuary/ActuaryUtilities.jl/blob/{commit}{path}#L{line}",
    sitename="ActuaryUtilities.jl",
    authors="Alec Loudenback",
    assets=String[],
)

deploydocs(;
    repo="github.com/JuliaActuary/ActuaryUtilities.jl",
)
