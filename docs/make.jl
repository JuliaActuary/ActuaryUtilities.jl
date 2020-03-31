using Documenter, ActuaryUtilities

makedocs(;
    modules=[ActuaryUtilities],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/alecloudenback/ActuaryUtilities.jl/blob/{commit}{path}#L{line}",
    sitename="ActuaryUtilities.jl",
    authors="Alec Loudenback",
    assets=String[],
)

deploydocs(;
    repo="github.com/alecloudenback/ActuaryUtilities.jl",
)
