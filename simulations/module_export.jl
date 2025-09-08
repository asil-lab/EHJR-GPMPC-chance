using Pkg
Pkg.activate(dirname(dirname(@__FILE__)))

# Include and import module
include("../src/CGPMPC_src.jl")
using .CGPMPC_src

# Print all names, including non-exported ones
println("\nAll names in module (including internal):")
@show names(CGPMPC_src, all=true)

# Print only exported names
println("\nExported names:")
@show names(CGPMPC_src)

# Try both direct and qualified access
println("\nTrying both access methods:")
config2 = CGPMPC_src.AgentConfig(1, 10, 0.5, (-2.0, 2.0), (-100.0, 100.0))
try
    # Direct access
    config1 = AgentConfig(1, 10, 0.5, (-2.0, 2.0), (-100.0, 100.0))
    println("Direct access works")
    
    # Qualified access
    config2 = CGPMPC_src.AgentConfig(1, 10, 0.5, (-2.0, 2.0), (-100.0, 100.0))
    println("Qualified access works")
catch e
    println("Access failed: ", e)
end
