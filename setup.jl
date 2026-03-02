"""
Activates the project environment and instantiates all dependencies.
"""

using Pkg

println("Setting up project...")


# Activate the project environment
project_dir = dirname(@__DIR__)
Pkg.activate(project_dir)
println("✓ Project environment activated")

# Instantiate dependencies
println("Installing dependencies...")
Pkg.instantiate()
println("✓ Dependencies installed")

