# Cooperative Gaussian Process-based Model Predictive Control for Safe Multi-Agent Navigation

This repository contains the code for reproducing the results in "Cooperative Gaussian Process-based Model Predictive Control for Safe Multi-Agent Navigation".

## Description

The code implements a cooperative control approach for multi-agent systems using Gaussian Process-based Model Predictive Control (GP-MPC) to ensure safe navigation in environments with dynamic obstacles.

- `./simulations`: scripts to run various experiments, convergence plots, distributed GP-MPC and linear GP-MPC cases.
- `./src`: core implementation including controllers, system dynamics, types, and utility functions.

## Key Points

- The paths specified in the code are relative to the location of the main code file. Adjust paths according to your system setup if needed.
- The main simulation scripts can be run with different parameters. Be sure to update save folders and configuration parameters as necessary.
- This project is implemented in Julia for high-performance numerical computation.

## Support and Questions

For questions and issues, please use the [Issues](../../issues) section of this repository.

## Supported Platforms

- Julia 1.0 and higher

## Citation

```bibtex
@article{Riemens2025,
  author = {Riemens, Ellen H. J. and van der Veen, Alle-Jan and Rajan, Raj T.},
  title = {Cooperative Gaussian Process-based Model Predictive Control for Safe Multi-Agent Navigation},
  journal = {},
  year = {2025}
}
```

## Authors

- **Ellen H. J. Riemens** - E.H.J.Riemens@tudelft.nl
- **Alle-Jan van der Veen** - A.J.vanderVeen@tudelft.nl
- **Raj T. Rajan** - R.T.Rajan@tudelft.nl

Delft University of Technology, Faculty of EEMCS, Mekelweg 4, 2628 CD Delft, The Netherlands

## Getting Started

The code is written in Julia.

### Setup

Run the setup script to automatically instantiate the project environment and install all required dependencies:

```julia
julia setup.jl
```

This will activate the project environment and install all packages listed in `Project.toml`.

### Running Simulations

The following simulation scripts are available in the `./simulations` directory:

#### `distributed_gp_mpc.jl`
Runs the main distributed GP-MPC algorithm for cooperative multi-agent navigation. Implements the centralized controller approach with Gaussian Process modeling of agent dynamics.

#### `lin_gp_mpc.jl`
Runs simulations using a linearized version of the GP-MPC controller. Useful for comparison and analysis of the impact of linearization approximations.

#### `montecarlo.jl`
Performs Monte Carlo simulations to evaluate the robustness and statistical performance of the distributed GP-MPC approach across multiple random scenarios.

#### `trajectorycomparion.jl`
Compares trajectories from different control approaches, generating visualizations and metrics to highlight differences between strategies.

#### `convergence_plots.jl`
Analyzes and plots the convergence behavior of the ADMM algorithm used in the optimization, showing primal and dual residuals.

#### `convergencerho.jl`
Studies the convergence properties with respect to the penalty parameter (rho) in the ADMM algorithm, useful for parameter tuning analysis.

#### `module_export.jl`
Utility script for exporting and managing the project modules and dependencies.

To run a specific simulation:

```julia
julia simulations/distributed_gp_mpc.jl
```
