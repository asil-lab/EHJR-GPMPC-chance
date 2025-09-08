module CGPMPC_src

using JuMP
using Ipopt
using GaussianProcesses
using LinearAlgebra
using Plots


# Re-export main types and functions



# Type definitions
include("types/GPModelTypes.jl")
include("types/AgentTypes.jl")
using .AgentTypes
using .GPModelTypes
include("types/controller_types.jl")
include("utils/GPUtils.jl")
using .ControllerTypes
using .GPUtils

# Core functionality
include("dynamics/SystemDynamics.jl")
using .SystemDynamics
include("controllers/BaseController.jl")
include("controllers/MPCController.jl")
include("controllers/DistributedMPC.jl")
using .BaseController
using .MPCController
using .DistributedMPC

# Utilities
include("utils/Visualization.jl")
include("utils/DataCollection.jl")
include("utils/Trajectories.jl")
include("utils/ReferenceUtils.jl")
using .Visualization
using .DataCollection
using .Trajectories
using .ReferenceUtils


export StandardMPC, GPBasedMPC  # Add controller types to exports
export Agent, AgentConfig
export DynamicsMode, Nominal, FullGP, LinearizedGP, Nonlinear
export reset_state!, get_state, apply_control!
export GPModel
export nominal_dynamics, true_dynamics, get_linear_system_matrices
export ControllerConfig, MPCParams
# Add Distributed MPC exports
export DistributedMPCController, Edge

export solve_distributed_mpc!, solve_local_problem!
export initialize_edges!, create_edge
export compute_control, compute_distributed_control, compute_centralized_control, compute_decentralized_control
export visualize_agents, collect_training_data
export collect_training_data!, train_gp!
export generate_oval_trajectory, generate_circle_trajectory, generate_figure8_trajectory, generate_crossing_trajectories, generate_merging_trajectories
export visualize_multi_agents
export predict_y_and_gradient
export create_reference_window  # Add to exports


end