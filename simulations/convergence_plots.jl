using Plots
using Random
using Statistics
using LinearAlgebra
include("../src/CGPMPC_src.jl")
using .CGPMPC_src

function plot_convergence_results(history)
    # Create figure with two subplots
    p = plot(layout=(2,1), size=(800, 600), dpi=300)
    
    # Plot primal residuals
    plot!(p[1], history["iterations"], history["primal_residuals"], 
          label="Primal Residual", 
          linewidth=2, 
          marker=:circle, 
          markersize=4,
          yscale=:log10,  # Log scale for better visualization
          title="ADMM Convergence - Primal Residual",
          xlabel="Iteration",
          ylabel="Residual (log scale)")
    
    # Plot dual residuals
    plot!(p[2], history["iterations"], history["dual_residuals"], 
          label="Dual Residual", 
          linewidth=2, 
          marker=:circle, 
          markersize=4,
          yscale=:log10,  # Log scale for better visualization
          title="ADMM Convergence - Dual Residual",
          xlabel="Iteration",
          ylabel="Residual (log scale)")
    
    # Add convergence threshold reference line if available
    # Assuming tolerance is 1e-4 (modify as needed)
    tolerance = 1e-4  # Replace with actual tolerance from your controller
    hline!(p[1], [tolerance], linestyle=:dash, label="Tolerance", color=:red)
    hline!(p[2], [tolerance], linestyle=:dash, label="Tolerance", color=:red)
    
    return p
end

function plot_trajectory_comparison(
    agents, 
    initial_positions, 
    final_positions, 
    references, 
    previous_paths,
    safety_distance)
    
    # Create figure
    p = plot(size=(800, 600), dpi=300, legend=:outertopright)
    
    # Colors for different agents
    colors = [:blue, :red, :green, :purple, :orange, :cyan, :magenta, :yellow]
    
    # Plot trajectories for each agent
    for (i, agent) in enumerate(agents)
        # Get color for this agent (cycle if more agents than colors)
        agent_color = colors[mod1(i, length(colors))]
        
        # Plot reference trajectory
        plot!(p, references[i][1, :], references[i][2, :], 
              linestyle=:dash, 
              color=agent_color, 
              linewidth=1.5, 
              label="Agent $i Reference")
        
        # Plot actual trajectory from previous_paths
        plot!(p, previous_paths[i][1, :], previous_paths[i][2, :], 
              color=agent_color, 
              linewidth=2, 
              label="Agent $i Actual")
        
        # Mark initial position
        scatter!(p, [initial_positions[1, i]], [initial_positions[2, i]], 
                marker=:square, 
                color=agent_color, 
                markersize=8, 
                label="Agent $i Start")
        
        # Mark final position
        scatter!(p, [final_positions[1, i]], [final_positions[2, i]], 
                marker=:star, 
                color=agent_color, 
                markersize=8, 
                label="Agent $i End")
    end
    
    # Visualize safety distance for a few points along the trajectories
    # This helps illustrate collision avoidance
    horizon = size(previous_paths[1], 2)
    sample_points = [1, div(horizon, 2), horizon]  # Sample at start, middle, end
    
    for t in sample_points
        for i in 1:length(agents)
            for j in (i+1):length(agents)
                # Draw safety circle around agent i at time t
                if t <= size(previous_paths[i], 2) && t <= size(previous_paths[j], 2)
                    dist = norm(previous_paths[i][:, t] - previous_paths[j][:, t])
                    if dist < safety_distance * 2  # Only draw if agents are close
                        # Draw line connecting the agents
                        plot!([previous_paths[i][1, t], previous_paths[j][1, t]], 
                              [previous_paths[i][2, t], previous_paths[j][2, t]], 
                              color=:gray, linestyle=:dot, label=nothing)
                        
                        # Add distance annotation
                        midpoint = (previous_paths[i][:, t] + previous_paths[j][:, t]) / 2
                        annotate!(midpoint[1], midpoint[2], text("d=$(round(dist, digits=2))", 8, :black))
                    end
                end
            end
        end
    end
    
    # Set titles and labels
    title!("Agent Trajectories with Distributed MPC")
    xlabel!("X Position")
    ylabel!("Y Position")
    
    return p
end

# Example usage (to be replaced with your actual simulation)
function run_simulation_example()
    # This is a placeholder for your actual simulation
    # You would replace this with your actual simulation code
    
    # Simulate some agents with the Distributed MPC controller
    num_agents = 3
    horizon = 10
    
    # Create simulated data
    history = Dict(
        "primal_residuals" => [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
        "dual_residuals" => [0.8, 0.3, 0.08, 0.04, 0.009, 0.004, 0.0008],
        "iterations" => [1, 2, 3, 4, 5, 6, 7]
    )
    
    # Create simulated agent data
    initial_positions = [0.0 2.0 4.0; 0.0 0.0 0.0]  # 2×num_agents matrix
    final_positions = [5.0 3.0 1.0; 5.0 5.0 5.0]    # 2×num_agents matrix
    
    # Simulated references and paths
    references = []
    previous_paths = []
    for i in 1:num_agents
        # Create a simple straight-line reference from initial to final position
        ref_x = range(initial_positions[1, i], final_positions[1, i], length=horizon)
        ref_y = range(initial_positions[2, i], final_positions[2, i], length=horizon)
        push!(references, [collect(ref_x)'; collect(ref_y)'])
        
        # Create a slightly different actual path
        path_x = ref_x .+ 0.2 * sin.(range(0, π, length=horizon))
        path_y = ref_y .+ 0.2 * sin.(range(0, π, length=horizon))
        push!(previous_paths, [collect(path_x)'; collect(path_y)'])
    end
    
    # Create dummy agent objects
    agents = [i for i in 1:num_agents]  # Just use indices as placeholders
    
    # Safety distance
    safety_distance = 1.0
    
    # Create plots
    convergence_plot = plot_convergence_results(history)
    trajectory_plot = plot_trajectory_comparison(agents, initial_positions, final_positions, references, previous_paths, safety_distance)
    
    # Save plots
    savefig(convergence_plot, "admm_convergence.png")
    savefig(trajectory_plot, "agent_trajectories.png")
    
    return convergence_plot, trajectory_plot
end

# convergence_plot, trajectory_plot = run_simulation_example()
# println("Plots generated successfully!")
# println("Convergence plot saved as: admm_convergence.png")
# println("Trajectory plot saved as: agent_trajectories.png")
# For actual implementation with your DistributedMPC code, you would:
# 1. Create a simulation that initializes agents and controller
# 2. Run solve_distributed_mpc! to get controls, history, and trajectories
# 3. Pass those to the plotting functions above

function run_actual_simulation(controller_config, use_gp, rho, max_iterations, tolerance)
    # Implement actual simulation with your DistributedMPC module
    # This is where you would integrate with your existing code
    
    # Example integration:
    controller = DistributedMPCController(controller_config, use_gp, rho, max_iterations, tolerance)
    agents = initialize_agents()
    references = generate_references(agents)
    previous_paths = initialize_paths(agents)
    
    for timestep in 1:simulation_steps
        controls, history, updated_paths = solve_distributed_mpc!(controller, agents, references, previous_paths)
        # Apply controls and update agent states
        apply_controls!(agents, controls)
        previous_paths = updated_paths
        
        # Record data for visualization
        record_data!(agents, controls, history)
    end
    
    Visualization
    convergence_plot = plot_convergence_results(history)
    trajectory_plot = plot_trajectory_comparison(agents, initial_positions, final_positions, references, previous_paths, safety_distance)
end

run_actual_simulation(
    ControllerConfig(MPCParams(10, (-2.0, 2.0), (-100.0, 100.0), [1.0, 1.0], Dict("print_level" => 0)), true, 0.0, 0.5),
    true,
    10.0,
    100,
    1e-9
)