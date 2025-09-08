using Pkg
Pkg.activate(dirname(dirname(@__FILE__))) # Activate the root project directory
include("../src/CGPMPC_src.jl")
using .CGPMPC_src 

using Plots
using Random
using Statistics

function create_reference_window(ref_traj::Matrix{Float64}, current_step::Int, horizon::Int)
    num_steps = size(ref_traj, 2)
    ref_window = zeros(size(ref_traj, 1), horizon)
    
    for i in 1:horizon
        # Wrap around to the beginning if we reach the end of trajectory
        step_idx = mod1(current_step + i - 1, num_steps)
        ref_window[:, i] = ref_traj[:, step_idx]
    end
    
    return ref_window
end

function run_single_phase(agent::Agent, controller, ref_traj::Matrix{Float64}, dynamics_mode::DynamicsMode, num_steps::Int64=50)
    # Initialize storage for results
    state_history = zeros(2, num_steps)
    control_history = zeros(2, num_steps)
    mse = zeros(num_steps)
    
    # Reset agent to initial state
    reset_state!(agent, ref_traj[:, 1] + 0.1 * randn(2))
    
    # Run simulation
    for t in 1:num_steps
        # Get current state
        current_state = get_state(agent)
        state_history[:, t] = current_state
        
        # Create reference window
        ref_window = create_reference_window(ref_traj, t, agent.config.horizon)
        
        # Compute control
        control = compute_control(
            controller, 
            current_state, 
            ref_window, 
            agent.gp_model,
            dynamics_mode=dynamics_mode
        )
        control_history[:, t] = control
        
        # Update state
        next_state = true_dynamics(current_state, control)
        reset_state!(agent, next_state)
        
        # Compute MSE for this step
        mse[t] = mean((current_state - ref_traj[:, t]).^2)
    end
    
    return (
        states=state_history,
        controls=control_history,
        mse=mse
    )
end

function run_multi_agent_phases(agents, controllers, ref_trajs; num_steps=50, centralized=false, dynamics_mode=LinearizedGP)
    # Initialize storage for results
    num_agents = length(agents)
    state_histories = [zeros(2, num_steps) for _ in 1:num_agents]
    control_histories = [zeros(2, num_steps) for _ in 1:num_agents]
    mse_histories = [zeros(num_steps) for _ in 1:num_agents]
    
    # Reset agents to initial states
    for i in 1:num_agents
        reset_state!(agents[i], ref_trajs[i][:, 1] + 0.1 * randn(2))
    end
    
    # Run simulation
    for t in 1:num_steps
        if centralized
            # Get all current states and reference windows
            current_states = [get_state(agent) for agent in agents]
            ref_windows = [create_reference_window(ref_trajs[i], t, agents[i].config.horizon) 
                          for i in 1:num_agents]
            
            # Compute centralized control
            controls = compute_centralized_control(
                controllers[1],  # Use first controller for centralized control
                agents,
                ref_windows;
                dynamics_mode=dynamics_mode
            )
            
            # Apply controls and update states
            for i in 1:num_agents
                state_histories[i][:, t] = current_states[i]
                control_histories[i][:, t] = controls[i]
                
                # Update state
                next_state = true_dynamics(current_states[i], controls[i])
                reset_state!(agents[i], next_state)
                
                # Compute MSE
                mse_histories[i][t] = mean((current_states[i] - ref_trajs[i][:, t]).^2)
            end
        else
            # Decentralized control: handle each agent separately
            for i in 1:num_agents
                current_agent = agents[i]
                
                # Create reference window for current agent
                ref_window = create_reference_window(ref_trajs[i], t, current_agent.config.horizon)
                
                # Get reference trajectories of other agents for collision avoidance
                other_agents_refs = Vector{Matrix{Float64}}()  # Specify concrete type
                for j in 1:num_agents
                    if j != i
                        other_ref = create_reference_window(ref_trajs[j], t, current_agent.config.horizon)
                        push!(other_agents_refs, other_ref)
                    end
                end
                
                # Compute control with other agents' reference trajectories
                control = compute_decentralized_control(
                    controllers[i],
                    current_agent,
                    ref_window,
                    other_agents_refs;  # Now has correct type Vector{Matrix{Float64}}
                    dynamics_mode=dynamics_mode
                )
                
                state_histories[i][:, t] = get_state(current_agent)
                control_histories[i][:, t] = control
                
                # Update state
                next_state = true_dynamics(get_state(current_agent), control)
                reset_state!(current_agent, next_state)
                
                # Compute MSE
                mse_histories[i][t] = mean((get_state(current_agent) - ref_trajs[i][:, t]).^2)
            end
        end
    end
    
    return (
        state_histories=state_histories,
        control_histories=control_histories,
        mse_histories=mse_histories
    )
end

function run_single_agent_comparison()
    # Configuration
    agent_config = AgentConfig(1, 10, 0.5, (-2.0, 2.0), (-100.0, 100.0))
    controller_config = ControllerConfig(
        MPCParams(10, (-2.0, 2.0), (-100.0, 100.0), [1.0, 1.0], Dict("print_level" => 0)), 
        true, 2.0, 0.5
    )
    
    # Create reference trajectory
    ref_traj = Matrix(generate_oval_trajectory(50))
    agent = Agent(agent_config, ref_traj[:, 1] + 0.1 * randn(2))
    
    # Create controllers for each dynamics mode
    standard_controller = StandardMPC(controller_config)
    full_gp_controller = GPBasedMPC(controller_config, false)
    linear_gp_controller = GPBasedMPC(controller_config, true)
    
    # Collect data and train GP
    collect_training_data!(agent, ref_traj)
    train_gp!(agent)
    
    # Run comparison for each dynamics mode
    standard_results = run_single_phase(agent, standard_controller, ref_traj, Nominal)
    full_gp_results = run_single_phase(agent, full_gp_controller, ref_traj, FullGP)
    linear_gp_results = run_single_phase(agent, linear_gp_controller, ref_traj, LinearizedGP)
    
    return (standard=standard_results, full_gp=full_gp_results, linear_gp=linear_gp_results), ref_traj
end

function run_multi_agent_comparison()
    # Create agents and reference trajectories
    num_agents = 3
    agents = [
        Agent(
            AgentConfig(i, 10, 0.5, (-2.0, 2.0), (-100.0, 100.0)),
            zeros(2)
        ) for i in 1:num_agents
    ]
    
    # Use crossing trajectories instead of ovals
    ref_trajs = generate_crossing_trajectories(50, num_agents, radius=2.0, spacing=1.0)
    
    # Train GPs for all agents
    for (agent, ref_traj) in zip(agents, ref_trajs)
        collect_training_data!(agent, ref_traj)
        train_gp!(agent)
    end
    
    # Run centralized and decentralized with LinearizedGP
    controller_config = ControllerConfig(
        MPCParams(10, (-2.0, 2.0), (-100.0, 100.0), [1.0, 1.0], Dict("print_level" => 0)), 
        true, 2.0, 0.5
    )
    
    controllers = [GPBasedMPC(controller_config, true) for _ in 1:num_agents]
    
    decentralized_results = run_multi_agent_phases(
        deepcopy(agents), controllers, ref_trajs,
        num_steps=50, centralized=false, dynamics_mode=LinearizedGP
    )
    
    centralized_results = run_multi_agent_phases(
        deepcopy(agents), controllers, ref_trajs,
        num_steps=50, centralized=true, dynamics_mode=LinearizedGP
    )
    
    return (decentralized=decentralized_results, centralized=centralized_results), ref_trajs
end

# Main execution
Random.seed!(1234)

# Run comparisons
single_results, single_ref = run_single_agent_comparison()
multi_results, multi_refs = run_multi_agent_comparison()

# Plot single agent results
p1 = plot(
    title="Single Agent Dynamics Mode Comparison",
    xlabel="x₁", ylabel="x₂",
    aspect_ratio=:equal,
    legend=:topright
)

plot!(p1, single_ref[1,:], single_ref[2,:], 
    label="Reference", color=:black, linestyle=:dash
)

labels = ["Standard", "Full GP", "Linear GP"]
colors = [:blue, :red, :green]
for (i, (label, results)) in enumerate(zip(labels, 
    [single_results.standard, single_results.full_gp, single_results.linear_gp]))
    plot!(p1, results.states[1,:], results.states[2,:],
        label=label, color=colors[i]
    )
end

# Plot MSE comparison
p2 = plot(
    title="Single Agent MSE Comparison",
    xlabel="Time step", ylabel="MSE",
    yscale=:log10,
    legend=:topright
)

for (i, (label, results)) in enumerate(zip(labels, 
    [single_results.standard, single_results.full_gp, single_results.linear_gp]))
    plot!(p2, results.mse,
        label=label, color=colors[i]
    )
end

# Plot multi-agent comparison
p3 = plot(
    title="Multi-Agent Crossing Trajectories (LinearizedGP)",
    xlabel="x₁", ylabel="x₂",
    aspect_ratio=:equal,
    legend=:outerright,  # Move legend outside to the right
    xlims=(-3, 3),
    ylims=(-3, 3),
    size=(600, 400)  # Adjust size to accommodate legend
)

for i in 1:3
    plot!(p3, multi_refs[i][1,:], multi_refs[i][2,:], 
        label="Reference $i", color=:black, linestyle=:dash
    )
end

for i in 1:3
    plot!(p3, multi_results.decentralized.state_histories[i][1,:], 
         multi_results.decentralized.state_histories[i][2,:],
         label="Decentralized Agent $i", color=colors[i], linestyle=:solid
    )
    plot!(p3, multi_results.centralized.state_histories[i][1,:], 
         multi_results.centralized.state_histories[i][2,:],
         label="Centralized Agent $i", color=colors[i], linestyle=:dash
    )
end

# Combine plots
final_plot = plot(p1, p2, p3,
    layout=(3,1), 
    size=(1000,1200),  # Increased width to accommodate legend
    margin=10Plots.mm
)

display(final_plot)

# Print MSE summary statistics
println("\nMSE Summary Statistics:")
println("Single Agent Comparison:")
# First get standard MSE as baseline
standard_mse = mean(single_results.standard.mse)
for (label, results) in zip(labels, [single_results.standard, single_results.full_gp, single_results.linear_gp])
    mean_mse = mean(results.mse)
    improvement = 100 * (standard_mse - mean_mse) / standard_mse  # Percentage improvement
    println("  $label: Mean MSE = $(round(mean_mse, digits=6)) ($(round(improvement, digits=2))% improvement)")
end

println("\nMulti-Agent Comparison (LinearizedGP):")
println("Decentralized Control:")
for i in 1:3
    mean_mse = mean(multi_results.decentralized.mse_histories[i])
    println("  Agent $i: Mean MSE = $(round(mean_mse, digits=6))")
end
println("Centralized Control:")
for i in 1:3
    mean_mse = mean(multi_results.centralized.mse_histories[i])
    # Calculate improvement over decentralized
    decentralized_mse = mean(multi_results.decentralized.mse_histories[i])
    improvement = 100 * (decentralized_mse - mean_mse) / decentralized_mse
    println("  Agent $i: Mean MSE = $(round(mean_mse, digits=6)) ($(round(improvement, digits=2))% vs decentralized)")
end

