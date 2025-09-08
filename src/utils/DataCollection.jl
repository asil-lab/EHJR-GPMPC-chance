module DataCollection

using GaussianProcesses
using Optim 
using Plots  # Add this import
using ..AgentTypes
using ..GPModelTypes
using ..SystemDynamics  # Add this import for true_dynamics

export collect_training_data!, train_gp!

function collect_training_data!(agent::Agent, ref_traj::Matrix{Float64}, num_samples=100)
    # Initialize data storage with correct dimensions
    X = zeros(4, num_samples)  # [x, y, u1, u2] as columns
    Y = zeros(2, num_samples)  # [dx, dy] as columns
    
    # Sample state space
    a = minimum(ref_traj[1, :])-1.0
    b = maximum(ref_traj[1, :])+1.0
    c = minimum(ref_traj[2, :])-1.0
    d = maximum(ref_traj[2, :])+1.0
    
    # Generate random points (store as columns)
    X[1, :] = a .+ (b - a) * rand(num_samples)  # x positions
    X[2, :] = c .+ (d - c) * rand(num_samples)  # y positions
    X[3:4, :] = 0.5 * randn(2, num_samples)     # controls
    
    # Collect dynamics data
    for i in 1:num_samples
        state = X[1:2, i]
        control = X[3:4, i]
        next_state = true_dynamics(state, control)
        Y[:, i] = next_state - state
    end
    
    # Visualize training data
    p1 = scatter(X[1, :], Y[1, :], 
        label="x dimension", 
        title="State vs State Difference",
        xlabel="x position",
        ylabel="dx")
        
    p2 = scatter(X[2, :], Y[2, :], 
        label="y dimension",
        xlabel="y position",
        ylabel="dy")
        
    p3 = scatter(X[3, :], Y[1, :], 
        label="x dimension",
        xlabel="u1",
        ylabel="dx")
        
    p4 = scatter(X[4, :], Y[2, :], 
        label="y dimension",
        xlabel="u2",
        ylabel="dy")
    
    plot(p1, p2, p3, p4, layout=(2,2), size=(800,600))
    savefig("training_data.png")
    
    # Initialize GP model
    model = GPModel(2, 2)  # 4 inputs, 2 outputs
    
    # Train separate GP for each dimension
    for dim in 1:2
        y = vec(Y[dim, :])
        
        # Create and optimize GP
        kern = SE(fill((1.0), 2), (0.1))
        model.models[dim] = GP(X[1:2,:], y, MeanZero(), kern, (1e-1))
        optimize!(model.models[dim])
    end
    
    # Assign the model to agent
    agent.gp_model = model
    
    return nothing
end

function generate_gp_model(X::Matrix{Float64}, y::Vector{Float64}, noise=1e4, σ=0.4)
    # Define kernel (RBF/SE kernel with ARD)
    # One length scale per input dimension
    l = ones(2)  # 4 input dimensions [x, y, u1, u2]
    
    m = MeanZero()
    kern = SE(log.(l), log(σ))
    
    # Create and train GP
    gp = GP(X, y, m, kern, log(noise))
    optimize!(gp)
    
    return gp
end

function train_gp!(agent::Agent)
    # Train each GP model if it exists
    if !isnothing(agent.gp_model)
        for gp in agent.gp_model.models
            if !isnothing(gp)
                optimize!(gp)
            end
        end
    end
    return nothing
end

end