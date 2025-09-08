module AgentTypes

using ..GPModelTypes: GPModel  # Add this import

export AgentConfig, Agent, reset_state!, get_state, apply_control!

struct AgentConfig
    id::Int  # Add this field
    horizon::Int
    dt::Float64
    input_bounds::Tuple{Float64, Float64}
    input_rate_bounds::Tuple{Float64, Float64}
end

mutable struct Agent
    id::Int  # Add this field
    config::AgentConfig
    state::Vector{Float64}
    gp_model::Union{Nothing, GPModel}
end

# Update constructor
function Agent(config::AgentConfig, initial_state::Vector{Float64})
    Agent(
        config.id,  # Pass id from config
        config,
        initial_state,
        nothing
    )
end

"""
Reset the agent's state to a given value
"""
function reset_state!(agent::Agent, new_state::Vector{Float64})
    agent.state = new_state
    return nothing
end

function get_state(agent::Agent)
    return agent.state
end

function apply_control!(agent::Agent, control::Vector{Float64})
    # Update agent state based on control input
    # This is where your system dynamics would be applied
    agent.state += control  # Simplified example - replace with actual dynamics
    return nothing
end

end

