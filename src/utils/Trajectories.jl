module Trajectories

using LinearAlgebra

export generate_oval_trajectory, generate_circle_trajectory, generate_figure8_trajectory, generate_crossing_trajectories, generate_merging_trajectories

"""
Generate an oval trajectory with optional phase shift
"""
function generate_oval_trajectory(num_points::Int; 
                                center::Vector{Float64}=[0.0, 0.0],
                                radii::Vector{Float64}=[2.0, 1.0],
                                phase_shift::Float64=0.0)
    # Generate time points
    t = range(0, 2π, length=num_points)
    
    # Create oval trajectory with phase shift
    x = center[1] .+ radii[1] * sin.(t .+ phase_shift)
    y = center[2] .+ radii[2] * cos.(t .+ phase_shift)
    
    return [x y]'
end

"""
Generate a circular reference trajectory
"""
function generate_circle_trajectory(num_points::Int;
                                 radius::Float64=5.0,
                                 center::Vector{Float64}=[0.0, 0.0])
    return generate_oval_trajectory(num_points, 
                                  center=center, 
                                  radii=[radius, radius])
end

"""
Generate a figure-8 reference trajectory
"""
function generate_figure8_trajectory(num_points::Int;
                                  width::Float64=10.0,
                                  height::Float64=5.0)
    t = range(0, 2π, length=num_points)
    x = width/2 * sin.(t)
    y = height/2 * sin.(2t)
    return [x y]'
end

"""
Generate crossing trajectories for multiple agents
"""
function generate_crossing_trajectories(num_points::Int, num_agents::Int=3;
                                     radius::Float64=2.0,
                                     spacing::Float64=1.0)
    trajectories = Vector{Matrix{Float64}}(undef, num_agents)
    
    for i in 1:num_agents
        # Calculate starting positions evenly distributed around a circle
        angle = 2π * (i-1) / num_agents
        start_pos = radius * [cos(angle), sin(angle)]
        end_pos = -start_pos  # Target is opposite point
        
        # Create trajectory from start to end point
        t = range(0, 1, length=num_points)
        
        # Simple linear interpolation to final position
        x = start_pos[1] .+ (end_pos[1] - start_pos[1]) .* min.(t, 1)
        y = start_pos[2] .+ (end_pos[2] - start_pos[2]) .* min.(t, 1)
        
        trajectories[i] = [x y]'
    end
    
    return trajectories
end

"""
Generate parallel trajectories with merging points
"""
function generate_merging_trajectories(num_points::Int, num_agents::Int=3;
                                    length::Float64=4.0,
                                    spacing::Float64=1.0)
    trajectories = Vector{Matrix{Float64}}(undef, num_agents)
    
    for i in 1:num_agents
        # Starting positions staggered along y-axis
        start_y = spacing * (i - (num_agents+1)/2)
        
        t = range(0, 1, length=num_points)
        
        # Create S-shaped paths that merge in the middle
        x = length * (t .- 0.5)
        y = start_y * cos.(π * t)
        
        trajectories[i] = [x y]'
    end
    
    return trajectories
end

end