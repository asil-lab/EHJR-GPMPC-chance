module ReferenceUtils

export create_reference_window

"""
Create a reference window that extends the final position if needed
"""
function create_reference_window(reference::Matrix{Float64}, current_step::Int, horizon::Int)
    num_steps = size(reference, 2)
    window = zeros(size(reference, 1), horizon)
    
    for t in 1:horizon
        # Index into reference, but stay at final position if beyond trajectory
        ref_idx = min(current_step + t - 1, num_steps)
        window[:, t] = reference[:, ref_idx]
    end
    
    return window
end

end # module