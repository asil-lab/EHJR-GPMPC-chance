module GPUtils

using GaussianProcesses
using LinearAlgebra
using Statistics

"""
Compute gradient of GP mean prediction at a point.

# Arguments
- `gp::GPE`: Trained Gaussian Process
- `X::Matrix{Float64}`: d×N matrix of training inputs (columns x_i)
- `y::Vector{Float64}`: N-vector of training targets
- `a::Vector{Float64}`: d-vector at which to evaluate gradient
"""
function predict_y_and_gradient(gp::GPE, a::Vector{Float64})
    # Get training data
    X = gp.x
    y = gp.y
    
    # Extract kernel and hyperparameters
    k = gp.kernel
    ℓ = exp.(k.iℓ2)  # Length scales
    
    # Get pre-computed α = K⁻¹y from GP
    α = gp.alpha
    
    # Compute kernel vector k(a,X)
    k_vec = vec(GaussianProcesses.cov(k, reshape(a, :, 1), X))
    
    # Compute gradient using vectorized operations
    A = a .- X  # d×N matrix of differences
    weights = k_vec .* α  # N-vector
    grad = (-1.0 ./ ℓ.^2) .* (A * weights)  # d-vector
    
    # Get mean prediction
    μ, _ = predict_f(gp, reshape(a, :, 1))
    
    return μ, grad
end

export predict_y_and_gradient

function grad_sigma2_at(gp::GPE, X::AbstractMatrix, a::AbstractVector)
    # extract kernel and length‐scale(s)
    k = gp.kernel
    ℓ = exp.(k.iℓ2)                 # scalar for SEIso, vector for SEArd
    # println(fieldnames(typeof(gp)))  # debug: print kernel fields
    # noise variance σ_n² = exp(2*logNoise)
    # println("logNoise: ", gp.logNoise)  # debug: print logNoise
    # println("params ", fieldnames(typeof(gp.logNoise)))  # debug: print kernel params
    lognoise = gp.logNoise.value
    σ_n2 = exp(2 * lognoise)

    # augmented training covariance
    K = GaussianProcesses.cov(k, X) + σ_n2 * I

    # Cholesky solve for β = K⁻¹ · k(X,a)
    L = cholesky(K)
    k_vec = vec(GaussianProcesses.cov(k, reshape(a, :, 1), X))  # N-vector = [k(a,xᵢ)]
    β = L \ (L' \ k_vec)

    # form differences A[:, i] = (a - xᵢ)
    A = a .- X                            # d×N

    # ARD weights w_j = 1/ℓ_j² (or scalar 1/ℓ²)
    w = 1.0 ./ (ℓ .^ 2)                   # d-vector or scalar

    # gradient: ∇σ²(a) = 2 * ∑_i w ⊙ (a - xᵢ) * β_i
    # implemented as row-wise weighting then matrix multiply
    grad = 2.0 * (A .* w) * β             # d-vector

    return grad
end
export grad_sigma2_at

"""
Get mean, standard deviation and gradient of standard deviation at a point.

# Arguments
- `gp::GPE`: Trained Gaussian Process
- `x::Vector{Float64}`: Point at which to evaluate
"""
function get_prediction_with_grad_sigma(gp::GPE, x::Vector{Float64})
    # Get mean and variance at point
    μ, var = predict_f(gp, reshape(x, :, 1))
    σ = sqrt.(var)[1]  # Convert variance to standard deviation
    
    # Get gradient of σ² at point
    grad_σ² = grad_sigma2_at(gp, gp.x, x)
    
    # Convert gradient of σ² to gradient of σ using chain rule:
    # ∇σ = (1/2σ)∇(σ²)
    grad_σ = grad_σ² ./ (2 * σ)
    
    return μ[1], σ, grad_σ
end

export get_prediction_with_grad_sigma

end
