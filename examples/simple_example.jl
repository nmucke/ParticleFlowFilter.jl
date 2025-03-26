
using Distributions
using Plots
using OrdinaryDiffEq

function observation_operator(x::Float32)
    return x.^2f0
end

function observation_operator_derivative(x::Float32)
    return 2f0 .* x
end

function likelihood(
    x::Float32, 
    y::Float32,
    noise::Distributions.Distribution,
    observation_operator::Function,
)
    return Distributions.pdf(noise, y .- observation_operator(x))
end

function grad_log_likelihood(
    x::Float32, 
    y::Float32,
    noise::Distributions.Distribution,
    observation_operator::Function,
    observation_operator_derivative::Function,
)
    observation_operator_derivative_particles = observation_operator_derivative.(x)
    observation_particles = observation_operator.(x)

    return observation_operator_derivative_particles .* 1f0./var(noise) .* (y .- observation_particles)
end

function grad_log_prior(
    x::Float32,
    prior::Distributions.Distribution,
)
    return Distributions.gradlogpdf.(prior, x)
end

function kernel(x::Float32, y::Float32; variance::Float32 = 1f0)
    return exp(-0.5f0 .* (x .- y).^2f0 ./ variance)
end

function kernel_derivative(x::Float32, y::Float32; variance::Float32 = 1f0)
    return  - (x .- y) ./ variance .* kernel(x, y; variance = variance)
end

function get_prior_variance(x::Vector{Float32})
    x_mean = mean(x)
    x_centered = x .- x_mean
    return 1f0 ./ (size(x, 1) - 1f0) .* x_centered' * x_centered
end


PRIOR_MEAN = 0.0f0
PRIOR_STD = 1f0
PRIOR_VAR = PRIOR_STD^2

NOISE_MEAN = 0f0
NOISE_STD = 0.5f0
NOISE_VAR = NOISE_STD^2

NUM_PARTICLES = 1000
NUM_STEPS = 250

prior = Distributions.Normal(PRIOR_MEAN, PRIOR_STD)
noise = Distributions.Normal(NOISE_MEAN, NOISE_STD)

prior_particles = rand(prior, (NUM_PARTICLES, 1))
observation_particles = observation_operator.(prior_particles) + rand(noise, NUM_PARTICLES)

obs = 2f0 .+ rand(noise, 1)

ds = 0.05f0

function get_flow(prior_particles, obs, noise, observation_operator, observation_operator_derivative)
    
    d_likelihood_dx = grad_log_likelihood.(
        prior_particles, 
        obs, 
        noise, 
        observation_operator, 
        observation_operator_derivative
    )

    d_prior_dx = grad_log_prior.(prior_particles, prior)
    d_posterior_dx = d_likelihood_dx .- d_prior_dx

    K = kernel.(prior_particles, prior_particles'; variance = PRIOR_VAR)
    dK = kernel_derivative.(prior_particles, prior_particles'; variance = PRIOR_VAR)
    I = sum(1f0 / NUM_PARTICLES .* (K .* d_posterior_dx' .+ dK), dims = 2)

    flow = PRIOR_VAR .* I
    prior_particles = prior_particles .+ ds .* flow

    return PRIOR_VAR .* I
end

function get_ode_RHS(obs, noise, observation_operator, observation_operator_derivative)
    RHS(u, p, t) = get_flow(u, obs, noise, observation_operator, observation_operator_derivative)
    return RHS
end

ode_RHS = get_ode_RHS(obs, noise, observation_operator, observation_operator_derivative)

prob = ODEProblem(ode_RHS, prior_particles, (0f0, NUM_STEPS * ds));#; dt = ds);
sol = solve(prob, Tsit5());

t = sol.t
posterior_particles = sol.u[end]
particle_flow = cat(sol.u..., dims = 2)

Plots.histogram(prior_particles; bins = 100, alpha = 0.5, label = "Prior")
Plots.histogram!(posterior_particles; bins = 100, alpha = 0.5, label = "Posterior")

Plots.plot(particle_flow'; legend = false)
