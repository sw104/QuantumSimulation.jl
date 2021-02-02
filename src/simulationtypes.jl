#!/usr/bin/julia
#
# Definitions of parameters for different simulation methods and types.
export SimParams, SimMethod, SimVariation, SplitStep, Lanczos;

"""
Simulation parameters for simulation completing the time evolution of a system.

These parameters are common to all the simulation methods.
"""
struct SimParams{T<:Number}
  "Time step."
  Δt::T;

  "Total number of time steps."
  steps::UInt;
  "Error tolerance parameter."
  ϵ::Number;

  function SimParams{T}(Δt::T, steps::Real, ϵ::Number=1e-8) where T
    cΔt = ustrip(Δt);
    # Error check the passed arguments.
    if imag(Δt) != 0
      if (real(cΔt) < 0 || imag(cΔt) < 0)
        throw(DomainError(Δt, "timestep must be positive"));
      elseif (real(cΔt) > 0 && imag(cΔt) > 0)
        throw(DomainError(Δt, "timestep must either be real or imaginary"));
      elseif abs(cΔt) == 0
        throw(DomainError(Δt, "timestep must not be zero"));
      end
    elseif cΔt <= 0
      throw(DomainError(Δt, "timestep must be positive"));
    end
    
    steps <= 0 && throw(DomainError(steps, "number of timesteps must be positive"));
    steps % 1 !== 0 && throw(DomainError(steps, "number of timesteps must be an integer"));

    ustrip(ϵ) < 0 && throw(DomainError(ϵ, "error must be positive"));

    new(Δt, steps, ϵ);
  end
end
SimParams(Δt::T, steps::Real, ϵ::Number=1e-8) where T<:Number = SimParams{T}(Δt, steps, ϵ);

"""
Simulation methods.

Concrete instances of SimMethod should implement all of the parameters
specific to the simulation method.
"""
abstract type SimMethod end

"Split step simulation method."
struct SplitStep <: SimMethod end

"Short iterative Lanczos method."
struct Lanczos <: SimMethod
  "Number of basis functions."
  basisno::UInt;

  "Number of steps before re-generating basis."
  basissteps::UInt;
end

"""
Simulation variations

Modifications to the SimMethod types.

Any variation specific to a SimMethod should inherit from an abstract type
with name '<SimMethod Name>Variation'.

See also: [`SimMethod`](@ref), [`AdaptiveΔt`](@ref), [`LanczosVariation`](@ref)
"""
abstract type SimVariation end;
