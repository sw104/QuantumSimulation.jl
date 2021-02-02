#!/usr/bin/julia
#
# Implementation of simulation variations common across multiple simulation methods.
#
export AdaptiveΔt;

"Adaptive timestep parameters."
struct AdaptiveΔt <: SimVariation
  "Maximum error before reducing Δt."
  ϵmax::Float64;
  "Minimum error before increasing Δt."
  ϵmin::Float64;

  "Factor to reduce Δt by."
  Δtdown::Float64;
  "Factor to increase Δt by."
  Δtup::Float64;
  "Maximum allowable Δt."
  Δtmax::Float64;
  "Minimum allowable Δt."
  Δtmin::Float64;

  "Maximum number of repeated evaluations per iteration."
  maxeval::UInt;

  """
  Maximum number successive iterations where the evaluation limit was reached
  before a different parameter should be altered.

  This might be undefined. As an example, in the Lanczos method, when this
  limit is reached the basis is regenerated.
  """
  maxevallimit::Union{UInt, Nothing}; 

  # TODO Implement inner constructor to check validity of passed arguments.
end
AdaptiveΔt(;ϵmax::Real=1e-8, ϵmin::Real=1e-12,
            Δtdown::Real=1.2, Δtup::Real=1.5, Δtmax::Real=0.5, Δtmin::Real=1e-5,
            maxeval::Real=5, maxevallimit::Real=3) =
  AdaptiveΔt(ϵmax, ϵmin, Δtdown, Δtup, Δtmax, Δtmin, maxeval, maxevallimit);

"""
    adapt_Δt(par::AdaptiveΔt, ϵ::Real, Δt::Number; debug::Bool=true)

Adapt the time step `Δt` using the error in the last propagation step, `ϵ`
according to the settings passed in `par`.
"""
function adapt_Δt(par::AdaptiveΔt, ϵ::Real, Δt::Number; debug::Bool=true)
  # Check passed arguments.
  if (ϵ <= 0)
    throw(DomainError(ϵ, "error must be positive."));
  elseif (imag(Δt) == 0 && real(Δt) <= 0)
    throw(DomainError(Δt, "timestep must be positive."));
  elseif (real(Δt) == 0 && imag(Δt) <= 0)
    throw(DomainError(Δt, "timestep must be positive."));
  elseif (real(Δt) > 0 && imag(Δt) > 0)
    throw(DomainError(Δt, "timestep must be either real or imaginary."));
  end

  # Increase timestep if less than the minimum error.
  if (ϵ < par.ϵmin)
    # Check that increasing the timestep is allowed.
    if (abs(Δt) > par.Δtmax)
      println("Reached maximum allowable timestep");
    else
      Δt *= par.Δtup;
      if (debug) println("Increasing timestep to ", Δt); end
    end
  elseif (ϵ > par.ϵmax)
    # Check that decreasing timestep is allowed.
    if (abs(Δt) < par.Δtmin)
      println("Reached minimum allowable timestep");
    # Decrease timestep.
    else
      Δt /= par.Δtdown;
      if (debug) println("Decreasing timestep to ", Δt); end
    end
  end

  return Δt;
end

"""
    adapt_Δt(par::AdaptiveΔt, ϵ::Real, Δt::Number, evals::Real; debug::Bool=true)

Return adapted time step and decision on if previous evaluation should be
repeated using the number of evaluations already completed.

This returns a `Tuple` with the first value being the new time step to use and
the second a `Boolean` indicating if re-evaluation is required.
"""
function adapt_Δt(par::AdaptiveΔt, ϵ::Real, Δt::Number, evals::Real;
                  debug::Bool=true)
  newΔt = adapt_Δt(par, ϵ, Δt, debug=debug);

  # Repeat evaluation if time step was reduced and maximum evaluations not
  # reached.
  if (newΔt < Δt && evals < par.maxeval) return (newΔt, true);
  else return (newΔt, false); end
end
"""
    adapt_Δt(par::AdaptiveΔt, ϵ::Real, Δt::Number, evals::Vector{T};
             debug::Bool=true) where T<:Union{Int,UInt}

Return adapted time step, re-evaluation decision and if the `maxevallimit` was
reached.

The `evals` vector contains the number of evaluations for all completed
propagation steps (including the current one) with zero values for steps not yet
completed.
"""
function adapt_Δt(par::AdaptiveΔt, ϵ::Real, Δt::Number, evals::Vector{T};
                  debug::Bool=true) where T<:Union{Int,UInt}
  # Make copy of `evals` to avoid changing its values.
  evalscopy = copy(evals);
  # Remove evaluations which haven't occurred yet (evaluation counter is zero).
  filter!(v->(v!=0), evalscopy);
  
  (Δt, reeval) = adapt_Δt(par, ϵ, Δt, evalscopy[end], debug=debug);

  # Forward result if re-evaluation was not required.
  if !reeval return (Δt, false, false); end

  maxevallimitreached = true;
  # Convert to Int, otherwise result will be cast to UInt and will overflow if result is negative.
  min = max(1, length(evalscopy)-convert(Int,par.maxevallimit));
  for i ∈ evalscopy[min]:evalscopy[end]
    if i < par.maxeval
      maxevallimitreached = false;
      break;
    end
  end
  
  return (Δt, reeval, maxevallimitreached);
end
"""
    adapt_Δt(par::AdaptiveΔt, ϵ::Real, Δt::Number, curreval::Real,
             evals::Vector{T}; debug::Bool=true) where T<:Union{Int,UInt}

Here `curreval` contains the index storing the number of evaluations for the
current propagation step within `evals`.
"""
function adapt_Δt(par::AdaptiveΔt, ϵ::Real, Δt::Number, curreval::Real,
                  evals::Vector{T}; debug::Bool=true) where T<:Union{Int,UInt}
  if par.maxevallimit === nothing
    error("maxium evalution limit was not initalised.");
  end

  (Δt, reeval) = adapt_Δt(par, ϵ, Δt, evals[curreval], debug=debug);

  # Forward result if re-evaluation was not required.
  if !reeval return (Δt, false, false); end

  # TODO: Check UInt overflow characteristics here.
  maxevallimitreached = true;
  for i ∈ max(curreval-par.maxevallimit, 1):curreval
    if evals[i] < par.maxeval
      maxevallimitreached = false;
      break;
    end
  end

  return (Δt, reeval, maxevallimitreached);
end
