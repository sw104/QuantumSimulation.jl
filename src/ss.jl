#!/usr/bin/julia
#
# Implementation of split step propagation method.
#
using Plots: Animation, plot, frame, gif;

export propagate, find_groundstate;

"""
    gen_opr(H::SeparableHamiltonian, par::SimParams)

Generate kinetic and potential operators in momentum and configuration space
(respectively) assuming that Strang splitting will be used.
"""
function gen_opr(H::SeparableHamiltonian, par::SimParams)
  # Compensate for factor of -1 introduced by imaginary time propagation.
  if (imag(par.Δt) != 0) fac=-1
  else fac = 1;
  end

  return (exp.(ustrip.(-im*H.T*par.Δt*fac)), exp.(ustrip.(-0.5im*H.V*par.Δt*fac)));
end

"""
    propagate(sim::SplitStep,
              xgrid::G=Grid(10, 0.01),
              kgrid::UniformGrid{M,1}=Grid(xgrid),
              par::SimParams=SimParams(0.05, 100, 1e-8),
              H::SeparableHamiltonian=SeparableHamiltonian(xgrid, kgrid, 0.5*xgrid.^2),
              ψ::WaveFunction{T,1,C,G}=WaveFunction(xgrid, exp.(-xgrid.^2/2));
              debug::Bool = true) where
        {T,C<:ConfigurationSpace,M<:MomentumSpace,G<:UniformGrid{C,1}}

Propagate the given wave function one time step using the split-step method.

# Arguments
- `sim::SplitStep`: the propagation method. Primarily used for dispatch, but can
  contain method specific simulation parameters.
- `xgrid::UniformGrid{ConfigurationSpace,1}`: grid of configuration space to evolve over.
- `kgrid::UniformGrid{M,1}`: 1D discretised momentum space.
- `par::SimParams`: Simulation parameters common to all simulation methods.
- `H::SeparableHamiltonian`: Hamiltonian to evolve system with respect to.
- `ψ::WaveFunction{T,1,C,G}`: 1D configuration space wave function to evolve.
- `debug::Bool`: If additional debug messages should be shown.
"""
function propagate(sim::SplitStep,
                   xgrid::G=Grid(10, 0.01),
                   kgrid::UniformGrid{M,1}=Grid(xgrid),
                   par::SimParams=SimParams(0.05, 100, 1e-8),
                   H::SeparableHamiltonian=SeparableHamiltonian(xgrid, kgrid, 0.5*xgrid.^2),
                   ψ::WaveFunction{T,1,C,G}=WaveFunction(xgrid, exp.(-xgrid.^2/2));
                   debug::Bool = false) where
        {T,C<:ConfigurationSpace,M<:MomentumSpace,G<:UniformGrid{C,1}}
  # Construct evolution operators from the passed Hamiltonian.
  (opT, opV) = gen_opr(H, par);

  # Work on a copy of the wave function passed with all units stripped.
  #ψ = WaveFunction{Complex{Float64},1,C,G}(ustrip.(ψi), ustrip(ψi.Δ));

  # Half-step in real space.
  ψ .*= opV;

  # FFT to momentum space.
  ϕ = transform(ψ, kgrid);

  # Full step in momentum space.
  ϕ .*= opT;

  # FFT back to position space.
  ψ = transform(ϕ, xgrid);

  # Half-step in real space.
  ψ .*= opV;

  # Renormalise wave function if running in imaginary time.
  if (ustrip(imag(par.Δt)) != 0)
    if debug println("Pre-normalisation norm was ", norm(ψ)); end
    normalise!(ψ);
  end

  # Return wave function of the same type as that passed.
  #return WaveFunction{T,1,C,G}(ψ, ψ.Δ);
  return ψ;
end

"""
    find_groundstate(sim::SplitStep,
                     xgrid::G,
                     kgrid::UniformGrid{M,1},
                     H::SeparableHamiltonian{C},
                     ψi::WaveFunction{T,1,C,G}=WaveFunction(xgrid, exp.(-xgrid.^2/2)),
                     par::SimParams=SimParams(0.1im, 250, 1e-8);
                     debug::Bool = false,
                     debugpath::String = "simrevamp/dev",
                     printstats::Bool = true) where
        {T,C<:ConfigurationSpace,M<:MomentumSpace,G<:UniformGrid{C,1}}


Determine the lowest energy eigenstate of the given potential.
"""
function find_groundstate(sim::SplitStep,
                         xgrid::G,
                         kgrid::UniformGrid{M,1},
                         H::SeparableHamiltonian{K,BT,B,C},
                         ψi::WaveFunction{T,1,C,G}=WaveFunction(xgrid, exp.(-xgrid.^2/2)),
                         par::SimParams=SimParams(0.1im, 250, 1e-8);
                         debug::Bool = false,
                         debugpath::String = "simrevamp/dev",
                         debugpltopts::SplitStepPlotOptions=SplitStepPlotOptions(true, 0.1),
                         printstats::Bool = true) where
        {T,K,BT,B,C<:ConfigurationSpace,M<:MomentumSpace,G<:UniformGrid{C,1}}

  if real(par.Δt) > 0*unit(par.Δt)
    throw(DomainError(par.Δt, "time step must be imaginary."));
  end

  # Create animation for convergence process if debugging.
  if debug
    anim = Animation();
    plt = SimPlot();
  end

  # Allow future expansion for determining multiple eigenstates.
  states = 1;

  # Define WaveBasis to store result in and make copy of passed WaveFunction for
  # operating on.
  Φ = WaveBasis{T}(xgrid, states);
  ψ = deepcopy(ψi);

  for i ∈ 1:states
    # Define vector to store wave function energy at each propagation step.
    E = zeros(typeof(1.0*unit(K)),par.steps+1);
    E[1] = energy(ψ, H);

    for j ∈ 1:par.steps
      if (debug && j%25 == 0) println("Iteration ", j); end

      ψ[:] = propagate(sim, xgrid, kgrid, par, H, ψ);
      E[j+1] = energy(ψ, H);

      if debug
        plot(plt, debugpltopts, xgrid, ψ, H,
            title=string("Step ", j, ", E = ", round(typeof(E[j+1]), E[j+1], sigdigits=6)));
        frame(anim);
      end

      # Stop propagation if energy has converged.
      if abs(E[j+1] - E[j]) < par.ϵ
        if printstats
          println("Converged to eigenstate ", i, " in ", j, " steps.");
          println("Energy found was ", round(typeof(E[j+1]), E[j+1], sigdigits=6));
        end
        break;
      end
    end
    normalise!(ψ);
    Φ[:,i] = ψ;
    #ψ[:] = orthog(ψi, Φ, i);
  end

  # Plot debug animation if requested.
  if debug
    check_path(debugpath);
    gif(anim, string(debugpath, gen_filename(par, xgrid, SplitStep(), debugpltopts), ".gif"), fps=16);
  end

  return Φ;
end
