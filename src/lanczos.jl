#!/usr/bin/julia
#
# Implementation of short iterative Lanczos method.
import LinearAlgebra: eigvals, eigvecs;
export KrylovBasis, ConvergenceVariation, propagate;

"Krylov basis one one-dimensional `UniformGrid`."
struct KrylovBasis{T,S,G<:UniformGrid{S,1}} <: WaveMatrix{T,S,G}
  data::Matrix{T};

  "Eigenvalues of Hamiltonian representation in Krylov space."
  λ::Vector{Float64};
  "Matrix of eigenvectors for the Hamiltonian representation in Krylov space."
  v::Matrix{Float64};

  """
  Grid spacing, requirement for WaveObject implementation and to compute inner
  products.
  """
  Δ::Number;
end
"""
    KrylovBasis{T}(grid::UniformGrid, size::Int)

Construct `KryovBasis` object with space for `size` basis vectors on
`grid`.
"""
KrylovBasis{T}(grid::G, size::Int) where {T,S,G<:UniformGrid{S}} =
  KrylovBasis{T,S,G}(zeros(T,length(grid),size), zeros(size),
                     zeros(size, size), grid.Δ);

"""
    Base.adjoint(Φ::KrylovBasis)

Compute adjoint of `Φ` and return the result as a new `KrylovBasis` object.
"""
Base.adjoint(Φ::KrylovBasis{T,S,G}) where {T,S,G} =
  KrylovBasis{T,S,G}(Base.adjoint(convert(Matrix,Φ)), Φ.λ, Φ.v, Φ.Δ);

"""
    KrylovBasis(grid::Uniform1DGrid, H::Hamiltonian, ψ::WaveFunction, size::Int;
                debug::Bool = false)

Construct `KrylovBasis` and fill with basis vectors and appropriate
eigenvalues/eigenvectors.
"""
function KrylovBasis(grid::G, H::Hamiltonian, ψ::WaveFunction{T,1,S,G},
                     size::Real; debug::Bool = false) where
          {T,S,G<:UniformGrid{S,1}}
  # Non-hermitian parameters: setting both to true prevents assumption of a
  # hermitian Hamiltonian.
  calcβ = false;  # Should the β[i] be explicitly calculated.
  orthog = false; # Explicitly orthogonalise basis at each step.

  if (size%1 != 0) throw(DomainError(size, "basis size must be an integer")); end
  if (size < 0) throw(DomainError(size, "basis size must be positive")); end
  size = convert(Int,size);

  if debug println("Generating Krylov space basis..."); end

  # Initialise empty basis which we'll now populate.
  Φ = KrylovBasis{eltype(ψ)}(grid, size);

  # Hamiltonian representation matrix elements.
  α = zeros(Complex{Float64}, size);    # <ϕ[i]|H|ϕ[i]>
  β = zeros(Complex{Float64}, size-1);  # <ϕ[i]|H|ϕ[i-1]>

  # Vector for storing explicitly calculated β for comparison with the implicit
  # calculation method.
  if (!calcβ && debug)
    β2 = zeros(eltype(ψ), size-1); # <ϕ[i-1]|H|ϕ[i]> 
  end

  # Normalise initial wave function if not already normalised.
  nm = norm(ψ);
  if (abs(1 - nm) > 1e-6)
    Φ[:,1] = ψ ./ nm;
    if (debug) println("Normalisation of initial ψ was required."); end
  else Φ[:,1] = ψ; end

  # Generate the basis vectors and Hamiltonian representation.
  for i ∈ 1:size
    # Apply Hamiltonian to previous basis element.
    # ψ = H|ϕ[i]>
    ψ[:] = ustrip(H * Φ[:,i])*unit(ψ[1]);

    # α[i] = <ϕ[i]|H|ϕ[i]>
    α[i] = ip(Φ, i, ψ);

    # β2[i-1] = <ϕ[i-1]|H|ϕ[i]>
    if (i != 1 && calcβ)
      β[i-1] = ip(Φ, i-1, ψ);
    elseif (i != 1 && !calcβ && debug)
      β2[i-1] = ip(ϕ, i-1, ψ);
    end

    # Stop after coefficient calculation for final run.
    if (i == size) break; end

    # Orthogonalise ψ over last two basis states to get new one.
    # Use modified Gram Schmidt for this procedure.
    if !orthog
      if (i != 1)
        Φ[:,i+1] = ψ - β[i-1] * Φ[:,i-1];
      else
        Φ[:,i+1] = ψ;
      end
      Φ[:,i+1] .-= α[i] * Φ[:,i];
    else # Orthogonalise over all previous basis states.
      orthog!(ψ, Φ, i); 
      Φ[:,i+1] = ψ;
    end

    # Normalise.
    if (!calcβ)
      β[i] = norm(Φ, i+1);
      Φ[:,i+1] ./= β[i];
    else
      if (debug) println("Prenorm Φ[", i+1, "] = ", norm(Φ, i+1)); end
      Φ[:,i+1] ./= norm(Φ, i+1);
    end

    # Debugging info.
    if (debug)
      println("|Φ[", i+1, "]| = ", norm(Φ, i+1));
      for j ∈ 1:i
        println("<ϕ[", j, "]|Φ[", i+1, "]> = ", ip(Φ, j, Φ, i+1));
      end
    end
  end
  # Reset ψ to its initial value.
  ψ[:] = Φ[:,1];
  
  # Print β values from both calculation methods for comparison.
  if (!calcβ && debug) println("β = ", β); println("β2 = ", β2); end

  # Construct Hamiltonian matrix representation from elements.
  Hn = LinearAlgebra.SymTridiagonal(real.(α), real.(β));
  # Debugging.
  if (debug) println("Σ Im(α) = ", sum(imag.(α)), "; Σ Im(β) = ", sum(imag.(β))); end

  # Calculate eigenvalues.
  Φ.λ[:] = LinearAlgebra.eigvals(Hn);
  # Remove extreme eigenvalues. 
  #filter!(e->abs(e)<500, λ);

  # Calculate respective eigenvectors.
  Φ.v[:,:] = LinearAlgebra.eigvecs(Hn, Φ.λ);

  return Φ;
end

"Simulation variations specific to the Lanczos method."
abstract type LanczosVariation <: SimVariation end;

"Convergence on basis application iterations."
struct ConvergenceVariation <: LanczosVariation
  ϵ::Real;
  maxeval::Int;
end

"""
    convert_to_krylov(ψ::WaveFunction{T,S,G}, Φ::KrylovBasis{K,S,G}) where
        {T,K,S,G}

Transform the supplied wave function into the Krylov basis.
"""
function convert_to_krylov(ψ::WaveFunction{T,1,S,G}, Φ::KrylovBasis{K,S,G}) where
        {T,K,S,G}
  ψv = Φ' * ψ;
  ψv ./ LinearAlgebra.norm(ψv);
end

"""
    estimate_error(sim::Lanczos, ψv::Vector, maxaccurate::Int)

Estimate the error in the Lanczos propagation step using the amplitude
of the basis vector components greater than `maxaccurate`. 
"""
function estimate_error(sim::Lanczos, ψv::Vector, maxaccurate::Real)
  if (maxaccurate%1 != 0) throw(DomainError(maxaccurate, "must be integer")); end
  if !(maxaccurate isa Int) maxaccurate = convert(Int, maxaccurate); end
  ϵ = 0.0;
  for i ∈ maxaccurate:sim.basisno
    ϵ += abs(ψv[i]);
  end
  return ϵ;
end


#=function propagate(sim::Lanczos,
                   xgrid::ConfUniform1DGrid=Grid(10, 0.01),
                   par::SimParams=SimParams(0.05, 100, 1e-8),
                   H::SeparableHamiltonian=SeparableHamiltonian(xgrid, 0.5*xgrid.^2),
                   ψ::ConfWaveFunction=WaveFunction(xgrid, exp.(-xgrid.^2/2));
                   debug::Bool = true)

  # Generate a basis for the propagation.
  Φ = KrylovBasis(xgrid, H, ψ, sim.basisno);

end=#

"""
    propagate(sim::Lanczos,
              Φ::KrylovBasis{T,S,G},
              ψ::WaveFunction{T,1,S,G},
              par::SimParams=SimParams(0.05, 100, 1e-8);
              debug::Bool = true) where {T,S,G<:UniformGrid{S}}

Propagate `ψ` using the short iterative Lanczos method with the Krylov basis
vectors `Φ` provided.
"""
function propagate(sim::Lanczos,
                   Φ::KrylovBasis{T,S,G},
                   ψ::WaveFunction{T,1,S,G},
                   par::SimParams=SimParams(0.05, 100, 1e-8);
                   debug::Bool = true) where {T,S,G<:UniformGrid{S}}
  # Explicitly construct complex-valued placeholder object to prevent implicit
  # conversion to types which will raise an inexact error.
  ψv = Vector{Complex{Float64}}(undef, size(Φ)[2])
  ψv[:] = convert_to_krylov(ψ, Φ);

  # Run propagation for the specified number of basis steps.
  for i ∈ 1:sim.basissteps
    ψv[:] = Φ.v * exp(-im * par.Δt * LinearAlgebra.Diagonal(Φ.λ)) * Φ.v' * ψv;
  end

  return WaveFunction{T,1,S,G}(Φ * ψv, ψ.Δ);
end

"""

Propagate `ψ` requiring convergence for each propagation step.
"""
function propagate(sim::Lanczos,
                   conv::ConvergenceVariation,
                   Φ::KrylovBasis{T,S,G},
                   ψ::WaveFunction{T,1,S,G},
                   par::SimParams=SimParams(0.05, 100, 1e-8);
                   adptΔt::Union{AdaptiveΔt,Nothing}=nothing,
                   startΔt::Number=0,
                   debug::Bool = true) where {T,S,G<:UniformGrid{S}}
  # Define containers for representation of ψ in Krylov space used in
  # propagation.
  ψv = Vector{Complex{Float64}}(undef, size(Φ)[2])
  ψvt = Vector{Complex{Float64}}(undef, size(Φ)[2])
  ψvtp = Vector{Complex{Float64}}(undef, size(Φ)[2])
  ψvall = Array{Complex{Float64}, 2}(undef, sim.basissteps, size(Φ)[2])

  ψv[:] = convert_to_krylov(ψ, Φ);

  Δt = zeros(typeof(par.Δt), sim.basissteps+1);
  if (startΔt != 0) Δt[1] = startΔt;
  else Δt[1] = par.Δt; end

  # Vectors to save statistics for each propagation step in.
  eval = zeros(UInt,sim.basissteps); # Number of evaluations of each time step.
  numeigen = zeros(UInt,sim.basissteps); # Number of eigenvectors used.
  error = zeros(sim.basissteps); # Estimated error in propagation.

  if (adptΔt !== nothing) stopprop = false; end

  for i ∈ 1:sim.basissteps
    evaluate = true;
    while (evaluate && eval[i] < conv.maxeval)
      fill!(ψvt, 0.0);
      eval[i] += 1;
      for j ∈ 1:sim.basisno
        ψvt[:] += Φ.v[:,j] * exp(ustrip(-im * Δt[i] * Φ.λ[j])) * Φ.v[:,j]' * ψv;

        # Check if result has converged.
        if (j == 1 || sum(abs2.(ψvt - ψvtp)) > conv.ϵ)
          # Store wave function for convergence analysis in subsequent iterations.
          if (j != sim.basisno)
            ψvtp[:] = ψvt;
          # We did not have convergence after all basis vectors were applied.
          else
            println("Failed to converge at Δt = ", Δt[i], " halving...");
            Δt[i] /= 2;
            # Save number of basis vector used if this is the last evaluation.
            if (eval[i] == conv.maxeval) numeigen[i] = j; end
          end
        else # Propagation converged to new wave function.
          # Estimate error for propagation step.
          error[i] = estimate_error(sim, ψv, j);
          
          # Adapt time step if required.
          if adptΔt !== nothing
            (newΔt, evaluate, stopprop) = adapt_Δt(adptΔt, error[i], Δt[i], i, eval);
            # Set new time step to appropriate index depending on re-evaluation
            # status.
            if evaluate Δt[i] = newΔt;
            elseif (i < sim.basissteps+1) Δt[i+1] = newΔt; end
          # Stop evaluation for this propagation step if not adapting Δt.
          else
            evaluate = false;
            if (i < sim.basissteps+1) Δt[i+1] = Δt[i]; end
          end

          # Save number of eigenvalues used in propagation and end propagation
          # step.
          if (!evaluate)
            numeigen[i] = j;
            break;
          end
        end
      end
    end
    # Assign ψv to the propagated wave function.
    ψv[:] = ψvt;

    # Normalise wavefunction to reduce fluctuations due to additional basis
    # vectors used.
    ψv ./= LinearAlgebra.norm(ψv);

    # Record historical propagation data.
    ψvall[i,:] = ψv;

    # Stop propagation if adapting time step and limit was reached.
    if (adptΔt !== nothing && stopprop)
      if debug println("Maximum evaluation limit reached."); end
      break;
    end
  end

  # Print propagation statistics if debugging.
  if debug
    #println(get_stats_str(eval, "evaluations"));
    #println(get_stats_str(numeigen, "eigenvectors used"));
    #println(get_stats_str(error, "error"));
    #println(get_stats_str(Δt, "time step"));
  end

  return (WaveFunction{T,1,S,G}(Φ * ψv, ψ.Δ), (Δt, error, numeigen, eval));
end
