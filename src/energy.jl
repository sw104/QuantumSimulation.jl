#!/usr/bin/julia
#
# Helper functions for constructing the Hamiltonian.
#
export sec_dv, energy, ke, pe;

"""
    sec_dv(grid::UniformGrid{S,1}) where S

Construct matrix representation of discretised second order partial derivative
acting on position space.
"""
function sec_dv(grid::UniformGrid{S,1}) where S
  rows = length(grid);
  # Second order method.
  #return LinearAlgebra.SymTridiagonal(repeat([-2.0/grid.Δ^2], rows),
  #                      repeat([1.0/grid.Δ^2], rows-1));

  # Second order method explicitly constructed.
  M = zeros(Float64, rows, rows);
  for i ∈ 1:rows
    # Set incomplete end values to zero to prevent introduction of spurious
    # boundary terms
    if i > 1 && i <= rows-1
      M[i,i-1] = -1;
    #end
    M[i,i] = 2;
    #if i <= rows-1
      M[i,i+1] = -1;
    end
  end
  #M *= -Rational(1/grid.Δ^2);
  M *= -(1/grid.Δ^2);
  return M;

  #=
  # Infinite order expression for 2nd derivative as per
  #J. Chem. Phys. 96, 1982 (1992) Daniel T. Colbert and William H. Miller 
  # https://aip-scitation-org.ezphost.dur.ac.uk/doi/10.1063/1.462100
  M = Matrix{Float64}(undef, rows, rows);
  for i ∈ 1:rows
    for j ∈ 1:rows
      if (i == j)
        M[i,i] = π^2/3;
      else
        M[i,j] = (-1)^(i-j) * 2/(i-j)^2;
      end
    end
  end
  M *= -1/grid.Δ^2;
  return M;=#

  # 5 point function
  #M = Matrix{Rational}(zero(Rational), rows, rows);
  #M = zeros(big(Rational), rows, rows);
  #=
  M = zeros(Float64, rows, rows);
  for i ∈ 3:rows-3
    M[i,i-2] = 1/12;
    M[i,i+2] = 1/12;
    M[i,i-1] = -4/3;
    M[i,i+1] = -4/3;
    M[i,i] = 5/2;
    #=
    if i > 1
      M[i,i-1] = -4//3;
      if (i > 2) M[i,i-2] = 1//12; end
    end
    M[i,i] = 5//2;
    if i <= rows-1
      M[i,i+1] = -4//3;
      if (i <= rows-2) M[i,i+2] = 1//12; end
    end=#
  end
  #M *= -Rational(1/grid.Δ^2);
  M *= -(1/grid.Δ^2);
  return M;=#
end

"""
    energy(ψ::WaveFunction{T,1,S,G}, H::SeparableHamiltonian) where
            {T,S,G<:UniformGrid}

Calculate energy of the given wave function with the kinetic energy calculated
in momentum space (uses a fast Fourier transform if wave function is in
configuration space).

See also: [`ke`](@ref), [`pe`](@ref)
"""
function energy(ψ::WaveFunction{T,1,S,G}, H::SeparableHamiltonian) where
            {T,S,G<:UniformGrid}
  return ke(ψ, H) + pe(ψ,H);
end
#In the future this function may implement a different underlying energy calculation.
function energy(ψ::WaveFunction{T,1,S,G}, H::SeparableHamiltonian) where
            {T,S<:ConfigurationSpace,G<:UniformGrid}
  return ke(ψ, H) + pe(ψ, H);
end

"""
    energy(ψ::WaveFunction{T,1,S,G}, H::SeparableHamiltonian,
           g::UniformGrid{S,1}) where {T,S<:ConfigurationSpace,G<:UniformGrid}

Calculate energy of the given wave function with the kinetic energy determined
using matrix multiplication in configuration space.

See also: [`ke`](@ref), [`pe`](@ref)
"""
function energy(ψ::WaveFunction{T,1,S,G}, H::SeparableHamiltonian,
                g::UniformGrid{S,1}) where {T,S<:ConfigurationSpace,G<:UniformGrid}
  return ke(ψ, g) + pe(ψ, H);
end
# TODO: Implement method of finding energy for non-separable Hamiltonians.

"""
    ke(ψ::WaveFunction{T,1,S,G}, H::SeparableHamiltonian) where
            {T,S<:MomentumSpace,G<:UniformGrid}

Compute kinetic energy fully in momentum space.

Uses a one-dimensional wave function defined on a `UniformGrid`.
"""
function ke(ψ::WaveFunction{T,1,S,G}, H::SeparableHamiltonian) where
            {T,S<:MomentumSpace,G<:UniformGrid}
  return real(sum(conj(ψ) .* H.T .* ψ)) * ψ.Δ;
end

"""
    ke(ψ::WaveFunction{T,1,S,G}, g::UniformGrid{S,1}) where
      {T,S<:ConfigurationSpace,G<:UniformGrid}

Compute kinetic energy in configuration space using E_k = \frac{1}{2} p^2
where the wave function is defined on a one-dimensional `UniformGrid`.
"""
function ke(ψ::WaveFunction{T,1,S,G}, g::UniformGrid{S,1}) where
            {T,S<:ConfigurationSpace,G<:UniformGrid}
  return real(ψ' * -0.5sec_dv(g) * ψ) * ψ.Δ;
end

"""
    ke(ψ::WaveFunction{T,1,S,G}, H::SeparableHamiltonian) where
      {T,S<:ConfigurationSpace,G<:UniformGrid}

Compute kinetic energy using a Fourier transform to momentum space and then
using the kinetic component of Hamiltonian.

The wave function will be defined on a one-dimensional `UniformGrid`.
"""
function ke(ψ::WaveFunction{T,1,S,G}, H::SeparableHamiltonian) where
            {T,S<:ConfigurationSpace,G<:UniformGrid}
  return real(sum(conj(ψ) .* FFTW.ifft(H.T .* FFTW.fft(ψ)))) * ψ.Δ;
end
function ke(ψ::WaveFunction{T,1,S,G}, H::SeparableHamiltonian{Q}) where
            {T<:Unitful.Quantity,S<:ConfigurationSpace,G<:UniformGrid,Q<:Unitful.Quantity}
  return ustrip(real(sum(conj(ψ) .* FFTW.ifft(ustrip(H.T) .* FFTW.fft(ψ)))) * ψ.Δ) * unit(Q);
end

"""
    pe(ψ::WaveFunction{T,1,S,G}, H::SeparableHamiltonian) where
            {T,S<:ConfigurationSpace,G<:UniformGrid}

Calculate potential energy of wave function.

Wave function is in configuration space and defined on one-dimensional uniform
grid.
"""
function pe(ψ::WaveFunction{T,1,S,G}, H::SeparableHamiltonian) where
            {T,S<:ConfigurationSpace,G<:UniformGrid}
  # Calculate energy due to spatial potential.
  Ex = real(sum(conj(ψ) .* H.V .* ψ)) * ψ.Δ;
end

# TODO: Implement determination of P.E. for momentum space wave function.
