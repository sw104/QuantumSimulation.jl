#!/usr/bin/julia
#
# File containing inner product operations on the grid space.
#
export ip, norm, norm2, normalise!, orthog!, orthog;

"""
    do_ip(v::AbstractVector, w::AbstractVector, Δ::Number)

Implementation of inner product calculation.
"""
function do_ip(v::AbstractVector, w::AbstractVector, Δ::Number)
  # Approximate integral over the discrete space.
  return sum(v' * w) * Δ;
end

"""
    check_ip_compatable(ψ::WaveObject{T,N,S,M,G},
                        ϕ::WaveObject{T,K,S,M,G}) where {T,N,S,M,K,G<:UniformGrid}

Check the lengths and uniform grid spacings of the `WaveObjects` are compatible
to allow the calculation of an inner product.
"""
function check_ip_compatable(ψ::WaveObject{T,N,S,M,G},
                             ϕ::WaveObject{P,K,S,M,G}) where {T,P,N,S,M,K,G<:UniformGrid}

  if (N === K) check_same_size(ψ, ϕ);
  else check_ip_compatable_dim(ψ,ϕ); end

  if (ψ.Δ != ϕ.Δ) error("grid spacings of WaveObjects are different"); end
end

"""
    check_same_size(ψ::WaveObject{T,N}, ϕ::WaveObject{P,N}) where {T,P,N}

Check `WaveObjects` of the same dimension have the same size.
"""
function check_same_size(ψ::WaveObject{T,N}, ϕ::WaveObject{P,N}) where {T,P,N}
  if (size(ψ) != size(ϕ))
    throw((DimensionMismatch, "wave functions must be of same size."));
  end
end
function check_same_size(ψ::WaveObject{T,1}, ϕ::WaveObject{P,1}) where {T,P}
  if (length(ψ) != length(ϕ))
    throw(DimensionMismatch("wave functions must be of same size."));
  end
end

"""
    check_ip_compatable_dim(ψ::WaveObject{T,1}, ϕ::WaveObject{P,2}) where {T,P}

Check that the length of the one-dimensional `WaveObject` is the same as the
length of a column of a two-dimensional `WaveObject`.
"""
function check_ip_compatable_dim(ψ::WaveObject{T,1}, ϕ::WaveObject{P,2}) where {T,P}
  if (length(ψ) != length(ϕ[:,1]))
    throw(DimensionMismatch("wave function must be same length as matrix column."));
  end
end
function check_ip_compatable_dim(ϕ::WaveObject{P,2}, ψ::WaveObject{T,1}) where {T,P}
  check_ip_compatable_dim(ψ, ϕ);
end

"""
    ip(ψ::O, ϕ::O) where {T,S,G<:UniformGrid,O<:WaveFunction{T,1,S,G}}

Inner product between two wave functions on the same one-dimensional
`UniformGrid` system.
"""
function ip(ψ::O, ϕ::O) where {T,S,G<:UniformGrid,O<:WaveFunction{T,1,S,G}}
  check_ip_compatable(ψ, ϕ);
  return do_ip(ψ, ϕ, ψ.Δ);
end

"""
    ip(ψ::WaveFunction{T,1,S,G}, ϕ::WaveMatrix{T,S,G}, pos::Int) where
            {T,S,G<:UniformGrid}

Inner product between `WaveFunction` and wave function in column `pos` of the
matrix-like object `ϕ`.

Objects are defined on a one-dimensional `UniformGrid`.
"""
function ip(ψ::WaveFunction{T,1,S,G}, ϕ::WaveMatrix{T,S,G}, pos::Int) where
            {T,S,G<:UniformGrid}
  check_ip_compatable(ψ, ϕ);
  return do_ip(ψ, ϕ[:,pos], ψ.Δ);
end
function ip(ϕ::WaveMatrix{T,S,G}, pos::Int, ψ::WaveFunction{T,1,S,G}) where
            {T,S,G<:UniformGrid}
  ip(ψ, ϕ, pos);
end

"""
    ip(ψ::O, pos::Int, ϕ::O, pos2::Int) where
          {T,S,G<:UniformGrid,O<:WaveMatrix{T,2,S,1,G}}

Inner product between columns of two wave matrices.
"""
function ip(ψ::O, pos::Int, ϕ::O, pos2::Int) where
          {T,S,G<:UniformGrid,O<:WaveMatrix{T,S,G}}
  check_ip_compatable(ψ, ϕ);
  return do_ip(ψ[:,pos], ϕ[:,pos2], ψ.Δ);
end

"""
    norm2(ψ::WaveFunction{T,1,S,G}) where {T,S,G<:UniformGrid}

Calculate norm² of a one-dimensional wave function on a `UniformGrid`.
"""
function norm2(ψ::WaveFunction{T,1,S,G}) where {T,S,G<:UniformGrid}
  return sum(abs2.(ψ) * ψ.Δ);
end

"""
    norm(ψ::WaveFunction{T,1,S,G}) where {T,S,G<:UniformGrid}

Calculate norm of a one-dimensional wave function on a `UniformGrid`.
"""
norm(ψ::WaveFunction{T,1,S,G}) where {T,S,G<:UniformGrid} = √norm2(ψ);

LinearAlgebra.norm(ψ::WaveFunction{T,1,S,G}) where {T,S,G<:UniformGrid} =
  QuantumSimulation.norm(ψ);

"""
    norm2(ψ::WaveMatrix{T,S,G}, pos::Int) where {T,S,G<:UniformGrid}

Calculate norm² of a matrix `WaveMatrix` element in column `pos` defined on a
one-dimensional `UniformGrid`.
"""
function norm2(ϕ::WaveMatrix{T,S,G}, pos::Int) where {T,S,G<:UniformGrid}
  return sum(abs2.(ϕ[:,pos]) * ϕ.Δ);
end

"""
    norm(ψ::WaveMatrix{T,S,G}, pos::Int) where {T,S,G<:UniformGrid}

Calculate norm of a matrix `WaveMatrix` element in column `pos` defined on a
one-dimensional `UniformGrid`.
"""
function norm(ϕ::WaveMatrix{T,S,G}, pos::Int) where {T,S,G<:UniformGrid}
  return √norm2(ϕ, pos);
end

"""
    normalise!(ψ::WaveFunction{T,1,S,G}) where {T,S,G<:UniformGrid}

Normalise the passed wave function defined on a one-dimensional `UniformGrid`.
"""
function normalise!(ψ::WaveFunction{T,1,S,G}) where {T,S,G<:UniformGrid}
  ψ ./= norm(ψ);
end

"""
    orthog!(ψ::WaveFunction{T,1,S,G}, ϕ::WaveMatrix{T,S,G},
                 max::Int=length(ϕ[1,:])) where {T,S,G<:UniformGrid}

Orthogonalise `ψ` with respect to the first `max` unit normalised wave functions
in ϕ.

Both `ψ` and `ϕ` are defined on a one-dimensional `UniformGrid`.

See also: [`orthog`](@ref)
"""
function orthog!(ψ::WaveFunction{T,1,S,G}, ϕ::WaveMatrix{T,S,G},
                 max::Int=length(ϕ[1,:])) where {T,S,G<:UniformGrid}
  # Implement modified Gram Schmidt to prevent accumulation of round-off errors.
  for i ∈ 1:max
    ψ[:] -= ip(ψ, ϕ, i) * ϕ[:,i];
  end
end
"""
    orthog(ψ::WaveFunction{T,1,S,G}, ϕ::WaveMatrix{T,S,G},
                 max::Int=length(ϕ[1,:])) where {T,S,G<:UniformGrid}

Return a copy of `ψ` orthogonalised with respect to the first `max` unit
normalised wave functions in ϕ.

Both `ψ` and `ϕ` are defined on a one-dimensional `UniformGrid`.

See also: [`orthog!`](@ref)
"""
function orthog(ψ::WaveFunction{T,1,S,G}, ϕ::WaveMatrix{T,S,G},
                 max::Int=length(ϕ[1,:])) where {T,S,G<:UniformGrid}
  ψcopy = deepcopy(ψ);
  orthog!(ψcopy, ϕ, max);
  return ψcopy;
end
