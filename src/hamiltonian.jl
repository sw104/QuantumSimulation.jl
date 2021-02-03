#!/usr/bin/julia
#
# Hamiltonian type definitions.
import LinearAlgebra: checksquare;
import Unitful: 𝐌, 𝐋;

export Hamiltonian, SeparableHamiltonian, set_min_zero;

"Abstract type of Hamiltonian matricies."
abstract type Hamiltonian{T<:Number,TM<:Number,M<:AbstractMatrix{TM}, S<:Space} <:
              AbstractMatrix{TM} end;
Base.size(H::Hamiltonian) = size(H.H);
Base.getindex(H::Hamiltonian, I::Int,J::Int) = H.H[I,J];
Base.setindex!(H::Hamiltonian, v, I::Int, J::Int) = (H.H[I,J] = v);

"Hamiltonian which can be separated into kinetic and potential parts."
struct SeparableHamiltonian{T,TM,M,S} <: Hamiltonian{T,TM,M,S}
  "Hamiltonian in the space `S`."
  H::M;

  "Diagonal components of potential part in configuration space."
  V::Vector{T};

  "Diagonal components on kinetic part in momentum space."
  T::Vector{T};

  """
      SeparableHamiltonian{T,TM,M,S}(H::M, V::AbstractVector,
                                    K::AbstractVector) where {T,TM,M,S}

  Re-implement default constructor with some error checking.
  """
  function SeparableHamiltonian{T,TM,M,S}(H::AbstractMatrix, V::AbstractVector,
                                          K::AbstractVector) where {T,TM,M,S}
    LinearAlgebra.checksquare(H);
    check_same_size(V, K);
    if size(H)[1] != length(V)
      throw(DimensionMismatch("passed matrix size does not match the vector length."));
    end
    new(H, V, K);
  end
end

"""
    SeparableHamiltonian{T}(g::Grid1D{S}) where {S,T}

Initialize blank Hamiltonian in the space `S` using given grid to determine
appropriate Hamiltonian size.
"""
function SeparableHamiltonian{T}(g::Grid1D{S}) where {T<:Number,S}
  len = length(g);
  SeparableHamiltonian{T,T,Matrix{T},S}(zeros(T, len, len),
                                      zeros(T, len), zeros(T, len));
end
SeparableHamiltonian(g::Grid1D) = SeparableHamiltonian{Complex{Float64}}(g);

"""
    SeparableHamiltonian(g::Grid1D{S}, H::AbstractMatrix{T}) where {S,T}

Initialise Hamiltonian in the space `S`, as the passed matrix.
"""
function SeparableHamiltonian(g::Grid1D{S}, H::AbstractMatrix{T}) where {S,T}
  LinearAlgebra.checksquare(H);
  check_same_size(g, H);
  len = length(g);
  SeparableHamiltonian{T,T,typeof(H),S}(H, zeros(T,len), zeros(T,len));
end

"""
    SeparableHamiltonian{T,,TM,M}(xgrid::Grid1D{ConfigurationSpace},
                        kgrid::Grid1D{MomentumSpace}) where M<:AbstractMatrix{T}

Construct Hamiltonian in configuration space with 'standard' KE component.

This assumes the mass of the particle and ħ are both 1.
"""
function SeparableHamiltonian{T,TM,M}(xgrid::Grid1D{C}, kgrid::Grid1D{S}) where
  {T,TM,M<:Matrix{TM},C<:ConfigurationSpace,S<:MomentumSpace}
  check_same_size(xgrid, kgrid);
  SeparableHamiltonian{T,TM,M,C}(-0.5sec_dv(xgrid), zeros(T,length(xgrid)), 0.5kgrid.^2);
end

"""
    function SeparableHamiltonian{T,TM,M}(xgrid::Grid1D{ConfigurationSpace,T1},
                                     kgrid::Grid1D{MomentumSpace,T2},
                                     mass::Number = 1) where
        {T,T1<:Unitful.Quantity,T2<:Unitful.Quantity}

Create Hamiltonian with energy units given by `T` in configuration space with
the 'standard' KE component.

This assumes ħ = 1.
"""
function SeparableHamiltonian{T,TM,M}(xgrid::Grid1D{ConfigurationSpace,T1},
                                 kgrid::Grid1D{MomentumSpace,T2}, m::Number=1) where
  {N<:Number,T<:Unitful.Quantity{N,Unitful.𝐌*Unitful.𝐋^2/Unitful.𝐓^2},TM<:Number,M<:Matrix{TM},
   T1<:Unitful.Quantity,T2<:Unitful.Quantity}
  check_same_size(xgrid, kgrid);

  SeparableHamiltonian{T,TM,M,ConfigurationSpace}(-0.5ustrip.(sec_dv(xgrid))/ustrip(m)*unit(T),
                                                zeros(T,length(xgrid)),
                                                0.5ustrip(kgrid).^2*unit(T));
end

"Convenience constructor for setting default matrix types."
SeparableHamiltonian{T}(xgrid::Grid1D{ConfigurationSpace},
                        kgrid::Grid1D{MomentumSpace}) where T =
  SeparableHamiltonian{T,T,Matrix{T}}(xgrid, kgrid);
SeparableHamiltonian{T,TM}(xgrid::Grid1D{ConfigurationSpace},
                           kgrid::Grid1D{MomentumSpace}) where {T,TM} =
  SeparableHamiltonian{T,TM,Matrix{TM}}(xgrid, kgrid);
SeparableHamiltonian{T}(xgrid::Grid1D{ConfigurationSpace},
                        kgrid::Grid1D{MomentumSpace},m::Number) where T =
  SeparableHamiltonian{T,T,Matrix{T}}(xgrid, kgrid, m);
SeparableHamiltonian{T,TM}(xgrid::Grid1D{ConfigurationSpace},
                           kgrid::Grid1D{MomentumSpace},m::Number) where {T,TM} =
  SeparableHamiltonian{T,TM,Matrix{TM}}(xgrid, kgrid, m);
SeparableHamiltonian(xgrid::Grid1D{ConfigurationSpace},
                     kgrid::Grid1D{MomentumSpace}) =
  SeparableHamiltonian{Complex{Float64},Complex{Float64},Matrix{Complex{Float64}}}(xgrid, kgrid);

"""
    SeparableHamiltonian{T,TM,M}(xgrid::Grid1D{ConfigurationSpace},
                            kgrid::Grid1D{MomentumSpace},
                            V::AbstractVector)

Construct Hamiltonian in configuration space with 'standard' KE component and
a potential given by `V`.

This assumes the mass of the particle and ħ are 1.
"""
function SeparableHamiltonian{T,TM,M}(xgrid::Grid1D{C}, kgrid::Grid1D{S},
                                   V::AbstractVector) where
    {T,TM,M<:Matrix{TM},C<:ConfigurationSpace,S<:MomentumSpace}
  check_same_size(xgrid, kgrid);
  check_same_size(xgrid, V);

  SeparableHamiltonian{T,TM,M,C}(-0.5sec_dv(xgrid) + LinearAlgebra.Diagonal(V),
                              V, 0.5kgrid.^2);
end

SeparableHamiltonian{T}(xgrid::Grid1D{ConfigurationSpace},
                        kgrid::Grid1D{MomentumSpace}, V::AbstractVector) where T =
  SeparableHamiltonian{T,T,Matrix{T}}(xgrid, kgrid, V);
SeparableHamiltonian{T,TM}(xgrid::Grid1D{ConfigurationSpace},
                           kgrid::Grid1D{MomentumSpace}, V::AbstractVector) where {T,TM} =
  SeparableHamiltonian{T,TM,Matrix{TM}}(xgrid, kgrid, V);
SeparableHamiltonian(xgrid::Grid1D{ConfigurationSpace},
                     kgrid::Grid1D{MomentumSpace}, V::AbstractVector{T}) where T =
  SeparableHamiltonian{T,T,Matrix{T}}(xgrid, kgrid, V);
  
"""
    SeparableHamiltonian{T,TM,M}(g::Grid1D{ConfigurationSpace}, V::AbstractVector)

Construct Hamiltonian in configuration space with the potential passed by `V`.
"""
function SeparableHamiltonian{T,TM,M}(g::Grid1D{S}, V::AbstractVector) where
      {T,TM,M<:Matrix{TM},S<:ConfigurationSpace}
  check_same_size(g, V);
  SeparableHamiltonian{T,TM,M,S}(LinearAlgebra.Diagonal(V), V, zeros(T,length(g)));
end

"""
    SeparableHamiltonian{T,TM,M}(g::Grid1D{MomentumSpace}, K::AbstractVector)

Construct Hamiltonian in momentum space with the kinetic component passed by `K`.
"""
function SeparableHamiltonian{T,TM,M}(g::Grid1D{S}, K::AbstractVector) where
      {T,TM,M<:Matrix{TM},S<:MomentumSpace}
  check_same_size(g, K);
  SeparableHamiltonian{T,TM,M,S}(LinearAlgebra.Diagonal(K), zeros(T,length(grid)), K);
end

"""
    SeparableHamiltonian(g::Grid1D{S}, V::AbstractVector{T}) where S<:Space

Construct Hamiltonian, with elements of type `T`, in the space `S` with
the momentum/potential component passed by `V` (depending on the space).
"""
SeparableHamiltonian(g::Grid1D, V::AbstractVector{T}) where T =
  SeparableHamiltonian{T,T,Matrix{T}}(g, V);
SeparableHamiltonian{T}(g::Grid1D, V::AbstractVector) where T =
  SeparableHamiltonian{T,T,Matrix{T}}(g, V);
SeparableHamiltonian{T,TM}(g::Grid1D, V::AbstractVector) where {T,TM} =
  SeparableHamiltonian{T,TM,Matrix{TM}}(g, V);

"""
    Base.:+(H₁::SeparableHamiltonian, H₂::SeparableHamiltonian)

Adding two `SeparableHamiltonian` objects of the same type returns a new
`SeparableHamiltonian`.
"""
Base.:+(H₁::SeparableHamiltonian{S,T,TM,M},
        H₂::SeparableHamiltonian{S,T,TM,M}) where {S,T,TM,M} =
  SeparableHamiltonian{S,T,TM,M}(H₁.H + H₂.H, H₁.V + H₂.V, H₁.T + H₂.T);
"""
    Base.:-(H₁::SeparableHamiltonian, H₂::SeparableHamiltonian)

Subtracting two `SeparableHamiltonian` objects of the same type returns a new
`SeparableHamiltonian`.
"""
Base.:-(H₁::SeparableHamiltonian{S,T,TM,M},
        H₂::SeparableHamiltonian{S,T,TM,M}) where {S,T,TM,M} =
  SeparableHamiltonian{S,T,TM,M}(H₁.H - H₂.H, H₁.V - H₂.V, H₁.T - H₂.T);

"""
    Base.:*(H::SeparableHamiltonian, m::Real)

Multiplying a `SeparableHamiltonian` object by a real scalar returns a
`SeparableHamiltonian` object.
"""
Base.:*(H::SeparableHamiltonian{S,T,TM,M}, m::Real) where {S,T,TM,M} =
  SeparableHamiltonian{S,T,TM,M}(H.H * m, H.V * m, H.T * m);
Base.:*(m::Real, H::SeparableHamiltonian) = Base.:*(H, m);

"""
    Base.:/(H::SeparableHamiltonian, m::Real)

Dividing a `SeparableHamiltonian` object by a real scalar returns a
`SeparableHamiltonian` object.
"""
Base.:/(H::SeparableHamiltonian{S,T,TM,M}, m::Real) where {S,T,TM,M} =
  SeparableHamiltonian{S,T,TM,M}(H.H / m, H.V / m, H.T / m);
Base.:/(m::Real, H::SeparableHamiltonian) = Base.:/(H, m);

"""
Hamiltonian constructors alias to the `SeparableHamiltonian` constructors.

See also: [`SeparableHamiltonian`](@ref)
"""
Hamiltonian(g::Grid1D) = SeparableHamiltonian(g);
Hamiltonian(xgrid::Grid1D{ConfigurationSpace}, kgrid::Grid1D{MomentumSpace}) =
  SeparableHamiltonian(xgrid, kgrid);
Hamiltonian(xgrid::Grid1D{ConfigurationSpace}, kgrid::Grid1D{MomentumSpace},
            V::AbstractVector) = SeparableHamiltonian(xgrid, kgrid, V);
Hamiltonian{T}(xgrid::Grid1D{ConfigurationSpace}, kgrid::Grid1D{MomentumSpace},
               V::AbstractVector) where T =
  SeparableHamiltonian{T}(xgrid, kgrid, V);
Hamiltonian(g::Grid1D, V::AbstractVector) = SeparableHamiltonian(g, V);
Hamiltonian{T}(g::Grid1D, V::AbstractVector) where T = SeparableHamiltonian{T}(g, V);

"""
    check_same_size(g::Grid1D, H::AbstractMatrix)

Check the passed matrix has first dimension equal to grid size.
"""
function check_same_size(g::Grid1D, H::AbstractMatrix)
  if (length(H[1,:]) != length(g))
    throw(DimensionMismatch("passed matrix size does not match grid size."));
  end
end
function check_same_size(H::AbstractMatrix, g::Grid1D)
  check_same_size(g, H)
end
# Note: functions implementing check_same_size() for Grid and AbstractArray
# are implemented in waveobjects.jl

"""
    set_min_zero(H::SeparableHamiltonian)

Shift the potential part of the Hamiltonian such that the minimum potential
value is zero.
"""
function set_min_zero(H::SeparableHamiltonian{S,T,M}) where {S,T,M}
  min = minimum(real.(H.V));
  V = H.V .- min;
  return SeparableHamiltonian{S,T,M}(H .- LinearAlgebra.Diagonal(V), V, H.T);
end
