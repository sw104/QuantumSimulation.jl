#!/usr/bin/julia
#
# Structures to contain wave functions and collections of wave functions.
#
import FFTW
export WaveObject, WaveFunction, WaveMatrix, WaveBasis, transform;

"""
    WaveObject{S<:Space, N, M, T<:Number, G<:Grid{S,M}} <: AbstractArray{T,N}

Abstract type from which implementations of discrete space representations of
objects related to quantum-mechanical wave functions are derived.

Grids should be defined on the same space and have the same element type as the
`WaveObject`.

`N` is the dimension of the `WaveObject` while `M` is the dimension of the grid.
"""
abstract type WaveObject{T<:Number, N, S, M, G<:Grid{S,M}} <: AbstractArray{T,N} end;
# Use the data array in the composite type implementation to act like a array.
Base.size(W::WaveObject) = size(W.data);
Base.getindex(W::WaveObject{T,N}, I::Vararg{Int,N}) where {T,N} =
  get(W.data, I, zero(T));
Base.setindex!(W::WaveObject{T,N}, v, I::Vararg{Int,N}) where {T,N} = 
  W.data[I...] = v;
# Use linear indexing for 1D `WaveObjects`.
Base.IndexStyle(::Type{<:WaveObject{T,1}}) where T = IndexLinear();
Base.getindex(W::WaveObject{T,1}, I::Int) where T = W.data[I];
Base.setindex!(W::WaveObject{T,1}, v, I::Int) where T = (W.data[I] = v);

"Type alias for matrix-like collections of 1D `WaveFunctions`."
WaveMatrix{T,S,G} = WaveObject{T,2,S,1,G}

"""
`WaveFunction` objects are same dimension as underlying `Grid` object they are
defined upon.
"""
struct WaveFunction{T,N,S,G} <: WaveObject{T,N,S,N,G}
  data::Array{T,N};

  "Spacing between the elements in the discrete grid."
  Δ::Array;

  """
      WaveFunction{T,N,S,G}(data::AbstractArray{K,N}, Δ::AbstractArray)

  Construct `WaveFunction` object with the provided data.
  """
  WaveFunction{T,N,S,G}(data::AbstractArray{K,N}, Δ::AbstractArray) where {T,K,N,S,G} =
    new(data, Δ);

  """
      WaveFunction{T,1,S,G}(data::AbstractVector, Δ::Number) where G<:UniformGrid

  Construct one dimensional `WaveFunction` on a `UniformGrid` which is a special
  case where there is only a single value for `Δ`.
  """
  WaveFunction{T,1,S,G}(data::AbstractVector, Δ::Number) where {T,S,G<:UniformGrid} =
    new(data, [Δ]);
  
end

"""
    WaveFunction{T}(grid::G) where {T,G<:Grid{S,N}} =

Construct `WaveFunction` with zero values everywhere on the grid.
"""
WaveFunction{T}(grid::G) where {T,N,S,G<:Grid{S,N}} =
  WaveFunction{T,N,S,G}(zeros(T,size(grid)), grid.Δ); 

"""
    WaveFunction(grid::Grid)

Construct `WaveFunction` with zero values of type `Complex{Float64}` everywhere.
"""
WaveFunction(grid::Grid) = WaveFunction{Complex{Float64}}(grid);


"""
    WaveFunction{T}(grid::G, ψ::AbstractArray{K,N}) where {T,K,N,S,G<:Grid{S,N}}

Construct wave function on the grid with values at the grid points given by ψ.
"""
function WaveFunction{T}(grid::G, ψ::AbstractArray{K,N}) where {T,K,N,S,G<:Grid{S,N}}
  check_same_size(grid, ψ);
  WaveFunction{T,N,S,G}(ψ, grid.Δ);
end
function WaveFunction{T}(grid::G, ψ::AbstractArray{K,N}) where {A,B,U,T<:Unitful.Quantity{A,B,U},K,N,S,G<:Grid{S,N}}
  check_same_size(grid, ψ);
  WaveFunction{T,N,S,G}(ustrip.(ψ)*unit(T), grid.Δ);
end
"""
    WaveFunction(grid::G, ψ::AbstractArray{T,N}) where {T,N,S,G<:Grid{S,N}}

Construct `WaveFunction` of the same type as the elements of `ψ`.
"""
WaveFunction(grid::G, ψ::AbstractArray{T,N}) where {T,N,S,G<:Grid{S,N}} =
  WaveFunction{T}(grid, ψ);

"""
    WaveFunction(ψ::WaveObject{T,N,S,M,G}) where {T,N,S,M,G}

Construct `WaveFunction` from a compatible `WaveObject`.

Works by dropping any empty dimensions until the `WaveObject` dimensions match
those of the `Grid` it is defined upon.
"""
function WaveFunction{TN}(ψ::WaveObject{T,N,S,M,G}) where {TN,T,N,S,M,G}
  Δ = ψ.Δ;
  for i ∈ N:-1:M+1
    ψ = dropdims(convert(Array{TN}, ψ), dims=i);
  end
  WaveFunction{TN,M,S,G}(ψ, Δ);
end
WaveFunction(ψ::WaveObject{T}) where T = WaveFunction{T}(ψ);

"""
    getproperty(ψ::WaveFunction{S,1,T,G}, s::Symbol) where {S,N,T,G<:UniformGrid}

Override `getproperty` so just one spacing is returned when `WaveFunction` is
defined on a one-dimensional `UniformGrid`.
"""
function Base.getproperty(ψ::WaveFunction{T,1,S,G}, s::Symbol) where
          {T,N,S,G<:UniformGrid}
  if s === :Δ return Base.getfield(ψ, s)[1];
  else return Base.getfield(ψ, s); end
end

"Generic basis of `WaveFunction` objects."
struct WaveBasis{T,N,S,M,G<:Grid{S,M}} <: WaveObject{T,N,S,M,G}
  data::Array{T,N};
  
  "Spacing between the elements in the discrete grid."
  Δ::Array;

  """
      WaveBasis{T,N,S,M,G}(data::AbstractArray{T,N}, Δ::AbstractArray)

  Construct `WaveBasis` ensuring that dimensionality is consistent with storing
  `WaveFunction` objects with `M` dimensions.
  """
  function WaveBasis{T,N,S,M,G}(data::AbstractArray{T,N}, Δ::AbstractArray) where
      {T,N,S,M,G}
    if N !== M+1
      throw(ArgumentError("WaveBasis must have exactly one more dimension than grid."));
    end
    new(data, Δ);
  end
  
  """
      WaveBasis{T,2,S,1,G}(data::AbstractMatrix{T}, Δ::Number) where G<:UniformGrid

  Construct `WaveBasis` which will contain one dimensional `WaveFunction`
  objects defined on a `UniformGrid`.

  For such objects the spacing, `Δ` is a single value.
  """
  WaveBasis{T,2,S,1,G}(data::AbstractMatrix{T}, Δ::Number) where {T,S,G<:UniformGrid} =
    new(data, [Δ]);
end

"""
    WaveBasis{T}(grid::G, size::Int) where {T,S,M::Number,G<:Grid{S,M}}

Construct `WaveBasis` object with space for `size` basis elements with entries
of type `T`, on `grid`.
"""
WaveBasis{T}(grid::G, size::Int) where {T,S,M,G<:Grid{S,M}} =
  WaveBasis{T,M+1,S,M,G}(zeros(T,length(grid),size), grid.Δ);

"""
    WaveBasis(grid::Grid, size::Int)

Construct `WaveBasis` with space for `size` basis elements of type
`Complex{Float64}` on `grid`.
"""
WaveBasis(grid::Grid, size::Int) = WaveBasis{Complex{Float64}}(grid, size);


"""
    getproperty(Φ::WaveBasis{T,2,S,1,G}, s::Symbol) where {T,S,G<:UniformGrid}

Override `getproperty` so just one spacing is returned when `WaveBasis` stores
one-dimensional `WaveFunction` objects on a `UniformGrid`.
"""
function Base.getproperty(Φ::WaveBasis{T,2,S,1,G}, s::Symbol) where
         {T,S,G<:UniformGrid}
  if s === :Δ return Base.getfield(Φ, s)[1];
  else return Base.getfield(Φ, s); end
end

"""
    check_same_size(grid::Grid, v::AbstractArray)

Check the size of the grid matches that of the passed array.
"""
function check_same_size(grid::Grid, a::AbstractArray)
  if size(a) != size(grid)
    throw(DimensionMismatch("array must be same size as the grid."));
  end
end
function check_same_size(grid::Grid{S,1}, v::Vector) where S
  if length(grid) != length(v)
    throw(DimensionMismatch("vector must be same length as the grid."));
  end
end
function check_same_size(v::AbstractArray, grid::Grid)
  check_same_size(grid, v)
end
function check_same_size(v::AbstractVector, w::AbstractVector)
  if length(v) != length(w)
    throw(DimensionMismatch("abstract vectors must be same length."));
  end
end

"""
    transform(ψ::WaveFunction{T,N,S}, g::Grid{M}) where
        {T,N,S<:ConfigurationSpace,M<:MomentumSpace}

Construct WaveFunction from `ψ` transformed onto momentum space.
"""
function transform(ψ::WaveFunction{T,N,S}, g::Grid{M}) where
            {T,N,S<:ConfigurationSpace,M<:MomentumSpace}
  return WaveFunction(g, FFTW.fft(ψ));
end
function transform(ψ::WaveFunction{T,N,S}, g::Grid{M,R,Q}) where
    {T<:Unitful.Quantity,N,S<:ConfigurationSpace,M<:MomentumSpace,
     R,A,D,U,Q<:Unitful.Quantity{A,D,U}}
    # Determine quantity of transformed WaveFunction.
    newquant = typeof((1.0+1.0im)*unit(Q)^(-1/2));
    return WaveFunction{newquant}(g, FFTW.fft(ustrip.(ψ)));
end
function transform(ψ::WaveFunction{T,N,S}, g::Grid{S}) where
            {T,N,S<:ConfigurationSpace}
  return transform(ψ, Grid(g));
end
"""
    transform(ψ::WaveFunction{T,N,S}, g::Grid{S}) where {T,N,S<:MomentumSpace}

Construct WaveFunction from `ψ` transformed onto configuration space.
"""
function transform(ψ::WaveFunction{T,N,S}, g::Grid{C}) where
            {T,N,S<:MomentumSpace,C<:ConfigurationSpace}
  return WaveFunction(g, FFTW.ifft(ψ));
end
function transform(ψ::WaveFunction{T,N,S}, g::Grid{C,R,Q}) where
    {T<:Unitful.Quantity,N,S<:MomentumSpace,C<:ConfigurationSpace,
     R,A,D,U,Q<:Unitful.Quantity{A,D,U}}
    # Determine quantity of transformed WaveFunction.
    newquant = typeof((1.0+1.0im)*unit(Q)^(-1/2));
    return WaveFunction{newquant}(g, FFTW.ifft(ustrip.(ψ)));
end
"""
    FFTW.fft(ψ::WaveFunction{Complex{Float64}})

Convert complex valued `WaveFunction` to `Vector` objects prior to calling
`FFTW.fft`.

Workaround for a dispatch bug in FFTW.
"""
FFTW.fft(ψ::WaveFunction{Complex{Float64}}) = FFTW.fft(convert(Vector, ψ));
"""
    FFTW.ifft(ψ::WaveFunction{Complex{Float64}})

Convert complex valued `WaveFunction` to `Vector` objects prior to calling
`FFTW.ifft`.

Workaround for a dispatch bug in FFTW.
"""
FFTW.ifft(ψ::WaveFunction{Complex{Float64}}) = FFTW.ifft(convert(Vector, ψ));

#Base.:*(Φ::WaveMatrix{T,S,G}, ψ::WaveFunction{K,1,S,G}) where {T,K,S,G} =
#  WaveFunction{T,1,S,G}(Base.:*(Φ,convert(Vector,ψ)), ψ);

# Take the real part of a wavefunction.
Base.real(ψ::WaveFunction{Complex{T},N,S,G}) where {T,N,S,G} =
  WaveFunction{Complex{T},N,S,G}(real.(ψ), ψ.Δ);
