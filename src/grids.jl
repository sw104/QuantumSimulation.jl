#!/usr/bin/julia
#
# Discrete grids to run simulations on.
export Grid, Grid1D, UniformGrid;

"General abstract type for the implementation of discretised spaces."
abstract type Grid{S<:Space,N,T<:Number} <: AbstractArray{T,N} end;
Base.size(g::Grid) = size(g.grid);
Base.getindex(g::Grid{S,N,T}, I::Vararg{Int,N}) where {S,N,T} =
  get(g.grid, I, zero(T));

"Define Grid1D alias to one dimensional grids for convenience."
Grid1D{S,T} = Grid{S,1,T};
# Use linear indexing for 1D grids and directly access underlying data.
Base.IndexStyle(::Type{<:Grid1D}) = IndexLinear();
Base.getindex(G::Grid1D, I::Int) = G.grid[I][1];
Base.setindex!(G::Grid1D, v, I::Int) = (G.grid[I][1] = v);

"Grid with equally spaced grid points."
struct UniformGrid{S,N,T} <: Grid{S,N,T}
  "Maximum spatial value in each dimension (the minimum is -max)."
  max::Vector{T};

  "Grid spacing in each dimension."
  Δ::Vector{T};
  
  "The grid points."
  grid::Array{Vector{T},N};

  """
      UniformGrid{S,N,T}(max::Vector{T}, Δ::Vector{T}) where {S,N,T}

  Construct uniform grid of two or more dimensions.

  TODO: Finish implementing.
  """
  function UniformGrid{S,N,T}(max::Vector{J}, Δ::Vector{J}) where
            {S,N,T,J<:Number,K<:Number}
    if (length(max) != length(Δ))
      throw(DimensionMismatch((length(max),length(Δ)),
                              "must have same number of maxium values and grid spacings"));
    elseif (length(max) != N)
      throw(DimensionMismatch(length(max), "size of vectors must match dimension of grid"));
    end

    values = LinRange.(-max,Δ,max);
    #grid = zeros(T, Tuple(length.(values)))
    grid = Array{Vector{T},N}(undef, Tuple(length.(values)));

    # Scope through each dimension.
    for i ∈ CartesianIndices(grid)
      for j ∈ 1:length(values) end
    end

    #=grid = zeros(T, Tuple(2floor.(max./Δ).+1));
    for i ∈ 1:N
      grid[i,:] = -max[i]:Δ[i]:max[i];
    end=#
    new(max, Δ, LinRange.(-max,Δ,max));
  end

  """
      UniformGrid{S,1,T}(max::Number, Δ::Number) where {S,T}

  Construct one-dimensional uniform grid.
  """
  function UniformGrid{S,1,T}(max::Number, Δ::Number) where {S,T}
    values = collect(-max:Δ:max);

    len = length(values);

    # Always have an odd number of elements.
    if (len%2 == 0) push!(values, values[len]+Δ); len+=1; end

    grid = Vector{Vector{T}}(undef, len);

    for i ∈ 1:len grid[i] = Vector{T}([values[i]]); end

    new([ max ], [ Δ ], grid);
  end
  
  """
      UniformGrid{S,T,1}(max::Number, Δ::Number) where {S<:MomentumSpace,T}

  Construct one-dimensional uniform grid in momentum space.

  The space is not stored sequentially to allow it to be used with Fast Fourier
  transform functions (such as those implemented by the FFTW package).
  """
  function UniformGrid{S,1,T}(max::Number, Δ::Number) where {S<:MomentumSpace,T}
    k = collect(-max:Δ:max);
    len = length(k);
    
    # Always have an odd number of elements.
    if (len%2 == 0) push!(k, k[len]+Δ); len+=1; end

    # Compute the index of the last negative entry in k.
    lastnegidx = floor(Int,len/2);

    # Split and re-order the array around the zero/sign-change point.
    k1 = [k[i] for i in 1:lastnegidx];
    k2 = [k[i] for i in lastnegidx+1:length(k)]
    k = vcat(k2, k1);
    
    grid = Vector{Vector{T}}(undef, len);

    for i ∈ 1:len grid[i] = Vector{T}([k[i]]); end

    new([ max ], [ Δ ], grid);
  end
end

"""
    getproperty(g::UniformGrid{S,T,1}, s::Symbol) where {S,T}

Overwrite `getproperty()` for one-dimensional `UniformGrid` objects to allow
direct access to the single value stored within the vector.
"""
function Base.getproperty(g::UniformGrid{S,1,T}, s::Symbol) where {S,T}
  if (s === :max || s === :Δ) return Base.getfield(g, s)[1];
  else return Base.getfield(g, s); end
end

"""
    Base.:*(G::UniformGrid{S,1}, m::Real) where {S<:Space}

Multiplying a `UniformGrid` object by a real scalar returns an appropriately
scaled `UniformGrid` object.
"""
Base.:*(G::UniformGrid{S,1,T}, m::Number) where {S,T} =
  UniformGrid{S,1,T}(G.max * m, G.Δ * m);
Base.:*(m::Number, G::UniformGrid) = Base.:*(G, m);

"""
    UniformGrid{S}(xgrid::UniformGrid{C,N,T}) where
      {S<:MomentumSpace,C<:ConfigurationSpace,N,T}

Construct discrete and uniform momentum space associated with the corresponding
position/configuration space given by `xgrid`.
"""
function UniformGrid{S}(xgrid::UniformGrid{C,N}) where
            {S<:MomentumSpace,C<:ConfigurationSpace,N}
  # Use Δk = (2πħ/x) where ħ = 1.
  Δk = π/xgrid.max;
  kgrid = UniformGrid{MomentumSpace,N,typeof(Δk*1.0)}(floor(Int, length(xgrid)/2) * Δk, Δk);  

  # Sanity check.
  check_same_size(xgrid, kgrid);

  return kgrid;
end

"""
    Uniform1DGrid(xmax::Number, Δx::Number; debug::Bool=false)

Construct appropriate discrete, uniform, one dimensional configuration and
momentum spaces using the parameters passed.

The `xmax` and `Δx` parameters refer to the properties of the configuration
space. An optional keyword boolean parameter `debug` may be passed to print
information about the constructed spaces.
"""
function UniformGrid(xmax::Number, Δx::Number; debug::Bool=false)

  xgrid = UniformGrid{ConfigurationSpace}(xmax, Δx);
  
  if debug println(xgrid, '\n', '\n'); end

  kgrid = UniformGrid{MomentumSpace}(xgrid);

  # Debugging for grids.
  if debug
    println(k, '\n', '\n');
    println("Grid size: x ", length(x), " k ", length(k), '\n');
  end

  return (xgrid, kgrid);
end

"Default to one dimensional grid with `Float64` values."
UniformGrid{S}(max::Number, Δ::Number) where S<:Space = UniformGrid{S,1,typeof(Δ*1.0)}(max,Δ);
"Implement using `Float64` type by default."
UniformGrid{S,1}(max::Number, Δ::Number) where {S<:Space,N} =
  UniformGrid{S,1,typeof(Δ*1.0)}(max,Δ);
"For convinence, specifying the grid space to convert to is optional."
UniformGrid(grid::UniformGrid{S}) where S<:ConfigurationSpace =
  UniformGrid{MomentumSpace}(grid);

"Default grid is one dimensional, uniform, configuration space grid."
Grid(max::Number, Δ::Number) = UniformGrid{ConfigurationSpace,1}(max, Δ);
Grid(grid::UniformGrid) = UniformGrid(grid);

"""
    check_same_size(g1::Grid1D, g2::Grid1D)

Check that the passed discrete, one dimensional spaces are the same size.
"""
function check_same_size(g1::Grid1D, g2::Grid1D)
  if (length(g1) != length(g2))
    throw(DimensionMismatch(string("grids are different sizes: ", length(g1), " and ", length(g2))));
  end
end
