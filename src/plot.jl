#!/usr/bin/julia
#
# Plotting functions.
#
using RecipesBase;
using Plots: @layout, grid;
#using IJulia; # Required to prevent plotting window opening and crashing.

export check_path, gen_filename, LanczosPlotOptions, SplitStepPlotOptions, SimPlot;

"""
Implementations contain mainly boolean objects specifying what should be
included within plots.

All implementations should at least implement all the properties of
`GeneralPlotOptions`.

See also: [`GeneralPlotOptions`](@ref).
"""
abstract type PlotOptions end;

"Plotting options for a simulation agnostic plot"
struct GeneralPlotOptions <: PlotOptions
  """
  If the real and imaginary parts of the wave function should be plotted in
  addition to the absolute square value.
  """
  components::Bool;

  """
  If the value of the wave function is less than the value of this property at
  the start or end of the plot range, then they will not be plotted.
  """
  extremaϵ::Number;

end
GeneralPlotOptions(;components::Bool=false, extremaϵ::Number=0.1) =
  GeneralPlotOptions(components, extremaϵ);

"Plotting options for split step simulations."
struct SplitStepPlotOptions <: PlotOptions
  components::Bool;
  extremaϵ::Number;
end
SplitStepPlotOptions(;components::Bool=false, extremaϵ::Number=0.1) =
  SplitStepPlotOptions(components, extremaϵ);

"Plotting options for Lanczos simulations."
struct LanczosPlotOptions <: PlotOptions
  components::Bool;
  extremaϵ::Number;
  "Plot basis functions."
  basis::Bool;
  "Plot eigenfunctions of the reduced Hamiltonian."
  eigenfct::Bool;
end

abstract type PlotType end;
struct SimPlot<:PlotType end;

"""
    check_path(path::String)

Check the path given exists and create it if it does not.
"""
function check_path(path::String)
  if (!isdir(path))
    mkpath(path)
  end
end

# Parameters common to all simulation methods.
function gen_filename(p::SimParams, grid::UniformGrid{S,1}) where S
  return string(real(p.Δt) == 0 ? "ti" : "td",
    round(typeof(grid.Δ), grid.Δ, sigdigits=4), "d",
    round(typeof(grid.max), grid.max, sigdigits=4), "mx",
    round(typeof(p.Δt), abs(p.Δt), sigdigits=4), "dt-", p.steps, "s");
end
# Simulation variation method.
function gen_filename(var::Vararg{SimVariation})
  name = string();

  # Iterate through simulation variations and append parameters to filename.
  for i ∈ var
    if i isa AdaptiveΔt
      name *= string("-", round(i.ϵmax, sigdigits=4), "emx",
                     round(i.ϵmin, sigdigits=4), "emn",
                     round(i.Δtdown, sigdigits=4), "tdn",
                     round(i.Δtup, sigdigits=4), "tup", 
                     round(i.Δtmax, sigdigits=4), "tmx",
                     round(i.Δtmin, sigdigits=4), "tmn", 
                     i.maxeval, "mxev");
      if i.maxevallimit != nothing
        name *= string(i.maxevallimit, "mxevlm");
      end
    elseif i isa ConvergenceVariation
      name *= string("-", i.ϵ, "ecnv");
    end
  end

  return name;
end
# Common plotting options.
function gen_filename(plt::PlotOptions)
  if plt.components
    return string("pcmpnt");
  end
  return string();
end
# Simulation specific methods.
function gen_filename(sim::Lanczos, plt::LanczosPlotOptions)
  name = string("-sil-", sim.basisno, "b", sim.basissteps, "bs-" );

  if plt.basis name *= string("pb"); end
  if plt.eigenfct name *= string("pefct"); end
  
  name *= gen_filename(plt);

  return name;
end
function gen_filename(sim::SplitStep, plt::SplitStepPlotOptions)
  name = string("-ss-");

  name *= gen_filename(plt);

  return name
end
"""
    gen_filename(p::SimParams, grid::UniformGrid{S,1},
                      sim::SimMethod, plt::PlotOptions,
                      var::Union{Vararg{SimVariation},Nothing}=nothing) where S

Generate an appropriate filename reflecting the parameters used in the
simulation.
"""
function gen_filename(p::SimParams, grid::UniformGrid{S,1},
                      sim::SimMethod, plt::PlotOptions,
                      var::Vararg{Union{Nothing,SimVariation}}=nothing) where S
  name = gen_filename(p, grid);
  name *= gen_filename(sim, plt);
  if (var[1] != nothing) name *= gen_filename(var...); end

  return name;
end

"""
    get_effective_minmax(v::AbstractVector, ϵ::Float64=0.1)

Return the minimum and maximum indices of `v` below and above which the value of
`v` does not exceed `ϵ`.
"""
function get_effective_minmax(v::AbstractVector, ϵ::Number=0.1)
  len = length(v);
  min = 1;
  max = len;
  # Find the first and last indices of v where v > ϵ.
  for i ∈ 1:len
    if abs(v[i]) > ϵ
      min = i;
      break;
    end;
  end
  for i ∈ len:-1:1
    if abs(v[i]) > ϵ
      max = i;
      break;
    end
  end

  return (min, max);
end
function get_effective_minmax(v::Vararg{AbstractVector}; ϵ=0.1)
  len = length(v);
  mins = Vector{Int}(undef, len);
  maxs = Vector{Int}(undef, len);

  for i ∈ 1:len
    (mins[i], maxs[i]) = get_effective_minmax(v[i], ϵ);
  end

  return (mins, maxs);
end
"""
    get_effective_minmax(m::AbstractMatrix, ϵ::Number=0.1)

Treat the columns of the matrix `m` as vectors.
"""
function get_effective_minmax(m::AbstractMatrix, ϵ::Number=0.1)
  numv = size(m)[2];
  mins = Vector{Int}(undef, numv);
  maxs = Vector{Int}(undef, numv);

  for i ∈ 1:numv
    (mins[i], maxs[i]) = get_effective_minmax(m[:,i], ϵ);
  end

  return (mins, maxs);
end

"""
    get_global_effective_minmax(v::Vararg{AbstractVector}; ϵ=0.1)

Returns the maximum and minimum values found by the respective
`get_effective_minmax()` function.

See also: [`get_effective_minmax`](@ref)
"""
function get_global_effective_minmax(v::Vararg{AbstractVector}; ϵ=0.1)
  (xmins, xmaxs) = get_effective_minmax(v..., ϵ=ϵ);
  return (minimum(xmins), maximum(xmaxs));
end
function get_global_effective_minmax(m::AbstractMatrix, ϵ::Number=0.1)
  (xmins, xmaxs) = get_effective_minmax(m, ϵ);
  return (minimum(xmins), maximum(xmaxs));
end

"""
    get_ranged_minmax(min::Int, max::Int, v::Vararg{AbstractVector})

Find the minimum and maximum of the `v` in the index range `min:max`.

Note: values of the `v` should be real to allow size comparison.
"""
function get_ranged_minmax(min::Int, max::Int, v::Vararg{AbstractVector})
  ymin = 0.0;
  ymax = 0.0;
  for i ∈ 2:numψ
    w = v[i][xmin:xmax];
    t = minimum(w);
    if (t < ymin || i == 1) ymin = t; end
    t = maxmum(w);
    if (t > ymmax || i ==1) ymax = t; end
  end

  return (ymin, ymax);
end

"""
    generate_wavefunction_labels(num::Int)

Generate labels to identify `num` wave functions.
"""
function generate_wavefunction_labels(num::Int)
  labels = Vector{String}(undef, num);
  if (num == 1) labels[1] = "|ψ|²";
  else
    for i ∈ 1:num labels[i] = string("|ψ", i, "|²"); end
  end
  return reshape(labels, 1, num);
end

"Set appropriate axis labels"
@recipe function f(::Type{O}, x::Vector, y::Array) where
                   {T,N,S<:ConfigurationSpace,O<:WaveObject{T,N,S,1}}
  xguide --> "r";
  (x, y)
end
@recipe function f(::Type{O}, x::Vector, y::Array) where
                   {T,N,S<:MomentumSpace,O<:WaveObject{T,N,S,1}}
  xguide --> "k";
  (x, y)
end

"""
    vectors_to_matrix(v::Tuple{Vararg{Vector}}, min::Int=1, max::Int=0)

Construct a matrix from the vectors `v` between the indices `min` and `max`.
"""
function vectors_to_matrix(v::Tuple{Vararg{AbstractVector}}, min::Int=1, max::Int=0)
  num = length(v);
  if (max == 0) max = num; end

  m = Matrix{Number}(undef, max - min + 1, num);
  for i ∈ 1:num m[:,i] = abs2.(v[i][min:max]); end

  return m;
end

"Plot absolute square of wave function across relevant range."
@recipe function f(grid::Grid1D, ψ::Vararg{WaveFunction{T,1}};
                   opts=GeneralPlotOptions()) where T
  (opts, grid, ψ...);
end
# Keyword arguments are not dispatched properly in the recipe chain,
# so define this alternative order of the arguments to implement the function.
# 2020/12/01
@recipe function f(opts::PlotOptions, grid::Grid1D,
                   ψ::Vararg{WaveFunction{T,1}}) where T
  label --> generate_wavefunction_labels(length(ψ));

  (min, max) = get_global_effective_minmax(ψ..., ϵ=opts.extremaϵ);

  (typeof(ψ[1]), grid[min:max], vectors_to_matrix(ψ, min, max));
end

"""
Plot potential function and absolute square of wave function across interesting
wave function range.
"""
@recipe function f(grid::Grid1D, H::SeparableHamiltonian,
                   ψ::Vararg{WaveFunction{T,1}};
                   opts=GeneralPlotOptions()) where T
  (opts, grid, H, ψ...);
end
# Workaround for same issue as above.
@recipe function f(opts::PlotOptions, grid::Grid1D, H::SeparableHamiltonian,
                   ψ::Vararg{WaveFunction{T,1}}) where T
  label --> hcat(["Potential"], generate_wavefunction_labels(length(ψ)));

  (min, max) = get_global_effective_minmax(ψ..., ϵ=opts.extremaϵ);

  (typeof(ψ[1]), grid[min:max], hcat(real.(H.V[min:max]),
                                     vectors_to_matrix(ψ, min, max)));
end

"""
Plot absolute square of wave function components of wave matrix across relevant
range.
"""
@recipe function f(grid::G, ϕ::WaveMatrix{T,S,G},
                   opts::PlotOptions=GeneralPlotOptions()) where {T,S,G}
  label --> generate_wavefunction_labels(size(ϕ)[2]);

  (min, max) = get_global_effective_minmax(ϕ, opts.extremaϵ);

  (typeof(ϕ), grid[min:max], abs2.(ϕ[min:max,:]))
end

"""
Plot potential function and absolute square of wave function components of wave
matrix across relevant range.
"""
@recipe function f(grid::G, H::SeparableHamiltonian, ϕ::WaveMatrix{T,S,G},
                   opts::PlotOptions=GeneralPlotOptions()) where {T,S,G}
  label --> hcat(["Potential"], generate_wavefunction_labels(size(ϕ)[2]));

  (min, max) = get_global_effective_minmax(ϕ, opts.extremaϵ);

  (typeof(ϕ), grid[min:max], hcat(real.(H.V[min:max]),
                                  abs2.(ϕ[min:max,:])))
end

"Plot potential function included in Hamiltonian."
@recipe function f(grid::Grid1D, H::SeparableHamiltonian)
  label --> "Potential"
  (grid, real.(H.V))
end

"Plot real and imaginary parts of wave functions alongside main plot."
@recipe function f(p::SimPlot, opts::PlotOptions, g::G, ψ::WaveFunction{T,1,S,G},
                   H::Union{SeparableHamiltonian,Nothing}=nothing) where {T,S,G}
  if opts.components
    #layout := @layout [ a{0.6w} grid(2,1) ]
    size := (900, 450);
    layout := @layout [ a{0.5w} b ]
    dpi := 200;

    @series begin
      subplot := 2;
      title := "Wave function decomposition";
      ylims := (-1,1);
      realψ = real.(ψ);
      (remin, remax) = get_effective_minmax(realψ, opts.extremaϵ);
      imagψ = imag.(ψ);
      (immin, immax) = get_effective_minmax(imagψ, opts.extremaϵ);
      min = Base.min(immin, remin);
      max = Base.max(immax, remax);
      label := hcat(["Re(ψ)"], ["Im(ψ)"]);
      (g[min:max], hcat(realψ[min:max], imagψ[min:max]))
    end
    # Allow negative y values on subplots.
    #=
    try
      subylims = ylims();
      if subylims[1] >= 0
        subylims = (-subylims[2], subylims[2]);
      end
    catch
    end=#

    #=@series begin
      subplot := 2;
      title := "Re(ψ)";
      ylims := (-1,1);
      realψ = real.(ψ);
      (min, max) = get_effective_minmax(realψ, opts.extremaϵ);
      (g[min:max], realψ[min:max])
    end
    
    @series begin
      subplot := 3;
      title := "Im(ψ)";
      ylims := (-1,1);
      imagψ = imag.(ψ);
      (min, max) = get_effective_minmax(imagψ, opts.extremaϵ);
      (g[min:max], imagψ[min:max])
    end=#
  end

  @series begin
    subplot := 1;
    if (H === nothing) return (opts, g, ψ);
    else return (opts, g, H, ψ); end
  end
end

"Plot Krylov basis and reduced Hamiltonian eigenfunctions from Lanczos method."
@recipe function f(p::SimPlot, opts::LanczosPlotOptions, g::G, ψ::WaveFunction{T,1,S,G},
                   Φ::KrylovBasis{T,S,G}, H::Union{SeparableHamiltonian,Nothing}=nothing) where {T,S,G}
  if opts.components && opts.basis && opts.eigenfct
    layout := @layout [ a{0.7w} grid(2,1); grid(1,2){0.4h}]
    size := (1400, 750);
  elseif !opts.components && opts.basis && opts.eigenfct
    layout := @layout [ a; grid(1,2)]
    size := (1400, 750);
  elseif !opts.components && !opts.basis && !opts.eigenfct
    size := (1400, 750);
  elseif opts.components && !opts.basis && !opts.eigenfct
    layout := @layout [ a{0.6w} grid(2,1) ]
    size := (1400, 750);
  else
    error("the requested plotting format has not yet been implemented.");
  end

  if opts.components
    @series begin
      subplot := 2;
      title := "Re(ψ)";
      realψ = real.(ψ);
      (min, max) = get_effective_minmax(realψ, opts.extremaϵ);
      (g[min:max], realψ[min:max])
    end
    @series begin
      subplot := 3;
      title := "Im(ψ)";
      imagψ = imag.(ψ);
      (min, max) = get_effective_minmax(imagψ, opts.extremaϵ);
      (g[min:max], imagψ[min:max])
    end
  end
  if opts.basis && opts.eigenfct
    # Plot basis functions.
    @series begin
      if !opts.components subplot := 2;
      else subplot := 4; end
      title := "Krylov Basis"
      (g, Φ, opts)
    end

    # Plot eigenfunctions.
    @series begin
      if !opts.components subplot := 3;
      else subplot := 5; end

      # Label the eigenstates by their energies.
      len = length(Φ.λ);
      labels = Vector{String}(undef, len)
      for i ∈ 1:len labels[i] = string("|ψ|² where E = ", round(Φ.λ[i], sigdigits=4)); end
      label := reshape(labels, 1, len);

      title := "Reduced Hamiltonian Eigenstates"
      (g, KrylovBasis{T,S,G}(Φ*Φ.v, Φ.λ, Φ.v, Φ.Δ), opts)
    end
  end

  @series begin
    subplot := 1;
    if (H === nothing) return (opts, g, ψ);
    else return (opts, g, H, ψ); end
  end
end
