module QuantumSimulation
# Include relevant packages. 'import' doesn't seem to be working well for
# individual function names.
import Base: IndexStyle, size, getindex, setindex!, min, max, adjoint,
       getproperty, getfield, +, -, *;
import LinearAlgebra#: Diagonal, SymTridiagonal, norm;
import Unitful#: Quantity, 𝐓;
using Unitful: ustrip, unit;

# Core datatypes and functions.
include("spaces.jl");
include("grids.jl");
include("waveobjects.jl");
include("hamiltonian.jl");
include("energy.jl");
include("ip.jl");
include("simulationtypes.jl");

# Useful potential functions.
include("potentials.jl");

# Basic statistics
include("stats.jl");

# Variations common across simulations.
include("simvariations.jl");

# Simulations
include("lanczos.jl");
include("plot.jl");
include("ss.jl");

end # module
