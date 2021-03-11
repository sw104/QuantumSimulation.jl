#!/usr/bin/julia
#
# Functions for statistical analysis.
#
import Statistics#: std;

export get_stats, get_stats!, get_stats_str, get_stats_str!, get_csv_stats_str;

"""
    get_stats(quantity::Vector)

Compute basic statistics (mean, minimum and maximum) on the collection of data
in `quantity`.

See also: [`get_stats!`](@ref)
"""
function get_stats(quantity::Vector)
  av = sum(quantity)/length(quantity);
  max = maximum(quantity);
  min = minimum(quantity);
  stdv = Statistics.std(quantity, mean=av);

  return (av, max, min, stdv);
end
"""
    get_stats!(quantity::Vector)

Remove zero elements from `quantity` then return `get_stats(quantity)`.

See also: [`get_stats`](@ref)
"""
function get_stats!(quantity::Vector)
  filter!(v->(v!=0), quantity);
  return get_stats(quantity);
end

"""
    get_stats_str(quantity::Vector, name::String="quantity"; long::Bool=true)

Construct a string containing a human readable presentation of the basic
statistics returned from `get_stats(quantity)` using the label `name`.

Passing `long=false` prints the statistics on a single (line feed terminated)
line.

See also: [`get_csv_stats_str`](@ref), [`get_stats_str!`](@ref), [`get_stats`](@ref)
"""
function get_stats_str(quantity::Vector, name::String="quantity"; long::Bool=true)
  return get_stats_str(get_stats(quantity)..., name, long=long);
end
function get_stats_str(av::Number, max::Number, min::Number, stdv::Number,
                       name::String="quantity"; long::Bool=true)
  # Construct string.
  if long
    return string("Average ", name, " = ", av, '\n',
                  "Standard deviation of ", name, " = ", stdv, '\n',
                  "Maximum ", name, " = ", max, '\n',
                  "Minimum ", name, " = ", min, '\n');
  else
    return string(uppercasefirst(name), " statistics:\n",
                  "μ=", av, ", σ=", stdv, ", max=", max, ", min=", min, '\n');
  end
end

"""
    get_stats_str!(quantity::Vector, name::String="quantity"; long::Bool=false)

Construct a string containing a human readable presentation of the basic
statistics returned from `get_stats!(quantity)` using the label `name`.

Passing `long=false` prints the statistics on a single (line feed terminated)
line.

See also: [`get_stats_str`](@ref), [`get_stats!`](@ref)
"""
function get_stats_str!(quantity::Vector, name::String="quantity";
                        long::Bool=true)
  return get_stats_str(get_stats!(quantity)..., name, long=long);
end

"""
    get_csv_stats_str(quantity::Vector, name::String="quantity";
                      label::Bool=true, newline::Bool=true)

Construct a string containing CSV values for the basic statistics returned from
`get_stats(quantity)` using the label `name` as the first CSV column entry.

See also: [`get_stats_str`](@ref), [`get_stats_str!`](@ref), [`get_stats`](@ref)
"""
get_csv_stats_str(quantity::Vector, name::String="quantity";
                  label::Bool=true, newline::Bool=true) =
  get_csv_stats_str(get_stats(quantity)..., name, label=label, newline=newline);
function get_csv_stats_str(av::Number, max::Number, min::Number, stdv::Number,
                           name::String="quantity"; label::Bool=true, newline::Bool=true)
  # Construct string.
  return string((label ? string(name,",") : ""),
                av,",",stdv,",",max,",",min,(newline ? '\n' : ','));
end
function get_csv_stats_str(av::T, max::T, min::T, stdv::T, name::String="quantity";
                           label::Bool=true, newline::Bool=true) where
  {T<:Unitful.Quantity}
  # Construct string.
  return string((label ? string(name," / ",unit(T),",") : ""),
                ustrip(av),",",ustrip(stdv),",", ustrip(max),",",ustrip(min),
                (newline ? '\n' : ','));
end
