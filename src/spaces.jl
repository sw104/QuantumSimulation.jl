#!/usr/bin/julia
#
# Abstract spaces types.
#
export ConfigurationSpace, MomentumSpace;

"""
User composite types which implement one of the subtypes of `Space` should be
defined as a `UnionAll` type including that space as one of the parameters.
"""
abstract type Space end;

"Singleton for implementation of configuration/position space."
struct ConfigurationSpace <: Space end;
"Singleton for implementation of momentum space."
struct MomentumSpace <: Space end;
