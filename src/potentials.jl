#!/usr/bin/julia
#
# Definitions of useful potential functions for use in simulations.
#
export radial_gaussian, radial_gaussian_harmonic_freq, simple_harmonic;

"""
    radial_gaussian(ω₀::Number, U₀::Number, r::AbstractVector{T}) where T<:Number

Returns the numerical values for the radial slice of a Gaussian function with
the parameters given defined over the values contained in `r`.
"""
function radial_gaussian(ω₀::Number, U₀::Number, r::AbstractVector{T}) where T<:Number
  return U₀ * (1 .- exp.(-2r.^2 / ω₀^2));
end

"""
    radial_gaussian_harmonic_freq(ω₀::Number, U₀::Number, m::Number)

Calculate the radial harmonic frequency for the Gaussian potential with the
given parameters
"""
function radial_gaussian_harmonic_freq(ω₀::Number, U₀::Number, m::Number)
  (4U₀/(m * ω₀^2))^(1/2)
end

"""
    simple_harmonic(m::Number, ω::Number, x::AbstractVector{T}) where T<:Number

Returns the numerical values for the simple harmonic function with the
parameters given defined over the values contained in `x`.
"""
function simple_harmonic(m::Number, ω::Number, x::AbstractVector{T}) where T<:Number
  return 0.5 * m * ω^2 * x.^2;
end
