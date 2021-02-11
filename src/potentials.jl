#!/usr/bin/julia
#
# Definitions of useful potential functions for use in simulations.
#
using UnitfulAtomic;
export gs_polarisation, radial_gaussian, radial_gaussian_harmonic_freq,
       tweezer_potential_depth, simple_harmonic;

"""
    gs_polarisation(atom::String, λ::Number)

Find the frequency dependent ground state polarisability for Rubidium and
Caesium at the given wavelength.
"""
function gs_polarisation(atom::String, λ::Number)
  # Apply the relevant data.
  # Data from M. S. Safronova et al., Phys. Rev. A 73, 022505 (2006).
  # https://doi-org.ezphost.dur.ac.uk/10.1103/PhysRevA.73.022505
  if atom == "Cs" || atom == "cs"
    dip12 = 4.4890;
    dip32 = 6.3238;
    δE12 = 0.050932;
    δE32 = 0.053456;
    A = 17.35;
  elseif atom == "Rb" || atom == "rb"
    dip12 = 4.231;
    dip32 = 5.977;
    δE12 = 0.057314;
    δE32 = 0.058396;
    A = 10.54;
  else throw(ArgumentError("only data for Rb and Cs has been implemented")); end

  # Calculate frequency in atomic units.
  ω = ustrip(auconvert(2π*Unitful.c0/λ));

  # Return the ground state dynamic polarisability.
  α = 1/3 * ((δE12 * dip12^2)/(δE12^2 - ω^2)
             + (δE32 * dip32^2)/(δE32^2 - ω^2))
      + A;

  # Apply the atomic units.
  return α * unit(1Unitful.u"a0_au^3");
end

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
    tweezer_potential_depth(P::Number, α::Number, ω₀::Number)

Calculate the trap depth, U₀, for an optical tweezer generated by a Gaussian
laser beam with power `P`, frequency-dependent atomic ground state
polarisability of `α` (in a₀³) and beam waist `ω₀`.

Uses the SI unit conversion of `α(C m²/V) = 4π ϵ₀ a₀³ α(a₀³)` with the potential
depth given by `U₀ = (P α)/(π c ϵ₀ ω₀²)`.
"""
function tweezer_potential_depth(P::Number, α::Number, ω₀::Number)
  return (4*ustrip(α)*Unitful.uconvert(Unitful.u"m", 1Unitful.u"a0_au")^3*P)/
         (Unitful.c0 * ω₀^2);
end

"""
    simple_harmonic(m::Number, ω::Number, x::AbstractVector{T}) where T<:Number

Returns the numerical values for the simple harmonic function with the
parameters given defined over the values contained in `x`.
"""
function simple_harmonic(m::Number, ω::Number, x::AbstractVector{T}) where T<:Number
  return 0.5 * m * ω^2 * x.^2;
end
