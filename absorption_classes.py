<<<<<<< HEAD
=======
"""
classes_new.py

Rewritten VenusAtmosphere. The core fix from before: number density AND
cross-section are now evaluated together, at the same altitude, at every
point along each line-of-sight ray -- rather than computing a single
column density and multiplying it by a single cross-section afterward.
That's what lets the 75-110 km altitude structure (both n(h) and, through
T(h)/P(h)-dependent line broadening, sigma(h, lambda)) actually shape the
resulting spectrum.

Geometry convention: everything is in metres, measured as radius from
Venus's centre (r), not "altitude above the cloud deck". This avoids the
R_base/R_top mismatch bug from the previous version, where alt_max_m
(a bare altitude, ~1.1e5 m) was compared directly against impact
parameters b (~6.1e6 m, a radius from the centre).

  R_base = R_venus + alt_min_km   (75 km -> fully opaque disk below this)
  R_top  = R_venus + alt_max_km   (110 km -> fully transparent above this)
"""

>>>>>>> 85825ef69bab8196adfad04e80e0a0bdaf35590b
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz

AU_M = 1.495978707e11


class VenusAtmosphere:
    def __init__(self, R_venus_m, alt_min_km, alt_max_km, species_profiles):
        """
        R_venus_m: Venus's solid-body radius, in METRES (e.g.
                   Venus.R.to_value(u.m) from poliastro/astropy -- don't
                   pass a bare .value, which may be in km).
        alt_min_km, alt_max_km: bounds of the atmospheric shell you're
                   modelling (75, 110 for the PH3 cloud-deck study).
        species_profiles: {species: {"n_interp": callable(r_m) -> m^-3}}
                   as produced by functions_new.build_species_profiles_for_atmosphere
                   / ph3_profile_from_vmr.
        """
        self.R_venus_m = R_venus_m
        self.alt_min_m = alt_min_km * 1e3
        self.alt_max_m = alt_max_km * 1e3

        self.R_base = self.R_venus_m + self.alt_min_m
        self.R_top = self.R_venus_m + self.alt_max_m

        self.species_profiles = species_profiles

    # ---- n(r) ----
    def number_density(self, r, species):
        """r = distance from Venus's centre, metres. Returns molecules/m^3."""
        return self.species_profiles[species]["n_interp"](r)

    # ---- tau(b) for one impact parameter, all species, one wavelength ----
    def tau_of_b(self, b, sigma_interp_dict, n_x=400):
        """
        Optical depth along the chord with impact parameter b (metres from
        Venus's centre). The ray runs from the tangent point (r=b) out to
        r=R_top, mirrored on both sides of the tangent point:

            r(x) = sqrt(b^2 + x^2),   x in [0, x_max],   x_max = sqrt(R_top^2 - b^2)

        Below R_base the disk is treated as fully opaque and never reaches
        this method (see effective_radius_m). Above R_top, transmission is
        defined as exactly 1 (tau = 0).

        Uses a fixed-grid trapezoidal integral (fast, vectorised) rather
        than scipy.quad -- accurate to a few % with n_x ~ 200-500 and much
        faster once you're looping over hundreds of wavelengths, each with
        hundreds of impact parameters.
        """
        if b >= self.R_top:
            return 0.0

        x_max = np.sqrt(self.R_top ** 2 - b ** 2)
        x = np.linspace(0.0, x_max, n_x)
        r = np.hypot(b, x)

        tau_half = np.zeros_like(x)
        for species, sigma_interp in sigma_interp_dict.items():
            n_vals = self.number_density(r, species)
            sigma_vals = sigma_interp(r)
            tau_half += n_vals * sigma_vals

        tau = 2.0 * _trapz(tau_half, x)
        return tau

    def transmission(self, b, sigma_interp_dict, n_x=400):
        return np.exp(-self.tau_of_b(b, sigma_interp_dict, n_x=n_x))

    # ---- R_eff(lambda) ----
    def effective_radius_m(self, sigma_interp_dict, n_b=150, n_x=400):
        """
        R_eff^2 = R_base^2 + 2 * integral_{R_base}^{R_top} b * (1 - T(b)) db

        - Below R_base (75 km): fully opaque, contributes R_base^2 directly.
        - R_base to R_top (75-110 km): partially transmissive annulus,
          weighted by how much starlight each impact parameter blocks.
        - Above R_top (110 km): fully transparent, contributes nothing.
        """
        b_grid = np.linspace(self.R_base, self.R_top, n_b)
        one_minus_T = np.array([
            1.0 - self.transmission(b, sigma_interp_dict, n_x=n_x)
            for b in b_grid
        ])
        annulus_integral = _trapz(b_grid * one_minus_T, b_grid)

        R_eff_sq = self.R_base ** 2 + 2.0 * annulus_integral
        return np.sqrt(R_eff_sq)

    def effective_radius_au(self, sigma_interp_dict, n_b=150, n_x=400):
        return self.effective_radius_m(sigma_interp_dict, n_b=n_b, n_x=n_x) / AU_M