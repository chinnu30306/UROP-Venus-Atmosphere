"""
absorption_functions.py

Rewritten support functions for the Venus transit / transmission-spectrum
pipeline. Key changes from the previous version:

  * All geometry below this module is in METERS and all cross-sections in
    m^2/molecule; conversions from HITRAN's cm^2/molecule happen at the
    single point they enter the pipeline (build_sigma_interp_dict).
  * Number-density profiles are turned into *interpolators over radius from
    Venus's centre* (r_m), not over a bare altitude array, so they can be
    evaluated directly at any point along a ray (see classes_new.py).
  * Added ph3_profile_from_vmr(): since PH3 (unlike CO/CO2/H2O/HCl/HF) may
    not have its own retrieved density profile, this builds one from a
    constant volume-mixing-ratio (default 20 ppb) applied to the total
    atmospheric density from VIRA T(h), P(h) via the ideal gas law.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import k as K_BOLTZMANN
from astropy.table import Table

import ssl
# Disable SSL certificate verification (HITRAN download workaround)
ssl._create_default_https_context = ssl._create_unverified_context

from hapi import db_begin, fetch, absorptionCoefficient_Voigt

AU_M = 1.495978707e11


# ---------------------------------------------------------------------------
# Toy cross-section models (kept for testing/sanity checks only)
# ---------------------------------------------------------------------------

def sigma_rayleigh(wavelength_nm, sigma_ref=5e-31, lambda_ref_nm=550):
    """Placeholder Rayleigh-scattering continuum, sigma ~ lambda^-4 (m^2)."""
    return sigma_ref * (lambda_ref_nm / wavelength_nm) ** 4


def sigma_with_band(wavelength_nm, band_center_nm, band_width_nm,
                     band_peak_sigma, sigma_ref=5e-31, lambda_ref_nm=550):
    """Rayleigh continuum plus one toy Gaussian absorption band -- for
    illustration/testing only, not a physical cross-section."""
    continuum = sigma_rayleigh(wavelength_nm, sigma_ref, lambda_ref_nm)
    band = band_peak_sigma * np.exp(
        -0.5 * ((wavelength_nm - band_center_nm) / band_width_nm) ** 2
    )
    return continuum + band


# ---------------------------------------------------------------------------
# HITRAN cross-sections
# ---------------------------------------------------------------------------

MOLECULE_IDS = {
    "H2O": 1,
    "CO2": 2,
    "CO": 5,
    "HF": 14,
    "HCL": 15,
    "PH3": 28,
}


def wavelength_grid_nm(wl_min_nm, wl_max_nm, resolution_cm):
    """
    Build a wavelength grid directly, evenly spaced in wavenumber (the
    natural HITRAN axis), without needing to run a throwaway HITRAN fetch
    just to get an x-axis (the old code called get_sigma_lambda_from_hitran
    purely to obtain wavelengths_nm and discarded its sigma_lambda).
    """
    wavenumber_min = 1.0 / (wl_max_nm * 1e-7)
    wavenumber_max = 1.0 / (wl_min_nm * 1e-7)
    wavenumber_cm = np.arange(wavenumber_min, wavenumber_max, resolution_cm)
    wavelengths_nm = 1e7 / wavenumber_cm
    return wavelengths_nm, wavenumber_cm


def get_sigma_lambda(
    molecule,
    temperature_K,
    pressure_atm,
    wl_min_nm=2000,
    wl_max_nm=5000,
    resolution_cm=1.0,
    isotopologue=1,
    cache_dir="hitran_data",
):
    """
    Absorption cross-section (cm^2/molecule) for `molecule` at a single
    (T, P), across [wl_min_nm, wl_max_nm]. Returns (wavelengths_nm, sigma_cm2)
    -- wavelengths_nm is DEscending in nm-space if wavenumber is ascending;
    it is returned sorted ascending in wavelength for convenience.
    """
    if molecule not in MOLECULE_IDS:
        raise ValueError(f"Unknown molecule '{molecule}'; add it to MOLECULE_IDS.")
    mol_id = MOLECULE_IDS[molecule]

    db_begin(cache_dir)

    wavenumber_min = 1.0 / (wl_max_nm * 1e-7)
    wavenumber_max = 1.0 / (wl_min_nm * 1e-7)

    fetch(molecule, mol_id, isotopologue, wavenumber_min, wavenumber_max)

    wavenumber_cm = np.arange(wavenumber_min, wavenumber_max, resolution_cm)

    nu, sigma_cm2 = absorptionCoefficient_Voigt(
        SourceTables=molecule,
        WavenumberGrid=wavenumber_cm,
        Environment={"p": pressure_atm, "T": temperature_K},
        HITRAN_units=True,  # cm^2/molecule
    )

    wavelengths_nm = 1e7 / nu

    # sort ascending in wavelength for downstream interp1d convenience
    order = np.argsort(wavelengths_nm)
    return wavelengths_nm[order], sigma_cm2[order]


def create_interpolator(wavelengths_nm, sigma_lambda):
    """Interpolator for sigma_lambda(wavelength_nm), clamped at the edges."""
    return interp1d(
        wavelengths_nm,
        sigma_lambda,
        kind="linear",
        bounds_error=False,
        fill_value=(sigma_lambda[0], sigma_lambda[-1]),
    )


# ---------------------------------------------------------------------------
# Profile ingestion
# ---------------------------------------------------------------------------

def extract_profile(filename):
    table = Table.read(filename, format="votable")
    profile = table[["alt", "p", "t", "n_sh0.5"]]
    profile.rename_column("n_sh0.5", "n")
    return profile


def build_species_profiles_for_atmosphere(interpolated_profiles, R_venus_m,
                                            n_units_per_cm3=True):
    """
    Convert {species: {"alt": km, "p", "t", "n"}} (as produced by your
    VIRA/SPICAV interpolation block) into the format VenusAtmosphere needs:
    {species: {"n_interp": callable(r_m) -> number density in molecules/m^3}}

    IMPORTANT: this assumes the input "n" arrays are in cm^-3 (the usual
    convention for these retrievals -- n_units_per_cm3=True applies the
    x1e6 conversion to m^-3). If your SPICAV-SOIR files are already in
    m^-3, call with n_units_per_cm3=False.
    """
    species_profiles = {}
    for species_name, prof in interpolated_profiles.items():
        alt_km = np.asarray(prof["alt"], dtype=float)
        n = np.asarray(prof["n"], dtype=float)
        if n_units_per_cm3:
            n = n * 1e6  # cm^-3 -> m^-3

        r_m = R_venus_m + alt_km * 1e3  # altitude (km) -> radius from centre (m)

        n_interp = interp1d(
            r_m, n, kind="linear", bounds_error=False,
            fill_value=(n[0], n[-1]),
        )
        species_profiles[species_name] = {"n_interp": n_interp}
    return species_profiles


def ph3_profile_from_vmr(alt_grid_km, temp_K, pres_atm, R_venus_m, vmr=20e-9):
    """
    Build a PH3 number-density profile from a constant volume mixing ratio
    (default 20 ppb, the reported Venus cloud-deck PH3 abundance) applied to
    the total atmospheric density implied by VIRA T(h), P(h):

        n_total(h) = P(h) / (k_B * T(h))      [ideal gas law, molecules/m^3]
        n_PH3(h)   = vmr * n_total(h)

    Use this if you don't have a dedicated PH3 density retrieval the way you
    do for CO/CO2/H2O/HCl/HF. Returns the same {"n_interp": ...} format as
    build_species_profiles_for_atmosphere, ready to drop into
    species_profiles["PH3"].
    """
    alt_grid_km = np.asarray(alt_grid_km, dtype=float)
    temp_K = np.asarray(temp_K, dtype=float)
    pres_pa = np.asarray(pres_atm, dtype=float) * 1.01325e5  # atm -> Pa

    n_total_m3 = pres_pa / (K_BOLTZMANN * temp_K)
    n_ph3_m3 = vmr * n_total_m3

    r_m = R_venus_m + alt_grid_km * 1e3
    n_interp = interp1d(
        r_m, n_ph3_m3, kind="linear", bounds_error=False,
        fill_value=(n_ph3_m3[0], n_ph3_m3[-1]),
    )
    return {"n_interp": n_interp}


def build_sigma_interp_dict(species_list, sigma_dict_alt, test_alts_km,
                              wl_idx, R_venus_m):
    """
    For a single wavelength index `wl_idx`, build
    {species: interp1d(r_m) -> cross-section (m^2/molecule)}
    from sigma_dict_alt[species] of shape (n_test_alts, n_wavelengths),
    which is in cm^2/molecule (HITRAN_units=True convention).

    This is what lets sigma vary *with altitude* (via T, P) as you walk
    along a ray in VenusAtmosphere.tau_of_b -- rather than picking one
    cross-section for the whole atmosphere.
    """
    r_test_m = R_venus_m + np.asarray(test_alts_km, dtype=float) * 1e3
    sigma_interp_dict = {}
    for species in species_list:
        sigma_cm2 = sigma_dict_alt[species][:, wl_idx]
        sigma_m2 = sigma_cm2 * 1e-4  # cm^2 -> m^2
        sigma_interp_dict[species] = interp1d(
            r_test_m, sigma_m2, kind="linear", bounds_error=False,
            fill_value=(sigma_m2[0], sigma_m2[-1]),
        )
    return sigma_interp_dict
