import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import k as K_BOLTZMANN
from astropy.table import Table
import miepython

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from hapi import db_begin, fetch, absorptionCoefficient_Voigt

AU_M = 1.495978707e11


def sigma_rayleigh(wavelength_nm, sigma_ref=6.5e-31, lambda_ref_nm=550):
    """Rayleigh-scattering for CO2, sigma ~ lambda^-4 (m^2)."""
    return sigma_ref * (lambda_ref_nm / wavelength_nm) ** 4


def mie_efficiencies(radius_um, wavelength_nm, refractive_index):
    """
    radius_um:        particle radius, microns
    wavelength_nm:     wavelength, nm
    refractive_index:  complex refractive index m = n - i*k
    """
    wavelength_um = wavelength_nm * 1e-3
    size_parameter = 2 * np.pi * radius_um / wavelength_um
    qext, qsca, qback, g = miepython.efficiencies_mx(refractive_index, size_parameter)
    return qext, qsca
 
 
def sigma_mie(radius_um, wavelength_nm, refractive_index):
    radius_m = radius_um * 1e-6
    qext, _ = mie_efficiencies(radius_um, wavelength_nm, refractive_index)
    return qext * np.pi * radius_m ** 2
 
 
def constant_sigma_interp(sigma_value):
    def _interp(r):
        return np.full_like(np.asarray(r, dtype=float), float(sigma_value))
    return _interp



MOLECULE_IDS = {
    "H2O": 1,
    "CO2": 2,
    "CO": 5,
    "SO2": 9,
    "HF": 14,
    "HCL": 15,
    "PH3": 28,
}


def wavelength_grid_nm(wl_min_nm, wl_max_nm, R=2700, oversample=1):
    """
    Generate wavelength grid with constant resolving power R.
    
    Parameters:
    -----------
    wl_min_nm : float
        Minimum wavelength in nm
    wl_max_nm : float
        Maximum wavelength in nm
    R : float
        Resolving power (λ/Δλ). Default is 2700 for G395H.
    oversample : int
        Oversampling factor (1 = Nyquist sampling, >1 oversamples)
        Recommended: oversample=3-5 for better line shapes
    
    Returns:
    --------
    wavelengths_nm : array
        Wavelengths in nm
    wavenumber_cm : array
        Wavenumbers in cm⁻¹
    """
    wavenumber_min = 1.0 / (wl_max_nm * 1e-7)  # cm⁻¹
    wavenumber_max = 1.0 / (wl_min_nm * 1e-7)   # cm⁻¹
    wavenumber_cm = [wavenumber_max]
    
    while wavenumber_cm[-1] > wavenumber_min:
        current_nu = wavenumber_cm[-1]
        delta_nu = current_nu / (R * oversample)
        next_nu = current_nu - delta_nu

        if next_nu < wavenumber_min:
            break
            
        wavenumber_cm.append(next_nu)
    
    wavenumber_cm = np.array(wavenumber_cm)
    wavelengths_nm = 1e7 / wavenumber_cm
    
    return wavelengths_nm, wavenumber_cm


def parse_species_isotopologue(species_key):
    base_molecule, iso_str = species_key.rsplit("_", 1)
    iso_id = int(iso_str)
    if iso_id == 0:
        iso_id = 1  # "_0" = main/default isotopologue = HITRAN iso 1
    return base_molecule, iso_id


def get_sigma_lambda(
    molecule,
    temperature_K,
    pressure_atm,
    wl_min_nm=2000,
    wl_max_nm=5000,
    R=2700,
    oversample=3,
    isotopologue=1,
    cache_dir="hitran_data",
):
    if molecule not in MOLECULE_IDS:
        raise ValueError(f"Unknown molecule '{molecule}'; add it to MOLECULE_IDS.")
    mol_id = MOLECULE_IDS[molecule]

    db_begin(cache_dir)

    wavenumber_min = 1.0 / (wl_max_nm * 1e-7)
    wavenumber_max = 1.0 / (wl_min_nm * 1e-7)

    table_name = f"{molecule}_{isotopologue}"

    fetch(table_name, mol_id, isotopologue, wavenumber_min, wavenumber_max)

    wavelengths_nm, wavenumber_cm = wavelength_grid_nm(wl_min_nm, wl_max_nm, R, oversample)

    nu, sigma_cm2 = absorptionCoefficient_Voigt(
        SourceTables=table_name,
        WavenumberGrid=wavenumber_cm,
        Environment={"p": pressure_atm, "T": temperature_K},
        HITRAN_units=True,
    )

    wavelengths_nm = 1e7 / nu
    order = np.argsort(wavelengths_nm)
    return wavelengths_nm[order], sigma_cm2[order]


def extract_profile(filename):
    table = Table.read(filename, format="votable")
    profile = table[["alt", "p", "t", "n_sh0.5"]]
    profile.rename_column("n_sh0.5", "n")
    return profile


def extract_aerosol_profile(filename):
    table = Table.read(filename, format="votable")
    return table[["alt", "p", "beta"]]
 
 
def aerosol_number_density_from_beta(beta_per_km, wavelength_ref_nm,
                                       radius_um, refractive_index):
    """
    beta(h) = n(h) * sigma_ext(radius, wavelength_ref)
    n(h)    = beta(h) / sigma_ext(radius, wavelength_ref)
    """
    beta_per_m = np.asarray(beta_per_km, dtype=float) / 1e3  # km^-1 -> m^-1
    sigma_ext_m2 = sigma_mie(radius_um, wavelength_ref_nm, refractive_index)
    return beta_per_m / sigma_ext_m2
 
 
def build_aerosol_profile_for_atmosphere(alt_km, beta_per_km, wavelength_ref_nm,
                                           radius_um, refractive_index, R_venus_m):
    n_m3 = aerosol_number_density_from_beta(
        beta_per_km, wavelength_ref_nm, radius_um, refractive_index
    )
    r_m = R_venus_m + np.asarray(alt_km, dtype=float) * 1e3
    n_interp = interp1d(
        r_m, n_m3, kind="linear", bounds_error=False,
        fill_value=(n_m3[0], n_m3[-1]),
    )
    return {"n_interp": n_interp}


def build_species_profiles_for_atmosphere(interpolated_profiles, R_venus_m,
                                            n_units_per_cm3=True):
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
    n_total(h) = P(h) / (k_B * T(h))      [ideal gas law, molecules/m^3]
    n_PH3(h)   = vmr * n_total(h)
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
