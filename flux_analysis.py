"""
flux_analysis.py

Flux-loss (ppm) vs wavelength for a Mars-observed Venus transit, using a
ray-traced 75-110 km atmospheric shell (VIRA T/P + SPICAV-SOIR-style
number-density profiles for CO, CO2, H2O, HCl, HF) plus a PH3 profile built
from a constant 20 ppb volume mixing ratio, since PH3 doesn't have its own
retrieved density file here.

Everything downstream of "build species_profiles / sigma_dict_alt" flows
through VenusAtmosphere's ray integral, so n(h) AND sigma(h, lambda) (via
altitude-dependent T, P) both shape the resulting spectrum -- this is the
piece that was missing/broken in the previous version.

NOTE: angle_between_vectors and circle_overlap_area are assumed to exist
in functions_new.py / wherever they did before -- they weren't part of the
code you pasted, so they're untouched and just re-imported here.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from astroquery.jplhorizons import Horizons
import astropy.units as u
from astropy.constants import R_sun
from poliastro.bodies import Venus
from scipy.interpolate import interp1d

from absorption_functions import (
    get_sigma_lambda,
    extract_profile,
    wavelength_grid_nm,
    build_species_profiles_for_atmosphere,
    ph3_profile_from_vmr,
    build_sigma_interp_dict,
)

from geometric_functions import angle_between_vectors, transit_depth
from absorption_classes import VenusAtmosphere

from astropy.table import Table

filename = "number_density_data/AERO.xml"

table = Table.read(filename, format="votable")

print(table.meta)
print(table.colnames)

for col in table.colnames:
    print(col, table[col].unit)
    print(col, table[col].description)

# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------

ALT_MIN_KM = 75.0
ALT_MAX_KM = 110.0
PH3_VMR = 0.8e-9  # strict upper limit of 0.8 ppb

WL_MIN_NM = 4000
WL_MAX_NM = 5000
RESOLUTION_CM = 1.0  # use 0.01 for better quality

DATA_DIR = Path("number_density_data")   # adjust to your actual path
VIRA_FILE = "VIRA_noheader.csv"

# ---------------------------------------------------------------------------
# 1. Venus radius, in METRES (bug fix: previous code risked using km)
# ---------------------------------------------------------------------------

R_venus_m = Venus.R.to_value(u.m)
R_sun_au = R_sun.to_value(u.au)
R_venus_au = Venus.R.to_value(u.au)

# ---------------------------------------------------------------------------
# 2. Load number-density profiles (CO, CO2, H2O, HCl, HF) from your VOTables
# ---------------------------------------------------------------------------

species_files = ["CO.xml", "CO2.xml", "H2O.xml", "HCL.xml", "HF.xml"]
raw_profiles = {}
for filename in species_files:
    filepath = DATA_DIR / filename
    raw_profiles[filepath.stem] = extract_profile(filepath)

# ---------------------------------------------------------------------------
# 3. Load VIRA T(h), P(h) and interpolate everything onto a common altitude
#    grid spanning exactly the 75-110 km shell you're modelling
# ---------------------------------------------------------------------------

alt_km_v, temp_K_v, pres_Pa_v, dens_kgpmc_v = np.loadtxt(
    VIRA_FILE, dtype=float, delimiter=",", unpack=True, usecols=(0, 1, 2, 3)
)
pres_atm_v = pres_Pa_v / 1.01325e5

alt_grid = np.arange(ALT_MIN_KM, ALT_MAX_KM + 0.1, 0.1)

temp_VIRA = interp1d(alt_km_v, temp_K_v, kind="linear",
                      bounds_error=False, fill_value="extrapolate")(alt_grid)
pres_VIRA = interp1d(alt_km_v, pres_atm_v, kind="linear",
                      bounds_error=False, fill_value="extrapolate")(alt_grid)

interpolated_profiles = {}
for species_name, profile in raw_profiles.items():
    altitude = np.asarray(profile["alt"], dtype=float)
    pressure = np.asarray(profile["p"], dtype=float)
    temperature = np.asarray(profile["t"], dtype=float)
    number_density = np.asarray(profile["n"], dtype=float)

    sort_idx = np.argsort(altitude)
    altitude = altitude[sort_idx]
    log_p = np.log(pressure[sort_idx])
    log_t = np.log(temperature[sort_idx])
    log_n = np.log(number_density[sort_idx])

    p_grid = np.exp(interp1d(altitude, log_p, kind="linear",
                              bounds_error=False, fill_value="extrapolate")(alt_grid))
    t_grid = np.exp(interp1d(altitude, log_t, kind="linear",
                              bounds_error=False, fill_value="extrapolate")(alt_grid))
    n_grid = np.exp(interp1d(altitude, log_n, kind="linear",
                              bounds_error=False, fill_value="extrapolate")(alt_grid))

    interpolated_profiles[species_name] = {
        "alt": alt_grid, "p": p_grid, "t": t_grid, "n": n_grid
    }

fig, axes = plt.subplots(3, 1, figsize=(9, 9))

axes[0].plot(profile["alt"], profile["p"], label='VIRA')
axes[0].plot(interpolated_profiles['CO']['alt'], interpolated_profiles['CO']['p'], label='SPICAV-SOIR')
axes[0].set_xlabel("Altitude / km")
axes[0].set_ylabel("Pressure / mbar")
axes[0].set_title("Pressure against altitude for 75-110km")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(profile["alt"], profile["t"], label='VIRA')
axes[1].plot(interpolated_profiles['CO']['alt'], interpolated_profiles['CO']['t'], label='SPICAV-SOIR')
axes[1].set_xlabel("Altitude / km")
axes[1].set_ylabel("Temperature / K")
axes[1].set_title("Temperature against altitude for 75-110km")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

for species, prof in interpolated_profiles.items():
    axes[2].plot(prof["alt"], prof["n"], label=species)

axes[2].set_title("Number density against altitude for 75-110km from SPICAV-SOIR")
axes[2].set_xlabel("Altitude / km")
axes[2].set_ylabel(r"Number density / cm$^{-3}$")
axes[2].set_yscale("log")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 4. Add PH3 from a constant 20 ppb VMR (no dedicated retrieval available)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5. Build the {species: {"n_interp": ...}} dict VenusAtmosphere needs
# ---------------------------------------------------------------------------

species_profiles = build_species_profiles_for_atmosphere(
    interpolated_profiles, R_venus_m, n_units_per_cm3=True
)

species_profiles["PH3"] = ph3_profile_from_vmr(
    alt_grid_km=alt_grid, temp_K=temp_VIRA, pres_atm=pres_VIRA,
    R_venus_m=R_venus_m, vmr=PH3_VMR,
)
 
species_list = list(interpolated_profiles.keys()) + ["PH3"]

atmosphere = VenusAtmosphere(
    R_venus_m=R_venus_m,
    alt_min_km=ALT_MIN_KM,
    alt_max_km=ALT_MAX_KM,
    species_profiles=species_profiles,
)

# ---------------------------------------------------------------------------
# 6. Venus/Mars geometry from JPL Horizons (unchanged)
# ---------------------------------------------------------------------------

s_v = Horizons(id=299, location="500@10",
               epochs={"start": "2030-08-19", "stop": "2030-08-21", "step": "1min"})
venus_vectors = s_v.vectors()
datetime_jd = venus_vectors["datetime_jd"]
s_v_pos = list(zip(venus_vectors["x"], venus_vectors["y"], venus_vectors["z"]))

s_m = Horizons(id=499, location="500@10",
               epochs={"start": "2030-08-19", "stop": "2030-08-21", "step": "1min"})
mars_vectors = s_m.vectors()
s_m_pos = list(zip(mars_vectors["x"], mars_vectors["y"], mars_vectors["z"]))

v_m_pos = np.array(s_v_pos) - np.array(s_m_pos)

theta, alpha_sun = [], []
for i in range(len(v_m_pos)):
    theta.append(angle_between_vectors(v_m_pos[i], -np.array(s_m_pos[i])))
    alpha_sun.append(np.arctan(R_sun_au / np.linalg.norm(s_m_pos[i])))

mid_transit_idx = int(np.argmin(theta))
v_mid = v_m_pos[mid_transit_idx]
alpha_sun_mid = alpha_sun[mid_transit_idx]
theta_mid = theta[mid_transit_idx]

flux_opaque_ppm = []
for i in range(len(theta)):
    depth = transit_depth(R_venus_au, v_m_pos[i], alpha_sun[i], theta[i])
    flux_opaque_ppm.append((1 - depth) * 1e6)

plt.figure(figsize=(9, 5))
plt.plot(datetime_jd, np.array(flux_opaque_ppm))
plt.xlabel("Julian Date")
plt.ylabel("Flux (ppm)")
plt.title("Solid-body transit depth")
plt.grid()
plt.show()

# ---------------------------------------------------------------------------
# 7. Wavelength grid (built directly -- no throwaway HITRAN fetch needed)
# ---------------------------------------------------------------------------

wavelengths_nm, wavenumber_cm = wavelength_grid_nm(WL_MIN_NM, WL_MAX_NM, RESOLUTION_CM)

# ---------------------------------------------------------------------------
# 8. Cross-sections sigma(species, altitude, wavelength) from HITRAN
#    -- computed once per species per test altitude, spanning the full
#    wavelength grid, at that altitude's VIRA T and P.
# ---------------------------------------------------------------------------

test_alts_km = [75., 80., 85., 90., 95., 100., 105., 110.]

sigma_dict_alt = {}  # species -> array [n_test_alts, n_wavelengths], cm^2/molecule
for species_name in species_list:
    sigma_all_altitudes = []
    for alt in test_alts_km:
        idx = np.argmin(np.abs(alt_grid - alt))
        T_alt = temp_VIRA[idx]
        P_alt = pres_VIRA[idx]

        wl, sigma = get_sigma_lambda(
            molecule=species_name,
            temperature_K=T_alt,
            pressure_atm=P_alt,
            wl_min_nm=WL_MIN_NM,
            wl_max_nm=WL_MAX_NM,
            resolution_cm=RESOLUTION_CM,
        )
    
        sigma_on_grid = np.interp(wavelengths_nm, wl, sigma)
        sigma_all_altitudes.append(sigma_on_grid)

    sigma_dict_alt[species_name] = np.array(sigma_all_altitudes)

# ---------------------------------------------------------------------------
# 9. Main loop: for each wavelength, build altitude-varying sigma
#    interpolators and ray-trace R_eff -> flux loss.
# ---------------------------------------------------------------------------

PH3_ONLY_LIST = ["PH3"]

R_eff_au = []
flux_transparent_ppm = []          # all species combined
flux_ph3_only_ppm = []             # PH3 alone, for visual comparison

for i in range(len(wavelengths_nm)):
    # -- full spectrum, all species --
    sigma_interp_dict_all = build_sigma_interp_dict(
        species_list, sigma_dict_alt, test_alts_km, wl_idx=i, R_venus_m=R_venus_m
    )
    R_eff = atmosphere.effective_radius_au(sigma_interp_dict_all)
    R_eff_au.append(R_eff)
    depth_mid = transit_depth(R_eff, v_mid, alpha_sun_mid, theta_mid)
    flux_transparent_ppm.append((1 - depth_mid) * 1e6)

    # -- PH3 only, same ray integral, restricted species list --
    sigma_interp_dict_ph3 = build_sigma_interp_dict(
        PH3_ONLY_LIST, sigma_dict_alt, test_alts_km, wl_idx=i, R_venus_m=R_venus_m
    )
    R_eff_ph3 = atmosphere.effective_radius_au(sigma_interp_dict_ph3)
    depth_ph3 = transit_depth(R_eff_ph3, v_mid, alpha_sun_mid, theta_mid)
    flux_ph3_only_ppm.append((1 - depth_ph3) * 1e6)

# ---------------------------------------------------------------------------
# 10. Plot -- full spectrum in blue, PH3-only contribution overlaid in a
#     distinct colour so its features are easy to pick out.
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(wavelengths_nm, flux_transparent_ppm, linewidth=1.5, color="blue",
         label="All species (CO, CO2, H2O, HCl, HF, PH3)")
ax.plot(wavelengths_nm, flux_ph3_only_ppm, linewidth=1.5, color="crimson",
         label="PH3 only (20 ppb VMR)")
ax.set_xlabel("Wavelength (nm)", fontsize=12)
ax.set_ylabel("Flux (ppm)", fontsize=12)
ax.set_title("Venus Transmission Spectrum, 75-110 km shell\n"
             "(altitude-dependent n and sigma, PH3 at 20 ppb VMR)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("venus_transmission_spectrum.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# Notes on runtime / accuracy trade-offs:
#
# - RESOLUTION_CM=1.0 keeps this runnable; drop to 0.01 (HITRAN-native) only
#   once the pipeline is validated, and expect a large slowdown -- consider
#   restricting WL_MIN_NM/WL_MAX_NM to a narrow window around the specific
#   PH3 feature you care about rather than scanning 2000-5000nm at high res.
# - n_b (impact parameters) and n_x (points along each ray) in
#   VenusAtmosphere control the ray-integral accuracy vs. speed; 150/400
#   is a reasonable starting point -- increase and confirm R_eff / spectrum
#   shape stops changing (convergence check) before trusting fine detail.
# - Sanity check before trusting the full spectrum: at one wavelength, print
#   atmosphere.tau_of_b(b, sigma_interp_dict) for b stepping from R_base to
#   R_top -- tau should be >> 1 near R_base and -> 0 near R_top.
# ---------------------------------------------------------------------------