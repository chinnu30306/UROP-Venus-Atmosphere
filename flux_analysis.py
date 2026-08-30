import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from functions_new import angle_between_vectors, transit_depth, get_sigma_lambda_from_hitran, extract_profile, get_sigma_lambda
from astropy.constants import R_sun
from poliastro.bodies import Mars, Venus
from classes_new import VenusAtmosphere
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d

import ssl
import urllib.request

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Now import HAPI and other modules
from hapi import db_begin, fetch, absorptionCoefficient_Voigt
# ... rest of your imports

# from astropy.table import Table

# filename = "number_density_data/CO2.xml"
# table = Table.read(filename, format="votable")
# print(table.colnames)

# for col in table.colnames:
#     print(col, "unit =", table[col].unit, "description =", table[col].description)


species = ['CO.xml', 'CO2.xml', 'H2O.xml', 'HCL.xml', 'HF.xml']
path = Path("/Users/chinmayib/git/UROP-Venus-Atmosphere/number_density_data")
profiles = {}
species_list = []

for filename in species:
    filepath = path / filename
    profile = extract_profile(filepath)
    species_name = filepath.stem
    profiles[species_name] = profile
    species_list.append(species_name)

alt_km, temp_K, pres_Pa, dens_kgpmc = np.loadtxt("VIRA_noheader.csv", dtype=float, delimiter=',', unpack=True, usecols=(0, 1, 2, 3))
pres_atm = pres_Pa / 1.01e5

alt_grid = np.arange(75.0, 110.0 + 5, 5)

temp_VIRA = interp1d(alt_km, temp_K, kind='linear', bounds_error=False, fill_value='extrapolate')(alt_grid)
pres_VIRA = interp1d(alt_km, pres_atm, kind='linear', bounds_error=False, fill_value='extrapolate')(alt_grid)

# alt_grid = alt_grid.tolist()
# temp_VIRA = temp_VIRA.tolist()
# pres_VIRA = pres_VIRA.tolist()

interpolated_profiles = {}

for species_name, profile in profiles.items():
    altitude = np.asarray(profile["alt"], dtype=float)
    pressure = np.asarray(profile["p"], dtype=float)
    temperature = np.asarray(profile["t"], dtype=float)
    number_density = np.asarray(profile["n"], dtype=float)

    sort_idx = np.argsort(altitude)
    altitude = altitude[sort_idx]
    log_p = np.log(pressure[sort_idx])
    log_t = np.log(temperature[sort_idx])
    log_n = np.log(number_density[sort_idx])

    p_grid = np.exp(interp1d(altitude, log_p, kind="linear", bounds_error=False, fill_value="extrapolate")(alt_grid))
    t_grid = np.exp(interp1d(altitude, log_t, kind="linear", bounds_error=False, fill_value="extrapolate")(alt_grid))
    n_grid = np.exp(interp1d(altitude, log_n, kind="linear", bounds_error=False, fill_value="extrapolate")(alt_grid))

    n_interp = interp1d(alt_grid, n_grid, kind="linear", bounds_error=False, fill_value=0.0)
    interpolated_profiles[species_name] = {"alt": alt_grid, "p": p_grid, "t": t_grid, "n": n_interp}


# fig, axes = plt.subplots(3, 1, figsize=(9, 9))

# axes[0].plot(profile["alt"], profile["p"], label='VIRA')
# axes[0].plot(interpolated_profiles['CO']['alt'], interpolated_profiles['CO']['p'], label='SPICAV-SOIR')
# axes[0].set_xlabel("Altitude / km")
# axes[0].set_ylabel("Pressure / mbar")
# axes[0].set_title("Pressure against altitude for 75-110km")
# axes[0].legend()
# axes[0].grid(True, alpha=0.3)

# axes[1].plot(profile["alt"], profile["t"], label='VIRA')
# axes[1].plot(interpolated_profiles['CO']['alt'], interpolated_profiles['CO']['t'], label='SPICAV-SOIR')
# axes[1].set_xlabel("Altitude / km")
# axes[1].set_ylabel("Temperature / K")
# axes[1].set_title("Temperature against altitude for 75-110km")
# axes[1].legend()
# axes[1].grid(True, alpha=0.3)

# for species, prof in interpolated_profiles.items():
#     axes[2].plot(prof["alt"], prof["n"], label=species)

# axes[2].set_title("Number density against altitude for 75-110km from SPICAV-SOIR")
# axes[2].set_xlabel("Altitude / km")
# axes[2].set_ylabel(r"Number density / cm$^{-3}$")
# axes[2].set_yscale("log")
# axes[2].legend()
# axes[2].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()


s_v = Horizons(id=299, location='500@10', epochs={'start': '2030-08-19', 'stop': '2030-08-21', 'step': '1min'})
venus_vectors = s_v.vectors()
datetime_jd = venus_vectors['datetime_jd']
s_v_pos = list(zip(venus_vectors['x'], venus_vectors['y'], venus_vectors['z']))

s_m = Horizons(id=499, location='500@10', epochs={'start': '2030-08-19', 'stop': '2030-08-21', 'step': '1min'})
mars_vectors = s_m.vectors()
s_m_pos = list(zip(mars_vectors['x'], mars_vectors['y'], mars_vectors['z']))

v_m_pos = np.array(s_v_pos) - np.array(s_m_pos)

R_sun_au = R_sun.to("au").value
R_venus_au = Venus.R.to("au").value
R_mars_au = Mars.R.to("au").value

theta, alpha_sun, alpha_venus = [], [], []
for i in range(len(v_m_pos)):
    theta.append(angle_between_vectors(v_m_pos[i], -np.array(s_m_pos[i])))
    alpha_sun.append(np.arctan(R_sun_au / np.linalg.norm(s_m_pos[i])))

mid_transit_idx = int(np.argmin(theta))

# flux_opaque_ppm = []
# for i in range(len(theta)):
#     depth = transit_depth(R_venus_au, v_m_pos[i], alpha_sun[i], theta[i])
#     flux_opaque_ppm.append((1 - depth) * 1e6)

# plt.figure(figsize=(9, 5))
# plt.plot(datetime_jd, np.array(flux_opaque_ppm))
# plt.xlabel("Julian Date")
# plt.ylabel("Flux (ppm)")
# plt.title("Solid-body transit depth")
# plt.grid()
# plt.show()


atmosphere = VenusAtmosphere(R_cloud_top_km=Venus.R.value + 70, alt_min_km=75, alt_max_km=110, species_profiles=interpolated_profiles,)
wavelengths_nm, sigma_lambda = np.array(get_sigma_lambda_from_hitran(resolution_cm=1.0))
sigma_dict_alt = {}
sigma_dict_all = {}

for species_name in interpolated_profiles.keys():
    
    sigma_all_altitudes = []

    for i in alt_grid:
        # Get temperature and pressure at this altitude
        idx = np.argmin(np.abs(alt_grid - i))
        T_alt = temp_VIRA[idx]
        P_alt = pres_VIRA[idx]
        
        wl, sigma = get_sigma_lambda(molecule=species_name, temperature_K=T_alt, pressure_atm=P_alt, resolution_cm=1.0)
        sigma_all_altitudes.append(sigma)
    
    # Store as [n_altitudes, n_wavelengths]
    sigma_dict_alt[species_name] = np.array(sigma_all_altitudes)


# R_eff_au, flux_transparent_ppm = [], []
# v_mid = v_m_pos[mid_transit_idx]
# alpha_sun_mid = alpha_sun[mid_transit_idx]
# theta_mid = theta[mid_transit_idx]

# for i in range(len(wavelengths_nm)):
#     sigma_dict = {}
#     for species in species_list:
#         sigma_dict[species] = sigma_dict_all[species][i]  # cm^2/molecule
    
#     R_eff = atmosphere.effective_radius_au(sigma_dict)
#     R_eff_au.append(R_eff)
    
#     depth_mid = transit_depth(R_eff, v_mid, alpha_sun_mid, theta_mid)
#     flux_transparent_ppm.append((1 - depth_mid) * 1e6)

# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(wavelengths_nm, flux_transparent_ppm, linewidth=2, color='blue')
# ax.set_xlabel("Wavelength (nm)", fontsize=12)
# ax.set_ylabel("Flux (ppm)", fontsize=12)
# ax.set_title("Transmission Spectrum of Venus", fontsize=14)
# ax.grid(True, alpha=0.3)
# plt.show()


# --- Calculate transmission spectrum with altitude-dependent cross-sections ---
species_list = list(interpolated_profiles.keys())
R_eff_au = []
flux_transparent_ppm = []

v_mid = v_m_pos[mid_transit_idx]
alpha_sun_mid = alpha_sun[mid_transit_idx]
theta_mid = theta[mid_transit_idx]

# Loop over wavelengths
for i in range(len(wavelengths_nm)):
    # For each wavelength, create sigma_dict with arrays of cross-sections
    # at each altitude
    sigma_dict = {}
    for species in species_list:
        # Get the cross-section at this wavelength for ALL altitudes
        sigma_dict[species] = sigma_dict_alt[species][:, i]  # Array of length n_altitudes
    
    # Calculate effective radius using altitude-dependent sigma
    R_eff = atmosphere.effective_radius_au(sigma_dict)
    R_eff_au.append(R_eff)
    
    depth_mid = transit_depth(R_eff, v_mid, alpha_sun_mid, theta_mid)
    flux_transparent_ppm.append((1 - depth_mid) * 1e6)
    
    if i % 100 == 0:
        print(f"Wavelength {i+1}/{len(wavelengths_nm)} done")

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(wavelengths_nm, flux_transparent_ppm, linewidth=2, color='blue')
ax.set_xlabel("Wavelength (nm)", fontsize=12)
ax.set_ylabel("Flux (ppm)", fontsize=12)
ax.set_title("Transmission Spectrum of Venus with Altitude-Dependent Cross-Sections", fontsize=14)
ax.grid(True, alpha=0.3)
plt.show()