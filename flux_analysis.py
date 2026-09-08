import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
import shutil
from astropy.io.votable import parse
from astroquery.jplhorizons import Horizons
import astropy.units as u
from astropy.constants import R_sun
from poliastro.bodies import Venus
from scipy.interpolate import interp1d
from astropy.io import fits
import pandas as pd
from astropy.table import Table

from absorption_functions import (
    get_sigma_lambda,
    extract_profile,
    extract_w23_long_lat,
    wavelength_grid_nm,
    parse_species_isotopologue,
    build_species_profiles_for_atmosphere,
    ph3_profile_from_vmr,
    build_sigma_interp_dict,
    extract_aerosol_profile,
    build_aerosol_profile_for_atmosphere,
    sigma_mie,
    sigma_rayleigh,
    constant_sigma_interp
)

from geometric_functions import angle_between_vectors, transit_depth, get_grazing_coordinates
from absorption_classes import VenusAtmosphere


filename = "number_density_data/SOIRProfiles_AERO.xml"
table = Table.read(filename, format="votable")
print(table.colnames)

# hdul = fits.open('lineplot_sn.fits')
# data = hdul[1].data  # The data is in the second HDU (index 1)
# df = pd.DataFrame(data)
# df.to_csv('snr_vs_wavelength.csv')

# wl, snr = np.loadtxt('snr_vs_wavelength.csv', skiprows=1, usecols=(1, 2), dtype='float', delimiter=',', unpack=True)

# sort_idx = np.argsort(wl)
# wl = wl[sort_idx]
# snr = snr[sort_idx]

# plt.plot(wl, snr)
# plt.xlabel(f"Wavelength / um")
# plt.ylabel("SNR")
# plt.title("Instrument SNR vs wavelength (raw ETC output)")
# plt.show()


ALT_MIN_KM = 75.0
ALT_MAX_KM = 110.0
PH3_VMR = 20e-6  # strict upper limit of 0.8 ppb

MIE_WAVELENGTH_REF_NM = 3000.0     # SOIR beta reference wavelength (3.0 um)
MIE_RADIUS_UM = 0.15               # assumed effective particle radius
MIE_REFRACTIVE_INDEX = complex(1.40, -0.01)  # H2SO4 droplets @ ~3.0 um

WL_MIN_NM = 4000
WL_MAX_NM = 5000
RESOLUTION_CM = 1.0  # use 0.01 for better quality

DATA_DIR = Path("number_density_data")   # adjust to your actual path
NEW_DATA_DIR = Path("number_density_data_new")
MIN_DATA_POINTS = 2
BASELINE_PPM = 1e6
VIRA_FILE = "VIRA_noheader.csv"


alt_grid = np.arange(ALT_MIN_KM, ALT_MAX_KM + 0.1, 0.1)
alt_km_v, temp_K_v, pres_Pa_v, dens_kgpmc_v = np.loadtxt(
    VIRA_FILE, dtype=float, delimiter=",", unpack=True, usecols=(0, 1, 2, 3)
)
pres_atm_v = pres_Pa_v / 1.01325e5

temp_VIRA = interp1d(alt_km_v, temp_K_v, kind="linear",
                      bounds_error=False, fill_value="extrapolate")(alt_grid)
pres_VIRA = interp1d(alt_km_v, pres_atm_v, kind="linear",
                      bounds_error=False, fill_value="extrapolate")(alt_grid)


R_venus_m = Venus.R.to_value(u.m)
R_sun_au = R_sun.to_value(u.au)
R_venus_au = Venus.R.to_value(u.au)

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

datetime_jd_mid = datetime_jd[mid_transit_idx]
mars_pos_au = np.array(s_m_pos[mid_transit_idx])
venus_pos_au = np.array(s_v_pos[mid_transit_idx])

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


species_folders = ["CO_0", "CO2_0", "H2O_1", "H2O_4", "HCL_1", "HCL_2", "HF_1"]

target_lon, target_lat = get_grazing_coordinates(
    datetime_jd_mid,
    mars_pos_au,
    venus_pos_au,
    R_venus_au
)

closest_files = {}

# for folder in species_folders:
#     folder_path = DATA_DIR / folder
#     file_pattern = os.path.join(folder_path, "*.xml")
#     files = glob.glob(file_pattern)
    
#     closest_file = None
#     min_distance = float('inf')
#     all_file_info = []
#     n_skipped_too_small = 0
    
#     for file_path in files:
#         votable = parse(file_path)
#         table = votable.get_first_table()
#         params = table.params

#         n_rows = len(table.array)
#         if n_rows < MIN_DATA_POINTS:
#             n_skipped_too_small += 1
#             continue
        
#         for param in params:
#             if param.name == 'longitude_120km':
#                 lon = float(param.value)
#             elif param.name == 'latitude_120km':
#                 lat = float(param.value)
        
#         distance = np.sqrt((lon - target_lon)**2 + (lat - target_lat)**2)
        
#         file_info = {
#             'file_path': file_path,
#             'file_name': os.path.basename(file_path),
#             'longitude': lon,
#             'latitude': lat,
#             'distance': distance,
#             'n_rows': n_rows
#         }
#         all_file_info.append(file_info)
        
#         if distance < min_distance:
#             min_distance = distance
#             closest_file = file_info

#     if n_skipped_too_small:
#         print(f"{folder:>10}: skipped {n_skipped_too_small} file(s) with "
#               f"< {MIN_DATA_POINTS} data points")
    
#     closest_files[folder] = closest_file
#     print(f"{folder:>10}: {closest_file['file_name']}")
#     print(f"             LON={closest_file['longitude']:.4f}°, LAT={closest_file['latitude']:.4f}°")
#     print(f"             Distance: {closest_file['distance']:.4f}°")


# w23_SOIRProfiles = ["SOIRProfiles_SO2_1.xml", "SOIRProfiles_AERO.xml"]
# w23_species = ["SO2_1", "AERO"]
# w23_long_lat = {}
# for i in range(2):
#     profile_path = DATA_DIR / w23_SOIRProfiles[i]
#     species = w23_species[i]
#     w23_long_lat[species] = extract_w23_long_lat(profile_path)

#     closest_file = None
#     min_distance = float('inf')
#     all_file_info = []
#     n_skipped_too_small = 0

#     SOIR_table = Table.read(profile_path, format='votable')

#     for j in range(len(SOIR_table)):
#         orbit = int(SOIR_table[j]["orbit"])
#         file_name = f"Orbit0{orbit}.1_{species}.xml"
#         file_path = DATA_DIR / species / file_name

#         table = Table.read(file_path, format='votable')
#         if len(table) < MIN_DATA_POINTS:
#             n_skipped_too_small += 1
#             continue

#         lon = (SOIR_table[j]["lon_max"] + SOIR_table[j]["lon_min"]) / 2
#         lat = (SOIR_table[j]["lat_max"] + SOIR_table[j]["lat_min"]) / 2

#         distance = np.sqrt((lon - target_lon)**2 + (lat - target_lat)**2)

#         file_info = {
#             "file_path": file_path,
#             "file_name": file_name,
#             "longitude": lon,
#             "latitude": lat,
#             "distance": distance,
#         }
#         all_file_info.append(file_info)

#         if distance < min_distance:
#             min_distance = distance
#             closest_file = file_info

#     if n_skipped_too_small:
#         print(f"{species:>10}: skipped {n_skipped_too_small} file(s) with "
#               f"< {MIN_DATA_POINTS} data points")
    
#     closest_files[species] = closest_file
#     print(f"{species:>10}: {closest_file['file_name']}")
#     print(f"             LON={closest_file['longitude']:.4f}°, LAT={closest_file['latitude']:.4f}°")
#     print(f"             Distance: {closest_file['distance']:.4f}°")


file_mapping = {}

for folder, file_info in closest_files.items():
    if file_info is None:
        print(f"{folder:>10}: No file found - skipping")
        continue
    
    original_path = file_info['file_path']
    
    new_filename = f"{folder}.xml"
    
    new_path = NEW_DATA_DIR / new_filename
    
    # Copy the file
    try:
        shutil.copy2(original_path, new_path)
        file_mapping[folder] = {
            'original': original_path,
            'new': new_path,
            'new_filename': new_filename,
            'longitude': file_info['longitude'],
            'latitude': file_info['latitude']
        }
        print(f"{folder:>10} → {new_filename}")
        
    except Exception as e:
        print(f"{folder}: Error copying file: {e}")
        

species_files = list(NEW_DATA_DIR.glob("*.xml"))
raw_profiles = {}
for filepath in species_files:
    stem = filepath.stem
    raw_profiles[stem] = extract_profile(filepath)


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
        "alt": alt_grid, "p": p_grid / 1013.25, "t": t_grid, "n": n_grid
    }

fig, axes = plt.subplots(3, 1, figsize=(9, 9))

axes[0].plot(alt_grid, pres_VIRA, label='VIRA')
axes[0].plot(interpolated_profiles['CO_0']['alt'], interpolated_profiles['CO_0']['p'], label='SPICAV-SOIR')
axes[0].set_xlabel("Altitude / km")
axes[0].set_ylabel("Pressure / atm")
axes[0].set_title("Pressure against altitude for 75-110km")
axes[0].set_yscale("log")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(alt_grid, temp_VIRA, label='VIRA')
axes[1].plot(interpolated_profiles['CO_0']['alt'], interpolated_profiles['CO_0']['t'], label='SPICAV-SOIR')
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


species_profiles = build_species_profiles_for_atmosphere(
    interpolated_profiles, R_venus_m, n_units_per_cm3=True
)

species_profiles["PH3"] = ph3_profile_from_vmr(
    alt_grid_km=alt_grid, temp_K=temp_VIRA, pres_atm=pres_VIRA,
    R_venus_m=R_venus_m, vmr=PH3_VMR,
)
 
species_list = list(interpolated_profiles.keys()) + ["PH3"]

aero_table = extract_aerosol_profile(DATA_DIR / "AERO.xml")
aero_alt = np.asarray(aero_table["alt"], dtype=float)
aero_beta = np.asarray(aero_table["beta"], dtype=float)
 
sort_idx = np.argsort(aero_alt)
aero_alt = aero_alt[sort_idx]
aero_beta = aero_beta[sort_idx]
 

beta_grid = interp1d(aero_alt, aero_beta, kind="linear",
                      bounds_error=False, fill_value="extrapolate")(alt_grid)
beta_grid = np.clip(beta_grid, a_min=0.0, a_max=None)  # extinction can't be negative
 
species_profiles["AERO"] = build_aerosol_profile_for_atmosphere(
    alt_km=alt_grid, beta_per_km=beta_grid,
    wavelength_ref_nm=MIE_WAVELENGTH_REF_NM, radius_um=MIE_RADIUS_UM,
    refractive_index=MIE_REFRACTIVE_INDEX, R_venus_m=R_venus_m,
)
species_profiles["RAYLEIGH_CO2"] = species_profiles["CO2_0"]  # same n(r), different sigma

atmosphere = VenusAtmosphere(
    R_venus_m=R_venus_m,
    alt_min_km=ALT_MIN_KM,
    alt_max_km=ALT_MAX_KM,
    species_profiles=species_profiles,
)


wavelengths_nm, wavenumber_cm = wavelength_grid_nm(WL_MIN_NM, WL_MAX_NM, RESOLUTION_CM)
test_alts_km = [75., 80., 85., 90., 95., 100., 105., 110.]

species_isotopologue_map = {}
for species_name in interpolated_profiles.keys():
    species_isotopologue_map[species_name] = parse_species_isotopologue(species_name)

sigma_dict_alt = {}  # species -> array [n_test_alts, n_wavelengths], cm^2/molecule
for species_name in species_list:
    if species_name == "PH3":
        base_molecule, iso_id = "PH3", 1
        t_arr = temp_VIRA
        p_arr = pres_VIRA
    else:
        base_molecule, iso_id = species_isotopologue_map[species_name]
        t_arr = interpolated_profiles[species_name]["t"]  # K
        p_arr = interpolated_profiles[species_name]["p"]  # atm
 
    sigma_all_altitudes = []
    for alt in test_alts_km:
        idx = np.argmin(np.abs(alt_grid - alt))
        T_alt = t_arr[idx]
        P_alt = p_arr[idx]

        if np.isnan(T_alt) or T_alt <= 0:
            print(f"  Warning: Invalid T={T_alt} at {alt} km, using 250 K")
            T_alt = 250.0

        wl, sigma = get_sigma_lambda(
            molecule=base_molecule,
            temperature_K=T_alt,
            pressure_atm=P_alt,
            wl_min_nm=WL_MIN_NM,
            wl_max_nm=WL_MAX_NM,
            resolution_cm=RESOLUTION_CM,
            isotopologue=iso_id,
        )
 
        sigma_on_grid = np.interp(wavelengths_nm, wl, sigma)
        sigma_all_altitudes.append(sigma_on_grid)
 
    sigma_dict_alt[species_name] = np.array(sigma_all_altitudes)



PH3_ONLY_LIST = ["PH3"]

R_eff_au = []
flux_total_ppm = []          # all species combined
flux_ph3_only_ppm = []             # PH3 alone, for visual comparison

for i in range(len(wavelengths_nm)):
    wl_nm = wavelengths_nm[i]

    # -- full spectrum, all species --
    sigma_interp_dict_all = build_sigma_interp_dict(
        species_list, sigma_dict_alt, test_alts_km, wl_idx=i, R_venus_m=R_venus_m
    )

    sigma_mie_val = sigma_mie(MIE_RADIUS_UM, wl_nm, MIE_REFRACTIVE_INDEX)
    sigma_rayleigh_val = sigma_rayleigh(wl_nm)
    sigma_interp_dict_all["AERO"] = constant_sigma_interp(sigma_mie_val)
    sigma_interp_dict_all["RAYLEIGH_CO2"] = constant_sigma_interp(sigma_rayleigh_val)

    R_eff = atmosphere.effective_radius_au(sigma_interp_dict_all)
    R_eff_au.append(R_eff)
    depth_mid = transit_depth(R_eff, v_mid, alpha_sun_mid, theta_mid)
    flux_total_ppm.append((1 - depth_mid) * 1e6)
    flux_total_ppm = np.asarray(flux_total_ppm, dtype=float)

    # -- PH3 only, same ray integral, restricted species list --
    sigma_interp_dict_ph3 = build_sigma_interp_dict(
        PH3_ONLY_LIST, sigma_dict_alt, test_alts_km, wl_idx=i, R_venus_m=R_venus_m
    )
    R_eff_ph3 = atmosphere.effective_radius_au(sigma_interp_dict_ph3)
    depth_ph3 = transit_depth(R_eff_ph3, v_mid, alpha_sun_mid, theta_mid)
    flux_ph3_only_ppm.append((1 - depth_ph3) * 1e6)


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(wavelengths_nm, flux_total_ppm, linewidth=1.5, color="blue",
         label="Total (absorption + Rayleigh + Mie)")
ax.plot(wavelengths_nm, flux_ph3_only_ppm, linewidth=1.5, color="crimson",
         label="PH3 absorption only (0.8 ppb VMR)")
ax.set_xlabel("Wavelength (nm)", fontsize=12)
ax.set_ylabel("Flux (ppm)", fontsize=12)
ax.set_title("Venus Transmission Spectrum at Mid-Transit, 75-110 km shell\n"
             "(altitude-dependent absorption + Rayleigh + Mie scattering)",
             fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("venus_transmission_spectrum.png", dpi=150)
plt.show()



# Load the SNR data from the FITS file
hdul = fits.open('lineplot_sn.fits')
data = hdul[1].data  # The data is in the second HDU (index 1)

# Create a pandas DataFrame and save as CSV
df = pd.DataFrame(data)
df.to_csv('snr_vs_wavelength.csv')

wl, snr = np.loadtxt('snr_vs_wavelength.csv', skiprows=1, usecols=(1, 2), dtype='float', delimiter=',', unpack=True)

sort_idx = np.argsort(wl)
wl = wl[sort_idx]
snr = snr[sort_idx]

plt.plot(wl, snr)
plt.xlabel(f"Wavelength / um")
plt.ylabel("SNR")
plt.title("Instrument SNR vs wavelength (raw ETC output)")
plt.show()

# ============ Simplified: Add Noise to Your Spectrum ============
snr_wl_nm = wl * 1000  # Convert microns to nm

snr_interp = interp1d(snr_wl_nm, snr, kind='linear', 
                       bounds_error=False, fill_value=np.nan)
snr_at_trans = snr_interp(wavelengths_nm)

in_band_mask = np.isfinite(snr_at_trans) & (snr_at_trans > 0)
noise_ppm = np.full_like(wavelengths_nm, np.nan)
noise_ppm[in_band_mask] = BASELINE_PPM / snr_at_trans[in_band_mask]
 
rng = np.random.default_rng(42)  # scoped RNG, doesn't mutate global numpy state
noisy_flux = flux_total_ppm.copy()
noisy_flux[in_band_mask] = (
    flux_total_ppm[in_band_mask] + rng.normal(0, noise_ppm[in_band_mask])
)

# ============ Plot Results ============
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(wavelengths_nm, flux_total_ppm, linewidth=2, color="blue",
        label="Noiseless spectrum")
ax.fill_between(wavelengths_nm[in_band_mask],
                 (flux_total_ppm - noise_ppm)[in_band_mask],
                 (flux_total_ppm + noise_ppm)[in_band_mask],
                 color='blue', alpha=0.15, label='±1σ instrument noise')
ax.scatter(wavelengths_nm[in_band_mask], noisy_flux[in_band_mask],
           s=8, color="red", alpha=0.6, label="Noisy realization")
ax.set_xlabel("Wavelength (nm)", fontsize=12)
ax.set_ylabel("Flux (ppm)", fontsize=12)
ax.set_title("Venus Transmission Spectrum with Instrument Noise", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("venus_transmission_with_noise.png", dpi=150)
plt.show()