import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from functions_new import angle_between_vectors, transit_depth, get_sigma_lambda_from_hitran, extract_profile
from astropy.constants import R_sun
from poliastro.bodies import Mars, Venus
from classes_new import VenusAtmosphere
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d

from astropy.table import Table

filename = "molecules_data/SO2.xml"

table = Table.read(filename, format="votable")

print(table.colnames)

for col in table.colnames:
    print(
        col,
        "unit =", table[col].unit,
        "description =", table[col].description
    )

species = ['CO.xml', 'CO2.xml', 'H2O.xml', 'HCL.xml', 'HF.xml', 'H2CO.xml', 'HCN.xml', 'HO2.xml', 'N2O.xml', 'NH3.xml', 'NO.xml', 'NO2.xml', 'O3.xml', 'CS.xml', 'CS2.xml', 'OCS.xml', 'SO2.xml', 'SO3.xml']
path = Path("/Users/chinmayib/git/UROP-Venus-Atmosphere/molecules_data")
profiles = {}

for filename in species:
    filepath = path / filename
    profile = extract_profile(filepath)
    species_name = filepath.stem
    profiles[species_name] = profile

alt_km, temp_K, pres_Pa, dens_kgpmc = np.loadtxt("VIRA_noheader.csv", dtype=float, delimiter=',', unpack=True, usecols=(0, 1, 2, 3))

alt_grid = np.arange(75.0, 110.0 + 0.1, 0.1)
interpolated_profiles = {}

for species_name, profile in profiles.items():
    altitude = np.asarray(profile["alt"], dtype=float)
    number_density = np.asarray(profile["n"], dtype=float)

    sort_idx = np.argsort(altitude)
    altitude = altitude[sort_idx]
    number_density = number_density[sort_idx]
    log_n = np.log(number_density)

    interpolation = interp1d(altitude, log_n, kind="linear", bounds_error=False, fill_value="extrapolate")
    log_n_grid = interpolation(alt_grid)
    n_grid = np.exp(log_n_grid)

    interpolated_profiles[species_name] = {"alt": alt_grid, "n": n_grid}


# plt.plot(interpolated_profiles["CO2"]["alt"], interpolated_profiles["CO2"]["n"], label='CO2')
# plt.plot(interpolated_profiles["SO2"]["alt"], interpolated_profiles["SO2"]["n"], label='SO2')
plt.plot(interpolated_profiles["CO2"]["alt"], interpolated_profiles["SO2"]["n"]/(interpolated_profiles["CO2"]["n"]*1e6), label='CO2')
plt.title("Number density against altitude")
plt.xlabel("Altitude/km")
plt.ylabel("Number density/cm^-3")
plt.legend()
plt.show()


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


# --- illustrative atmosphere: replace these numbers with real Venus values ---
atmosphere = VenusAtmosphere(
    R_cloud_top_km=6098,   # ~cloud-top altitude above the ~6052 km solid body
    T_atm=230,             # K, upper-atmosphere/haze temperature
    mu=43.45 * 1.66054e-27,  # kg/molecule, ~CO2-dominated
    g=8.87,                # m/s^2
    n0=1e23,                # molecules/m^3 at the cloud top
)

# --- transmission spectrum at mid-transit: loop steps 1-5 over wavelength ---
wavelengths_nm, sigma_lambda = np.array(get_sigma_lambda_from_hitran(resolution_cm=1.0))
venus_molecules = {'CO2': 0.965, 'N2':  0.035, 'SO2': 1e-4, 'H2O': 1e-5, 'CO':  1e-5}
sigma_total = []

sigma_total = np.zeros(len(wavelengths_nm))

for molecule, mixing_ratio in venus_molecules.items():
    wl, sigma = get_sigma_lambda_from_hitran(molecule=molecule, resolution_cm=1.0)
    sigma_total += mixing_ratio * sigma
    
R_eff_au, flux_transparent_ppm = [], []
v_mid = v_m_pos[mid_transit_idx]
alpha_sun_mid = alpha_sun[mid_transit_idx]
theta_mid = theta[mid_transit_idx]

for sigma in sigma_total:
    R_eff = atmosphere.effective_radius_au(sigma)
    R_eff_au.append(R_eff)
    depth_mid = transit_depth(R_eff, v_mid, alpha_sun_mid, theta_mid)
    flux_transparent_ppm.append((1 - depth_mid) * 1e6)


fig1, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(wavelengths_nm, np.array(flux_transparent_ppm))
plt.xlabel("Wavelength (nm)")
plt.ylabel("Flux (ppm)")
plt.title("With-atmosphere transit depth")
ax1.grid()
plt.show()