import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from functions_new import angle_between_vectors, transit_depth, get_sigma_lambda_from_hitran
from astropy.constants import R_sun
from poliastro.bodies import Mars, Venus
from classes_new import VenusAtmosphere


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

depth = transit_depth(R_venus_au, v_m_pos, alpha_sun, theta)

flux_opaque_ppm = (1 - np.array(depth)) * 1e6
plt.figure(figsize=(9, 5))
plt.plot(datetime_jd, flux_opaque_ppm)
plt.xlabel("Julian Date")
plt.ylabel("Flux (ppm)")
plt.title("Solid-body Venus transit depth as seen from Mars")
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
wavelengths_nm, sigma_lambda = np.array(get_sigma_lambda_from_hitran())

R_eff_au, depth_ppm = [], []
for i in range(len(wavelengths_nm)):
    R_eff = atmosphere.effective_radius_au(sigma_lambda[i])
    R_eff_au.append(R_eff)
    depth_mid = transit_depth(R_eff, v_m_pos, alpha_sun, theta)[mid_transit_idx]
    depth_ppm.append((1 - depth_mid) * 1e6)


fig1, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(wavelengths_nm, np.array(depth_ppm))
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Transit depth (ppm)")
ax1.set_title("Transmission spectrum at mid-transit (Venus atmosphere, toy model)")
ax1.grid()
plt.show()