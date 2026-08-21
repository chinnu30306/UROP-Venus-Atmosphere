import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from functions_new import angle_between_vectors, circle_overlap_area, transit_depth, sigma_with_band
from astropy.constants import R_sun
from poliastro.bodies import Mars, Venus
from classes_new import VenusAtmosphere
# from hapi import db_begin, fetch, absorptionCoefficient_Voigt

# # Initialize a local database folder
# db_begin('your_hitran_data_folder')

# # Fetch CO2 data for a specific wavelength range
# fetch('CO2', 1, 1, 2000, 2100) # Example wavenumber range (cm^-1)

# # Environment: pressure in atm, temperature in Kelvin
# nu, coef = absorptionCoefficient_Voigt(SourceTables='CO2', 
#                                        Environment={'p': 1.0, 'T': 296.0},
#                                        OmegaStep=0.01) # Spectral resolution

# # Example: Load a pre-computed cross-section file
# data = np.load('co2_cross_sections.npz')
# wavelengths = data['wavelengths']  # in microns or nm
# cross_sections = data['cross_sections']  # in m^2 per molecule


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


theta, alpha_sun, alpha_venus, v_m_dist_au = [], [], [], []
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
wavelengths_nm = np.linspace(300, 1000, 40)
R_eff_by_wavelength_au = np.array([atmosphere.effective_radius_au(sigma_with_band(w, band_center_nm=760, band_width_nm=15, band_peak_sigma=3e-29)) for w in wavelengths_nm])

depth_spectrum_ppm = np.array([(1 - transit_depth(R_eff, v_m_pos, alpha_sun, theta)[mid_transit_idx]) * 1e6 for R_eff in R_eff_by_wavelength_au])

fig1, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(wavelengths_nm, depth_spectrum_ppm)
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Transit depth (ppm)")
ax1.set_title("Transmission spectrum at mid-transit (Venus atmosphere, toy model)")
ax1.grid()

# --- light curves at two representative wavelengths: continuum vs in-band ---
R_eff_continuum_au = atmosphere.effective_radius_au(
    sigma_with_band(550, band_center_nm=760, band_width_nm=15, band_peak_sigma=3e-29)
)
R_eff_inband_au = atmosphere.effective_radius_au(
    sigma_with_band(760, band_center_nm=760, band_width_nm=15, band_peak_sigma=3e-29)
)

flux_loss_continuum_ppm = (1 - np.array(transit_depth(R_eff_continuum_au, v_m_pos, alpha_sun, theta))) * 1e6
flux_loss_inband_ppm = (1 - np.array(transit_depth(R_eff_inband_au, v_m_pos, alpha_sun, theta))) * 1e6

fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.plot(datetime_jd, flux_loss_continuum_ppm, label="550 nm (continuum)")
ax2.plot(datetime_jd, flux_loss_inband_ppm, label="760 nm (in absorption band)")
ax2.set_xlabel("Julian Date")
ax2.set_ylabel("Flux loss (ppm)")
ax2.set_title("Venus transit light curve as seen from Mars, with atmosphere")
ax2.legend()
ax2.grid()

plt.show()