import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from functions_new import angle_between_vectors, circle_overlap_area
from astropy.constants import R_sun
from poliastro.bodies import Mars, Venus


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
    alpha_venus.append(np.arctan(R_venus_au / np.linalg.norm(v_m_pos[i])))

transit_depth = []
for i in range(len(theta)):
    overlap = circle_overlap_area(alpha_sun[i], alpha_venus[i], theta[i])
    sun_area = np.pi * alpha_sun[i]**2
    fraction = overlap / sun_area
    transit_depth.append(fraction)


flux_loss_ppm = (1 - np.array(transit_depth)) * 1e6
plt.figure(figsize=(9, 5))
plt.plot(datetime_jd, flux_loss_ppm)
plt.xlabel("Julian Date")
plt.ylabel("Flux loss (ppm)")
plt.title("Solid-body Venus transit depth as seen from Mars")
plt.grid()
plt.show()