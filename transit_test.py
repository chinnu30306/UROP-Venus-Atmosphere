import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from astropy.time import Time
from astropy import units as u
from astropy.constants import R_sun
from functions import angle_between_vectors
from classes import Planet
from poliastro.bodies import Mars


# Get the position vectors of Venus and Mars from JPL Horizons
m_v = Horizons(id=299, location='499', epochs={'start':'2030-08-19', 'stop':'2030-08-21', 'step':'1min'})
venus_vectors = m_v.vectors()
m_v_x = venus_vectors['x']
m_v_y = venus_vectors['y']
m_v_z = venus_vectors['z']
m_v_pos = list(zip(m_v_x, m_v_y, m_v_z))
datetime_jd = venus_vectors['datetime_jd']

m_s = Horizons(id=10, location='499', epochs={'start':'2030-08-19', 'stop':'2030-08-21', 'step':'1min'})
sun_vectors = m_s.vectors()
m_s_x = sun_vectors['x']
m_s_y = sun_vectors['y']
m_s_z = sun_vectors['z']
m_s_pos = list(zip(m_s_x, m_s_y, m_s_z))

s_v = Horizons(id=299, location='500@10', epochs={'start':'2030-08-19', 'stop':'2030-08-21', 'step':'1min'})
venus_vectors = s_v.vectors()
s_v_x = venus_vectors['x']
s_v_y = venus_vectors['y']
s_v_z = venus_vectors['z']
s_v_pos = list(zip(s_v_x, s_v_y, s_v_z))

s_m = Horizons(id=499, location='500@10', epochs={'start':'2030-08-19', 'stop':'2030-08-21', 'step':'1min'})
mars_vectors = s_m.vectors()
s_m_x = mars_vectors['x']
s_m_y = mars_vectors['y']
s_m_z = mars_vectors['z']
s_m_pos = list(zip(s_m_x, s_m_y, s_m_z))

new1 = np.array(s_v_pos) - np.array(s_m_pos)


# Use the cross product to find the angle between the two position vectors
theta = []
alpha_sun = []
alpha_venus = []
for i in range(len(new1)):
    theta.append(angle_between_vectors(new1[i], -np.array(s_m_pos[i])))
    alpha_sun.append(np.arctan(R_sun / np.linalg.norm(s_m_pos[i])))
    alpha_venus.append(np.arctan(R_sun / np.linalg.norm(s_v_pos[i])))

transit_threshold = alpha_sun + alpha_venus

min_index = np.argmin(theta)
sun_dist_au = np.linalg.norm(s_m_pos[min_index]) * u.au
sun_dist_m = sun_dist_au.to(u.m)
min_transit_angle_rad = 2 * np.arctan(R_sun / sun_dist_m)
print(R_sun)
min_transit_angle_deg = np.degrees(min_transit_angle_rad.value)
print(min_transit_angle_deg)

plt.plot(datetime_jd, theta, label='Angle')
plt.xlabel('Julian Date')
plt.ylabel('Angle (rad)')
plt.title('Angle between Venus and Sun over Time')
plt.axhline(y=min_transit_angle_rad.value, color='r', linestyle='--', label=f'Minimum Transit Angle ({min_transit_angle_rad.value:.4f} rad)')
plt.legend()
plt.grid()
plt.show()