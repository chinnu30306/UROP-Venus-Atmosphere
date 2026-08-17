import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from classes import Planet, Grid, Cone
from astropy.constants import R_sun, R_earth
# from poliastro.bodies import Earth, Moon
from matplotlib.patches import Rectangle
from functions import  angle_between_vectors
from astropy import units as u
R_moon = 1737.4 * u.km


# Get the position vectors of moon and earth from JPL Horizons
moon = Horizons(id=301, location='500@10', epochs={'start':'2026-08-12', 'stop':'2026-08-13', 'step':'1min'})
moon_vectors = moon.vectors()
moon_x = moon_vectors['x']
moon_y = moon_vectors['y']
moon_z = moon_vectors['z']
moon_position = list(zip(moon_x, moon_y, moon_z))
datetime_jd = moon_vectors['datetime_jd']

earth = Horizons(id=399, location='500@10', epochs={'start':'2026-08-12', 'stop':'2026-08-13', 'step':'1min'})
earth_vectors = earth.vectors()
earth_x = earth_vectors['x']
earth_y = earth_vectors['y']
earth_z = earth_vectors['z']
earth_position = list(zip(earth_x, earth_y, earth_z))

e_m_vec = np.array(moon_position) - np.array(earth_position)
# print(f'earth to moon distance: {np.linalg.norm(e_m_vec[1068])}')
# print(f'earth to sun distance: {np.linalg.norm(earth_position[1068])}')


R_sun_au = R_sun.to("au").value
R_moon_au = R_moon.to("au").value
R_earth_au = R_earth.to("au").value

## PLOT OF ANGLE##
theta = []
transit_threshold = []
alpha_sun_array  = np.zeros(shape = len(e_m_vec))
alpha_moon_array = np.zeros(shape = len(e_m_vec))

for i in range(len(e_m_vec)):
    theta.append(angle_between_vectors(e_m_vec[i], -np.array(earth_position[i])))
    alpha_sun_array[i] = np.arctan(R_sun_au / np.linalg.norm(earth_position[i])) 
    alpha_moon_array[i] = np.arctan(R_moon_au / np.linalg.norm(e_m_vec[i]))

transit_threshold = 2 * alpha_sun_array + alpha_moon_array
min_theta = min(theta)
min_index = theta.index(min_theta)
min_jd = datetime_jd[min_index]
print(f'min_index: {min_index}')
print(f'min_theta: {theta[min_index] * (180 / np.pi)}')
print(f'min_jd: {min_jd}')
print(f'alpha_sun: {alpha_sun_array[min_index] * (180 / np.pi)}')
print(f'alpha_moon: {alpha_moon_array[min_index] * (180 / np.pi)}')

# plt.plot(datetime_jd, theta, label='Angle')
# plt.plot(datetime_jd, transit_threshold, label='Transit Threshold')
# plt.xlabel('Julian Date')
# plt.ylabel('Angle (rad)')
# plt.title('Angle between moon and Sun over Time')
# # plt.axhline(y=min(transit_threshold), color='r', linestyle='--', label=f'Minimum Transit Angle ({min(transit_threshold):.4f} rad)')
# plt.legend()
# plt.grid()
# plt.show()

## PLOT OF ECLIPSE ON SURFACE ##
fig, ax = plt.subplots()
ax.set_xlim(-180,180)
ax.set_ylim(-90,90)
ax.set_xlabel("Longitude (deg)")
ax.set_ylabel("Latitude (deg)")
ax.set_title("Visibility Map of moon from earth")

num = 60
time = []
moon_pos = []
earth_pos = []
for i, t in enumerate(datetime_jd):
    if t > 2461265.25 - num * 0.0006944:
        if t < 2461265.25 + num * 0.0006944:
            time.append(t)
            moon_pos.append(moon_position[i])
            earth_pos.append(earth_position[i])

time = time[::10]
moon_pos = moon_pos[::10]
earth_pos = earth_pos[::10]

earth = Planet(pos = earth_position[0], radius = R_earth_au)
antumbra_cone = Cone(R_sun_au, R_moon_au, moon_position[0])
penumbra_cone = Cone(R_sun_au, R_moon_au, moon_position[0])

## for all points in time##
for i, time in enumerate(time):
    earth.set_pos(earth_pos[i])
    antumbra_cone.set_r12(moon_pos[i])
    earth.isInCone(antumbra_cone, 1, time)

for i in range(earth.grid.get_rows()):
    for j in range(earth.grid.get_cols()):
        value = earth.grid.get_value(i, j)
        xy = (j-180, i-90)
        if value == 1:
            ax.add_patch(Rectangle(xy,1,1, fc = 'r'))  # Penumbra
        elif value == 2:
            ax.add_patch(Rectangle(xy,1,1, fc = 'b'))  # Antumbra

plt.show()

## at minimum point ##
# earth.set_pos(earth_position[min_index])
# antumbra_cone.set_r12(moon_position[min_index])
# earth.isInCone(antumbra_cone, 1, datetime_jd[min_index])

# for i in range(earth.grid.get_rows()):
#     for j in range(earth.grid.get_cols()):
#         value = earth.grid.get_value(i, j)
#         xy = (j-180, i-90)
#         if value == 1:
#             ax.add_patch(Rectangle(xy,1,1, fc = 'r'))  # Penumbra
#         elif value == 2:
#             ax.add_patch(Rectangle(xy,1,1, fc = 'b'))  # Antumbra

# plt.show()
