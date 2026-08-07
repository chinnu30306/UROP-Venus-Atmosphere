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
venus = Horizons(id=299, location='500@10', epochs={'start':'2030-08-19', 'stop':'2030-08-21', 'step':'1min'})
venus_vectors = venus.vectors()
venus_x = venus_vectors['x']
venus_y = venus_vectors['y']
venus_z = venus_vectors['z']
venus_position = list(zip(venus_x, venus_y, venus_z))
datetime_jd = venus_vectors['datetime_jd']

mars = Horizons(id=499, location='500@10', epochs={'start':'2030-08-19', 'stop':'2030-08-21', 'step':'1min'})
mars_vectors = mars.vectors()
mars_x = mars_vectors['x']
mars_y = mars_vectors['y']
mars_z = mars_vectors['z']
mars_position = list(zip(mars_x, mars_y, mars_z))


# Use the cross product to find the angle between the two position vectors
angle = []
for i in range(len(venus_position)):
    angle.append(angle_between_vectors(venus_position[i], mars_position[i]))

min_index = np.argmin(angle)
mars_dist_au = np.linalg.norm(mars_position[min_index]) * u.au
mars_dist_m = mars_dist_au.to(u.m)
min_transit_angle_rad = 2 * np.arcsin(R_sun / mars_dist_m)

# plt.plot(datetime_jd, angle, label='Angle')
# plt.xlabel('Julian Date')
# plt.ylabel('Angle (rad)')
# plt.title('Angle between Venus and Mars over Time')
# plt.axhline(y=min_transit_angle_rad.value, color='r', linestyle='--', label=f'Minimum Transit Angle ({min_transit_angle_rad.value:.4f} rad)')
# plt.legend()
# plt.grid()
# plt.show()

# Plotting the visibility map of Venus from Mars
mars_pos = []
venus_pos = []
time = []
for i in range(len(angle)):
    if angle[i] <= min_transit_angle_rad.value:
        time.append(Time(datetime_jd[i], format='jd'))
        mars_pos.append(mars_position[i])
        venus_pos.append(venus_position[i])

R_mars = Mars.R.to("m").value
R_mars_au = Mars.R.to(u.au).value
Mars = Planet(pos = mars_position[0], radius = R_mars_au)

latLongArray = []
for i in range(len(datetime_jd)):
    Mars.set_pos(mars_position[i])
    latLong = Mars.longLat(venus_position[i], datetime_jd[i])
    if latLong is not None:
        latLongArray.append(latLong)

longitude = []
latitude = []
for p in latLongArray:
    latitude.append(p[0])
    longitude.append(p[1])

fig, ax = plt.subplots()
ax.set_xlim(-180,180)
ax.set_ylim(-90,90)
ax.plot(longitude,latitude)
ax.set_xlabel("Longitude (deg)")
ax.set_ylabel("Latitude (deg)")
ax.set_title("Visibility Map of Venus from Mars")
plt.grid()
plt.show()