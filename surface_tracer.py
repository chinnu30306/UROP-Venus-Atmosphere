import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from classes import Planet, Grid, Cone
from astropy.constants import R_sun
from poliastro.bodies import Mars, Venus
from matplotlib.patches import Rectangle


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


# Plotting the visibility map of Venus from Mars
R_mars_au = Mars.R.to("au").value
Mars = Planet(pos = mars_position[0], radius = R_mars_au, rows=180, cols=360)

latLongArray = []
for i in range(len(datetime_jd)):
    Mars.set_pos(mars_position[i])
    latLong = Mars.lat_long_new(venus_position[i], datetime_jd[i])
    if latLong != None:
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

time = []
venus_pos = []
mars_pos = []
for i, time in enumerate(datetime_jd):
    if time > 2462733.3:
        if time < 2462733.6:
            time.append[time]
            venus_pos.append[venus_position[i]]
            mars_pos.append[mars_position[i]]

R_sun_au = R_sun.to("au").value
R_venus_au = Venus.R.to("au").value
R_mars_au = Mars.R.to("au").value

Mars = Planet(pos = mars_position[0], radius = R_mars_au, rows=180, cols=360)
antumbra_cone = Cone(R_sun_au, R_venus_au, venus_pos[0])

for i, time in enumerate(time):
    Mars.set_pos(mars_pos[i])
    antumbra_cone.set_r12(venus_pos[i])
    Mars.isInCone(antumbra_cone, 1, time)

for i in range(Mars.grid.get_rows()):
    for j in range(Mars.grid.get_cols()):
        value = Mars.grid.get_value(i, j)
        xy = (j-180, i-90)
        if value == 1:
            ax.add_patch(Rectangle(xy,1,1, fc = 'r'))  # Penumbra
        elif value == 2:
            ax.add_patch(Rectangle(xy,1,1, fc = 'b'))  # Antumbra
