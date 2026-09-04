import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from geometric_classes import Planet, Cone
from geometric_functions import transit_visible_from_mars
from astropy.constants import R_sun
from poliastro.bodies import Mars, Venus
from matplotlib.patches import Rectangle


venus = Horizons(id=299, location='500@10',
                  epochs={'start': '2030-08-19', 'stop': '2030-08-21', 'step': '1min'})
venus_vectors = venus.vectors(refplane='ecliptic')

mars = Horizons(id=499, location='500@10',
                 epochs={'start': '2030-08-19', 'stop': '2030-08-21', 'step': '1min'})
mars_vectors = mars.vectors(refplane='ecliptic')

datetime_jd = venus_vectors['datetime_jd']
venus_position = list(zip(venus_vectors['x'], venus_vectors['y'], venus_vectors['z']))
mars_position = list(zip(mars_vectors['x'], mars_vectors['y'], mars_vectors['z']))

# --- restrict to the transit window ---
time_jd, venus_pos, mars_pos = [], [], []
for i, t in enumerate(datetime_jd):
    if 2462733.3 < t < 2462733.6:
        time_jd.append(t)
        venus_pos.append(venus_position[i])
        mars_pos.append(mars_position[i])

R_sun_au = R_sun.to("au").value
R_venus_au = Venus.R.to("au").value
R_mars_au = Mars.R.to("au").value

mars_body = Planet(pos=mars_pos[0], radius=R_mars_au)

# --- trace the shadow's centre-line across the whole window ---
# lat_path, lon_path = [], []
# for i, t in enumerate(time_jd):
#     if not transit_visible_from_mars(R_sun_au, R_venus_au, R_mars_au,
#                                       venus_pos[i], mars_pos[i]):
#         continue
#     mars_body.set_pos(np.array(mars_pos[i]))
#     result = mars_body.sub_axis_point(venus_pos[i], t)
#     if result is not None:
#         lat_path.append(result[0])
#         lon_path.append(result[1])

# fig, ax = plt.subplots()
# ax.set_xlim(-180, 180)
# ax.set_ylim(-90, 90)
# ax.plot(lon_path, lat_path, '.-', color='crimson')
# ax.set_xlabel("Longitude (deg)")
# ax.set_ylabel("Latitude (deg)")
# ax.set_title("Path of Venus's Transit Shadow across Mars (19 Aug 2030)")
# plt.grid()
# plt.show()

# --- optional: instantaneous antumbral footprint at one moment, as a check ---
mars_body = Planet(pos=mars_pos[0], radius=R_mars_au)

for i, t in enumerate(time_jd):
    if not transit_visible_from_mars(R_sun_au, R_venus_au, R_mars_au,
                                      venus_pos[i], mars_pos[i]):
        continue   # skip times when Venus isn't in transit as seen from Mars

    mars_body.set_pos(np.array(mars_pos[i]))
    antumbra_cone = Cone(R_sun_au, R_venus_au, venus_pos[i])
    mars_body.isInCone(antumbra_cone, 2, t)      # 2 = antumbral; accumulates, never cleared

fig, ax = plt.subplots()
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
for r in range(mars_body.grid.get_rows()):
    for c in range(mars_body.grid.get_cols()):
        if mars_body.grid.get_value(r, c) == 2:
            ax.add_patch(Rectangle((c - 180, r - 90), 1, 1, fc='b'))
ax.set_xlabel("Longitude (deg)")
ax.set_ylabel("Latitude (deg)")
ax.set_title("Swept path of Venus's antumbral shadow across Mars (19 Aug 2030)")
plt.show()