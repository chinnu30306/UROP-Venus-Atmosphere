import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from astropy.time import Time


# Get the position vectors of Venus and Mars from JPL Horizons
venus = Horizons(id=299, location='500@10', epochs={'start':'2030-08-19', 'stop':'2030-08-21', 'step':'1min'})
venus_vectors = venus.vectors()

venus_x = venus_vectors['x']
venus_y = venus_vectors['y']
venus_z = venus_vectors['z']
venus_datetime_jd = venus_vectors['datetime_jd']

venus_position = list(zip(venus_x, venus_y, venus_z))
normalised_venus_position = [(x/np.linalg.norm([x, y, z]), y/np.linalg.norm([x, y, z]), z/np.linalg.norm([x, y, z])) for x, y, z in venus_position]

mars = Horizons(id=499, location='500@10', epochs={'start':'2030-08-19', 'stop':'2030-08-21', 'step':'1min'})
mars_vectors = mars.vectors()

mars_x = mars_vectors['x']
mars_y = mars_vectors['y']
mars_z = mars_vectors['z']

mars_position = list(zip(mars_x, mars_y, mars_z))
normalised_mars_position = [(x/np.linalg.norm([x, y, z]), y/np.linalg.norm([x, y, z]), z/np.linalg.norm([x, y, z])) for x, y, z in mars_position]


# Use the cross product to find the angle between the two position vectors
def angle_between_vectors(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

angle = []
for i in range(len(normalised_venus_position)):
    angle.append(angle_between_vectors(normalised_venus_position[i], normalised_mars_position[i]))

plt.plot(venus_datetime_jd, angle, label='Angle')
plt.xlabel('Julian Date')
plt.ylabel('Angle (deg)')
plt.title('Angle between Venus and Mars over Time')
plt.axhline(y=0.35, color='r', linestyle='--', label='0.35 deg')
plt.legend()
plt.grid()
plt.show()

t = Time(venus_datetime_jd[np.argmin(angle)], format='jd')
print(t.iso)