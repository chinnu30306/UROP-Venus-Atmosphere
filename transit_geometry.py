import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from astropy.time import Time
from astropy import units as u
from astropy.constants import R_sun
import spiceypy as spice

spice.furnsh("kernels/naif0012.tls")
spice.furnsh("kernels/pck00011.tpc")
spice.furnsh("kernels/de440s.bsp")


def angle_between_vectors(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return angle_rad


# Get the position vectors of Venus and Mars from JPL Horizons
venus = Horizons(id=299, location='500@10', epochs={'start':'2030-08-19', 'stop':'2030-08-21', 'step':'1min'})
venus_vectors = venus.vectors()

venus_x = venus_vectors['x']
venus_y = venus_vectors['y']
venus_z = venus_vectors['z']
datetime_jd = venus_vectors['datetime_jd']

venus_position = list(zip(venus_x, venus_y, venus_z))

mars = Horizons(id=499, location='500@10', epochs={'start':'2030-08-19', 'stop':'2030-08-21', 'step':'1min'})
mars_vectors = mars.vectors()
mars_eph = mars.ephemerides(extra_precision=True)

mars_x = mars_vectors['x']
mars_y = mars_vectors['y']
mars_z = mars_vectors['z']

mars_position = list(zip(mars_x, mars_y, mars_z))

longitude = mars_eph['PDSunLon']
latitude = mars_eph['PDSunLat']

# Use the cross product to find the angle between the two position vectors
angle = []
for i in range(len(venus_position)):
    angle.append(angle_between_vectors(venus_position[i], mars_position[i]))

min_index = np.argmin(angle)
mars_dist_au = np.linalg.norm(mars_position[min_index]) * u.au
mars_dist_m = mars_dist_au.to(u.m)
min_transit_angle_rad = 2 * np.arcsin(R_sun / mars_dist_m)

plt.plot(datetime_jd, angle, label='Angle')
plt.xlabel('Julian Date')
plt.ylabel('Angle (rad)')
plt.title('Angle between Venus and Mars over Time')
plt.axhline(y=min_transit_angle_rad.value, color='r', linestyle='--', label=f'Minimum Transit Angle ({min_transit_angle_rad.value:.4f} rad)')
plt.legend()
plt.grid()
plt.show()

t = Time(datetime_jd[min_index], format='jd')
print(t.iso)

transit_longitude = longitude[min_index]
transit_latitude = latitude[min_index]
print(transit_longitude, transit_latitude)

def rotate_to_mars_frame(datetime_jd, r_eclip):
    for i in datetime_jd:
        t = Time(i, format='jd')
        et = spice.str2et(t.iso)
        R = spice.pxform("ECLIPJ2000", "IAU_MARS", et)
        r_eclip = np.array([1, 2, 3])  # Replace with actual ecliptic coordinates of the point of interest
        r_mars = R @ r_eclip
        x_prime, y_prime, z_prime = r_mars