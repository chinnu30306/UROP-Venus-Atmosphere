import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from astropy.time import Time
import spiceypy as spice

spice.furnsh("kernels/naif0012.tls")
spice.furnsh("kernels/pck00011.tpc")
spice.furnsh("kernels/de440s.bsp")


venus = Horizons(id=299, location='500@10', epochs={'start':'2030-08-19', 'stop':'2030-08-21', 'step':'1min'})
venus_vectors = venus.vectors()
datetime_jd = venus_vectors['datetime_jd']


def rotate_to_mars_frame(datetime_jd, r_eclip):
    for i in datetime_jd:
        t = Time(i, format='jd')
        et = spice.str2et(t.iso)
        R = spice.pxform("ECLIPJ2000", "IAU_MARS", et)
        r_eclip = np.array([1, 2, 3])  # Replace with actual ecliptic coordinates of the point of interest
        r_mars = R @ r_eclip
        x_prime, y_prime, z_prime = r_mars