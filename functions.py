import numpy as np
from astropy.time import Time
import spiceypy as spice

def angle_between_vectors(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return angle_rad

spice.furnsh("kernels/naif0012.tls")
spice.furnsh("kernels/pck00011.tpc")
spice.furnsh("kernels/de440s.bsp")

def rotate_to_mars_frame(datetime_jd, r_eclip):
    t = Time(datetime_jd, format='jd')
    et = spice.str2et(t.iso)
    R = spice.pxform("ECLIPJ2000", "IAU_MARS", et)
    r_mars = R @ r_eclip
    return r_mars