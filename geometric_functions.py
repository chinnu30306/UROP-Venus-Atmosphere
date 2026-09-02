import numpy as np
from astropy.time import Time
import spiceypy as spice


def angle_between_vectors(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

spice.furnsh("kernels/naif0012.tls")
spice.furnsh("kernels/pck00011.tpc")
spice.furnsh("kernels/de440s.bsp")

def rotate_to_frame(datetime_jd, r, initial_frame, final_frame):
    t = Time(datetime_jd, format='jd')
    et = spice.str2et(t.iso)
    R = spice.pxform(initial_frame, final_frame, et)
    return R @ r

def quadric(coeff, point):
    """
    Evaluate the general quadric (Veras 2019, eq. 7):
        A x^2 + B y^2 + C z^2 + 2D xy + 2E yz + 2F xz + 2G x + 2H y + 2J z + K
    `point` must be expressed in the SAME frame the coefficients were
    derived in (heliocentric / ECLIPJ2000-oriented, origin at the Sun).
    """
    x, y, z = point
    A, B, C, D, E, F, G, H, J, K = coeff
    return (A*x*x + B*y*y + C*z*z
            + 2*(D*x*y + E*y*z + F*x*z + G*x + H*y + J*z)
            + K)

def transit_visible_from_mars(R1, R2, R3, r12, r13):
    """
    Veras (2019) eqs. (37), (53)-(58): is any part of the target
    (Mars, radius R3) inside the primary-occulter shadow cone?

    R1, R2, R3 : radii of Sun, Venus, Mars (consistent units, e.g. AU)
    r12        : Sun -> Venus vector (heliocentric)
    r13        : Sun -> Mars vector (heliocentric)
    """
    r12 = np.array(r12, dtype=float)
    r13 = np.array(r13, dtype=float)
    r12_mag = np.linalg.norm(r12)
    r13_mag = np.linalg.norm(r13)

    delta = (R1 - R2) / r12_mag                      # eq. (14)

    # eq. (37): depends only on R1, delta, r13, R3 -- not on Venus's
    # exact direction, only its distance (via delta)
    Kint = R1**2 + (1 - delta**2) * (r13_mag**2 - R3**2)
    if Kint < 0:
        # Mars is closer to the Sun than the cone's own radius allows
        # for a tangent solution to exist -- treat as "not in transit"
        return False

    sqrtK = np.sqrt(Kint)

    # eq. (53): tangent-point distance from the Sun
    r_tan = (Kint + R1**2 - 2*delta*R1*sqrtK) / (1 - delta**2)

    # eq. (54)
    cos_xi_tan = sqrtK / r_tan

    # eq. (55)
    numerator = Kint**1.5 - delta * R1 * Kint
    denom = r13_mag * np.sqrt((1 - delta**2) * Kint * (r_tan * (1 - delta**2)))
    cos_kappa_tan = numerator / denom

    xi_tan = np.arccos(np.clip(cos_xi_tan, -1.0, 1.0))
    kappa_tan = np.arccos(np.clip(cos_kappa_tan, -1.0, 1.0))
    psi_bullet = abs(xi_tan) + abs(kappa_tan)          # eq. (58)

    # eq. (56)
    cos_psi = np.dot(r12, r13) / (r12_mag * r13_mag)

    # eq. (57)
    return cos_psi >= np.cos(psi_bullet)

def circle_overlap_area(R1, R2, d):
    """
    Calculate the area of overlap between two circles.

    R1 : radius of circle 1
    R2 : radius of circle 2
    d  : distance between their centres

    All quantities must use the same units.
    """

    # No overlap
    if d >= R1 + R2:
        return 0.0

    # One circle completely inside the other
    if d <= abs(R1 - R2):
        return np.pi * min(R1, R2)**2

    # Partial overlap
    alpha = np.arccos((d**2 + R1**2 - R2**2) / (2 * d * R1))
    beta = np.arccos((d**2 + R2**2 - R1**2) / (2 * d * R2))
    area = (R1**2 * alpha + R2**2 * beta - 0.5 * np.sqrt((-d + R1 + R2) * (d + R1 - R2) * (d - R1 + R2) * (d + R1 + R2)))
    return area

def transit_depth(R_venus_au, v_m_pos, alpha_sun, theta):
    """
    R_venus_au: apparent radius of the occulting body (Venus or R_eff), in AU
                *as seen from Mars* -- i.e. already an angular-equivalent
                radius, matching how alpha_sun is computed.
    v_m_pos:    Venus-minus-Mars position vector (au), used only for its norm.
    alpha_sun, theta: angular sun radius and separation angle, radians.
    """
    alpha_venus = R_venus_au / np.linalg.norm(v_m_pos)
    overlap = circle_overlap_area(alpha_sun, alpha_venus, theta)  # noqa: F821
    sun_area = np.pi * alpha_sun ** 2
    depth = overlap / sun_area
    return depth