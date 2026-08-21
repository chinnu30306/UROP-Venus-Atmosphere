import numpy as np
from functions_new import angle_between_vectors, rotate_to_mars_frame, rotate_to_sun_frame, quadric
from scipy.integrate import quad
from scipy.constants import Boltzmann as k_B
from astropy import units as u

class Grid():
    def __init__(self, rows, cols, R):
        self.rows = rows
        self.cols = cols
        self.value = np.zeros((rows, cols))
        self.R_prime = np.zeros((rows, cols, 3))
        lats = np.linspace(-90, 90, rows)
        lons = np.linspace(-180, 180, cols)
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                lat_rad, lon_rad = np.radians(lat), np.radians(lon)
                x = R * np.cos(lat_rad) * np.cos(lon_rad)
                y = R * np.cos(lat_rad) * np.sin(lon_rad)
                z = R * np.sin(lat_rad)
                self.R_prime[i, j] = [x, y, z]

    def get_rows(self):
        return self.rows

    def get_cols(self):
        return self.cols

    def set_value(self, row, col, value):
        self.value[row, col] = value

    def get_value(self, row, col):
        return self.value[row, col]


class Planet():
    """
    A spherical body. `pos` is ALWAYS heliocentric (Sun at the origin),
    matching Veras (2019) Section 2.1: "The centre of the primary...
    resides at the origin."
    """
    def __init__(self, pos=[0, 0, 0], radius=0, rows=180, cols=360):
        self.__pos = np.array(pos, dtype=float)
        self.__r = radius
        self.grid = Grid(rows, cols, self.__r)

    def pos(self):
        return self.__pos

    def rad(self):
        return self.__r

    def set_pos(self, newPos):
        self.__pos = np.array(newPos, dtype=float)

    def sub_axis_point(self, venus_pos, time):
        """
        Centre-line point of the shadow: intersect the Sun->Venus ray
        (the cone's own axis) with the near side of Mars's sphere, and
        return the (lat, lon) of that point on Mars.

        This is what you actually want for "the path of Venus's shadow":
        the track of the point where the Sun-Venus-Mars alignment is
        closest to perfect, i.e. Veras's psi -> 0 (eq. 56).
        """
        venus_pos = np.array(venus_pos, dtype=float)
        r13 = self.pos()          # Sun -> Mars
        R3 = self.rad()

        # angular test: is Mars even close enough to the Sun-Venus line
        # to be plausibly in transit? (cheap early-out, not required
        # for correctness since the quadratic below returns None anyway)
        # Solve |t*venus_pos - r13|^2 = R3^2 for t (Sun->Venus line
        # parametrised as t*venus_pos, since the Sun is the origin).
        a = np.dot(venus_pos, venus_pos)
        b = -2 * np.dot(venus_pos, r13)
        c = np.dot(r13, r13) - R3 * R3
        disc = b * b - 4 * a * c
        if disc < 0:
            return None

        t = min(np.roots([a, b, c]))     # near-side (Sun-facing) root
        Q = t * venus_pos                # heliocentric point on Mars's surface
        R_m_vec = Q - r13                # relative to Mars's centre
        R_prime = rotate_to_mars_frame(time, R_m_vec)

        lat = np.degrees(np.arctan2(R_prime[2], np.hypot(R_prime[0], R_prime[1])))
        lon = np.degrees(np.arctan2(R_prime[1], R_prime[0]))
        return (lat, lon)

    def isInCone(self, cone, value, time):
        """
        Mark every grid point on Mars's sunlit hemisphere that lies
        inside the umbral/antumbral cone (Veras 2019, Section 3.2/4).
        """
        coeffs = cone.quadric_coeffs()
        sun_dir_from_mars = -self.pos()          # Mars -> Sun

        for i in range(self.grid.get_rows()):
            for j in range(self.grid.get_cols()):
                # body-fixed offset -> ECLIPJ2000-oriented offset (still
                # centred on Mars)
                R_vec = rotate_to_sun_frame(time, self.grid.R_prime[i, j])

                # skip Mars's own night side: a point that isn't facing
                # the Sun can't be showing a transit shadow regardless
                # of cone geometry
                if angle_between_vectors(R_vec, sun_dir_from_mars) > np.pi / 2:
                    continue

                point = R_vec + self.pos()        # heliocentric position

                # restrict to the physically relevant nappe (Veras 2019,
                # footnote 2): points with cos(xi) < 0 sit mathematically
                # inside the same quadric but on the far side of the Sun
                cos_xi = np.dot(point, cone.r12) / (
                    np.linalg.norm(point) * cone.r12_mag
                )
                if cos_xi <= 0:
                    continue

                Q = quadric(coeffs, point)
                if Q <= 0:          # interior of the cone (see derivation)
                    self.grid.set_value(i, j, value)


class Cone():
    """
    The umbral/antumbral radiation cone formed by the primary (Sun,
    R1) and the occulter (Venus, R2), separated by r12 (Sun->Venus).
    All formulas are Veras (2019) eqs (10)-(26).
    """
    def __init__(self, R1, R2, r12):
        self.R1 = R1
        self.R2 = R2
        self.set_r12(r12)

    def set_r12(self, r12):
        self.r12 = np.array(r12, dtype=float)
        self.r12_mag = np.linalg.norm(self.r12)
        self.delta = (self.R1 - self.R2) / self.r12_mag       # eq (14), = sin(alpha)
        self.alpha = np.arcsin(np.clip(self.delta, -1.0, 1.0))
        self.apex_dist = self.R1 / self.delta                 # h+n, eq (13)
        self.n = self.R1 * self.delta
        self.h = self.apex_dist - self.n                      # eq (11)
        self.Rc = self.R1 * np.sqrt(max(0.0, 1 - self.delta**2))  # eq (12)

    def quadric_coeffs(self):
        """Coefficients A..K of the double-napped cone quadric (eqs 17-26)."""
        x12, y12, z12 = self.r12
        r12 = self.r12_mag
        d2 = self.delta ** 2
        A = 1 - d2 - (x12 / r12) ** 2
        B = 1 - d2 - (y12 / r12) ** 2
        C = 1 - d2 - (z12 / r12) ** 2
        D = -x12 * y12 / r12 ** 2
        E = -y12 * z12 / r12 ** 2
        F = -x12 * z12 / r12 ** 2
        G = self.R1 * self.delta * x12 / r12
        H = self.R1 * self.delta * y12 / r12
        J = self.R1 * self.delta * z12 / r12
        K = -self.R1 ** 2
        return [A, B, C, D, E, F, G, H, J, K]


class VenusAtmosphere:
    def __init__(self, R_cloud_top_km, T_atm, mu, g, n0):
        """
        R_cloud_top_km : radius of the opaque cloud deck (km) -- the
                          hard edge below which T=0 at every wavelength
        T_atm          : temperature of the semi-transparent haze/upper
                          atmosphere above the cloud deck (K)
        mu             : mean molecular mass of that layer (kg/molecule)
        g              : gravitational acceleration used for the scale
                          height (m/s^2)
        n0             : number density at the cloud top (molecules/m^3)
        """
        self.R_cloud_top = R_cloud_top_km * 1e3   # km -> m
        self.T_atm = T_atm
        self.mu = mu
        self.g = g
        self.n0 = n0
        self.H = k_B * T_atm / (mu * g)            # scale height (m)

    # ---- step 1: n(h) ----
    def number_density(self, h):
        """h = altitude above the cloud deck (m), h >= 0."""
        return self.n0 * np.exp(-h / self.H)

    # ---- step 2: N(b) ----
    def column_density(self, b):
        """h(x) = sqrt(b^2 + x^2) - R_cloud_top along the ray."""
        if b < self.R_cloud_top:
            return np.inf
        integrand = lambda x: self.number_density(np.hypot(b, x) - self.R_cloud_top)
        x_max = 50 * self.H   # density is negligible well beyond this
        half, _ = quad(integrand, 0, x_max)
        return 2 * half       # symmetric ray, +x and -x

    # ---- step 3: tau_lambda(b) ----
    def optical_depth(self, b, sigma_lambda):
        """sigma_lambda: cross-section (m^2/molecule) at this wavelength."""
        if b < self.R_cloud_top:
            return np.inf
        return self.column_density(b) * sigma_lambda

    # ---- step 4: T_lambda(b) ----
    def transmission(self, b, sigma_lambda):
        tau = self.optical_depth(b, sigma_lambda)
        return np.exp(-tau) if np.isfinite(tau) else 0.0

    # ---- step 5: R_eff(lambda) ----
    def effective_radius_au(self, sigma_lambda, n_scale_heights=15):
        """
        opaque disk area plus the partially-transmissive annulus above it,
        weighted by (1-T(b)); returns AU.
        """
        b_max = self.R_cloud_top + n_scale_heights * self.H
        integrand = lambda b: b * (1 - self.transmission(b, sigma_lambda))
        annulus_integral, _ = quad(integrand, self.R_cloud_top, b_max)
        R_eff_sq = self.R_cloud_top**2 + 2 * annulus_integral
        return np.sqrt(R_eff_sq) / 1.495978707e11