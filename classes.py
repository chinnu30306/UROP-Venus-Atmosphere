import numpy as np
from functions import angle_between_vectors, rotate_to_mars_frame, intercept, rotate_to_sun_frame
from astropy.constants import R_sun
from poliastro.bodies import Venus

class Planet():
    def __init__(self, pos = [0,0,0], radius = 0, rows=180, cols=360):
        self.__pos = np.array(pos)
        self.__r = radius
        self.grid = Grid(rows, cols, self.__r)
  
    def pos(self):
        return self.__pos

    def rad(self):
        return self.__r

    def set_pos(self, newPos):
        self.__pos = newPos

    # getting the longitude and latitude 
    def longLat(self, objectPos, datetime_jd):
        # coefficients of the quadratic
        venusPos = np.array(objectPos)
        r_vm = self.__pos - venusPos
        rayVector = np.array([venusPos[0], venusPos[1], 0])
        R_m_sq = self.__r * self.__r
        v_sq = np.dot(rayVector,rayVector) 
        r_vm_sq = np.dot(r_vm,r_vm) 
        r_vmDotV = np.dot(rayVector, r_vm)
        a = v_sq
        b = - 2 * r_vmDotV
        c = r_vm_sq - R_m_sq
        print("Quadratic Coefficients: a =", a, ", b =", b, ", c =", c)

        #  finding R vector
        if b * b < 4 * a *c:
            return None
        root = min(np.roots([a, b, c]))
        R_m = root * rayVector - self.__pos

        # rotating R vector    
        R_prime = rotate_to_mars_frame(datetime_jd, R_m)

        # converting the newR into latitude longitude 
        lat = np.arctan2(R_prime[2], np.sqrt(R_prime[0]*R_prime[0] + R_prime[1]*R_prime[1])) * (180 / np.pi)
        long = np.arctan2(R_prime[1], R_prime[0]) * (180 / np.pi)

        return (lat, long)

    def lat_long_new(self, pos, time):
        R_m_au = self.rad()
        R_sun_au = R_sun.to("au").value
        R_venus_au = Venus.R.to("au").value
        r_sm = self.pos()
        r_vm = - np.array(pos) + r_sm

        theta = angle_between_vectors(-np.array(r_vm), -np.array(r_sm))
        alpha_sun = np.arctan(R_sun_au / np.linalg.norm(r_sm))
        alpha_venus = np.arctan(R_venus_au / np.linalg.norm(r_vm))
        transit_threshold = alpha_sun + alpha_venus
        if theta > transit_threshold:
            return None

        # (r_sm)^2*x^2 - 2*(r_sm . r_vm)*x + (r_vm)^2 - R_m^2 = 0
        a = np.dot(r_sm, r_sm)
        b = -2 * np.dot(r_sm, r_vm)
        c = np.dot(r_vm, r_vm) - R_m_au * R_m_au

        if b*b - 4*a*c < 0:
            return None

        root = min(np.roots([a, b, c]))
        R_m_vec = -r_vm + root * r_sm
        R_prime = rotate_to_mars_frame(time, R_m_vec)

        lat = np.arctan2(R_prime[2], np.sqrt(R_prime[0]**2 + R_prime[1]**2)) * (180 / np.pi)
        long = np.arctan2(R_prime[1], R_prime[0]) * (180 / np.pi)

        return (lat, long)

    def isInCone(self, other, value, time):
        coeffs = other.cone_sphere_intersect(self)
        for i in range(self.grid.get_rows()):
            for j in range(self.grid.get_cols()):
                R_vec = rotate_to_sun_frame(time, self.grid.R_prime[i, j])

                if angle_between_vectors(R_vec, - np.array(self.pos())) > np.pi/2:
                    pass
                else:
                    eqn = intercept(coeffs, R_vec + self.pos())
                    if eqn >= 0:
                        self.grid.set_value(i, j, value)


class Grid():
    def __init__(self, rows, cols, R):
        self.rows = rows
        self.cols = cols
        self.value = np.zeros((rows, cols))
        self.R_prime = np.zeros((rows, cols, 3))

        # Create arrays of latitudes and longitudes
        lats = np.linspace(-90, 90, rows)    # rows points from -90° to +90°
        lons = np.linspace(-180, 180, cols)  # cols points from -180° to +180°
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                lat_rad = np.radians(lat)
                lon_rad = np.radians(lon)
                
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
                    
        
# class Sphere():
#     def __init__(self, R3, r_13):
#         self.R3 = R3
#         self.r_13 = np.array(r_13)

class Cone():
    def __init__(self, R1, R2, r_12):
        self.R1 = R1
        self.R2 = R2
        self.r_12 = np.array(r_12)
        self.r_12_mag = np.linalg.norm(r_12)
        self.delta = (self.R1 - self.R2) / self.r_12_mag

    def set_r12(self, r_12):
        self.r_12 = np.array(r_12)
        self.r_12_mag = np.linalg.norm(r_12)
        self.delta = (self.R1 - self.R2) / self.r_12_mag

    def set_r12(self, r_12):
        self.r_12 = np.array(r_12)
        self.r_12_mag = np.linalg.norm(r_12)
        self.h = self.R1 * (self.r_12_mag*self.r_12_mag - (self.R1 - self.R2)*(self.R1 - self.R2)) / (self.r_12_mag * (self.R1 - self.R2))
        self.R_c = self.R1 / self.r_12_mag * np.sqrt(self.r_12_mag*self.r_12_mag - (self.R1 - self.R2)*(self.R1 - self.R2))
        self.delta = (self.R1 - self.R2) / self.r_12_mag
        self.n = self.R1 / self.delta - self.h


    def cone_sphere_intersect(self, other): # defines coefficients for quadric that needs to be solved to find the intersect between 
        A = (self.r_12[0] * self.r_12[0]) / (self.r_12_mag * self.r_12_mag)
        B = (self.r_12[1] * self.r_12[1]) / (self.r_12_mag * self.r_12_mag)
        C = (self.r_12[2] * self.r_12[2]) / (self.r_12_mag * self.r_12_mag)
        D = (self.r_12[0] * self.r_12[1]) / (self.r_12_mag * self.r_12_mag)
        E = (self.r_12[1] * self.r_12[2]) / (self.r_12_mag * self.r_12_mag)
        F = (self.r_12[0] * self.r_12[2]) / (self.r_12_mag * self.r_12_mag)
        G = -other.pos()[0] * (1 - self.delta * self.delta) - (self.delta * self.R1 * self.r_12[0]) / self.r_12_mag
        H = -other.pos()[1] * (1 - self.delta * self.delta) - (self.delta * self.R1 * self.r_12[1]) / self.r_12_mag
        J = -other.pos()[2] * (1 - self.delta * self.delta) - (self.delta * self.R1 * self.r_12[2]) / self.r_12_mag
        K = self.R1 * self.R1 + (1 - self.delta * self.delta) * (np.dot(other.pos(), other.pos()) - other.rad() * other.rad())

        return [A, B, C, D, E, F, G, H, J, K]

    