import numpy as np
from functions import angle_between_vectors, rotate_to_mars_frame
from astropy.constants import R_sun
from poliastro.bodies import Venus

class Planet():
    def __init__(self, pos = [0,0,0], radius = 0):
        self.__pos = np.array(pos)
        self.__r = radius

    #position getter   
    def pos(self):
        return self.__pos

    #radius getter
    def rad(self):
        return self.__r

    #position setter
    def set_pos(self, newPos):
        self.__pos = newPos

    # getting the longitude and latitude 
    def longLat(self, objectPos, datetime_jd):
        ''' 
        Calculate the longitude and latitude on the planet's surface where a
        line-of-sight vector from the planet intersects its surface.
        
        This method solves for the intersection of a ray (from the planet center
        through the given direction) with the planet's spherical surface. It then
        rotates the intersection point into the planet's body-fixed reference
        frame (accounting for planetary rotation) and converts to longitude/latitude.
        
        Parameters
        ----------
        objectPos : array-like
            3D position vector of the target object [km]
        datetime_jd : float
            Julian Date (JD) used to determine the planet's rotation angle for
            converting from inertial to body-fixed coordinates
            
        Returns
        -------
        tuple
            (latitude, longitude) in degrees, where:
            - latitude is in range [-90°, +90°]
            - longitude is in range [-180°, +180°]
        '''

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


class Grid():
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.value = np.zeros((rows, cols))

    def get_rows(self):
        return self.rows

    def get_cols(self):
        return self.cols

    def set_value(self, row, col, value):
        self.value[row, col] = value

    def get_value(self, row, col):
        return self.value[row, col]

    def isInCone(self, cone, value):
        for i in range(self.rows()):
            for j in range(self.cols()):
                if cone:
                    self.set_value(i, j, value)
        
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
        self.h = self.R1 * (self.r_12_mag*self.r_12_mag - (self.R1 - self.R2)*(self.R1 - self.R2)) / (self.r_12_mag * (self.R1 - self.R2))
        self.R_c = self.R1 / self.r_12_mag * np.sqrt(self.r_12_mag*self.r_12_mag - (self.R1 - self.R2)*(self.R1 - self.R2))
        self.delta = (self.R1 - self.R2) / self.r_12_mag
        self.n = self.R1 / self.delta - self.h

    def cone_sphere_intersect(self, other):
        A = self.r_12[0]*self.r_12[0] / self.r_12_mag*self.r_12_mag
        B = self.r_12[1]*self.r_12[1] / self.r_12_mag*self.r_12_mag
        C = 
        K = self.R1*self.R1 + (1 - self.delta*self.delta) * (other.pos()*other.pos() - other.rad()*other.rad())
    