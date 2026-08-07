import numpy as np
from functions import rotate_to_mars_frame


class Planet():
    '''
    A class representing a celestial body (planet) with position, radius, and orientation.
    
    This class provides functionality to calculate the apparent position of other
    celestial objects as seen from the planet's surface, accounting for the planet's
    rotation and surface geometry.
    
    Attributes
    ----------
    __pos : np.ndarray
        3D position vector of the planet in the solar system barycenter frame [km]
    __r : float
        Radius of the planet [km]
    __axes : np.ndarray
        3D unit vector pointing to the planet's north pole
        
    Methods
    -------
    pos()
        Returns the current position vector
    rad()
        Returns the planet's radius
    pole()
        Returns the north pole direction vector
    set_pos(newPos)
        Updates the planet's position
    longLat(rayVector, datetime_jd)
        Calculates the surface coordinates where a line-of-sight intersects the planet
    '''
    def __init__(self, pos = [0,0,0], radius = 0):
        '''
        Initialize a Planet object with position, radius, and orientation.
        
        Parameters
        ----------
        pos : list or array-like, optional
            Initial 3D position vector of the planet [km]. Default is [0,0,0].
        radius : float, optional
            Radius of the planet [km]. Default is 0.
        northPole : list or array-like, optional
            3D unit vector indicating the planet's north pole direction.
            Default is [0,0,1] (Z-axis aligned).
        '''
        self.__pos = np.array(pos)
        self.__r = radius

    #position getter   
    def pos(self):
        '''
        Get the planet's current position vector.
        
        Returns
        -------
        np.ndarray
            3D position vector of the planet [km]
        '''
        return self.__pos

    #radius getter
    def rad(self):
        '''
        Get the planet's radius.
        
        Returns
        -------
        float
            Radius of the planet [m]
        '''
        return self.__r

    #position setter
    def set_pos(self, newPos):
        '''
        Update the planet's position in space.
        
        Parameters
        ----------
        newPos : list or array-like
            New 3D position vector for the planet [km]
        '''
        self.__pos = newPos

    # getting the longitude and latitude 
    def longLat(self, rayVector, datetime_jd):
        ''' 
        Calculate the longitude and latitude on the planet's surface where a
        line-of-sight vector from the planet intersects its surface.
        
        This method solves for the intersection of a ray (from the planet center
        through the given direction) with the planet's spherical surface. It then
        rotates the intersection point into the planet's body-fixed reference
        frame (accounting for planetary rotation) and converts to longitude/latitude.
        
        Parameters
        ----------
        rayVector : array-like
            3D vector pointing from the planet toward the target object 
            (e.g., another planet or star) [km]
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
        rayVector = np.array(rayVector)
        R_m_sq = self.__r * self.__r
        v_sq = np.dot(rayVector,rayVector) 
        r_sm_sq = np.dot(self.__pos,self.__pos) 
        r_smDotV = np.dot(rayVector, self.__pos)
        a = v_sq
        b = - 2 * r_smDotV
        c = r_sm_sq - R_m_sq
        print(f"Quadratic Coefficients: a={a}, b={b}, c={c}")

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