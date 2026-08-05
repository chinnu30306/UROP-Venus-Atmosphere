import numpy as np


class Planet():
    '''
    
    '''

    def __init__(self, pos = [0,0,0], radius = 0, northPole = [0,0,1]):
        '''
        
        '''
        self.__pos = np.array(pos)
        self.__r = radius
        self.__axes = np.array(northPole)

    #position getter   
    def pos(self):
        '''
        
        '''
        return self.__pos

    #radius getter
    def rad(self):
        '''
        
        '''
        return self.__r

    # pole getter 
    def pole(self):
        '''
        
        '''
        return self.__axes

    #position setter
    def set_pos(self, newPos):
        '''
        
        '''
        self.__pos = newPos

    # getting the longitude and latitude 
    def longLat(self, rayVector):
        ''' 
        this returns the longitude and latitude centre of a transit from the blocked light by a given planet. 
        '''
        # coefficients of the quadratic
        R_m_sq = self.__r * self.__r
        v_sq = np.dot(rayVector,rayVector)
        r_sm_sq = np.dot(self.__pos,self.__pos)
        r_smDotV = np.dot(rayVector, self.__pos)

        #  finding R vector
        root = min(np.roots([v_sq, - 2 * r_smDotV, r_sm_sq - R_m_sq]))
        R = root * rayVector - self.__pos

        # rotating R vector    
        newR = R # THIS NEEDS WORK 

        # converting the newR into latitude longitude 
        lat = np.arctan(newR[2]/(newR[0] * newR[0] + newR[1] * newR[1])) * (180/np.pi)
        Long = np.atan2(newR[1], newR[0]) * (180/np.pi)

        return (lat, Long)