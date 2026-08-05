from astroquery.jplhorizons import Horizons
from astropy.constants import R_mars
from planet import Planet

# Get the position vectors of Venus and Mars from JPL Horizons
venus = Horizons(id=299, location='500@10', epochs={'start':'2030-08-19', 'stop':'2030-08-21', 'step':'1min'})
venus_vectors = venus.vectors()

venus_x = venus_vectors['x']
venus_y = venus_vectors['y']
venus_z = venus_vectors['z']
venus_datetime_jd = venus_vectors['datetime_jd']

venus_position = list(zip(venus_x, venus_y, venus_z))
mars = Horizons(id=499, location='500@10', epochs={'start':'2030-08-19', 'stop':'2030-08-21', 'step':'1min'})
mars_vectors = mars.vectors()

mars_x = mars_vectors['x']
mars_y = mars_vectors['y']
mars_z = mars_vectors['z']

mars_position = list(zip(mars_x, mars_y, mars_z))

Mars = Planet(pos = mars_position[0], radius = R_mars)

mars_datetime_jd = mars_vectors['datetime_jd']

for i, value in enumerate(mars_datetime_jd):
    Mars.set_pos(mars_position[i])
    [] = Mars.longLat() 