from datetime import datetime
from jwst_gtvt import get_gtvt_windows
from astroquery.jplhorizons import Horizons

obj = Horizons(
    id='499',
    location='500@-170',
    epochs={'start':'2030-08-19', 'stop':'2030-08-21', 'step':'1hour'}
)

eph = obj.ephemerides()
ra_mars = eph['RA_app'].value
dec_mars = eph['DEC_app'].value
datetime_jd = eph['datetime_jd']

def jwst_visible(ra, dec, observation_time):
    """Determine whether a fixed target is visible to JWST."""

    # Run GTVT calculation
    visibility_windows = get_gtvt_windows(ra, dec)

    for start, end in visibility_windows:
        if start <= observation_time <= end:
            return True

    return False

visible = []
for i in range(len(ra_mars)):
    visible.append(jwst_visible(
        ra = ra_mars[i],
        dec = dec_mars[i],
        observation_time = datetime_jd[i])
        )

print(visible)