import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from functions_new import angle_between_vectors
from astropy.constants import R_sun
from poliastro.bodies import Venus, Mars
from classes_new import Cone

# Get the position vectors of Venus and Mars from JPL Horizons
m_v = Horizons(id=299, location='499', epochs={'start': '2030-08-19', 'stop': '2030-08-21', 'step': '1min'})
venus_vectors = m_v.vectors()
m_v_pos = list(zip(venus_vectors['x'], venus_vectors['y'], venus_vectors['z']))
datetime_jd = venus_vectors['datetime_jd']

m_s = Horizons(id=10, location='499', epochs={'start': '2030-08-19', 'stop': '2030-08-21', 'step': '1min'})
sun_vectors = m_s.vectors()
m_s_pos = list(zip(sun_vectors['x'], sun_vectors['y'], sun_vectors['z']))

s_v = Horizons(id=299, location='500@10', epochs={'start': '2030-08-19', 'stop': '2030-08-21', 'step': '1min'})
venus_vectors = s_v.vectors()
s_v_pos = list(zip(venus_vectors['x'], venus_vectors['y'], venus_vectors['z']))

s_m = Horizons(id=499, location='500@10', epochs={'start': '2030-08-19', 'stop': '2030-08-21', 'step': '1min'})
mars_vectors = s_m.vectors()
s_m_pos = list(zip(mars_vectors['x'], mars_vectors['y'], mars_vectors['z']))

v_m_pos = np.array(s_v_pos) - np.array(s_m_pos)

R_sun_au = R_sun.to("au").value
R_venus_au = Venus.R.to("au").value
R_mars_au = Mars.R.to("au").value

# --- angle measured at Mars (your original approach) ---
theta = []
transit_threshold = []
for i in range(len(v_m_pos)):
    theta.append(angle_between_vectors(v_m_pos[i], -np.array(s_m_pos[i])))
    # NOTE: alpha_sun/alpha_venus recomputed fresh each iteration before
    # being summed and appended, so this was already correct despite
    # reusing the same variable name as a scalar rather than a list.
    alpha_sun = np.arctan(R_sun_au / np.linalg.norm(s_m_pos[i]))
    alpha_venus = np.arctan(R_venus_au / np.linalg.norm(v_m_pos[i]))
    transit_threshold.append(alpha_sun + alpha_venus)

min_theta = min(theta)
min_index = theta.index(min_theta)
min_jd = datetime_jd[min_index]
print("Angle-at-Mars method:")
print(f"  min_index = {min_index}")
print(f"  min_jd    = {min_jd}")

# --- angle measured at the cone's vertex (this session's addition) ---
angle_at_vertex = []
vertex_threshold = []
for i in range(len(s_m_pos)):
    cone = Cone(R_sun_au, R_venus_au, s_v_pos[i])
    r12_hat = cone.r12 / cone.r12_mag
    vertex_pos = cone.apex_dist * r12_hat                     # Sun -> vertex
    vec_vertex_to_mars = np.array(s_m_pos[i]) - vertex_pos
    dist_vertex_to_mars = np.linalg.norm(vec_vertex_to_mars)

    angle_at_vertex.append(angle_between_vectors(vec_vertex_to_mars, r12_hat))
    alpha_mars_at_vertex = np.arctan(R_mars_au / dist_vertex_to_mars)
    vertex_threshold.append(cone.alpha + alpha_mars_at_vertex)

min_angle_vertex = min(angle_at_vertex)
min_index_vertex = angle_at_vertex.index(min_angle_vertex)
print("Angle-at-vertex method:")
print(f"  min_index = {min_index_vertex}")
print(f"  min_jd    = {datetime_jd[min_index_vertex]}")

# --- plot both ---
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 8))

ax1.plot(datetime_jd, theta, label='Angle (at Mars)')
ax1.plot(datetime_jd, transit_threshold, label='Transit threshold (at Mars)')
ax1.set_ylabel('Angle (rad)')
ax1.set_title('Measured at Mars')
ax1.legend()
ax1.grid()

ax2.plot(datetime_jd, angle_at_vertex, label='Angle (at vertex)')
ax2.plot(datetime_jd, vertex_threshold, label='Transit threshold (at vertex)')
ax2.set_xlabel('Julian Date')
ax2.set_ylabel('Angle (rad)')
ax2.set_title('Measured at the antumbral cone\'s vertex')
ax2.legend()
ax2.grid()

fig.suptitle('Venus-Mars transit alignment: two equivalent geometric views')
plt.tight_layout()
plt.show()