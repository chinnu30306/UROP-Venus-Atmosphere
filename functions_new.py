import numpy as np
from astropy.time import Time
import spiceypy as spice
# import hapi

# # Use HTTPS directly instead of HTTP
# hapi.GLOBAL_HOST = 'https://hitran.org'

from hapi import db_begin, fetch, absorptionCoefficient_Voigt

def angle_between_vectors(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

spice.furnsh("kernels/naif0012.tls")
spice.furnsh("kernels/pck00011.tpc")
spice.furnsh("kernels/de440s.bsp")

def rotate_to_mars_frame(datetime_jd, r_eclip):
    """Rotate a Mars-centred offset vector from ECLIPJ2000 into IAU_MARS."""
    t = Time(datetime_jd, format='jd')
    et = spice.str2et(t.iso)
    R = spice.pxform("ECLIPJ2000", "IAU_MARS", et)
    return R @ r_eclip

def rotate_to_sun_frame(datetime_jd, r_mars_body):
    """Rotate a Mars-centred offset vector from IAU_MARS into ECLIPJ2000."""
    t = Time(datetime_jd, format='jd')
    et = spice.str2et(t.iso)
    R = spice.pxform("IAU_MARS", "ECLIPJ2000", et)
    return R @ r_mars_body

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

import numpy as np

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
    alpha_venus = R_venus_au / np.linalg.norm(v_m_pos)
    overlap = circle_overlap_area(alpha_sun, alpha_venus, theta)
    sun_area = np.pi * alpha_sun ** 2
    depth = overlap / sun_area
    return depth


# --- toy cross-section models: REPLACE with real line-list / HITRAN data ---

def sigma_rayleigh(wavelength_nm, sigma_ref=5e-31, lambda_ref_nm=550):
    """Placeholder Rayleigh-scattering continuum, sigma ~ lambda^-4 (m^2)."""
    return sigma_ref * (lambda_ref_nm / wavelength_nm) ** 4


def sigma_with_band(wavelength_nm, band_center_nm, band_width_nm,
                     band_peak_sigma, sigma_ref=5e-31, lambda_ref_nm=550):
    """Rayleigh continuum plus one toy Gaussian absorption band, standing
    in for a real molecular feature (e.g. SO2/CO2) -- for illustration
    only, not a physical cross-section."""
    continuum = sigma_rayleigh(wavelength_nm, sigma_ref, lambda_ref_nm)
    band = band_peak_sigma * np.exp(
        -0.5 * ((wavelength_nm - band_center_nm) / band_width_nm) ** 2
    )
    return continuum + band



def get_sigma_lambda_from_hitran(
    molecule='CO2',
    isotopologue=1,
    wl_min_nm=1200,
    wl_max_nm=2600,
    temperature=230,
    pressure=1.0,
    resolution_cm=0.01,
    cache_dir='hitran_data',
    return_wavenumber=False
):
    """
    Calculate absorption cross-section (sigma_lambda) for a given molecule
    using the HITRAN database via HAPI.
    
    Parameters:
    -----------
    molecule : str
        Molecule name (e.g., 'CO2', 'CH4', 'H2O')
    isotopologue : int
        Isotopologue ID (1 = main isotope)
    wl_min_nm, wl_max_nm : float
        Wavelength range in nanometers
    temperature : float
        Temperature in Kelvin (e.g., 230 K for Venus upper atmosphere)
    pressure : float
        Pressure in atmospheres (e.g., 1.0 atm)
    resolution_cm : float
        Spectral resolution in wavenumber (cm^-1)
        Lower values = higher resolution (e.g., 0.01 for high-res)
    cache_dir : str
        Directory to store downloaded HITRAN data
    return_wavenumber : bool
        If True, also return wavenumber array in cm^-1
    
    Returns:
    --------
    wavelengths_nm : numpy.ndarray
        Wavelengths in nanometers
    sigma_lambda : numpy.ndarray
        Absorption cross-section in cm²/molecule
    (Optional) wavenumber_cm : numpy.ndarray
        Wavenumber in cm^-1
    """
    
    # Initialize HAPI database
    db_begin(cache_dir)
    
    # Convert wavelength (nm) to wavenumber (cm^-1)
    # 1 cm^-1 = 1 / (wavelength in cm) = 1 / (wavelength in nm * 1e-7)
    wavenumber_min = 1.0 / (wl_max_nm * 1e-7)
    wavenumber_max = 1.0 / (wl_min_nm * 1e-7)
    
    # print(f"Fetching {molecule} data from {wavenumber_min:.1f} to {wavenumber_max:.1f} cm^-1...")
    # print(f"  This corresponds to {wl_min_nm:.0f} to {wl_max_nm:.0f} nm")
    
    # Download data from HITRAN
    # Molecule IDs: 2 = CO2, 6 = CH4, 1 = H2O, etc.
    molecule_ids = {
        'CO2': 2,
        'CH4': 6,
        'H2O': 1,
        'CO': 5,
        'SO2': 9,
        'N2O': 4,
        'O3': 3,
        'N2': 7
    }
    
    mol_id = molecule_ids.get(molecule, 2)  # Default to CO2
    
    try:
        fetch(molecule, mol_id, isotopologue, wavenumber_min, wavenumber_max)
        print(f"Successfully fetched {molecule} data")
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Make sure you have an internet connection.")
        print("If this keeps failing, try using a different wavenumber range.")
        raise
    
    # Create wavenumber grid
    wavenumber_cm = np.arange(wavenumber_min, wavenumber_max, resolution_cm)
    
    # Calculate absorption coefficient using Voigt profile
    # print(f"Calculating absorption coefficient at T={temperature}K, P={pressure}atm...")
    # print(f"  Resolution: {resolution_cm} cm^-1 ({len(wavenumber_cm)} points)")
    
    try:
        # Calculate absorption coefficient
        # HITRAN_units=True returns cross-section in cm²/molecule
        nu, coef = absorptionCoefficient_Voigt(
            SourceTables=molecule,
            WavenumberGrid=wavenumber_cm,
            Environment={'p': pressure, 'T': temperature},
            HITRAN_units=True  # Returns cm²/molecule
        )
    except TypeError:
        # Fallback for older HAPI versions
        nu, coef = absorptionCoefficient_Voigt(
            SourceTables=molecule,
            WavenumberGrid=wavenumber_cm,
            Environment={'p': pressure, 'T': temperature}
        )
    
    # Convert wavenumber (cm^-1) to wavelength (nm)
    # wavenumber = 1 / (wavelength in cm)
    # wavelength (nm) = 1 / (wavenumber * 10000)
    # wavelengths_nm = 1.0 / (nu * 1e7 / 1e9)  # 1 cm^-1 = 10^7 / nm
    wavelengths_nm = 1e7 / nu
    
    # Cross-section is already in cm²/molecule from HITRAN_units=True
    sigma_lambda = coef * 1e-4
    
    # print(f"Cross-section range: {sigma_lambda.min():.2e} to {sigma_lambda.max():.2e} cm²/molecule")
    # print(f"Wavelength range: {wavelengths_nm.min():.1f} to {wavelengths_nm.max():.1f} nm")
    
    if return_wavenumber:
        return wavelengths_nm, sigma_lambda, nu
    return wavelengths_nm, sigma_lambda


def create_interpolator(wavelengths_nm, sigma_lambda):
    """
    Create an interpolation function for sigma_lambda at arbitrary wavelengths.
    
    Parameters:
    -----------
    wavelengths_nm : numpy.ndarray
        Wavelengths in nanometers
    sigma_lambda : numpy.ndarray
        Absorption cross-section in cm²/molecule
    
    Returns:
    --------
    function : Interpolation function sigma(wavelength_nm)
    """
    from scipy.interpolate import interp1d
    
    # Create interpolation function
    # Fill with small value outside range (or nearest)
    sigma_interp = interp1d(
        wavelengths_nm,
        sigma_lambda,
        kind='linear',
        bounds_error=False,
        fill_value=sigma_lambda.min()  # Use minimum as fill value
    )
    
    return sigma_interp
