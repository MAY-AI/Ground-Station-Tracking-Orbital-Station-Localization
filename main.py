## Gabriel Agostine
# ASEN 6044 - Final Project
# Last modified: 4/28/2026

## Imports
#############################################################
#############################################################

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## Constants
#############################################################
#############################################################

# Simulation
MU = 3.986e14       # gravitational parameter [m^3/s^2]
R_E = 6.3781e6      # Earth radius [m]
OMEGA_E = 7.2921e-5 # Earth rotation rate [rad/s]

# Tracking station (lat [deg], long [deg]), altitude [m]
TRACKING_STATION_ANGLES = [
    (45.0, 0.0),    # South Western France
    (45.0, 120.0),  # North Eastern China
    (45.0, -120.0), # Northern Oregon, US
    (-30.0, 20.0),  # Southern Africa
    (-45.0, 170.0), # New Zeland
    (-45.0, -70.0), # Southern Argentina
    (10.0, -70.0),  # North Western Venezuela
    (10.0, 0.0),    # North Eastern Ghana
    (7.0, 80.0),    # Western Sri Lanka
    (19.7, -155.5)  # Big Hawaii Island
]
TRACKING_STATION_ALTITUDES = [
    34.21,          # South Western France
    1_029.92,       # North Eastern China
    1_350.93,       # Northern Oregon, US
    976.83,         # Southern Africa
    638,            # New Zeland
    554.11,         # Southern Argentina
    829.48,         # North Western Venezuela
    180.89,         # North Eastern Ghana
    31.68,          # Western Sri Lanka
    2_013.42        # Big Hawaii Island
]

# Tracking station (lat [deg], lon [deg]), altitude [m]
TRACKING_STATION_ANGLES_US = [
    (42.9424, -71.6361),  # New Boston, NH
    (38.8033, -104.5256), # Schriever, CO
    (34.7373, -120.5843), # Vandenberg, CA
    (21.5664, -158.2528), # Kaena Point, HI
    (64.8048, -147.5109), # Fairbanks, AK
    (35.4260, -116.8900), # Goldstone, CA
    (32.5429, -106.6133), # White Sands, NM
    (37.9333, -75.4678),  # Wallops Island, VA
    (28.5383, -80.6503),  # Cape Canaveral, FL
    (13.6144,  144.8677), # Guam
]
TRACKING_STATION_ALTITUDES_US = [
    187.0,                # New Boston, NH
    1901.0,               # Schriever, CO
    112.2,                # Vandenberg, CA
    457.2,                # Kaena Point, HI
    155.2,                # Fairbanks, AK
    966.0,                # Goldstone, CA
    1442.6,               # White Sands, NM
    13.0,                 # Wallops Island, VA
    3.0,                  # Cape Canaveral, FL
    150.0,                # Guam
]

# Measurement
EL_CUTOFF = np.radians(10.0) # line-of-sight elevation cutoff [rad]

# Noise
MEASUREMENT_STD = np.array([np.radians(1.0), np.radians(1.0), 1_000.0]) # az, el, range
PROCESS_STD = np.array([0.1,  0.1,  0.1,
                        1e-3, 1e-3, 1e-3])
X0_STD = np.array([10e3, 10e3, 10e3,  # position std [m]
                   10.0, 10.0, 10.0]) # velocity std [m/s]

# Particle filter
N_PARTICLES = 1_000

# Plots
DARK_IMAGES: bool = False
if DARK_IMAGES:
    IMG_BG_COLOR = '#181818'
else:
    IMG_BG_COLOR = '#FFFFFF'

## Functions
#############################################################
#############################################################

def two_body_ode(t: float,
                 X: np.ndarray) -> np.ndarray:
    """
    ### Inputs:
    - Time
    - State
    ### Output:
    - State derivative
    """

    # Unpack state
    x, y, z = X[0:3]

    # Calculate distance
    D = np.sqrt(x**2 + y**2 + z**2)

    # Calculate and return state derivative
    ax = -MU / D**3 * x
    ay = -MU / D**3 * y
    az = -MU / D**3 * z
    return np.array([X[3], X[4], X[5], ax, ay, az])

def rk4_step(f,
             t: float,
             X: np.ndarray,
             dt: float) -> np.ndarray:
    """
    ### Inputs:
    - Function for integration
    - Time
    - State
    - Timestep
    ### Output:
    - Integrated state
    """

    # Calculate RK4 integration constants and integrate
    k1 = f(t, X)
    k2 = f(t + dt/2, X + dt/2 * k1)
    k3 = f(t + dt/2, X + dt/2 * k2)
    k4 = f(t + dt, X + dt * k3)
    return X + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

def simulate(X0: np.ndarray,
             t0: float,
             tf: float,
             dt: float,
             noise_std: float = 0.0) -> tuple[np.ndarray,
                                              np.ndarray]:
    """
    ### Inputs:
    - Initial state
    - Starting time
    - Ending time
    - Timestep
    ### Optional inputs:
    - Noise standard distribution
    ### Output:
    - Time and state history
    """

    # Initialize time and state histories
    times = np.arange(t0, tf + dt, dt)
    X_hist = np.zeros((len(times), 6))
    X_hist[0] = X0

    # Simulate orbital station
    for i in range(1, len(times)):
        X_hist[i] = rk4_step(two_body_ode, times[i - 1], X_hist[i - 1], dt)
        if noise_std > 0:
            X_hist[i, 3:] += np.random.randn(3) * noise_std
    return times, X_hist

def latlon_to_ecef(lat_deg: float,
                   lon_deg: float,
                   alt: float) -> np.ndarray:
    """
    ### Inputs:
    - Latitude [deg]
    - Longitude [deg]
    - Surface altitude [m]
    ### Output:
    - Position vector in ECEF frame
    """

    # Unpack lat and long angles as radians
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    # Calculate radius from center of Earth
    r = R_E + alt

    # Return ECEF position
    return np.array([
        r * np.cos(lat) * np.cos(lon),
        r * np.cos(lat) * np.sin(lon),
        r * np.sin(lat)
    ])

def initial_state_from_latlon(lat_deg: float,
                              lon_deg: float,
                              alt: float,
                              inc: float) -> np.ndarray:
    """
    ### Inputs:
    - Latitude [deg]
    - Longitude [deg]
    - Altitude [m]
    - Orbital inclination [rad]
    ### Output:
    - Initial ECI state vector (assumes ascending pass at given lat/lon)
    """

    # Position from lat/lon
    pos = latlon_to_ecef(lat_deg, lon_deg, alt)

    # Local east and north unit vectors at this lat/lon
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    e_east = np.array([-np.sin(lon),
                       np.cos(lon),
                       0.0])
    e_north = np.array([-np.sin(lat)*np.cos(lon),
                        -np.sin(lat)*np.sin(lon),
                        np.cos(lat)])

    # Build and return x_0
    r0 = np.linalg.norm(pos)
    v_circ = np.sqrt(MU / r0)
    vel = v_circ * (np.sin(inc) * e_north + np.cos(inc) * e_east)
    return np.concatenate([pos, vel])

def Rz(theta: float) -> np.ndarray:
    """
    ### Inputs:
    - Z axis rotation angle
    ### Output:
    - Rotation matrix from ECEF to ECI (for positive angle)
    """

    # Calculate and return DCM
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,  s, 0],
                     [-s, c, 0],
                     [0,  0, 1]])

def ned_rotation(lat_deg, lon_deg):
    """
    ### Inputs:
    - Latitude [deg]
    - Longitude [deg]
    ### Output:
    - Rotation matrix from ECEF to NED
    """

    # Unpack lat and long angles as radians
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    # Return DCM
    return np.array([
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat) ],
        [-np.sin(lon),             np.cos(lon),              0           ],
        [-np.cos(lat)*np.cos(lon), -np.cos(lat)*np.sin(lon), -np.sin(lat)]
    ])

def make_stations(lat_lons: list[tuple[float, float]],
                  alts: list[float]) -> list[dict]:
    """
    ### Inputs:
    - List of latitudes and longitudes
    - List of altitudes from the surface
    ### Output:
    - Dictionary of station data
        - [id, latitude [deg], longitude [deg], altitude [m], ECEF position, R_ECEF^NED]
    """

    # Initialize list of dictionary of stations
    stations = []

    # Loop through all stations and fill station data
    for i, (lat_deg, lon_deg) in enumerate(lat_lons):
        stations.append({
            'id':         i,
            'lat_deg':    lat_deg,
            'lon_deg':    lon_deg,
            'alt':        alts[i],
            'ecef':       latlon_to_ecef(lat_deg, lon_deg, alts[i]),
            'R_ned_ecef': ned_rotation(lat_deg, lon_deg)
        })
    return stations

def delta_ned(X_eci: np.ndarray,
              station: dict,
              t: float) -> np.ndarray:
    """
    ### Inputs:
    - State
    - Station dict
    - Time
    ### Output:
    - Difference vector from station to orbital station in NED frame
    """

    # Calculate difference vector
    pos_ecef = Rz(OMEGA_E * t) @ X_eci[0:3]
    diff_ecef = pos_ecef - station['ecef']
    return station['R_ned_ecef'] @ diff_ecef

def has_los(X_eci: np.ndarray,
            station: dict,
            t: float) -> bool:
    """
    ### Inputs:
    - State (ECI frame)
    - Station dict
    - Time
    ### Output:
    - Line of sight (True/False)
    """

    # Calculate delta vector
    d = delta_ned(X_eci, station, t)

    # Check for line of sight
    rho = np.linalg.norm(d)
    if rho < 1e-6:
        return False
    el = np.arcsin(-d[2] / rho)
    return el > EL_CUTOFF

def measure(X_eci: np.ndarray,
            station: dict,
            t: float,
            noise_std: np.ndarray | None = None) -> tuple[np.ndarray | None,
                                                          bool]:
    """
    ### Inputs:
    - State
    - Station dict
    - Time
    ### Optional inputs:
    - Noise array
        - [az_std_rad, el_std_rad, range_std_m]
    ### Output:
    - Measurement z
        - [azimuth, elevation, range]
    - Line of Sight (True/False)
    """

    # Check for line of sight
    if not has_los(X_eci, station, t):
        return None, False
    
    # Get delta vector and calculate observation
    d = delta_ned(X_eci, station, t)
    rho = np.linalg.norm(d)
    az = np.arctan2(d[1], d[0])
    el = np.arcsin(-d[2] / rho)
    z = np.array([az, el, rho])

    # Account for noise
    if noise_std is not None:
        z += np.random.randn(3) * noise_std
    return z, True

def measure_all_stations(X_eci: np.ndarray,
                         stations: list[dict],
                         t: float,
                         noise_std: np.ndarray | None = None) -> list:
    """
    ### Inputs:
    - State
    - List of station dicts
    - Time
    ### Optional inputs:
    - Noise array
        - [az_std_rad, el_std_rad, range_std_m]
    ### Output:
    - List of (station_id, z) tuples for all stations with LOS
    """

    # Initialize measurements
    measurements = []

    # Get station measurements
    for s in stations:
        z, los = measure(X_eci, s, t, noise_std)
        if los:
            measurements.append((s['id'], z))
    return measurements

def likelihood(z_meas: np.ndarray,
               z_pred: np.ndarray | None,
               noise_std: np.ndarray) -> float:
    """
    ### Inputs:
    - Observed measurement
    - Predicted measurement
    - Noise array
        - [az_std_rad, el_std_rad, range_std_m]
    ### Output:
    - Likelihood of measurement
    """

    # Calculate difference between observed and predicted measurement
    diff = z_meas - z_pred

    # Wrap azimuth difference to [-pi, pi]
    diff[0] = (diff[0] + np.pi) % (2 * np.pi) - np.pi

    # Calculate likelihood
    exponent = -0.5 * np.sum((diff / noise_std)**2)
    norm = np.prod(1.0 / (np.sqrt(2 * np.pi) * noise_std))
    return norm * np.exp(exponent)

def resample_particles(particles: np.ndarray,
                       weights: np.ndarray,
                       top_frac: float = 0.5):
    """
    ### Inputs:
    - Particle set (N x 6)
    - Particle weights (1 x N)
    ### Optional inputs:
    - Fraction of lowest-weight particles to replace (default: 0.5)
    ### Output:
    - Resampled particle set
    """

    # Build pool from top fraction by weight
    N = len(weights)
    d = particles.shape[1]
    sorted_idx = np.argsort(weights)
    n_top = int(N * (1.0 - top_frac))
    high_idx = sorted_idx[N - n_top:]
    high_weights = weights[high_idx]
    high_weights = high_weights / high_weights.sum()

    # Draw ALL N parents proportionally from high-weight pool
    parents = np.random.choice(high_idx, size=N, p=high_weights)

    # Jitter using per-dimension std of the surviving pool
    h = (4.0 / (N * (d + 2))) ** (1.0 / (d + 4))
    sigma = np.std(particles[high_idx], axis=0) * h + 1e-10
    return particles[parents] + np.random.randn(N, d) * sigma

def eff_sample_size(weights: np.ndarray) -> float:
    """
    ### Inputs:
    - Particle weights (1 x N)
    ### Output:
    - Effective sample size
    """

    # Return effective sample size
    return 1.0 / np.sum(weights**2)

class ParticleFilter:
    def __init__(self,
                 N: int,
                 x0_mean: np.ndarray,
                 x0_std: np.ndarray,
                 proc_std: np.ndarray,
                 meas_std: np.ndarray,
                 dt: float,
                 ess_thresh: float = 0.5):
        """
        ### Inputs:
        - Number of particles
        - Initial state mean
        - Initial state std
        - Process noise std
        - Measurement noise std
        - Timestep
        ### Optional Inputs:
        - Effective sample size threshold (resample when ess is below threshold, default: 0.5)
        """

        # Define self variables
        self.N = N
        self.proc_std = proc_std
        self.meas_std = meas_std
        self.dt = dt
        self.ess_thresh = ess_thresh

        # Initialize particles uniformly around x0_mean
        self.particles = x0_mean + np.random.randn(N, 6) * x0_std
        self.weights = np.ones(N) / N

    def predict(self):
        """
        ### Inputs:
        - None
        ### Output:
        - Updated particles
        """

        # Loop through all particles and update them
        for i in range(self.N):
            self.particles[i] = rk4_step(two_body_ode, 0, self.particles[i], self.dt)
            self.particles[i, :] += np.random.randn(6) * self.proc_std

    def update(self,
               measurements: list,
               stations: list[dict],
               t: float):
        """
        ### Inputs:
        - List of (station_id, z) tuples for all stations with LOS
        - List of station dicts
        - Time
        ### Output:
        - Updated particle weights
        """

        # Do nothing if no tracking stations have LOS
        if len(measurements) == 0:
            return
        
        # Update each particles weight
        for i in range(self.N):
            # Initialize new particle weight as 1.0
            w = 1.0

            # Loop through all measurements
            for sid, z_meas in measurements:
                # Get predicted measurement for particle and station pair
                z_pred, los = measure(self.particles[i], stations[sid], t)

                # Update weight
                if los:
                    w *= likelihood(z_meas, z_pred, self.meas_std)
                else:
                    w *= 1e-300
            self.weights[i] *= w

        # Normalize
        total = np.sum(self.weights)
        if total < 1e-300:
            self.weights = np.ones(self.N) / self.N
        else:
            self.weights /= total

    def resample(self):
        """
        ### Inputs:
        - None
        ### Output:
        - Resampled particles and particle weights
        """

        # Check if need to resample particles
        if eff_sample_size(self.weights) < self.ess_thresh * self.N:
            self.particles = resample_particles(self.particles, self.weights)
            self.weights = np.ones(self.N) / self.N

    def estimate(self):
        """
        ### Inputs:
        - None
        ### Output:
        - Mean state estimate
        """

        # Return mean state estimate
        return np.average(self.particles, weights=self.weights, axis=0)

def plot_orbit(times: np.ndarray,
               Xh: np.ndarray,
               PFh: np.ndarray,
               stations: list[dict]) -> None:
    """
    ### Inputs:
    - Time history
    - State history
    - Particle Filter state history
    - List of station dicts
    ### Output:
    - Ground station projection of Orbital Station
    - 3D orbit
    """

    # Color stuff
    if DARK_IMAGES:
        axcol = 'white'
    else:
        axcol = 'black'

    # Unpack state history to latitude and longitude vectors (degrees)
    raw_lon = np.degrees(np.arctan2(Xh[:,1], Xh[:,0])) - np.degrees(OMEGA_E * times)
    lons = ((raw_lon + 180) % 360) - 180
    lats = np.degrees(np.arctan2(Xh[:,2], np.sqrt(Xh[:,0]**2 + Xh[:,1]**2)))

    # Unpack PF state history to latitude and longitude vectors (degrees)
    raw_lon_PF = np.degrees(np.arctan2(PFh[:,1], PFh[:,0])) - np.degrees(OMEGA_E * times)
    lons_PF = ((raw_lon_PF + 180) % 360) - 180
    lats_PF = np.degrees(np.arctan2(PFh[:,2], np.sqrt(PFh[:,0]**2 + PFh[:,1]**2)))

    # Split vector to account for angle wrapping
    dlon = np.abs(np.diff(lons))
    dlon_PF = np.abs(np.diff(lons_PF))
    breaks = dlon > 90
    breaks_PF = dlon_PF > 90

    # Insert NaNs to break the line
    lons_plot = lons.copy()
    lats_plot = lats.copy()
    lons_plot[1:][breaks] = np.nan
    lats_plot[1:][breaks] = np.nan
    lons_plot_PF = lons_PF.copy()
    lats_plot_PF = lats_PF.copy()
    lons_plot_PF[1:][breaks_PF] = np.nan
    lats_plot_PF[1:][breaks_PF] = np.nan

    # Create figure
    fig, ax = plt.subplots(figsize=(19.2, 10.8), constrained_layout=True, facecolor=IMG_BG_COLOR)
    img = Image.open('./Images/earth-map.jpg')
    ax.imshow(img, extent=(-180.0, 180.0, -90.0, 90.0), aspect='auto')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    # Grid
    for lon in range(-180, 181, 30):
        ax.axvline(lon, color='white', lw=0.3, alpha=0.15)
    for lat in range(-90, 91, 30):
        ax.axhline(lat, color='white', lw=0.3, alpha=0.15)
    ax.axhline(0, color='white', lw=0.6, alpha=0.15)
    ax.axvline(0, color='white', lw=0.6, alpha=0.15)

    # Station markers
    for i, s in enumerate(stations):
        ax.plot(s['lon_deg'], s['lat_deg'], 'k^', markersize=15,
                label='Tracking Stations' if i == 0 else '_nolegend_')
        
    # Axes
    ax.set_title('Ground Station Tracking Network', color=axcol, fontsize=14)
    ax.set_xlabel('Longitude [deg]', color=axcol, fontsize=10)
    ax.set_ylabel('Latitude [deg]', color=axcol, fontsize=10)
    ax.tick_params(colors=axcol, labelsize=9)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.legend()
    fig.savefig('Images/Ground_Station_Network.png', dpi=150)
    plt.close()

    # Ground Track
    fig, ax = plt.subplots(figsize=(19.2, 10.8), constrained_layout=True, facecolor=IMG_BG_COLOR)
    img = Image.open('./Images/earth-map.jpg')
    ax.imshow(img, extent=(-180.0, 180.0, -90.0, 90.0), aspect='auto')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    # Grid
    for lon in range(-180, 181, 30):
        ax.axvline(lon, color='white', lw=0.3, alpha=0.15)
    for lat in range(-90, 91, 30):
        ax.axhline(lat, color='white', lw=0.3, alpha=0.15)
    ax.axhline(0, color='white', lw=0.6, alpha=0.15)
    ax.axvline(0, color='white', lw=0.6, alpha=0.15)

    # Plot ground track
    ax.plot(lons_plot, lats_plot, 'k', label='True Ground Track')
    ax.plot(lons_plot_PF, lats_plot_PF, 'k--', label='Estimated Ground Track')

    # Station markers
    for i, s in enumerate(stations):
        ax.plot(s['lon_deg'], s['lat_deg'], 'k^', markersize=15,
                label='Tracking Stations' if i == 0 else '_nolegend_')

    # Start and end marker
    ax.plot(lons_plot[0], lats_plot[0], 'go', label='True Start location')
    ax.plot(lons_plot[-1], lats_plot[-1], 'ro', label='True End location')
    ax.plot(lons_plot_PF[0], lats_plot_PF[0], 'gx', label='PF Start location')
    ax.plot(lons_plot_PF[-1], lats_plot_PF[-1], 'rx', label='PF End location')

    # Axes
    ax.set_title('Ground Projection', color=axcol, fontsize=14)
    ax.set_xlabel('Longitude [deg]', color=axcol, fontsize=10)
    ax.set_ylabel('Latitude [deg]', color=axcol, fontsize=10)
    ax.tick_params(colors=axcol, labelsize=9)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.legend()
    fig.savefig('Images/Ground_Projection.png', dpi=150)
    plt.close()

    # 3D Trajectory
    fig = plt.figure(constrained_layout=True, facecolor=IMG_BG_COLOR)
    ax = fig.add_subplot(111, projection='3d')

    # Earth sphere
    u_ = np.linspace(0, 2*np.pi, 80)
    v_ = np.linspace(0, np.pi, 60)
    xe = R_E * np.outer(np.cos(u_), np.sin(v_)) / 1000.0
    ye = R_E * np.outer(np.sin(u_), np.sin(v_)) / 1000.0
    ze = R_E * np.outer(np.ones(80), np.cos(v_)) / 1000.0
    ax.plot_surface(xe, ye, ze, color='#1a6fbd', alpha=0.5)

    # Orbit
    x, y, z = Xh[:, 0] / 1000.0, Xh[:, 1] / 1000.0, Xh[:, 2] / 1000.0
    if DARK_IMAGES:
        ax.plot(x, y, z, 'w--', lw=1.2)
    else:
        ax.plot(x, y, z, 'k--', lw=1.2)
    ax.scatter(x[0],  y[0],  z[0],  marker='o', c='g')
    ax.scatter(x[-1], y[-1], z[-1], marker='x', c='r')

    # Black panes and no grid
    ax.set_title('3D Orbit', color=axcol, fontsize=14)
    ax.tick_params(colors=axcol)
    ax.set_xlabel(r'$x_{ECI}$ [km]', color=axcol, fontsize=10)
    ax.set_ylabel(r'$y_{ECI}$ [km]', color=axcol, fontsize=10)
    ax.set_zlabel(r'$z_{ECI}$ [km]', color=axcol, fontsize=10)
    ax.grid(False)
    fig.savefig("Images/3D_Path.png", dpi=150)
    plt.close()

def nees(x_est: np.ndarray,
         x_true: np.ndarray,
         P_est: np.ndarray) -> float:
    """
    ### Inputs:
    - Estimated state
    - True state
    - Estimated covariance
    ### Output:
    - NEES scalar
    """

    # Find state estimate error
    e = x_est - x_true

    # Return NEES error
    P_reg = P_est + np.eye(6) * 1e-6
    return float(e @ np.linalg.solve(P_reg, e))

def particle_covariance(particles: np.ndarray,
                        weights: np.ndarray) -> np.ndarray:
    """
    ### Inputs:
    - Particle set (N x 6)
    - Particle weights (1 x N)
    ### Output:
    - Particle covariance matrix (6 x 6)
    """

    # Calculate particle mean
    mean = np.average(particles, weights=weights, axis=0)

    # Find particle error from mean
    diff = particles - mean

    # Return covariance
    return (weights[:, None] * diff).T @ diff

def plot_nees(est_hist: np.ndarray,
              true_hist: np.ndarray,
              pf_hist: list,
              times: np.ndarray) -> None:
    """
    ### Inputs:
    - Estimated state history
    - True state history
    - List of (particles, weights) snapshots at each timestep
    - Time history
    ### Output:
    - NEES history
    """

    # Color stuff
    if DARK_IMAGES:
        axcol = 'white'
    else:
        axcol = 'black'

    # Initialize nees histories
    T = len(est_hist)
    nees_hist = np.zeros(T)

    # Loop through each timestep
    for k in range(T):
        # Unpack particle and weights
        particles, weights = pf_hist[k]

        # Calculate covariance for particle and weight pairs
        P = particle_covariance(particles, weights)

        # Attempt to calculate NEES error for pair
        try:
            nees_hist[k] = nees(est_hist[k], true_hist[k], P)
        except np.linalg.LinAlgError:
            nees_hist[k] = np.nan

    # Plot
    fig, ax = plt.subplots(figsize=(19.2, 10.8), constrained_layout=True, facecolor=IMG_BG_COLOR)
    if DARK_IMAGES:
        ax.set_facecolor('#0e2a47')
    nees_clipped = np.clip(nees_hist, 0, np.nanpercentile(nees_hist, 99))
    if DARK_IMAGES:
        ax.semilogy(times, nees_clipped + 1, 'w')
    else:
        ax.semilogy(times, nees_clipped + 1)
    ax.set_title('NEES evolution over simulation time', color=axcol, fontsize=14)
    ax.set_xlabel('Simulation time [s]', color=axcol, fontsize=10)
    ax.set_ylabel('NEES error (log scale)', color=axcol, fontsize=10)
    ax.tick_params(colors=axcol)
    ax.grid(True, alpha=0.3)
    fig.savefig('Images/NEES_evolution.png', dpi=150)
    plt.close()

def plot_particle_cloud(true_hist: np.ndarray,
                        pf_hist: list,
                        times: np.ndarray) -> None:
    """
    ### Inputs:
    - True state history (T x 6)
    - List of (particles, weights) snapshots at each timestep
    - Time history
    ### Output:
    - Particle distribution history over time
    """

    # Color stuff
    if DARK_IMAGES:
        axcol = 'white'
    else:
        axcol = 'black'

    # Plot
    labels = [r'x [km]', r'y [km]', r'z [km]', r'$\dot{x}$ [km/s]', r'$\dot{y}$ [km/s]', r'$\dot{z}$ [km/s]']
    fig, ax = plt.subplots(3, 2,
                           figsize=(19.2, 10.8),
                           constrained_layout=True, 
                           facecolor=IMG_BG_COLOR,
                           sharex=True)
    ax = ax.flatten(order='F')
    for i, a in enumerate(ax):
        if DARK_IMAGES:
            a.set_facecolor('#0e2a47')
            a.plot(times, true_hist[:, i] / 1e3,
                    'w--',
                    linewidth=2,
                    label='True State')
        else:
            a.plot(times, true_hist[:, i] / 1e3,
                    'k--',
                    linewidth=2,
                    label='True State')
        for t, (particles, weights) in zip(times, pf_hist):
            states = np.array(particles) / 1e3
            weights = np.array(weights)
            w_scaled = weights / (weights.max() + 1e-30)
            w_alpha = 0.05 + 0.85 * w_scaled
            a.scatter(
                np.full(states.shape[0], t), states[:, i],
                c='r',
                alpha=w_alpha,
                marker='x',
                s=10,
                label='Particles' if t == 0 else '_nolegend_'
            )
        a.set_ylabel(labels[i], color=axcol, fontsize=10)
        a.tick_params(colors=axcol)
        if i == 3:
            if DARK_IMAGES:
                a.legend(facecolor='#0e2a47', labelcolor=axcol, loc='upper right')
            else:
                a.legend(loc='upper right')
        a.grid(True, alpha=0.3)
    for a in ax[4:]:
        a.set_xlabel('Simulation time [s]', color=axcol, fontsize=10)
    fig.suptitle('Particle State History', color=axcol, fontsize=14)
    fig.savefig('Images/Particle_History.png', dpi=150)
    plt.close()

def plot_rtn_error(true_hist: np.ndarray,
                   est_hist: np.ndarray,
                   times: np.ndarray) -> None:
    """
    ### Inputs:
    - True state history  (T x 6)
    - Estimated state history  (T x 6)
    - Time history
    ### Output:
    - Position error decomposed into Radial, Transverse (along-track), Normal components
    ###
    ### Transverse error is the along-track lag/lead — invisible in Cartesian plots.
    ### A converged filter should show R and N errors < 1 km, T error < ~10 km.
    """

    T = len(times)
    e_R = np.zeros(T)   # radial (altitude error)
    e_T = np.zeros(T)   # transverse / along-track
    e_N = np.zeros(T)   # normal / cross-track

    for k in range(T):
        r     = true_hist[k, :3]
        v     = true_hist[k, 3:]
        r_hat = r / np.linalg.norm(r)
        n_hat = np.cross(r, v); n_hat /= np.linalg.norm(n_hat)
        t_hat = np.cross(n_hat, r_hat)

        dp    = est_hist[k, :3] - r
        e_R[k] = np.dot(dp, r_hat) / 1e3    # km
        e_T[k] = np.dot(dp, t_hat) / 1e3    # km  ← this is the lag
        e_N[k] = np.dot(dp, n_hat) / 1e3    # km

    fig, axes = plt.subplots(3, 1, figsize=(19.2, 10.8),
                             constrained_layout=True,
                             facecolor="#181818", sharex=True)
    labels = ['Radial error [km]', 'Along-track error [km]', 'Cross-track error [km]']
    data   = [e_R, e_T, e_N]
    colors = ['#4fc3f7', '#ff6644', '#81c784']

    for i, a in enumerate(axes):
        a.set_facecolor('#0e2a47')
        a.axhline(0, color='white', lw=0.6, alpha=0.4)
        a.plot(times, data[i], color=colors[i], lw=1.2)
        a.set_ylabel(labels[i], color='white', fontsize=10)
        a.tick_params(colors='white')
        a.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Simulation time [s]', color='white', fontsize=10)
    fig.suptitle('RTN Frame Error (PF Estimate − True)', color='white', fontsize=14)
    fig.savefig('Images/RTN_Error.png', dpi=150)
    plt.close()

## Main
#############################################################
#############################################################

if __name__ == '__main__':
    # Orbit altitude
    alt = 2000e3
    r0 = R_E + alt

    # Tangential velocity
    v_circ = np.sqrt(MU / r0)

    # Inclenation
    inc = np.radians(51.6)

    # Initial state
    X0 = initial_state_from_latlon(0.0, -60.0, alt, inc)

    # Orbital period
    T_orb = 2 * np.pi * r0 / v_circ

    # Simulate
    dt = 10.0
    print(f'\nSimulating True orbit...')
    times, Xh = simulate(X0, 0.0, 3.5*T_orb, dt)
    print(f'                     ...Finished!')

    # Initialize stations
    stations = make_stations(TRACKING_STATION_ANGLES, TRACKING_STATION_ALTITUDES)
    
    # Particle filter
    pf = ParticleFilter(
        N=N_PARTICLES,
        x0_mean=X0,
        x0_std=X0_STD,
        proc_std=PROCESS_STD,
        meas_std=MEASUREMENT_STD,
        dt=dt,
        ess_thresh=0.5
    )
    
    # Run PF
    print(f'\nRunning Particle Filter...')
    PFh = np.zeros_like(Xh)
    pf_hist = []
    for k, t in enumerate(times):
        # Get noisy measurements from all stations with LOS
        z_all = measure_all_stations(Xh[k], stations, t, noise_std=MEASUREMENT_STD)

        # PF update
        pf.update(z_all, stations, t)

        # Estimate and save PF state history 
        PFh[k] = pf.estimate()
        pf_hist.append((pf.particles.copy(), pf.weights.copy()))

        # Resample particles
        pf.resample()

        # Predict to next timestep
        if k < len(times) - 1:
            pf.predict()
    print(f'                       ...Finished!')

    # Plot
    print(f'\nPlotting Ground Projection...')
    plot_orbit(times, Xh, PFh, stations)
    print(f'Plotting NEES Error...')
    plot_nees(PFh, Xh, pf_hist, times)
    print(f'Plotting Particle Cloud...\n')
    plot_particle_cloud(Xh, pf_hist, times)
