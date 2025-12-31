"""
Helios Parameters - Atmosphere parameter structures.

Ported from atmosphere/model.h by Eric Bruneton
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .constants import (
    EARTH_RADIUS,
    EARTH_TOP_RADIUS,
    SUN_ANGULAR_RADIUS,
    RAYLEIGH_SCALE_HEIGHT,
    RAYLEIGH_SCATTERING_COEFFICIENTS,
    MIE_SCALE_HEIGHT,
    MIE_SCATTERING_COEFFICIENT,
    MIE_EXTINCTION_COEFFICIENT,
    MIE_PHASE_FUNCTION_G,
    OZONE_CENTER_ALTITUDE,
    OZONE_WIDTH,
    OZONE_ABSORPTION_COEFFICIENTS,
    DEFAULT_GROUND_ALBEDO,
    MAX_SUN_ZENITH_ANGLE,
    LAMBDA_R, LAMBDA_G, LAMBDA_B,
)


@dataclass
class DensityProfileLayer:
    """
    An atmosphere layer whose density is defined as:
        exp_term * exp(exp_scale * h) + linear_term * h + constant_term
    clamped to [0, 1], where h is the altitude in meters.
    
    Attributes:
        width: Layer width in meters (ignored for top layer)
        exp_term: Exponential term coefficient (unitless)
        exp_scale: Exponential scale in m^-1
        linear_term: Linear term coefficient in m^-1
        constant_term: Constant term (unitless)
    """
    width: float = 0.0
    exp_term: float = 0.0
    exp_scale: float = 0.0
    linear_term: float = 0.0
    constant_term: float = 0.0
    
    def get_density(self, altitude: float) -> float:
        """Compute density at given altitude within this layer."""
        density = (
            self.exp_term * np.exp(self.exp_scale * altitude) +
            self.linear_term * altitude +
            self.constant_term
        )
        return np.clip(density, 0.0, 1.0)


@dataclass
class AtmosphereParameters:
    """
    Complete atmosphere parameters for the Bruneton model.
    
    All spatial values are in meters unless otherwise noted.
    Scattering/extinction coefficients are in m^-1.
    Wavelengths are in nanometers.
    """
    
    # Wavelengths for spectral data (nm)
    wavelengths: np.ndarray = field(default_factory=lambda: np.array([LAMBDA_R, LAMBDA_G, LAMBDA_B]))
    
    # Solar irradiance at top of atmosphere (W/m^2/nm) at each wavelength
    solar_irradiance: np.ndarray = field(default_factory=lambda: np.array([1.474, 1.8504, 1.91198]))
    
    # Sun angular radius (radians)
    sun_angular_radius: float = SUN_ANGULAR_RADIUS
    
    # Planet geometry
    bottom_radius: float = EARTH_RADIUS  # Planet surface radius (m)
    top_radius: float = EARTH_TOP_RADIUS  # Top of atmosphere radius (m)
    
    # Rayleigh scattering (air molecules)
    rayleigh_density: List[DensityProfileLayer] = field(default_factory=lambda: [
        DensityProfileLayer(
            width=0.0,
            exp_term=1.0,
            exp_scale=-1.0 / RAYLEIGH_SCALE_HEIGHT,
            linear_term=0.0,
            constant_term=0.0
        )
    ])
    rayleigh_scattering: np.ndarray = field(
        default_factory=lambda: RAYLEIGH_SCATTERING_COEFFICIENTS.copy()
    )
    
    # Mie scattering (aerosols)
    mie_density: List[DensityProfileLayer] = field(default_factory=lambda: [
        DensityProfileLayer(
            width=0.0,
            exp_term=1.0,
            exp_scale=-1.0 / MIE_SCALE_HEIGHT,
            linear_term=0.0,
            constant_term=0.0
        )
    ])
    mie_scattering: np.ndarray = field(
        default_factory=lambda: np.array([MIE_SCATTERING_COEFFICIENT] * 3)
    )
    mie_extinction: np.ndarray = field(
        default_factory=lambda: np.array([MIE_EXTINCTION_COEFFICIENT] * 3)
    )
    mie_phase_function_g: float = MIE_PHASE_FUNCTION_G
    
    # Absorption (ozone layer)
    absorption_density: List[DensityProfileLayer] = field(default_factory=lambda: [
        # Lower layer (below ozone peak)
        DensityProfileLayer(
            width=OZONE_CENTER_ALTITUDE,
            exp_term=0.0,
            exp_scale=0.0,
            linear_term=1.0 / OZONE_WIDTH,
            constant_term=-2.0 / 3.0
        ),
        # Upper layer (above ozone peak)
        DensityProfileLayer(
            width=0.0,  # Extends to top
            exp_term=0.0,
            exp_scale=0.0,
            linear_term=-1.0 / OZONE_WIDTH,
            constant_term=8.0 / 3.0
        )
    ])
    absorption_extinction: np.ndarray = field(
        default_factory=lambda: OZONE_ABSORPTION_COEFFICIENTS.copy()
    )
    
    # Ground albedo at each wavelength
    ground_albedo: np.ndarray = field(
        default_factory=lambda: np.array([DEFAULT_GROUND_ALBEDO] * 3)
    )
    
    # Maximum sun zenith angle for precomputation (radians)
    max_sun_zenith_angle: float = MAX_SUN_ZENITH_ANGLE
    
    # Length unit used in shaders (1.0 = meters, 1000.0 = kilometers)
    length_unit_in_meters: float = 1000.0  # Use km for better numerical precision
    
    # Precomputation options
    num_precomputed_wavelengths: int = 15  # For accurate luminance, use 15-50
    combine_scattering_textures: bool = True
    half_precision: bool = False  # Use float32 for VFX quality
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.wavelengths = np.asarray(self.wavelengths, dtype=np.float64)
        self.solar_irradiance = np.asarray(self.solar_irradiance, dtype=np.float64)
        self.rayleigh_scattering = np.asarray(self.rayleigh_scattering, dtype=np.float64)
        self.mie_scattering = np.asarray(self.mie_scattering, dtype=np.float64)
        self.mie_extinction = np.asarray(self.mie_extinction, dtype=np.float64)
        self.absorption_extinction = np.asarray(self.absorption_extinction, dtype=np.float64)
        self.ground_albedo = np.asarray(self.ground_albedo, dtype=np.float64)
    
    @classmethod
    def earth_default(cls, use_ozone: bool = True) -> 'AtmosphereParameters':
        """Create default Earth atmosphere parameters."""
        params = cls()
        if not use_ozone:
            params.absorption_extinction = np.zeros(3, dtype=np.float64)
        return params
    
    @classmethod
    def from_artistic_controls(
        cls,
        rayleigh_density_scale: float = 1.0,
        mie_density_scale: float = 1.0,
        mie_phase_g: float = MIE_PHASE_FUNCTION_G,
        rayleigh_height: float = RAYLEIGH_SCALE_HEIGHT,
        mie_height: float = MIE_SCALE_HEIGHT,
        ground_albedo: float = DEFAULT_GROUND_ALBEDO,
        use_ozone: bool = True,
        ozone_density: float = 1.0,
        mie_angstrom_beta: float = 0.04,
    ) -> 'AtmosphereParameters':
        """
        Create atmosphere parameters from artistic control values.
        
        Args:
            rayleigh_density_scale: Multiplier for air molecule density
            mie_density_scale: Multiplier for aerosol density
            mie_phase_g: Mie phase function asymmetry parameter
            rayleigh_height: Scale height for air molecules (meters)
            mie_height: Scale height for aerosols (meters)
            ground_albedo: Ground reflectivity (0-1)
            use_ozone: Include ozone absorption layer
            ozone_density: Multiplier for ozone absorption (affects sunset colors)
            mie_angstrom_beta: Aerosol optical thickness (higher = denser haze)
        """
        params = cls()
        
        # Adjust Rayleigh
        params.rayleigh_scattering = RAYLEIGH_SCATTERING_COEFFICIENTS * rayleigh_density_scale
        params.rayleigh_density = [
            DensityProfileLayer(
                width=0.0,
                exp_term=1.0,
                exp_scale=-1.0 / rayleigh_height,
                linear_term=0.0,
                constant_term=0.0
            )
        ]
        
        # Adjust Mie
        params.mie_scattering = np.array([MIE_SCATTERING_COEFFICIENT * mie_density_scale] * 3)
        params.mie_extinction = np.array([MIE_EXTINCTION_COEFFICIENT * mie_density_scale] * 3)
        params.mie_phase_function_g = np.clip(mie_phase_g, -0.999, 0.999)
        params.mie_density = [
            DensityProfileLayer(
                width=0.0,
                exp_term=1.0,
                exp_scale=-1.0 / mie_height,
                linear_term=0.0,
                constant_term=0.0
            )
        ]
        
        # Ground albedo
        params.ground_albedo = np.array([ground_albedo] * 3)
        
        # Ozone - scale absorption by ozone_density multiplier
        if not use_ozone or ozone_density <= 0:
            params.absorption_extinction = np.zeros(3, dtype=np.float64)
        else:
            params.absorption_extinction = OZONE_ABSORPTION_COEFFICIENTS * ozone_density
        
        # Mie Angstrom Beta - scale Mie scattering/extinction by beta ratio
        # Default beta is 0.04, so if user sets 0.08, double the Mie
        # The formula is: mie_coefficient = (beta / 0.04) * base_coefficient
        if mie_angstrom_beta > 0:
            beta_scale = mie_angstrom_beta / 0.04
            params.mie_scattering = params.mie_scattering * beta_scale
            params.mie_extinction = params.mie_extinction * beta_scale
        
        return params
    
    def get_atmosphere_height(self) -> float:
        """Return the atmosphere thickness in meters."""
        return self.top_radius - self.bottom_radius
    
    @classmethod
    def from_blender_settings(cls, settings) -> 'AtmosphereParameters':
        """
        Create atmosphere parameters from Blender Helios settings PropertyGroup.
        
        Args:
            settings: HeliosAtmosphereSettings PropertyGroup
        """
        return cls.from_artistic_controls(
            rayleigh_density_scale=settings.rayleigh_density,
            mie_density_scale=settings.mie_density,
            mie_phase_g=settings.mie_phase_g,
            rayleigh_height=settings.rayleigh_scale_height,
            mie_height=settings.mie_scale_height,
            ground_albedo=settings.ground_albedo,
            use_ozone=settings.use_ozone,
            ozone_density=getattr(settings, 'ozone_density', 1.0),
            mie_angstrom_beta=getattr(settings, 'mie_angstrom_beta', 0.04),
        )


@dataclass 
class RenderingParameters:
    """
    Runtime rendering parameters (not requiring LUT rebuild).
    Mirrors the structure from atmospheric_renderer.h
    """
    
    # Camera position in meters (relative to planet center at origin)
    camera_position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, EARTH_RADIUS + 1.0])
    )
    
    # Camera rotation in degrees
    camera_pitch: float = 0.0  # Rotation around X axis
    camera_yaw: float = 0.0    # Rotation around Y axis (azimuth)
    camera_roll: float = 0.0   # Rotation around Z axis
    
    # Field of view in degrees
    fov: float = 50.0
    
    # Sun direction (zenith/azimuth in radians)
    sun_zenith: float = np.pi / 3.0  # 60 degrees
    sun_azimuth: float = 0.0
    sun_intensity: float = 1.0
    
    # Exposure and color
    exposure: float = 10.0
    white_balance: bool = True
    
    # Render mode
    render_mode: int = 0  # 0=Perspective, 1=Latlong
    
    @property
    def sun_direction(self) -> np.ndarray:
        """Compute unit sun direction vector (pointing towards sun)."""
        # Z-up coordinate system
        cos_zenith = np.cos(self.sun_zenith)
        sin_zenith = np.sin(self.sun_zenith)
        cos_azimuth = np.cos(self.sun_azimuth)
        sin_azimuth = np.sin(self.sun_azimuth)
        
        return np.array([
            sin_zenith * sin_azimuth,
            sin_zenith * cos_azimuth,
            cos_zenith
        ])
