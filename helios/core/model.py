"""
Helios Atmosphere Model - Core atmospheric scattering model.

Ported from atmosphere/model.cc by Eric Bruneton

This module handles:
- LUT precomputation (transmittance, scattering, irradiance)
- Spectral to RGB conversion
- Model initialization and state management
"""

import sys
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from .constants import (
    TRANSMITTANCE_TEXTURE_WIDTH,
    TRANSMITTANCE_TEXTURE_HEIGHT,
    SCATTERING_TEXTURE_WIDTH,
    SCATTERING_TEXTURE_HEIGHT,
    SCATTERING_TEXTURE_DEPTH,
    SCATTERING_TEXTURE_R_SIZE,
    SCATTERING_TEXTURE_MU_SIZE,
    SCATTERING_TEXTURE_MU_S_SIZE,
    SCATTERING_TEXTURE_NU_SIZE,
    IRRADIANCE_TEXTURE_WIDTH,
    IRRADIANCE_TEXTURE_HEIGHT,
    LAMBDA_R, LAMBDA_G, LAMBDA_B,
    convert_spectrum_to_linear_srgb,
)
from .parameters import AtmosphereParameters, DensityProfileLayer


@dataclass
class PrecomputedTextures:
    """Container for precomputed LUT textures."""
    transmittance: np.ndarray  # Shape: (H, W, 3)
    scattering: np.ndarray     # Shape: (D, H, W, 4) - RGBA for combined, or RGB
    irradiance: np.ndarray     # Shape: (H, W, 3)
    single_mie_scattering: Optional[np.ndarray] = None  # Shape: (D, H, W, 3) if separate


class AtmosphereModel:
    """
    Main atmosphere model class.
    
    Handles precomputation of lookup tables and provides shader uniforms.
    """
    
    def __init__(self, params: Optional[AtmosphereParameters] = None):
        """
        Initialize the atmosphere model.
        
        Args:
            params: Atmosphere parameters. Uses Earth defaults if None.
        """
        self.params = params or AtmosphereParameters.earth_default()
        self.textures: Optional[PrecomputedTextures] = None
        self._is_initialized = False
        
        # Cache for solar irradiance in RGB
        self._solar_irradiance_rgb: Optional[np.ndarray] = None
    
    @property
    def is_initialized(self) -> bool:
        """Check if LUTs have been precomputed."""
        return self._is_initialized
    
    def init(self, num_scattering_orders: int = 4, progress_callback=None) -> None:
        """
        Precompute the atmosphere LUT textures.
        
        Args:
            num_scattering_orders: Number of scattering orders to compute (default 4)
            progress_callback: Optional callback(progress, message) for progress updates
        """
        if progress_callback:
            progress_callback(0.0, "Initializing atmosphere model...")
        
        # Compute solar irradiance in RGB
        self._compute_solar_irradiance_rgb()
        
        # Allocate textures
        transmittance = np.zeros(
            (TRANSMITTANCE_TEXTURE_HEIGHT, TRANSMITTANCE_TEXTURE_WIDTH, 3),
            dtype=np.float32
        )
        scattering = np.zeros(
            (SCATTERING_TEXTURE_DEPTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_WIDTH, 4),
            dtype=np.float32
        )
        irradiance = np.zeros(
            (IRRADIANCE_TEXTURE_HEIGHT, IRRADIANCE_TEXTURE_WIDTH, 3),
            dtype=np.float32
        )
        
        single_mie = None
        if not self.params.combine_scattering_textures:
            single_mie = np.zeros(
                (SCATTERING_TEXTURE_DEPTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_WIDTH, 3),
                dtype=np.float32
            )
        
        # Precompute transmittance
        if progress_callback:
            progress_callback(0.1, "Computing transmittance LUT...")
        self._precompute_transmittance(transmittance)
        
        # Precompute direct irradiance
        if progress_callback:
            progress_callback(0.2, "Computing direct irradiance...")
        delta_irradiance = np.zeros_like(irradiance)
        self._precompute_direct_irradiance(transmittance, delta_irradiance)
        
        # Precompute single scattering
        if progress_callback:
            progress_callback(0.3, "Computing single scattering...")
        delta_rayleigh = np.zeros(
            (SCATTERING_TEXTURE_DEPTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_WIDTH, 3),
            dtype=np.float32
        )
        delta_mie = np.zeros_like(delta_rayleigh)
        self._precompute_single_scattering(transmittance, delta_rayleigh, delta_mie, scattering, progress_callback)
        
        # Copy single Mie if not combining
        if single_mie is not None:
            single_mie[:] = delta_mie
        
        # Multiple scattering orders
        for order in range(2, num_scattering_orders + 1):
            if progress_callback:
                progress = 0.3 + 0.6 * (order - 2) / (num_scattering_orders - 1)
                progress_callback(progress, f"Computing scattering order {order}...")
            
            # Compute scattering density
            delta_scattering_density = np.zeros_like(delta_rayleigh)
            self._precompute_scattering_density(
                transmittance, delta_rayleigh, delta_mie,
                scattering, delta_irradiance,
                order, delta_scattering_density
            )
            
            # Compute indirect irradiance
            delta_indirect_irradiance = np.zeros_like(irradiance)
            self._precompute_indirect_irradiance(
                delta_rayleigh, delta_mie, scattering, order,
                delta_indirect_irradiance
            )
            irradiance += delta_indirect_irradiance
            
            # Compute multiple scattering
            delta_multiple = np.zeros_like(delta_rayleigh)
            self._precompute_multiple_scattering(
                transmittance, delta_scattering_density, delta_multiple
            )
            
            # Accumulate
            scattering[..., :3] += delta_multiple
            delta_rayleigh[:] = delta_multiple
        
        # Add direct irradiance
        irradiance += delta_irradiance
        
        if progress_callback:
            progress_callback(1.0, "Precomputation complete.")
        
        self.textures = PrecomputedTextures(
            transmittance=transmittance,
            scattering=scattering,
            irradiance=irradiance,
            single_mie_scattering=single_mie
        )
        self._is_initialized = True
    
    def _compute_solar_irradiance_rgb(self) -> None:
        """Convert solar irradiance spectrum to RGB."""
        self._solar_irradiance_rgb = convert_spectrum_to_linear_srgb(
            self.params.wavelengths,
            self.params.solar_irradiance
        )
    
    def _get_texture_coord_from_unit_range(self, x: float, texture_size: int) -> float:
        """Convert unit range [0,1] to texture coordinate."""
        return 0.5 / texture_size + x * (1.0 - 1.0 / texture_size)
    
    def _get_unit_range_from_texture_coord(self, u: float, texture_size: int) -> float:
        """Convert texture coordinate to unit range [0,1]."""
        return (u - 0.5 / texture_size) / (1.0 - 1.0 / texture_size)
    
    def _distance_to_top_atmosphere_boundary(self, r, mu):
        """
        Compute distance from point at radius r looking in direction with cosine mu
        to the top atmosphere boundary. Supports scalar and array inputs.
        """
        top_radius = self.params.top_radius
        discriminant = r * r * (mu * mu - 1.0) + top_radius * top_radius
        return np.maximum(0.0, -r * mu + np.sqrt(np.maximum(0.0, discriminant)))
    
    def _distance_to_bottom_atmosphere_boundary(self, r, mu):
        """
        Compute distance from point at radius r looking in direction with cosine mu
        to the bottom atmosphere boundary (ground). Supports scalar and array inputs.
        """
        bottom_radius = self.params.bottom_radius
        discriminant = r * r * (mu * mu - 1.0) + bottom_radius * bottom_radius
        return np.maximum(0.0, -r * mu - np.sqrt(np.maximum(0.0, discriminant)))
    
    def _ray_intersects_ground(self, r: float, mu: float) -> bool:
        """Check if a ray from radius r with direction cosine mu hits the ground."""
        bottom_radius = self.params.bottom_radius
        return mu < 0.0 and r * r * (mu * mu - 1.0) + bottom_radius * bottom_radius >= 0.0
    
    def _get_layer_density(self, layer: DensityProfileLayer, altitude: float) -> float:
        """Get density at altitude for a single layer."""
        density = (
            layer.exp_term * np.exp(layer.exp_scale * altitude) +
            layer.linear_term * altitude +
            layer.constant_term
        )
        return np.clip(density, 0.0, 1.0)
    
    def _get_profile_density(self, layers: list, altitude) -> np.ndarray:
        """Get density at altitude for a density profile (multiple layers).
        Supports both scalar and array inputs."""
        if not layers:
            return np.zeros_like(altitude) if hasattr(altitude, '__len__') else 0.0
        
        altitude = np.asarray(altitude)
        scalar_input = altitude.ndim == 0
        if scalar_input:
            altitude = altitude.reshape(1)
        
        result = np.zeros_like(altitude, dtype=np.float64)
        
        # Process each layer
        current_altitude = 0.0
        for i, layer in enumerate(layers):
            if i < len(layers) - 1:
                # Not the top layer - check if altitude is in this layer
                layer_mask = (altitude >= current_altitude) & (altitude < current_altitude + layer.width)
                layer_alt = altitude - current_altitude
            else:
                # Top layer - all remaining altitudes
                layer_mask = altitude >= current_altitude
                layer_alt = altitude - current_altitude
            
            # Compute density for this layer
            density = (
                layer.exp_term * np.exp(layer.exp_scale * layer_alt) +
                layer.linear_term * layer_alt +
                layer.constant_term
            )
            density = np.clip(density, 0.0, 1.0)
            
            result = np.where(layer_mask, density, result)
            current_altitude += layer.width
        
        if scalar_input:
            return result[0]
        return result
    
    def _compute_optical_length_to_top_atmosphere_boundary(
        self, r: float, mu: float, layers: list, scattering_coeff: np.ndarray
    ) -> np.ndarray:
        """
        Compute optical length (integral of extinction) from point at radius r
        along direction mu to the top of the atmosphere.
        """
        # Number of integration samples
        SAMPLE_COUNT = 500
        
        dx = self._distance_to_top_atmosphere_boundary(r, mu) / SAMPLE_COUNT
        result = np.zeros(3, dtype=np.float64)
        
        for i in range(SAMPLE_COUNT):
            d_i = (i + 0.5) * dx
            # Distance from planet center
            r_i = np.sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r)
            # Altitude above ground
            altitude = r_i - self.params.bottom_radius
            # Density at this point
            density = self._get_profile_density(layers, altitude)
            result += density * scattering_coeff * dx
        
        return result
    
    def _compute_transmittance_to_top_atmosphere_boundary(
        self, r: float, mu: float
    ) -> np.ndarray:
        """
        Compute transmittance from point at radius r along direction mu
        to the top of atmosphere.
        """
        # Rayleigh
        rayleigh_optical = self._compute_optical_length_to_top_atmosphere_boundary(
            r, mu, self.params.rayleigh_density, self.params.rayleigh_scattering
        )
        
        # Mie (extinction, not scattering)
        mie_optical = self._compute_optical_length_to_top_atmosphere_boundary(
            r, mu, self.params.mie_density, self.params.mie_extinction
        )
        
        # Absorption (ozone)
        absorption_optical = self._compute_optical_length_to_top_atmosphere_boundary(
            r, mu, self.params.absorption_density, self.params.absorption_extinction
        )
        
        total_optical = rayleigh_optical + mie_optical + absorption_optical
        return np.exp(-total_optical)
    
    def _get_r_mu_from_transmittance_texture_uv(
        self, u: float, v: float
    ) -> Tuple[float, float]:
        """Convert transmittance texture UV to (r, mu) parameters."""
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        H = np.sqrt(top * top - bottom * bottom)
        
        # Map v to rho (distance from bottom to top at horizon)
        rho = H * self._get_unit_range_from_texture_coord(v, TRANSMITTANCE_TEXTURE_HEIGHT)
        r = np.sqrt(rho * rho + bottom * bottom)
        
        # Map u to d (distance to atmosphere boundary)
        d_min = top - r
        d_max = rho + H
        d = d_min + (d_max - d_min) * self._get_unit_range_from_texture_coord(
            u, TRANSMITTANCE_TEXTURE_WIDTH
        )
        
        # Compute mu from r and d
        mu = (H * H - rho * rho - d * d) / (2.0 * r * d) if d > 0 else 1.0
        mu = np.clip(mu, -1.0, 1.0)
        
        return r, mu
    
    def _precompute_transmittance(self, transmittance: np.ndarray) -> None:
        """Precompute transmittance lookup table (fully vectorized)."""
        import sys
        
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        H = np.sqrt(top * top - bottom * bottom)
        
        # Create coordinate grids
        j_coords = np.arange(TRANSMITTANCE_TEXTURE_HEIGHT)
        i_coords = np.arange(TRANSMITTANCE_TEXTURE_WIDTH)
        jj, ii = np.meshgrid(j_coords, i_coords, indexing='ij')
        
        u = (ii + 0.5) / TRANSMITTANCE_TEXTURE_WIDTH
        v = (jj + 0.5) / TRANSMITTANCE_TEXTURE_HEIGHT
        
        # Convert UV to r, mu (vectorized)
        rho = H * self._get_unit_range_from_texture_coord(v, TRANSMITTANCE_TEXTURE_HEIGHT)
        r = np.sqrt(rho * rho + bottom * bottom)
        
        d_min = top - r
        d_max = rho + H
        d = d_min + (d_max - d_min) * self._get_unit_range_from_texture_coord(u, TRANSMITTANCE_TEXTURE_WIDTH)
        
        mu = np.where(d > 0, (H * H - rho * rho - d * d) / (2.0 * r * d), 1.0)
        mu = np.clip(mu, -1.0, 1.0)
        
        # Compute distance to top atmosphere boundary (vectorized)
        discriminant = r * r * (mu * mu - 1.0) + top * top
        dist_to_top = np.maximum(0.0, -r * mu + np.sqrt(np.maximum(0.0, discriminant)))
        
        # Numerical integration (vectorized)
        SAMPLE_COUNT = 500
        optical_depth = np.zeros((TRANSMITTANCE_TEXTURE_HEIGHT, TRANSMITTANCE_TEXTURE_WIDTH, 3), dtype=np.float64)
        
        dx = dist_to_top / SAMPLE_COUNT
        
        print("  Computing transmittance integration...")
        sys.stdout.flush()
        
        for s in range(SAMPLE_COUNT):
            d_i = (s + 0.5) * dx
            
            # Radius at sample point
            r_i = np.sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r)
            altitude = r_i - bottom
            
            # Get densities (vectorized)
            rayleigh_density = self._get_profile_density(self.params.rayleigh_density, altitude)
            mie_density = self._get_profile_density(self.params.mie_density, altitude)
            absorption_density = self._get_profile_density(self.params.absorption_density, altitude)
            
            # Accumulate optical depth
            optical_depth += (
                rayleigh_density[..., np.newaxis] * self.params.rayleigh_scattering[:3] +
                mie_density[..., np.newaxis] * self.params.mie_extinction[:3] +
                absorption_density[..., np.newaxis] * self.params.absorption_extinction[:3]
            ) * dx[..., np.newaxis]
        
        transmittance[:] = np.exp(-optical_depth).astype(np.float32)
        print(f"  Transmittance done, max: {transmittance.max():.4f}, min: {transmittance.min():.6f}")
        sys.stdout.flush()
    
    def _precompute_direct_irradiance(
        self, transmittance: np.ndarray, delta_irradiance: np.ndarray
    ) -> None:
        """Precompute direct solar irradiance at ground level (vectorized)."""
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        
        # Create coordinate grids
        j_coords = np.arange(IRRADIANCE_TEXTURE_HEIGHT)
        i_coords = np.arange(IRRADIANCE_TEXTURE_WIDTH)
        jj, ii = np.meshgrid(j_coords, i_coords, indexing='ij')
        
        v = (jj + 0.5) / IRRADIANCE_TEXTURE_HEIGHT
        u = (ii + 0.5) / IRRADIANCE_TEXTURE_WIDTH
        
        r = bottom + (top - bottom) * v
        mu_s = -0.2 + u * 1.4  # Sun zenith cosine range
        
        # Sample transmittance (vectorized)
        trans = self._sample_transmittance(transmittance, r, mu_s)
        
        # Direct irradiance = solar_irradiance * transmittance * cos(zenith)
        # Only when sun is above horizon (mu_s > 0)
        irradiance = self.params.solar_irradiance[:3] * trans * mu_s[..., np.newaxis]
        delta_irradiance[:] = np.where(mu_s[..., np.newaxis] > 0, irradiance, 0.0).astype(np.float32)
    
    def _sample_transmittance(
        self, transmittance: np.ndarray, r, mu
    ) -> np.ndarray:
        """Sample the transmittance texture with bilinear interpolation.
        Supports both scalar and array inputs for r and mu."""
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        H = np.sqrt(top * top - bottom * bottom)
        
        # Handle both scalar and array inputs - broadcast to common shape
        r = np.asarray(r)
        mu = np.asarray(mu)
        scalar_input = r.ndim == 0 and mu.ndim == 0
        
        # Broadcast r and mu to common shape
        r, mu = np.broadcast_arrays(r, mu)
        
        rho = np.sqrt(np.maximum(0.0, r * r - bottom * bottom))
        
        # Compute texture coordinates
        v = self._get_texture_coord_from_unit_range(rho / H, TRANSMITTANCE_TEXTURE_HEIGHT)
        
        # Distance to top atmosphere boundary (vectorized)
        discriminant = r * r * (mu * mu - 1.0) + top * top
        d = np.maximum(0.0, -r * mu + np.sqrt(np.maximum(0.0, discriminant)))
        
        d_min = top - r
        d_max = rho + H
        
        u_range = np.where(d_max > d_min, (d - d_min) / (d_max - d_min), 0.0)
        u = self._get_texture_coord_from_unit_range(u_range, TRANSMITTANCE_TEXTURE_WIDTH)
        
        # Bilinear sample (vectorized)
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
        
        ui = u * (TRANSMITTANCE_TEXTURE_WIDTH - 1)
        vi = v * (TRANSMITTANCE_TEXTURE_HEIGHT - 1)
        
        i0 = np.floor(ui).astype(int)
        j0 = np.floor(vi).astype(int)
        i1 = np.minimum(i0 + 1, TRANSMITTANCE_TEXTURE_WIDTH - 1)
        j1 = np.minimum(j0 + 1, TRANSMITTANCE_TEXTURE_HEIGHT - 1)
        
        fu = ui - i0
        fv = vi - j0
        
        # Sample texture at all four corners
        result = (
            transmittance[j0, i0] * ((1 - fu) * (1 - fv))[..., np.newaxis] +
            transmittance[j0, i1] * (fu * (1 - fv))[..., np.newaxis] +
            transmittance[j1, i0] * ((1 - fu) * fv)[..., np.newaxis] +
            transmittance[j1, i1] * (fu * fv)[..., np.newaxis]
        )
        
        if scalar_input:
            return result[0]
        return result
    
    def _get_r_mu_mu_s_nu_from_scattering_texture_uvwz(
        self, u: float, v: float, w: float, z: float
    ) -> Tuple[float, float, float, float, bool]:
        """
        Convert scattering texture 4D coordinates to physical parameters.
        Returns (r, mu, mu_s, nu, ray_r_mu_intersects_ground)
        """
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        H = np.sqrt(top * top - bottom * bottom)
        
        # Unpack z into r
        rho = H * self._get_unit_range_from_texture_coord(z, SCATTERING_TEXTURE_R_SIZE)
        r = np.sqrt(rho * rho + bottom * bottom)
        
        # Unpack w into mu (with ground intersection handling)
        if w < 0.5:
            # Ray intersects ground
            d_min = r - bottom
            d_max = rho
            d = d_min + (d_max - d_min) * self._get_unit_range_from_texture_coord(
                1.0 - 2.0 * w, SCATTERING_TEXTURE_MU_SIZE // 2)
            mu = (d == 0.0) and -1.0 or -(rho * rho + d * d) / (2.0 * r * d)
            ray_intersects_ground = True
        else:
            # Ray doesn't intersect ground  
            d_min = top - r
            d_max = rho + H
            d = d_min + (d_max - d_min) * self._get_unit_range_from_texture_coord(
                2.0 * w - 1.0, SCATTERING_TEXTURE_MU_SIZE // 2)
            mu = (d == 0.0) and 1.0 or (H * H - rho * rho - d * d) / (2.0 * r * d)
            ray_intersects_ground = False
        
        mu = np.clip(mu, -1.0, 1.0)
        
        # Unpack v into mu_s
        x_mu_s = self._get_unit_range_from_texture_coord(v, SCATTERING_TEXTURE_MU_S_SIZE)
        d_min = top - bottom
        d_max = H
        mu_s_min = -0.2  # cos(102 degrees)
        D = self._distance_to_top_atmosphere_boundary(bottom, mu_s_min)
        A = (D - d_min) / (d_max - d_min)
        a = (A - x_mu_s * A) / (1.0 + x_mu_s * A)
        d = d_min + min(a, A) * (d_max - d_min)
        mu_s = (d == 0.0) and 1.0 or (H * H - d * d) / (2.0 * bottom * d)
        mu_s = np.clip(mu_s, -1.0, 1.0)
        
        # Unpack u into nu
        nu = self._get_unit_range_from_texture_coord(u, SCATTERING_TEXTURE_NU_SIZE)
        nu = nu * 2.0 - 1.0  # Map [0,1] to [-1,1]
        
        return r, mu, mu_s, nu, ray_intersects_ground
    
    def _compute_single_scattering_integrand(
        self, transmittance: np.ndarray,
        r: float, mu: float, mu_s: float, nu: float, d: float,
        ray_r_mu_intersects_ground: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute single scattering integrand at distance d along the view ray.
        Returns (rayleigh, mie) scattering contributions.
        """
        r_d = np.sqrt(d * d + 2.0 * r * mu * d + r * r)
        r_d = np.clip(r_d, self.params.bottom_radius, self.params.top_radius)
        mu_s_d = (r * mu_s + d * nu) / r_d
        mu_s_d = np.clip(mu_s_d, -1.0, 1.0)
        
        # Transmittance from camera to point
        trans_to_point = self._get_transmittance(transmittance, r, mu, d, ray_r_mu_intersects_ground)
        
        # Transmittance from point to sun
        trans_to_sun = self._sample_transmittance(transmittance, r_d, mu_s_d)
        
        total_trans = trans_to_point * trans_to_sun
        
        # Get density at this altitude
        altitude = r_d - self.params.bottom_radius
        rayleigh_density = self._get_profile_density(self.params.rayleigh_density, altitude)
        mie_density = self._get_profile_density(self.params.mie_density, altitude)
        
        rayleigh = total_trans * self.params.rayleigh_scattering[:3] * rayleigh_density
        mie = total_trans * self.params.mie_scattering[:3] * mie_density
        
        return rayleigh, mie
    
    def _get_transmittance(
        self, transmittance: np.ndarray,
        r, mu, d, ray_intersects_ground
    ) -> np.ndarray:
        """Get transmittance between two points along a ray.
        Supports both scalar and array inputs."""
        # Broadcast all inputs to common shape
        r = np.asarray(r)
        mu = np.asarray(mu)
        d = np.asarray(d)
        ray_intersects_ground = np.asarray(ray_intersects_ground)
        
        # Broadcast to common shape
        r, mu, d, ray_intersects_ground = np.broadcast_arrays(r, mu, d, ray_intersects_ground)
        
        r_d = np.sqrt(d * d + 2.0 * r * mu * d + r * r)
        mu_d = np.where(r_d > 0, (r * mu + d) / r_d, mu)
        
        # For ground-intersecting rays
        trans_ground = (
            self._sample_transmittance(transmittance, r_d, -mu_d) /
            np.maximum(1e-10, self._sample_transmittance(transmittance, r, -mu))
        )
        
        # For sky rays
        trans_sky = (
            self._sample_transmittance(transmittance, r, mu) /
            np.maximum(1e-10, self._sample_transmittance(transmittance, r_d, mu_d))
        )
        
        # Select based on ray type
        return np.where(ray_intersects_ground[..., np.newaxis], trans_ground, trans_sky)
    
    def _compute_single_scattering(
        self, transmittance: np.ndarray,
        r: float, mu: float, mu_s: float, nu: float,
        ray_r_mu_intersects_ground: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute single scattering for given parameters by numerical integration.
        Returns (rayleigh, mie) scattering values.
        """
        SAMPLE_COUNT = 50
        
        # Distance to atmosphere boundary (or ground)
        if ray_r_mu_intersects_ground:
            d_max = self._distance_to_bottom_atmosphere_boundary(r, mu)
        else:
            d_max = self._distance_to_top_atmosphere_boundary(r, mu)
        
        dx = d_max / SAMPLE_COUNT
        
        rayleigh_sum = np.zeros(3, dtype=np.float64)
        mie_sum = np.zeros(3, dtype=np.float64)
        
        for i in range(SAMPLE_COUNT):
            d_i = (i + 0.5) * dx
            rayleigh, mie = self._compute_single_scattering_integrand(
                transmittance, r, mu, mu_s, nu, d_i, ray_r_mu_intersects_ground)
            rayleigh_sum += rayleigh * dx
            mie_sum += mie * dx
        
        return rayleigh_sum, mie_sum
    
    def _precompute_single_scattering(
        self, transmittance: np.ndarray,
        delta_rayleigh: np.ndarray, delta_mie: np.ndarray,
        scattering: np.ndarray,
        progress_callback=None
    ) -> None:
        """Precompute single scattering lookup table using vectorized numpy."""
        import sys
        
        total_slices = SCATTERING_TEXTURE_DEPTH
        SAMPLE_COUNT = 50
        
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        H = np.sqrt(top * top - bottom * bottom)
        mu_s_min = -0.2
        
        # Iterate through depth slices (r dimension) - vectorize the inner dimensions
        for k in range(SCATTERING_TEXTURE_DEPTH):
            if progress_callback:
                progress_callback(0.3 + 0.5 * k / total_slices, f"Computing scattering slice {k+1}/{total_slices}")
            
            z = (k + 0.5) / SCATTERING_TEXTURE_DEPTH
            
            # Compute r for this slice
            rho = H * self._get_unit_range_from_texture_coord(z, SCATTERING_TEXTURE_R_SIZE)
            r = np.sqrt(rho * rho + bottom * bottom)
            
            # Create coordinate grids for this slice
            j_coords = np.arange(SCATTERING_TEXTURE_HEIGHT)
            i_coords = np.arange(SCATTERING_TEXTURE_WIDTH)
            jj, ii = np.meshgrid(j_coords, i_coords, indexing='ij')
            
            w = (jj + 0.5) / SCATTERING_TEXTURE_HEIGHT
            
            nu_idx = ii // SCATTERING_TEXTURE_MU_S_SIZE
            mu_s_idx = ii % SCATTERING_TEXTURE_MU_S_SIZE
            u = (nu_idx + 0.5) / SCATTERING_TEXTURE_NU_SIZE
            v = (mu_s_idx + 0.5) / SCATTERING_TEXTURE_MU_S_SIZE
            
            # Compute mu from w (vectorized)
            ray_intersects_ground = w < 0.5
            
            # For rays that intersect ground
            d_min_ground = r - bottom
            d_max_ground = rho
            w_ground = self._get_unit_range_from_texture_coord(1.0 - 2.0 * w, SCATTERING_TEXTURE_MU_SIZE // 2)
            d_ground = d_min_ground + (d_max_ground - d_min_ground) * w_ground
            mu_ground = np.where(d_ground == 0.0, -1.0, -(rho * rho + d_ground * d_ground) / (2.0 * r * d_ground))
            
            # For rays that don't intersect ground
            d_min_sky = top - r
            d_max_sky = rho + H
            w_sky = self._get_unit_range_from_texture_coord(2.0 * w - 1.0, SCATTERING_TEXTURE_MU_SIZE // 2)
            d_sky = d_min_sky + (d_max_sky - d_min_sky) * w_sky
            mu_sky = np.where(d_sky == 0.0, 1.0, (H * H - rho * rho - d_sky * d_sky) / (2.0 * r * d_sky))
            
            mu = np.where(ray_intersects_ground, mu_ground, mu_sky)
            mu = np.clip(mu, -1.0, 1.0)
            
            # Compute mu_s from v (vectorized)
            x_mu_s = self._get_unit_range_from_texture_coord(v, SCATTERING_TEXTURE_MU_S_SIZE)
            d_min_s = top - bottom
            d_max_s = H
            D = self._distance_to_top_atmosphere_boundary(bottom, mu_s_min)
            A = (D - d_min_s) / (d_max_s - d_min_s)
            a_val = (A - x_mu_s * A) / (1.0 + x_mu_s * A)
            d_s = d_min_s + np.minimum(a_val, A) * (d_max_s - d_min_s)
            mu_s = np.where(d_s == 0.0, 1.0, (H * H - d_s * d_s) / (2.0 * bottom * d_s))
            mu_s = np.clip(mu_s, -1.0, 1.0)
            
            # Compute nu from u (vectorized)
            nu = self._get_unit_range_from_texture_coord(u, SCATTERING_TEXTURE_NU_SIZE)
            nu = nu * 2.0 - 1.0
            
            # Compute distance to boundary
            d_max = np.where(
                ray_intersects_ground,
                self._distance_to_bottom_atmosphere_boundary(r, mu),
                self._distance_to_top_atmosphere_boundary(r, mu)
            )
            
            # Numerical integration (vectorized over spatial dimensions)
            rayleigh_sum = np.zeros((SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_WIDTH, 3), dtype=np.float64)
            mie_sum = np.zeros_like(rayleigh_sum)
            
            dx = d_max / SAMPLE_COUNT
            
            for s in range(SAMPLE_COUNT):
                d_i = (s + 0.5) * dx
                
                # Distance from planet center at sample point
                r_d = np.sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r)
                r_d = np.clip(r_d, bottom, top)
                
                # Sun angle at sample point
                mu_s_d = (r * mu_s + d_i * nu) / r_d
                mu_s_d = np.clip(mu_s_d, -1.0, 1.0)
                
                # Transmittance from camera to point (fully vectorized)
                t_to_point = self._get_transmittance(
                    transmittance, r, mu, d_i, ray_intersects_ground)
                
                # Transmittance from point to sun (fully vectorized)
                t_to_sun = self._sample_transmittance(transmittance, r_d, mu_s_d)
                
                trans_total = t_to_point * t_to_sun
                
                # Density at sample altitude (fully vectorized)
                altitude = r_d - bottom
                rayleigh_density = self._get_profile_density(self.params.rayleigh_density, altitude)
                mie_density = self._get_profile_density(self.params.mie_density, altitude)
                
                # Accumulate scattering
                rayleigh_contrib = trans_total * self.params.rayleigh_scattering[:3] * rayleigh_density[..., np.newaxis]
                mie_contrib = trans_total * self.params.mie_scattering[:3] * mie_density[..., np.newaxis]
                
                rayleigh_sum += rayleigh_contrib * dx[..., np.newaxis]
                mie_sum += mie_contrib * dx[..., np.newaxis]
            
            delta_rayleigh[k] = rayleigh_sum.astype(np.float32)
            delta_mie[k] = mie_sum.astype(np.float32)
            
            print(f"  Slice {k+1}/{total_slices} done, max rayleigh: {rayleigh_sum.max():.6f}")
            sys.stdout.flush()
        
        # Combine into scattering texture
        scattering[..., :3] = delta_rayleigh + delta_mie
        if self.params.combine_scattering_textures:
            scattering[..., 3] = delta_mie[..., 0]
    
    def _sample_scattering(
        self, scattering: np.ndarray,
        r: float, mu: float, mu_s: float, nu: float,
        ray_r_mu_intersects_ground: bool
    ) -> np.ndarray:
        """Sample the 4D scattering texture with trilinear interpolation."""
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        H = np.sqrt(top * top - bottom * bottom)
        
        # Clamp inputs
        r = np.clip(r, bottom, top)
        mu = np.clip(mu, -1.0, 1.0)
        mu_s = np.clip(mu_s, -1.0, 1.0)
        nu = np.clip(nu, -1.0, 1.0)
        
        rho = np.sqrt(np.maximum(0.0, r * r - bottom * bottom))
        
        # Compute texture coordinates
        # z coordinate (r)
        z = self._get_texture_coord_from_unit_range(rho / H, SCATTERING_TEXTURE_R_SIZE)
        
        # w coordinate (mu) - split for ground intersection
        r_mu = r * mu
        discriminant = r_mu * r_mu - r * r + bottom * bottom
        
        if ray_r_mu_intersects_ground:
            d = -r_mu - np.sqrt(np.maximum(0.0, discriminant))
            d_min = r - bottom
            d_max = rho
            w_range = (d_max - d_min) > 0 and (d - d_min) / (d_max - d_min) or 0.0
            w = 0.5 - 0.5 * self._get_texture_coord_from_unit_range(w_range, SCATTERING_TEXTURE_MU_SIZE // 2)
        else:
            d = -r_mu + np.sqrt(np.maximum(0.0, r_mu * r_mu - r * r + top * top))
            d_min = top - r
            d_max = rho + H
            w_range = (d_max - d_min) > 0 and (d - d_min) / (d_max - d_min) or 0.0
            w = 0.5 + 0.5 * self._get_texture_coord_from_unit_range(w_range, SCATTERING_TEXTURE_MU_SIZE // 2)
        
        # v coordinate (mu_s)
        d_min_s = top - bottom
        d_max_s = H
        mu_s_min = -0.2
        D = self._distance_to_top_atmosphere_boundary(bottom, mu_s_min)
        A = (D - d_min_s) / (d_max_s - d_min_s)
        d_s = self._distance_to_top_atmosphere_boundary(bottom, mu_s)
        a = (d_s - d_min_s) / (d_max_s - d_min_s)
        x_mu_s = a / (1.0 + a * A - a) if (1.0 + a * A - a) != 0 else 0.0
        v = self._get_texture_coord_from_unit_range(x_mu_s, SCATTERING_TEXTURE_MU_S_SIZE)
        
        # u coordinate (nu)
        u = self._get_texture_coord_from_unit_range((nu + 1.0) / 2.0, SCATTERING_TEXTURE_NU_SIZE)
        
        # Sample with trilinear interpolation
        # scattering shape: (depth, height, width, 3) where width = nu_size * mu_s_size
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
        w = np.clip(w, 0.0, 1.0)
        z = np.clip(z, 0.0, 1.0)
        
        # Combined u-v into texture width
        tex_u = u * SCATTERING_TEXTURE_NU_SIZE / SCATTERING_TEXTURE_WIDTH + \
                v * (SCATTERING_TEXTURE_WIDTH - SCATTERING_TEXTURE_NU_SIZE) / SCATTERING_TEXTURE_WIDTH
        
        xi = tex_u * (SCATTERING_TEXTURE_WIDTH - 1)
        yi = w * (SCATTERING_TEXTURE_HEIGHT - 1)
        zi = z * (SCATTERING_TEXTURE_DEPTH - 1)
        
        x0, y0, z0 = int(xi), int(yi), int(zi)
        x1 = min(x0 + 1, SCATTERING_TEXTURE_WIDTH - 1)
        y1 = min(y0 + 1, SCATTERING_TEXTURE_HEIGHT - 1)
        z1 = min(z0 + 1, SCATTERING_TEXTURE_DEPTH - 1)
        
        fx, fy, fz = xi - x0, yi - y0, zi - z0
        
        # Trilinear interpolation
        result = (
            scattering[z0, y0, x0] * (1-fx) * (1-fy) * (1-fz) +
            scattering[z0, y0, x1] * fx * (1-fy) * (1-fz) +
            scattering[z0, y1, x0] * (1-fx) * fy * (1-fz) +
            scattering[z0, y1, x1] * fx * fy * (1-fz) +
            scattering[z1, y0, x0] * (1-fx) * (1-fy) * fz +
            scattering[z1, y0, x1] * fx * (1-fy) * fz +
            scattering[z1, y1, x0] * (1-fx) * fy * fz +
            scattering[z1, y1, x1] * fx * fy * fz
        )
        return result[:3]
    
    def _sample_irradiance(
        self, irradiance: np.ndarray, r: float, mu_s: float
    ) -> np.ndarray:
        """Sample the irradiance texture with bilinear interpolation."""
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        
        x_r = (r - bottom) / (top - bottom)
        x_mu_s = mu_s * 0.5 + 0.5
        
        u = self._get_texture_coord_from_unit_range(x_mu_s, IRRADIANCE_TEXTURE_WIDTH)
        v = self._get_texture_coord_from_unit_range(x_r, IRRADIANCE_TEXTURE_HEIGHT)
        
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
        
        xi = u * (IRRADIANCE_TEXTURE_WIDTH - 1)
        yi = v * (IRRADIANCE_TEXTURE_HEIGHT - 1)
        
        x0, y0 = int(xi), int(yi)
        x1 = min(x0 + 1, IRRADIANCE_TEXTURE_WIDTH - 1)
        y1 = min(y0 + 1, IRRADIANCE_TEXTURE_HEIGHT - 1)
        
        fx, fy = xi - x0, yi - y0
        
        result = (
            irradiance[y0, x0] * (1-fx) * (1-fy) +
            irradiance[y0, x1] * fx * (1-fy) +
            irradiance[y1, x0] * (1-fx) * fy +
            irradiance[y1, x1] * fx * fy
        )
        return result
    
    def _rayleigh_phase(self, nu: float) -> float:
        """Rayleigh phase function."""
        k = 3.0 / (16.0 * np.pi)
        return k * (1.0 + nu * nu)
    
    def _mie_phase(self, g: float, nu: float) -> float:
        """Mie phase function (Cornette-Shanks)."""
        k = 3.0 / (8.0 * np.pi) * (1.0 - g * g) / (2.0 + g * g)
        return k * (1.0 + nu * nu) / np.power(1.0 + g * g - 2.0 * g * nu, 1.5)
    
    def _precompute_scattering_density(
        self, transmittance: np.ndarray,
        delta_rayleigh: np.ndarray, delta_mie: np.ndarray,
        scattering: np.ndarray, irradiance: np.ndarray,
        order: int, delta_scattering_density: np.ndarray
    ) -> None:
        """
        Precompute scattering density for multiple scattering.
        
        This computes the source term for order n scattering by integrating
        over all incident directions the radiance from order n-1 scattering.
        """
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        H = np.sqrt(top * top - bottom * bottom)
        g = self.params.mie_phase_function_g
        
        # Reduced samples for faster computation (8 instead of 16)
        # Full accuracy would use 16, but that's ~4x slower
        SPHERE_SAMPLES = 8
        dphi = np.pi / SPHERE_SAMPLES
        dtheta = np.pi / SPHERE_SAMPLES
        
        print(f"  Computing scattering density for order {order} ({SPHERE_SAMPLES}x{2*SPHERE_SAMPLES} samples)...")
        sys.stdout.flush()
        
        # Iterate over all texels
        for k in range(SCATTERING_TEXTURE_DEPTH):
            z = (k + 0.5) / SCATTERING_TEXTURE_DEPTH
            rho = H * self._get_unit_range_from_texture_coord(z, SCATTERING_TEXTURE_R_SIZE)
            r = np.sqrt(rho * rho + bottom * bottom)
            
            for j in range(SCATTERING_TEXTURE_HEIGHT):
                w = (j + 0.5) / SCATTERING_TEXTURE_HEIGHT
                
                # Decode mu from w
                if w < 0.5:
                    d_min = r - bottom
                    d_max = rho
                    w_unit = self._get_unit_range_from_texture_coord(1.0 - 2.0 * w, SCATTERING_TEXTURE_MU_SIZE // 2)
                    d = d_min + (d_max - d_min) * w_unit
                    mu = -1.0 if d == 0.0 else -(rho * rho + d * d) / (2.0 * r * d)
                    ray_intersects_ground = True
                else:
                    d_min = top - r
                    d_max = rho + H
                    w_unit = self._get_unit_range_from_texture_coord(2.0 * w - 1.0, SCATTERING_TEXTURE_MU_SIZE // 2)
                    d = d_min + (d_max - d_min) * w_unit
                    mu = 1.0 if d == 0.0 else (H * H - rho * rho - d * d) / (2.0 * r * d)
                    ray_intersects_ground = False
                
                mu = np.clip(mu, -1.0, 1.0)
                
                for i in range(SCATTERING_TEXTURE_WIDTH):
                    # Decode mu_s and nu from i
                    nu_idx = i % SCATTERING_TEXTURE_NU_SIZE
                    mu_s_idx = i // SCATTERING_TEXTURE_NU_SIZE
                    
                    v = (mu_s_idx + 0.5) / SCATTERING_TEXTURE_MU_S_SIZE
                    x_mu_s = self._get_unit_range_from_texture_coord(v, SCATTERING_TEXTURE_MU_S_SIZE)
                    d_min_s = top - bottom
                    d_max_s = H
                    mu_s_min = -0.2
                    D = self._distance_to_top_atmosphere_boundary(bottom, mu_s_min)
                    A = (D - d_min_s) / (d_max_s - d_min_s)
                    a_val = (A - x_mu_s * A) / (1.0 + x_mu_s * A) if (1.0 + x_mu_s * A) != 0 else 0.0
                    d_s = d_min_s + min(a_val, A) * (d_max_s - d_min_s)
                    mu_s = 1.0 if d_s == 0.0 else (H * H - d_s * d_s) / (2.0 * bottom * d_s)
                    mu_s = np.clip(mu_s, -1.0, 1.0)
                    
                    u = (nu_idx + 0.5) / SCATTERING_TEXTURE_NU_SIZE
                    nu = self._get_unit_range_from_texture_coord(u, SCATTERING_TEXTURE_NU_SIZE) * 2.0 - 1.0
                    
                    # Build direction vectors
                    sin_mu = np.sqrt(max(0.0, 1.0 - mu * mu))
                    omega = np.array([sin_mu, 0.0, mu])
                    
                    sun_dir_x = 0.0 if sin_mu == 0.0 else (nu - mu * mu_s) / sin_mu
                    sun_dir_y = np.sqrt(max(0.0, 1.0 - sun_dir_x * sun_dir_x - mu_s * mu_s))
                    omega_s = np.array([sun_dir_x, sun_dir_y, mu_s])
                    
                    # Integrate over all incident directions
                    rayleigh_mie = np.zeros(3)
                    
                    for l in range(SPHERE_SAMPLES):
                        theta = (l + 0.5) * dtheta
                        cos_theta = np.cos(theta)
                        sin_theta = np.sin(theta)
                        
                        ray_r_theta_intersects_ground = self._ray_intersects_ground(r, cos_theta)
                        
                        # Transmittance to ground (if applicable)
                        if ray_r_theta_intersects_ground:
                            d_to_ground = self._distance_to_bottom_atmosphere_boundary(r, cos_theta)
                            t_to_ground = self._get_transmittance(transmittance, r, cos_theta, d_to_ground, True)
                            ground_albedo = self.params.ground_albedo[:3]
                        else:
                            t_to_ground = np.zeros(3)
                            ground_albedo = np.zeros(3)
                        
                        for m in range(2 * SPHERE_SAMPLES):
                            phi = (m + 0.5) * dphi
                            omega_i = np.array([np.cos(phi) * sin_theta, np.sin(phi) * sin_theta, cos_theta])
                            domega_i = dtheta * dphi * sin_theta
                            
                            # Get incident radiance from previous order
                            nu1 = np.dot(omega_s, omega_i)
                            
                            if order == 2:
                                # Use single scattering (delta_rayleigh + delta_mie with phase)
                                rayleigh_i = self._sample_scattering(delta_rayleigh, r, omega_i[2], mu_s, nu1, ray_r_theta_intersects_ground)
                                mie_i = self._sample_scattering(delta_mie, r, omega_i[2], mu_s, nu1, ray_r_theta_intersects_ground)
                                incident = rayleigh_i * self._rayleigh_phase(nu1) + mie_i * self._mie_phase(g, nu1)
                            else:
                                # Use accumulated multiple scattering
                                incident = self._sample_scattering(scattering, r, omega_i[2], mu_s, nu1, ray_r_theta_intersects_ground)
                            
                            # Add ground reflection contribution
                            if ray_r_theta_intersects_ground and order >= 2:
                                ground_normal = np.array([0.0, 0.0, 1.0]) * r + omega_i * d_to_ground
                                ground_normal = ground_normal / np.linalg.norm(ground_normal)
                                ground_mu_s = np.dot(ground_normal, omega_s)
                                ground_irr = self._sample_irradiance(irradiance, bottom, ground_mu_s)
                                incident = incident + t_to_ground * ground_albedo * (1.0 / np.pi) * ground_irr
                            
                            # Phase function for scattering from omega_i to -omega
                            nu2 = np.dot(omega, omega_i)
                            altitude = r - bottom
                            rayleigh_density = self._get_profile_density(self.params.rayleigh_density, altitude)
                            mie_density = self._get_profile_density(self.params.mie_density, altitude)
                            
                            rayleigh_mie += incident * (
                                self.params.rayleigh_scattering[:3] * rayleigh_density * self._rayleigh_phase(nu2) +
                                self.params.mie_scattering[:3] * mie_density * self._mie_phase(g, nu2)
                            ) * domega_i
                    
                    delta_scattering_density[k, j, i, :3] = rayleigh_mie
            
            if (k + 1) % 4 == 0 or k == 0:  # Print every 4th slice
                print(f"    Density slice {k+1}/{SCATTERING_TEXTURE_DEPTH}")
                sys.stdout.flush()
    
    def _precompute_indirect_irradiance(
        self, delta_rayleigh: np.ndarray, delta_mie: np.ndarray,
        scattering: np.ndarray, order: int,
        delta_indirect_irradiance: np.ndarray
    ) -> None:
        """
        Precompute indirect (sky) irradiance at ground level.
        
        Integrates sky radiance over the upper hemisphere.
        """
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        g = self.params.mie_phase_function_g
        
        # Reduced samples for faster computation (16 instead of 32)
        HEMI_SAMPLES = 16
        dphi = np.pi / HEMI_SAMPLES
        dtheta = np.pi / HEMI_SAMPLES
        
        print(f"  Computing indirect irradiance for order {order} ({HEMI_SAMPLES//2}x{2*HEMI_SAMPLES} samples)...")
        sys.stdout.flush()
        
        for j in range(IRRADIANCE_TEXTURE_HEIGHT):
            v = (j + 0.5) / IRRADIANCE_TEXTURE_HEIGHT
            x_r = self._get_unit_range_from_texture_coord(v, IRRADIANCE_TEXTURE_HEIGHT)
            r = bottom + x_r * (top - bottom)
            
            for i in range(IRRADIANCE_TEXTURE_WIDTH):
                u = (i + 0.5) / IRRADIANCE_TEXTURE_WIDTH
                x_mu_s = self._get_unit_range_from_texture_coord(u, IRRADIANCE_TEXTURE_WIDTH)
                mu_s = np.clip(2.0 * x_mu_s - 1.0, -1.0, 1.0)
                
                omega_s = np.array([np.sqrt(max(0.0, 1.0 - mu_s * mu_s)), 0.0, mu_s])
                
                result = np.zeros(3)
                
                # Integrate over upper hemisphere
                for jj in range(HEMI_SAMPLES // 2):
                    theta = (jj + 0.5) * dtheta
                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)
                    
                    for ii in range(2 * HEMI_SAMPLES):
                        phi = (ii + 0.5) * dphi
                        omega = np.array([np.cos(phi) * sin_theta, np.sin(phi) * sin_theta, cos_theta])
                        domega = dtheta * dphi * sin_theta
                        
                        nu = np.dot(omega, omega_s)
                        
                        if order == 1:
                            # Use single scattering with phase
                            rayleigh_i = self._sample_scattering(delta_rayleigh, r, omega[2], mu_s, nu, False)
                            mie_i = self._sample_scattering(delta_mie, r, omega[2], mu_s, nu, False)
                            sky_rad = rayleigh_i * self._rayleigh_phase(nu) + mie_i * self._mie_phase(g, nu)
                        else:
                            # Use accumulated scattering
                            sky_rad = self._sample_scattering(scattering, r, omega[2], mu_s, nu, False)
                        
                        # Weight by cosine (omega.z = cos_theta)
                        result += sky_rad * omega[2] * domega
                
                delta_indirect_irradiance[j, i] = result
        
        print(f"    Indirect irradiance complete")
        sys.stdout.flush()
    
    def _precompute_multiple_scattering(
        self, transmittance: np.ndarray,
        delta_scattering_density: np.ndarray,
        delta_multiple: np.ndarray
    ) -> None:
        """
        Precompute multiple scattering contribution.
        
        Integrates the scattering density along each view ray.
        """
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        H = np.sqrt(top * top - bottom * bottom)
        
        SAMPLE_COUNT = 50
        
        print(f"  Computing multiple scattering integration...")
        sys.stdout.flush()
        
        for k in range(SCATTERING_TEXTURE_DEPTH):
            z = (k + 0.5) / SCATTERING_TEXTURE_DEPTH
            rho = H * self._get_unit_range_from_texture_coord(z, SCATTERING_TEXTURE_R_SIZE)
            r = np.sqrt(rho * rho + bottom * bottom)
            
            for j in range(SCATTERING_TEXTURE_HEIGHT):
                w = (j + 0.5) / SCATTERING_TEXTURE_HEIGHT
                
                # Decode mu
                if w < 0.5:
                    d_min = r - bottom
                    d_max = rho
                    w_unit = self._get_unit_range_from_texture_coord(1.0 - 2.0 * w, SCATTERING_TEXTURE_MU_SIZE // 2)
                    d = d_min + (d_max - d_min) * w_unit
                    mu = -1.0 if d == 0.0 else -(rho * rho + d * d) / (2.0 * r * d)
                    ray_intersects_ground = True
                else:
                    d_min = top - r
                    d_max = rho + H
                    w_unit = self._get_unit_range_from_texture_coord(2.0 * w - 1.0, SCATTERING_TEXTURE_MU_SIZE // 2)
                    d = d_min + (d_max - d_min) * w_unit
                    mu = 1.0 if d == 0.0 else (H * H - rho * rho - d * d) / (2.0 * r * d)
                    ray_intersects_ground = False
                
                mu = np.clip(mu, -1.0, 1.0)
                
                # Distance to boundary
                if ray_intersects_ground:
                    d_max_ray = self._distance_to_bottom_atmosphere_boundary(r, mu)
                else:
                    d_max_ray = self._distance_to_top_atmosphere_boundary(r, mu)
                
                dx = d_max_ray / SAMPLE_COUNT
                
                for i in range(SCATTERING_TEXTURE_WIDTH):
                    # Decode nu
                    nu_idx = i % SCATTERING_TEXTURE_NU_SIZE
                    mu_s_idx = i // SCATTERING_TEXTURE_NU_SIZE
                    
                    v = (mu_s_idx + 0.5) / SCATTERING_TEXTURE_MU_S_SIZE
                    x_mu_s = self._get_unit_range_from_texture_coord(v, SCATTERING_TEXTURE_MU_S_SIZE)
                    d_min_s = top - bottom
                    d_max_s = H
                    mu_s_min = -0.2
                    D = self._distance_to_top_atmosphere_boundary(bottom, mu_s_min)
                    A = (D - d_min_s) / (d_max_s - d_min_s)
                    a_val = (A - x_mu_s * A) / (1.0 + x_mu_s * A) if (1.0 + x_mu_s * A) != 0 else 0.0
                    d_s = d_min_s + min(a_val, A) * (d_max_s - d_min_s)
                    mu_s = 1.0 if d_s == 0.0 else (H * H - d_s * d_s) / (2.0 * bottom * d_s)
                    mu_s = np.clip(mu_s, -1.0, 1.0)
                    
                    u = (nu_idx + 0.5) / SCATTERING_TEXTURE_NU_SIZE
                    nu = self._get_unit_range_from_texture_coord(u, SCATTERING_TEXTURE_NU_SIZE) * 2.0 - 1.0
                    
                    # Integrate along ray (trapezoidal rule)
                    rayleigh_mie_sum = np.zeros(3)
                    
                    for s in range(SAMPLE_COUNT + 1):
                        d_i = s * dx
                        
                        # Position along ray
                        r_i = np.sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r)
                        r_i = np.clip(r_i, bottom, top)
                        mu_i = (r * mu + d_i) / r_i if r_i > 0 else mu
                        mu_i = np.clip(mu_i, -1.0, 1.0)
                        mu_s_i = (r * mu_s + d_i * nu) / r_i if r_i > 0 else mu_s
                        mu_s_i = np.clip(mu_s_i, -1.0, 1.0)
                        
                        # Sample scattering density at this point
                        scatter_i = self._sample_scattering(
                            delta_scattering_density, r_i, mu_i, mu_s_i, nu, ray_intersects_ground
                        )
                        
                        # Transmittance from start to this point
                        trans_i = self._get_transmittance(transmittance, r, mu, d_i, ray_intersects_ground)
                        
                        # Trapezoidal weight
                        weight = 0.5 if (s == 0 or s == SAMPLE_COUNT) else 1.0
                        
                        rayleigh_mie_sum += scatter_i * trans_i * dx * weight
                    
                    delta_multiple[k, j, i, :3] = rayleigh_mie_sum
            
            print(f"    Multiple scattering slice {k+1}/{SCATTERING_TEXTURE_DEPTH}")
            sys.stdout.flush()
    
    def _ray_intersects_ground(self, r: float, mu: float) -> bool:
        """Check if a ray from radius r in direction mu intersects the ground."""
        bottom = self.params.bottom_radius
        return mu < 0.0 and r * r * (mu * mu - 1.0) + bottom * bottom >= 0.0
    
    def get_shader_uniforms(self) -> dict:
        """
        Get dictionary of uniform values for shaders.
        
        Returns:
            Dictionary with uniform names and values
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call init() first.")
        
        p = self.params
        return {
            'bottom_radius': p.bottom_radius / p.length_unit_in_meters,
            'top_radius': p.top_radius / p.length_unit_in_meters,
            'rayleigh_scattering': p.rayleigh_scattering * p.length_unit_in_meters,
            'mie_scattering': p.mie_scattering * p.length_unit_in_meters,
            'mie_extinction': p.mie_extinction * p.length_unit_in_meters,
            'mie_phase_function_g': p.mie_phase_function_g,
            'absorption_extinction': p.absorption_extinction * p.length_unit_in_meters,
            'ground_albedo': p.ground_albedo,
            'sun_angular_radius': p.sun_angular_radius,
            'solar_irradiance': self._solar_irradiance_rgb,
        }
    
    def save_textures(self, filepath: str) -> None:
        """Save precomputed textures to a file (NumPy format)."""
        if not self._is_initialized:
            raise RuntimeError("Model not initialized.")
        
        np.savez_compressed(
            filepath,
            transmittance=self.textures.transmittance,
            scattering=self.textures.scattering,
            irradiance=self.textures.irradiance,
            single_mie=self.textures.single_mie_scattering
        )
    
    def save_textures_exr(self, output_dir: str) -> None:
        """
        Save precomputed textures as EXR files for OSL shader.
        
        Creates:
        - transmittance.exr (2D: 256x64)
        - scattering.exr (3D stored as tiled 2D)
        - single_mie_scattering.exr (3D stored as tiled 2D)
        - irradiance.exr (2D: 64x16)
        
        Args:
            output_dir: Directory to save EXR files
        """
        import os
        
        if not self._is_initialized:
            raise RuntimeError("Model not initialized.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save 2D textures directly
        self._save_2d_exr(
            os.path.join(output_dir, "transmittance.exr"),
            self.textures.transmittance
        )
        self._save_2d_exr(
            os.path.join(output_dir, "irradiance.exr"),
            self.textures.irradiance
        )
        
        # Save 3D textures as tiled 2D (for OSL texture() sampling)
        # Scattering texture: (D, H, W, 4) -> tile D slices horizontally
        self._save_3d_as_tiled_exr(
            os.path.join(output_dir, "scattering.exr"),
            self.textures.scattering
        )
        
        # Single Mie scattering (if separate)
        if self.textures.single_mie_scattering is not None:
            self._save_3d_as_tiled_exr(
                os.path.join(output_dir, "single_mie_scattering.exr"),
                self.textures.single_mie_scattering
            )
    
    def _save_2d_exr(self, filepath: str, data: np.ndarray) -> None:
        """Save a 2D texture as EXR using Blender's image API."""
        import os
        
        try:
            import bpy
            
            height, width = data.shape[:2]
            channels = data.shape[2] if len(data.shape) > 2 else 1
            
            # Create a new image in Blender
            img_name = os.path.basename(filepath).replace('.exr', '_temp')
            
            # Remove existing image if it exists
            if img_name in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[img_name])
            
            # Create image with alpha channel (RGBA)
            img = bpy.data.images.new(img_name, width, height, alpha=True, float_buffer=True)
            
            # Prepare pixel data (Blender expects RGBA, flattened, bottom-to-top)
            pixels = np.zeros((height, width, 4), dtype=np.float32)
            if channels >= 3:
                pixels[:, :, :3] = data[:, :, :3]
            elif channels == 1:
                pixels[:, :, 0] = data[:, :, 0] if len(data.shape) > 2 else data
                pixels[:, :, 1] = pixels[:, :, 0]
                pixels[:, :, 2] = pixels[:, :, 0]
            if channels >= 4:
                pixels[:, :, 3] = data[:, :, 3]
            else:
                pixels[:, :, 3] = 1.0
            
            # Flip vertically (Blender stores bottom-to-top)
            pixels = np.flipud(pixels)
            
            # Flatten and assign
            img.pixels[:] = pixels.flatten()
            
            # Save as EXR
            img.file_format = 'OPEN_EXR'
            img.filepath_raw = filepath
            img.save()
            
            # Clean up
            bpy.data.images.remove(img)
            print(f"Helios: Saved EXR: {filepath}")
            
        except Exception as e:
            print(f"Helios: Failed to save EXR {filepath}: {e}")
            # Fallback to raw
            self._save_raw_with_header(filepath, data)
    
    def _save_3d_as_tiled_exr(self, filepath: str, data: np.ndarray) -> None:
        """
        Save a 3D texture as a tiled 2D EXR.
        
        The 3D texture (D, H, W, C) is converted to 2D by tiling depth slices.
        Layout: W*D width, H height (slices placed side by side)
        
        This allows OSL to sample it with modified UV coordinates.
        """
        import os
        
        depth, height, width = data.shape[:3]
        channels = data.shape[3] if len(data.shape) > 3 else 1
        
        # Debug: check input data
        fname = os.path.basename(filepath)
        print(f"  [DEBUG] _save_3d_as_tiled_exr {fname}: input shape={data.shape}, min={data.min():.6f}, max={data.max():.6f}")
        print(f"  [DEBUG] {fname} has inf: {np.any(np.isinf(data))}, has nan: {np.any(np.isnan(data))}")
        
        # Create tiled 2D image: all depth slices side by side
        tiled_width = width * depth
        tiled = np.zeros((height, tiled_width, channels), dtype=np.float32)
        
        for d in range(depth):
            x_start = d * width
            x_end = (d + 1) * width
            if channels == 1:
                tiled[:, x_start:x_end, 0] = data[d, :, :]
            else:
                tiled[:, x_start:x_end, :] = data[d, :, :, :channels]
        
        # Debug: check tiled data
        print(f"  [DEBUG] {fname} tiled: shape={tiled.shape}, min={tiled.min():.6f}, max={tiled.max():.6f}")
        print(f"  [DEBUG] {fname} tiled has inf: {np.any(np.isinf(tiled))}, has nan: {np.any(np.isnan(tiled))}")
        
        # Save as 2D EXR
        self._save_2d_exr(filepath, tiled)
    
    def _save_raw_with_header(self, filepath: str, data: np.ndarray) -> None:
        """Fallback: save as raw binary with shape header."""
        with open(filepath + '.raw', 'wb') as f:
            # Write shape as header
            shape_bytes = np.array(data.shape, dtype=np.int32).tobytes()
            f.write(len(data.shape).to_bytes(4, 'little'))
            f.write(shape_bytes)
            f.write(data.astype(np.float32).tobytes())
    
    def load_textures(self, filepath: str) -> None:
        """Load precomputed textures from a file."""
        data = np.load(filepath)
        self.textures = PrecomputedTextures(
            transmittance=data['transmittance'],
            scattering=data['scattering'],
            irradiance=data['irradiance'],
            single_mie_scattering=data['single_mie'] if 'single_mie' in data else None
        )
        self._is_initialized = True
