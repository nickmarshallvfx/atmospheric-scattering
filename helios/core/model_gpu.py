"""
GPU-Accelerated Atmosphere Model using CuPy.

This module provides GPU-accelerated precomputation of atmospheric scattering LUTs.
Falls back to NumPy if CuPy is not available.
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
)
from .parameters import AtmosphereParameters

# Try to import CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


@dataclass
class PrecomputedTextures:
    """Container for precomputed LUT textures."""
    transmittance: np.ndarray
    scattering: np.ndarray
    irradiance: np.ndarray
    single_mie_scattering: Optional[np.ndarray] = None


class GPUAtmosphereModel:
    """
    GPU-accelerated atmosphere model using CuPy.
    
    Processes all texels in parallel on the GPU for massive speedup.
    """
    
    def __init__(self, params: Optional[AtmosphereParameters] = None, use_gpu: bool = True):
        self.params = params or AtmosphereParameters.earth_default()
        self.textures: Optional[PrecomputedTextures] = None
        self._is_initialized = False
        
        # Select backend
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        if self.use_gpu:
            device = cp.cuda.Device()
            print(f"[Helios] GPU Backend: {device.name.decode()} ({device.mem_info[1] // 1024**2} MB)")
        else:
            if use_gpu and not CUPY_AVAILABLE:
                print("[Helios] CuPy not available, using CPU (NumPy)")
            else:
                print("[Helios] Using CPU backend (NumPy)")
        
        self._solar_irradiance_rgb: Optional[np.ndarray] = None
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    def _to_gpu(self, arr):
        """Move array to GPU if using CuPy."""
        if self.use_gpu:
            return cp.asarray(arr)
        return arr
    
    def _to_cpu(self, arr):
        """Move array to CPU."""
        if self.use_gpu:
            return cp.asnumpy(arr)
        return np.asarray(arr)
    
    def _sync(self):
        """Synchronize GPU."""
        if self.use_gpu:
            cp.cuda.Stream.null.synchronize()
    
    def init(self, num_scattering_orders: int = 4, progress_callback=None) -> None:
        """Precompute atmosphere LUTs with GPU acceleration."""
        xp = self.xp
        
        if progress_callback:
            progress_callback(0.0, "Initializing GPU atmosphere model...")
        
        # Compute solar irradiance RGB
        from .constants import convert_spectrum_to_linear_srgb
        self._solar_irradiance_rgb = convert_spectrum_to_linear_srgb(
            self.params.wavelengths,
            self.params.solar_irradiance
        )
        
        # Allocate textures on GPU
        transmittance = xp.zeros(
            (TRANSMITTANCE_TEXTURE_HEIGHT, TRANSMITTANCE_TEXTURE_WIDTH, 3),
            dtype=xp.float32
        )
        scattering = xp.zeros(
            (SCATTERING_TEXTURE_DEPTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_WIDTH, 4),
            dtype=xp.float32
        )
        irradiance = xp.zeros(
            (IRRADIANCE_TEXTURE_HEIGHT, IRRADIANCE_TEXTURE_WIDTH, 3),
            dtype=xp.float32
        )
        
        # Precompute transmittance
        if progress_callback:
            progress_callback(0.1, "Computing transmittance LUT (GPU)...")
        print("[Helios] Computing transmittance...")
        self._precompute_transmittance_gpu(transmittance)
        self._sync()
        
        # Precompute direct irradiance
        if progress_callback:
            progress_callback(0.2, "Computing direct irradiance (GPU)...")
        print("[Helios] Computing direct irradiance...")
        delta_irradiance = xp.zeros_like(irradiance)
        self._precompute_direct_irradiance_gpu(transmittance, delta_irradiance)
        self._sync()
        
        # Precompute single scattering
        if progress_callback:
            progress_callback(0.3, "Computing single scattering (GPU)...")
        print("[Helios] Computing single scattering...")
        delta_rayleigh = xp.zeros(
            (SCATTERING_TEXTURE_DEPTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_WIDTH, 3),
            dtype=xp.float32
        )
        delta_mie = xp.zeros_like(delta_rayleigh)
        self._precompute_single_scattering_gpu(transmittance, delta_rayleigh, delta_mie, scattering)
        self._sync()
        
        # Multiple scattering orders
        for order in range(2, num_scattering_orders + 1):
            if progress_callback:
                progress = 0.3 + 0.6 * (order - 2) / max(1, num_scattering_orders - 1)
                progress_callback(progress, f"Computing scattering order {order} (GPU)...")
            
            print(f"[Helios] Computing scattering order {order}...")
            
            # Scattering density
            delta_scattering_density = xp.zeros_like(delta_rayleigh)
            self._precompute_scattering_density_gpu(
                transmittance, delta_rayleigh, delta_mie,
                scattering, delta_irradiance,
                order, delta_scattering_density
            )
            self._sync()
            
            # Indirect irradiance
            delta_indirect = xp.zeros_like(irradiance)
            self._precompute_indirect_irradiance_gpu(
                delta_rayleigh, delta_mie, scattering, order, delta_indirect
            )
            irradiance += delta_indirect
            self._sync()
            
            # Multiple scattering integration
            delta_multiple = xp.zeros_like(delta_rayleigh)
            self._precompute_multiple_scattering_gpu(
                transmittance, delta_scattering_density, delta_multiple
            )
            
            scattering[..., :3] += delta_multiple
            delta_rayleigh[:] = delta_multiple
            self._sync()
        
        # Add direct irradiance
        irradiance += delta_irradiance
        
        if progress_callback:
            progress_callback(1.0, "Precomputation complete.")
        
        # Move results to CPU
        self.textures = PrecomputedTextures(
            transmittance=self._to_cpu(transmittance),
            scattering=self._to_cpu(scattering),
            irradiance=self._to_cpu(irradiance),
            single_mie_scattering=None
        )
        self._is_initialized = True
        print("[Helios] Precomputation complete!")
    
    def _get_unit_range_from_texture_coord(self, u, texture_size):
        """Convert texture coordinate to unit range [0,1]."""
        return (u - 0.5 / texture_size) / (1.0 - 1.0 / texture_size)
    
    def _get_texture_coord_from_unit_range(self, x, texture_size):
        """Convert unit range [0,1] to texture coordinate."""
        return 0.5 / texture_size + x * (1.0 - 1.0 / texture_size)
    
    def _precompute_transmittance_gpu(self, transmittance):
        """Fully vectorized transmittance computation."""
        xp = self.xp
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        H = xp.sqrt(top * top - bottom * bottom)
        
        # Create coordinate grids for all texels
        j_coords = xp.arange(TRANSMITTANCE_TEXTURE_HEIGHT)
        i_coords = xp.arange(TRANSMITTANCE_TEXTURE_WIDTH)
        jj, ii = xp.meshgrid(j_coords, i_coords, indexing='ij')
        
        u = (ii + 0.5) / TRANSMITTANCE_TEXTURE_WIDTH
        v = (jj + 0.5) / TRANSMITTANCE_TEXTURE_HEIGHT
        
        # Convert UV to r, mu
        rho = H * self._get_unit_range_from_texture_coord(v, TRANSMITTANCE_TEXTURE_HEIGHT)
        r = xp.sqrt(rho * rho + bottom * bottom)
        
        d_min = top - r
        d_max = rho + H
        d = d_min + (d_max - d_min) * self._get_unit_range_from_texture_coord(u, TRANSMITTANCE_TEXTURE_WIDTH)
        
        mu = xp.where(d > 0, (H * H - rho * rho - d * d) / (2.0 * r * d), 1.0)
        mu = xp.clip(mu, -1.0, 1.0)
        
        # Distance to top atmosphere
        discriminant = r * r * (mu * mu - 1.0) + top * top
        dist_to_top = xp.maximum(0.0, -r * mu + xp.sqrt(xp.maximum(0.0, discriminant)))
        
        # Numerical integration
        SAMPLE_COUNT = 500
        optical_depth = xp.zeros((TRANSMITTANCE_TEXTURE_HEIGHT, TRANSMITTANCE_TEXTURE_WIDTH, 3), dtype=xp.float64)
        dx = dist_to_top / SAMPLE_COUNT
        
        # Move coefficients to GPU
        rayleigh_scat = self._to_gpu(self.params.rayleigh_scattering[:3])
        mie_ext = self._to_gpu(self.params.mie_extinction[:3])
        abs_ext = self._to_gpu(self.params.absorption_extinction[:3])
        
        for s in range(SAMPLE_COUNT):
            d_i = (s + 0.5) * dx
            r_i = xp.sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r)
            altitude = r_i - bottom
            
            # Get densities (vectorized)
            rayleigh_density = self._get_profile_density_gpu(self.params.rayleigh_density, altitude)
            mie_density = self._get_profile_density_gpu(self.params.mie_density, altitude)
            abs_density = self._get_profile_density_gpu(self.params.absorption_density, altitude)
            
            optical_depth += (
                rayleigh_density[..., None] * rayleigh_scat +
                mie_density[..., None] * mie_ext +
                abs_density[..., None] * abs_ext
            ) * dx[..., None]
        
        transmittance[:] = xp.exp(-optical_depth).astype(xp.float32)
    
    def _get_profile_density_gpu(self, layers, altitude):
        """Get density at altitude for a density profile (vectorized GPU)."""
        xp = self.xp
        
        if not layers:
            return xp.zeros_like(altitude)
        
        result = xp.zeros_like(altitude, dtype=xp.float64)
        current_altitude = 0.0
        
        for i, layer in enumerate(layers):
            if i < len(layers) - 1:
                layer_mask = (altitude >= current_altitude) & (altitude < current_altitude + layer.width)
                layer_alt = altitude - current_altitude
            else:
                layer_mask = altitude >= current_altitude
                layer_alt = altitude - current_altitude
            
            density = (
                layer.exp_term * xp.exp(layer.exp_scale * layer_alt) +
                layer.linear_term * layer_alt +
                layer.constant_term
            )
            density = xp.clip(density, 0.0, 1.0)
            result = xp.where(layer_mask, density, result)
            current_altitude += layer.width
        
        return result
    
    def _precompute_direct_irradiance_gpu(self, transmittance, delta_irradiance):
        """Vectorized direct irradiance computation."""
        xp = self.xp
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        
        j_coords = xp.arange(IRRADIANCE_TEXTURE_HEIGHT)
        i_coords = xp.arange(IRRADIANCE_TEXTURE_WIDTH)
        jj, ii = xp.meshgrid(j_coords, i_coords, indexing='ij')
        
        v = (jj + 0.5) / IRRADIANCE_TEXTURE_HEIGHT
        u = (ii + 0.5) / IRRADIANCE_TEXTURE_WIDTH
        
        r = bottom + (top - bottom) * v
        mu_s = -0.2 + u * 1.4
        
        trans = self._sample_transmittance_gpu(transmittance, r, mu_s)
        solar = self._to_gpu(self.params.solar_irradiance[:3])
        
        irr = solar * trans * mu_s[..., None]
        delta_irradiance[:] = xp.where(mu_s[..., None] > 0, irr, 0.0).astype(xp.float32)
    
    def _sample_transmittance_gpu(self, transmittance, r, mu):
        """Sample transmittance texture with bilinear interpolation (vectorized)."""
        xp = self.xp
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        H = xp.sqrt(top * top - bottom * bottom)
        
        rho = xp.sqrt(xp.maximum(0.0, r * r - bottom * bottom))
        v = self._get_texture_coord_from_unit_range(rho / H, TRANSMITTANCE_TEXTURE_HEIGHT)
        
        discriminant = r * r * (mu * mu - 1.0) + top * top
        d = xp.maximum(0.0, -r * mu + xp.sqrt(xp.maximum(0.0, discriminant)))
        
        d_min = top - r
        d_max = rho + H
        x_mu = xp.where(d_max > d_min, (d - d_min) / (d_max - d_min), 0.0)
        u = self._get_texture_coord_from_unit_range(x_mu, TRANSMITTANCE_TEXTURE_WIDTH)
        
        u = xp.clip(u, 0.0, 1.0)
        v = xp.clip(v, 0.0, 1.0)
        
        # Bilinear interpolation
        xi = u * (TRANSMITTANCE_TEXTURE_WIDTH - 1)
        yi = v * (TRANSMITTANCE_TEXTURE_HEIGHT - 1)
        
        x0 = xp.floor(xi).astype(xp.int32)
        y0 = xp.floor(yi).astype(xp.int32)
        x1 = xp.minimum(x0 + 1, TRANSMITTANCE_TEXTURE_WIDTH - 1)
        y1 = xp.minimum(y0 + 1, TRANSMITTANCE_TEXTURE_HEIGHT - 1)
        
        fx = xi - x0
        fy = yi - y0
        
        # Sample all 4 corners
        v00 = transmittance[y0, x0]
        v01 = transmittance[y0, x1]
        v10 = transmittance[y1, x0]
        v11 = transmittance[y1, x1]
        
        result = (
            v00 * (1-fx)[..., None] * (1-fy)[..., None] +
            v01 * fx[..., None] * (1-fy)[..., None] +
            v10 * (1-fx)[..., None] * fy[..., None] +
            v11 * fx[..., None] * fy[..., None]
        )
        return result
    
    def _precompute_single_scattering_gpu(self, transmittance, delta_rayleigh, delta_mie, scattering):
        """Vectorized single scattering computation."""
        xp = self.xp
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        H = xp.sqrt(top * top - bottom * bottom)
        
        rayleigh_scat = self._to_gpu(self.params.rayleigh_scattering[:3])
        mie_scat = self._to_gpu(self.params.mie_scattering[:3])
        mie_ext = self._to_gpu(self.params.mie_extinction[:3])
        solar = self._to_gpu(self.params.solar_irradiance[:3])
        
        SAMPLE_COUNT = 50
        
        # Process one depth slice at a time to manage memory
        for k in range(SCATTERING_TEXTURE_DEPTH):
            if (k + 1) % 8 == 0:
                print(f"  Single scattering: {k+1}/{SCATTERING_TEXTURE_DEPTH}")
                sys.stdout.flush()
            
            z = (k + 0.5) / SCATTERING_TEXTURE_DEPTH
            z_unit = self._get_unit_range_from_texture_coord(z, SCATTERING_TEXTURE_R_SIZE)
            rho = H * z_unit
            r_k = float(xp.sqrt(rho * rho + bottom * bottom))
            
            # Create grids for this slice
            j_coords = xp.arange(SCATTERING_TEXTURE_HEIGHT)
            i_coords = xp.arange(SCATTERING_TEXTURE_WIDTH)
            jj, ii = xp.meshgrid(j_coords, i_coords, indexing='ij')
            
            # Decode mu from j
            w = (jj + 0.5) / SCATTERING_TEXTURE_HEIGHT
            
            # Below horizon (w < 0.5)
            d_min_below = r_k - bottom
            d_max_below = rho
            w_unit_below = self._get_unit_range_from_texture_coord(1.0 - 2.0 * w, SCATTERING_TEXTURE_MU_SIZE // 2)
            d_below = d_min_below + (d_max_below - d_min_below) * w_unit_below
            mu_below = xp.where(d_below == 0, -1.0, -(rho * rho + d_below * d_below) / (2.0 * r_k * d_below))
            
            # Above horizon (w >= 0.5)
            d_min_above = top - r_k
            d_max_above = rho + H
            w_unit_above = self._get_unit_range_from_texture_coord(2.0 * w - 1.0, SCATTERING_TEXTURE_MU_SIZE // 2)
            d_above = d_min_above + (d_max_above - d_min_above) * w_unit_above
            mu_above = xp.where(d_above == 0, 1.0, (H * H - rho * rho - d_above * d_above) / (2.0 * r_k * d_above))
            
            mu = xp.where(w < 0.5, mu_below, mu_above)
            mu = xp.clip(mu, -1.0, 1.0)
            ray_intersects_ground = w < 0.5
            
            # Decode mu_s and nu from i
            nu_idx = ii % SCATTERING_TEXTURE_NU_SIZE
            mu_s_idx = ii // SCATTERING_TEXTURE_NU_SIZE
            
            v_s = (mu_s_idx + 0.5) / SCATTERING_TEXTURE_MU_S_SIZE
            x_mu_s = self._get_unit_range_from_texture_coord(v_s, SCATTERING_TEXTURE_MU_S_SIZE)
            d_min_s = top - bottom
            d_max_s = H
            mu_s_min = -0.2
            
            D_s = self._distance_to_top_gpu(bottom, mu_s_min)
            A = (D_s - d_min_s) / (d_max_s - d_min_s)
            denom = 1.0 + x_mu_s * A
            a_val = xp.where(denom != 0, (A - x_mu_s * A) / denom, 0.0)
            d_s = d_min_s + xp.minimum(a_val, A) * (d_max_s - d_min_s)
            mu_s = xp.where(d_s == 0, 1.0, (H * H - d_s * d_s) / (2.0 * bottom * d_s))
            mu_s = xp.clip(mu_s, -1.0, 1.0)
            
            u_nu = (nu_idx + 0.5) / SCATTERING_TEXTURE_NU_SIZE
            nu = self._get_unit_range_from_texture_coord(u_nu, SCATTERING_TEXTURE_NU_SIZE) * 2.0 - 1.0
            
            # Distance to boundary
            disc_ground = r_k * r_k * (mu * mu - 1.0) + bottom * bottom
            d_ground = xp.where(disc_ground >= 0, xp.maximum(0.0, -r_k * mu - xp.sqrt(xp.maximum(0.0, disc_ground))), 1e10)
            
            disc_top = r_k * r_k * (mu * mu - 1.0) + top * top
            d_top = xp.maximum(0.0, -r_k * mu + xp.sqrt(xp.maximum(0.0, disc_top)))
            
            d_max_ray = xp.where(ray_intersects_ground, d_ground, d_top)
            dx = d_max_ray / SAMPLE_COUNT
            
            # Integrate along ray
            rayleigh_sum = xp.zeros((SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_WIDTH, 3), dtype=xp.float64)
            mie_sum = xp.zeros_like(rayleigh_sum)
            
            for s in range(SAMPLE_COUNT):
                d_i = (s + 0.5) * dx
                
                # Position along ray
                r_i = xp.sqrt(d_i * d_i + 2.0 * r_k * mu * d_i + r_k * r_k)
                altitude_i = r_i - bottom
                
                # Transmittance from origin to sample point
                mu_i = xp.where(r_i > 0, (r_k * mu + d_i) / r_i, mu)
                mu_i = xp.clip(mu_i, -1.0, 1.0)
                trans_to_i = self._get_transmittance_gpu(transmittance, r_k, mu, d_i, ray_intersects_ground)
                
                # Transmittance from sample to sun
                mu_s_i = xp.where(r_i > 0, (r_k * mu_s + d_i * nu) / r_i, mu_s)
                mu_s_i = xp.clip(mu_s_i, -1.0, 1.0)
                trans_to_sun = self._sample_transmittance_gpu(transmittance, r_i, mu_s_i)
                
                # Densities
                rayleigh_density = self._get_profile_density_gpu(self.params.rayleigh_density, altitude_i)
                mie_density = self._get_profile_density_gpu(self.params.mie_density, altitude_i)
                
                # Accumulate
                common = trans_to_i * trans_to_sun * dx[..., None]
                rayleigh_sum += common * rayleigh_density[..., None] * rayleigh_scat
                mie_sum += common * mie_density[..., None] * mie_scat
            
            # Store results
            delta_rayleigh[k] = (rayleigh_sum * solar).astype(xp.float32)
            delta_mie[k] = (mie_sum * solar).astype(xp.float32)
            
            # Combined scattering (Rayleigh in RGB, Mie in A)
            scattering[k, :, :, :3] = delta_rayleigh[k]
            scattering[k, :, :, 3] = delta_mie[k, :, :, 0]
    
    def _distance_to_top_gpu(self, r, mu):
        """Distance to top atmosphere boundary."""
        xp = self.xp
        top = self.params.top_radius
        r = xp.asarray(r)
        mu = xp.asarray(mu)
        discriminant = r * r * (mu * mu - 1.0) + top * top
        return xp.maximum(0.0, -r * mu + xp.sqrt(xp.maximum(0.0, discriminant)))
    
    def _get_transmittance_gpu(self, transmittance, r, mu, d, ray_intersects_ground):
        """Get transmittance along a ray segment."""
        xp = self.xp
        r = xp.asarray(r)
        mu = xp.asarray(mu)
        d = xp.asarray(d)
        
        r_d = xp.sqrt(d * d + 2.0 * r * mu * d + r * r)
        mu_d = xp.where(r_d > 0, (r * mu + d) / r_d, mu)
        
        trans_full = self._sample_transmittance_gpu(transmittance, r, mu)
        trans_d = self._sample_transmittance_gpu(transmittance, r_d, mu_d)
        
        # Handle direction (looking up vs down)
        result = xp.where(
            mu > 0,
            xp.minimum(trans_full / xp.maximum(trans_d, 1e-10), 1.0),
            xp.minimum(trans_d / xp.maximum(trans_full, 1e-10), 1.0)
        )
        return result
    
    def _precompute_scattering_density_gpu(self, transmittance, delta_rayleigh, delta_mie,
                                           scattering, irradiance, order, delta_density):
        """Compute scattering density for multiple scattering (GPU vectorized)."""
        xp = self.xp
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        H = xp.sqrt(top * top - bottom * bottom)
        g = self.params.mie_phase_function_g
        
        rayleigh_scat = self._to_gpu(self.params.rayleigh_scattering[:3])
        mie_scat = self._to_gpu(self.params.mie_scattering[:3])
        ground_alb = self._to_gpu(self.params.ground_albedo[:3])
        
        SPHERE_SAMPLES = 8
        dphi = float(xp.pi) / SPHERE_SAMPLES
        dtheta = float(xp.pi) / SPHERE_SAMPLES
        
        # Process one depth slice at a time
        for k in range(SCATTERING_TEXTURE_DEPTH):
            if (k + 1) % 8 == 0:
                print(f"  Scattering density order {order}: {k+1}/{SCATTERING_TEXTURE_DEPTH}")
                sys.stdout.flush()
            
            z = (k + 0.5) / SCATTERING_TEXTURE_DEPTH
            rho = H * self._get_unit_range_from_texture_coord(z, SCATTERING_TEXTURE_R_SIZE)
            r_k = float(xp.sqrt(rho * rho + bottom * bottom))
            
            # Create grids for this slice
            j_coords = xp.arange(SCATTERING_TEXTURE_HEIGHT)
            i_coords = xp.arange(SCATTERING_TEXTURE_WIDTH)
            jj, ii = xp.meshgrid(j_coords, i_coords, indexing='ij')
            
            # Decode mu, mu_s, nu (same as single scattering)
            w = (jj + 0.5) / SCATTERING_TEXTURE_HEIGHT
            d_min_below = r_k - bottom
            d_max_below = rho
            w_unit_below = self._get_unit_range_from_texture_coord(1.0 - 2.0 * w, SCATTERING_TEXTURE_MU_SIZE // 2)
            d_below = d_min_below + (d_max_below - d_min_below) * w_unit_below
            mu_below = xp.where(d_below == 0, -1.0, -(rho * rho + d_below * d_below) / (2.0 * r_k * d_below))
            
            d_min_above = top - r_k
            d_max_above = rho + H
            w_unit_above = self._get_unit_range_from_texture_coord(2.0 * w - 1.0, SCATTERING_TEXTURE_MU_SIZE // 2)
            d_above = d_min_above + (d_max_above - d_min_above) * w_unit_above
            mu_above = xp.where(d_above == 0, 1.0, (H * H - rho * rho - d_above * d_above) / (2.0 * r_k * d_above))
            
            mu = xp.where(w < 0.5, mu_below, mu_above)
            mu = xp.clip(mu, -1.0, 1.0)
            
            nu_idx = ii % SCATTERING_TEXTURE_NU_SIZE
            mu_s_idx = ii // SCATTERING_TEXTURE_NU_SIZE
            
            v_s = (mu_s_idx + 0.5) / SCATTERING_TEXTURE_MU_S_SIZE
            x_mu_s = self._get_unit_range_from_texture_coord(v_s, SCATTERING_TEXTURE_MU_S_SIZE)
            d_min_s = top - bottom
            d_max_s = H
            D_s = self._distance_to_top_gpu(bottom, -0.2)
            A = (D_s - d_min_s) / (d_max_s - d_min_s)
            denom = 1.0 + x_mu_s * A
            a_val = xp.where(denom != 0, (A - x_mu_s * A) / denom, 0.0)
            d_s = d_min_s + xp.minimum(a_val, A) * (d_max_s - d_min_s)
            mu_s = xp.where(d_s == 0, 1.0, (H * H - d_s * d_s) / (2.0 * bottom * d_s))
            mu_s = xp.clip(mu_s, -1.0, 1.0)
            
            u_nu = (nu_idx + 0.5) / SCATTERING_TEXTURE_NU_SIZE
            nu = self._get_unit_range_from_texture_coord(u_nu, SCATTERING_TEXTURE_NU_SIZE) * 2.0 - 1.0
            
            # Build direction vectors
            sin_mu = xp.sqrt(xp.maximum(0.0, 1.0 - mu * mu))
            sun_dir_x = xp.where(sin_mu == 0, 0.0, (nu - mu * mu_s) / sin_mu)
            sun_dir_y = xp.sqrt(xp.maximum(0.0, 1.0 - sun_dir_x * sun_dir_x - mu_s * mu_s))
            
            # Densities at this altitude
            altitude = r_k - bottom
            rayleigh_density = self._get_profile_density_gpu(self.params.rayleigh_density, xp.asarray(altitude))
            mie_density = self._get_profile_density_gpu(self.params.mie_density, xp.asarray(altitude))
            
            # Integrate over sphere
            result = xp.zeros((SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_WIDTH, 3), dtype=xp.float64)
            
            for l in range(SPHERE_SAMPLES):
                theta = (l + 0.5) * dtheta
                cos_theta = float(xp.cos(theta))
                sin_theta = float(xp.sin(theta))
                
                for m in range(2 * SPHERE_SAMPLES):
                    phi = (m + 0.5) * dphi
                    cos_phi = float(xp.cos(phi))
                    sin_phi = float(xp.sin(phi))
                    
                    domega = dtheta * dphi * sin_theta
                    
                    # Incident direction
                    omega_i_x = cos_phi * sin_theta
                    omega_i_y = sin_phi * sin_theta
                    omega_i_z = cos_theta
                    
                    # nu1 = dot(omega_s, omega_i)
                    nu1 = sun_dir_x * omega_i_x + sun_dir_y * omega_i_y + mu_s * omega_i_z
                    
                    # nu2 = dot(omega, omega_i)
                    nu2 = sin_mu * omega_i_x + mu * omega_i_z
                    
                    # Sample incident radiance
                    if order == 2:
                        rayleigh_i = self._sample_scattering_gpu(delta_rayleigh, r_k, omega_i_z, mu_s, nu1)
                        mie_i = self._sample_scattering_gpu(delta_mie, r_k, omega_i_z, mu_s, nu1)
                        incident = rayleigh_i * self._rayleigh_phase(nu1) + mie_i * self._mie_phase(g, nu1)
                    else:
                        incident = self._sample_scattering_gpu(scattering, r_k, omega_i_z, mu_s, nu1)
                    
                    # Phase functions
                    phase_r = self._rayleigh_phase(nu2)
                    phase_m = self._mie_phase(g, nu2)
                    
                    result += incident * (
                        rayleigh_scat * float(rayleigh_density) * phase_r +
                        mie_scat * float(mie_density) * phase_m
                    ) * domega
            
            delta_density[k, :, :, :3] = result.astype(xp.float32)
    
    def _sample_scattering_gpu(self, scattering, r, mu, mu_s, nu):
        """Sample 3D scattering texture (simplified for batch processing)."""
        xp = self.xp
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        H = xp.sqrt(top * top - bottom * bottom)
        
        r = xp.asarray(r)
        mu = xp.asarray(mu)
        mu_s = xp.asarray(mu_s)
        nu = xp.asarray(nu)
        
        rho = xp.sqrt(xp.maximum(0.0, r * r - bottom * bottom))
        
        # z coordinate (r)
        u_r = self._get_texture_coord_from_unit_range(rho / H, SCATTERING_TEXTURE_R_SIZE)
        
        # w coordinate (mu)
        r_mu = r * mu
        discriminant_ground = r * r * (mu * mu - 1.0) + bottom * bottom
        ray_hits_ground = (mu < 0) & (discriminant_ground >= 0)
        
        d_min = xp.where(ray_hits_ground, r - bottom, top - r)
        d_max = xp.where(ray_hits_ground, rho, rho + H)
        d = xp.where(ray_hits_ground,
            -r_mu - xp.sqrt(xp.maximum(0.0, discriminant_ground)),
            -r_mu + xp.sqrt(xp.maximum(0.0, r * r * (mu * mu - 1.0) + top * top))
        )
        
        x_mu = xp.where(d_max > d_min, (d - d_min) / (d_max - d_min), 0.0)
        u_mu_base = self._get_texture_coord_from_unit_range(x_mu, SCATTERING_TEXTURE_MU_SIZE // 2)
        u_mu = xp.where(ray_hits_ground, 0.5 - 0.5 * u_mu_base, 0.5 + 0.5 * u_mu_base)
        
        # v coordinate (mu_s)
        d_min_s = top - bottom
        d_max_s = H
        D = self._distance_to_top_gpu(bottom, -0.2)
        A = (D - d_min_s) / (d_max_s - d_min_s)
        d_s = self._distance_to_top_gpu(bottom, mu_s)
        x_mu_s = xp.where(A > 0, (d_s - d_min_s) / (A * (d_max_s - d_min_s)), 0.0)
        x_mu_s = xp.clip(x_mu_s, 0.0, 1.0)
        u_mu_s = self._get_texture_coord_from_unit_range(x_mu_s, SCATTERING_TEXTURE_MU_S_SIZE)
        
        # u coordinate (nu)
        x_nu = (nu + 1.0) / 2.0
        u_nu = self._get_texture_coord_from_unit_range(x_nu, SCATTERING_TEXTURE_NU_SIZE)
        
        # Combined x coordinate
        u_x = (u_mu_s * SCATTERING_TEXTURE_MU_S_SIZE * SCATTERING_TEXTURE_NU_SIZE + 
               u_nu * SCATTERING_TEXTURE_NU_SIZE) / SCATTERING_TEXTURE_WIDTH
        
        u_x = xp.clip(u_x, 0.0, 1.0)
        u_mu = xp.clip(u_mu, 0.0, 1.0)
        u_r = xp.clip(u_r, 0.0, 1.0)
        
        # Trilinear interpolation
        xi = u_x * (SCATTERING_TEXTURE_WIDTH - 1)
        yi = u_mu * (SCATTERING_TEXTURE_HEIGHT - 1)
        zi = u_r * (SCATTERING_TEXTURE_DEPTH - 1)
        
        x0 = xp.floor(xi).astype(xp.int32)
        y0 = xp.floor(yi).astype(xp.int32)
        z0 = xp.floor(zi).astype(xp.int32)
        
        x1 = xp.minimum(x0 + 1, SCATTERING_TEXTURE_WIDTH - 1)
        y1 = xp.minimum(y0 + 1, SCATTERING_TEXTURE_HEIGHT - 1)
        z1 = xp.minimum(z0 + 1, SCATTERING_TEXTURE_DEPTH - 1)
        
        fx = xi - x0
        fy = yi - y0
        fz = zi - z0
        
        # Sample all 8 corners
        result = (
            scattering[z0, y0, x0, :3] * (1-fx)[..., None] * (1-fy)[..., None] * (1-fz)[..., None] +
            scattering[z0, y0, x1, :3] * fx[..., None] * (1-fy)[..., None] * (1-fz)[..., None] +
            scattering[z0, y1, x0, :3] * (1-fx)[..., None] * fy[..., None] * (1-fz)[..., None] +
            scattering[z0, y1, x1, :3] * fx[..., None] * fy[..., None] * (1-fz)[..., None] +
            scattering[z1, y0, x0, :3] * (1-fx)[..., None] * (1-fy)[..., None] * fz[..., None] +
            scattering[z1, y0, x1, :3] * fx[..., None] * (1-fy)[..., None] * fz[..., None] +
            scattering[z1, y1, x0, :3] * (1-fx)[..., None] * fy[..., None] * fz[..., None] +
            scattering[z1, y1, x1, :3] * fx[..., None] * fy[..., None] * fz[..., None]
        )
        return result
    
    def _rayleigh_phase(self, nu):
        """Rayleigh phase function."""
        xp = self.xp
        k = 3.0 / (16.0 * float(xp.pi))
        return k * (1.0 + nu * nu)
    
    def _mie_phase(self, g, nu):
        """Mie phase function (Cornette-Shanks)."""
        xp = self.xp
        k = 3.0 / (8.0 * float(xp.pi)) * (1.0 - g * g) / (2.0 + g * g)
        denom = xp.power(1.0 + g * g - 2.0 * g * nu, 1.5)
        return k * (1.0 + nu * nu) / xp.maximum(denom, 1e-10)
    
    def _precompute_indirect_irradiance_gpu(self, delta_rayleigh, delta_mie, scattering, order, delta_irradiance):
        """Compute indirect irradiance (GPU vectorized)."""
        xp = self.xp
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        g = self.params.mie_phase_function_g
        
        HEMI_SAMPLES = 16
        dphi = float(xp.pi) / HEMI_SAMPLES
        dtheta = float(xp.pi) / HEMI_SAMPLES
        
        j_coords = xp.arange(IRRADIANCE_TEXTURE_HEIGHT)
        i_coords = xp.arange(IRRADIANCE_TEXTURE_WIDTH)
        jj, ii = xp.meshgrid(j_coords, i_coords, indexing='ij')
        
        v = (jj + 0.5) / IRRADIANCE_TEXTURE_HEIGHT
        u = (ii + 0.5) / IRRADIANCE_TEXTURE_WIDTH
        
        x_r = self._get_unit_range_from_texture_coord(v, IRRADIANCE_TEXTURE_HEIGHT)
        r = bottom + x_r * (top - bottom)
        
        x_mu_s = self._get_unit_range_from_texture_coord(u, IRRADIANCE_TEXTURE_WIDTH)
        mu_s = xp.clip(2.0 * x_mu_s - 1.0, -1.0, 1.0)
        
        sin_mu_s = xp.sqrt(xp.maximum(0.0, 1.0 - mu_s * mu_s))
        
        result = xp.zeros((IRRADIANCE_TEXTURE_HEIGHT, IRRADIANCE_TEXTURE_WIDTH, 3), dtype=xp.float64)
        
        for jj_s in range(HEMI_SAMPLES // 2):
            theta = (jj_s + 0.5) * dtheta
            cos_theta = float(xp.cos(theta))
            sin_theta = float(xp.sin(theta))
            
            for ii_s in range(2 * HEMI_SAMPLES):
                phi = (ii_s + 0.5) * dphi
                cos_phi = float(xp.cos(phi))
                sin_phi = float(xp.sin(phi))
                
                omega_x = cos_phi * sin_theta
                omega_y = sin_phi * sin_theta
                omega_z = cos_theta
                
                domega = dtheta * dphi * sin_theta
                
                nu = sin_mu_s * omega_x + mu_s * omega_z
                
                if order == 1:
                    rayleigh_i = self._sample_scattering_gpu(delta_rayleigh, r, omega_z, mu_s, nu)
                    mie_i = self._sample_scattering_gpu(delta_mie, r, omega_z, mu_s, nu)
                    sky_rad = rayleigh_i * self._rayleigh_phase(nu) + mie_i * self._mie_phase(g, nu)
                else:
                    sky_rad = self._sample_scattering_gpu(scattering, r, omega_z, mu_s, nu)
                
                result += sky_rad * omega_z * domega
        
        delta_irradiance[:] = result.astype(xp.float32)
    
    def _precompute_multiple_scattering_gpu(self, transmittance, delta_density, delta_multiple):
        """Integrate scattering density along rays (GPU vectorized)."""
        xp = self.xp
        bottom = self.params.bottom_radius
        top = self.params.top_radius
        H = xp.sqrt(top * top - bottom * bottom)
        
        SAMPLE_COUNT = 50
        
        for k in range(SCATTERING_TEXTURE_DEPTH):
            if (k + 1) % 8 == 0:
                print(f"  Multiple scattering: {k+1}/{SCATTERING_TEXTURE_DEPTH}")
                sys.stdout.flush()
            
            z = (k + 0.5) / SCATTERING_TEXTURE_DEPTH
            rho = H * self._get_unit_range_from_texture_coord(z, SCATTERING_TEXTURE_R_SIZE)
            r_k = float(xp.sqrt(rho * rho + bottom * bottom))
            
            j_coords = xp.arange(SCATTERING_TEXTURE_HEIGHT)
            i_coords = xp.arange(SCATTERING_TEXTURE_WIDTH)
            jj, ii = xp.meshgrid(j_coords, i_coords, indexing='ij')
            
            # Decode mu
            w = (jj + 0.5) / SCATTERING_TEXTURE_HEIGHT
            d_min_below = r_k - bottom
            d_max_below = rho
            w_unit_below = self._get_unit_range_from_texture_coord(1.0 - 2.0 * w, SCATTERING_TEXTURE_MU_SIZE // 2)
            d_below = d_min_below + (d_max_below - d_min_below) * w_unit_below
            mu_below = xp.where(d_below == 0, -1.0, -(rho * rho + d_below * d_below) / (2.0 * r_k * d_below))
            
            d_min_above = top - r_k
            d_max_above = rho + H
            w_unit_above = self._get_unit_range_from_texture_coord(2.0 * w - 1.0, SCATTERING_TEXTURE_MU_SIZE // 2)
            d_above = d_min_above + (d_max_above - d_min_above) * w_unit_above
            mu_above = xp.where(d_above == 0, 1.0, (H * H - rho * rho - d_above * d_above) / (2.0 * r_k * d_above))
            
            mu = xp.where(w < 0.5, mu_below, mu_above)
            mu = xp.clip(mu, -1.0, 1.0)
            ray_intersects_ground = w < 0.5
            
            # Decode nu
            nu_idx = ii % SCATTERING_TEXTURE_NU_SIZE
            u_nu = (nu_idx + 0.5) / SCATTERING_TEXTURE_NU_SIZE
            nu = self._get_unit_range_from_texture_coord(u_nu, SCATTERING_TEXTURE_NU_SIZE) * 2.0 - 1.0
            
            # Decode mu_s
            mu_s_idx = ii // SCATTERING_TEXTURE_NU_SIZE
            v_s = (mu_s_idx + 0.5) / SCATTERING_TEXTURE_MU_S_SIZE
            x_mu_s = self._get_unit_range_from_texture_coord(v_s, SCATTERING_TEXTURE_MU_S_SIZE)
            d_min_s = top - bottom
            d_max_s = H
            D_s = self._distance_to_top_gpu(bottom, -0.2)
            A = (D_s - d_min_s) / (d_max_s - d_min_s)
            denom = 1.0 + x_mu_s * A
            a_val = xp.where(denom != 0, (A - x_mu_s * A) / denom, 0.0)
            d_s = d_min_s + xp.minimum(a_val, A) * (d_max_s - d_min_s)
            mu_s = xp.where(d_s == 0, 1.0, (H * H - d_s * d_s) / (2.0 * bottom * d_s))
            mu_s = xp.clip(mu_s, -1.0, 1.0)
            
            # Distance to boundary
            disc_ground = r_k * r_k * (mu * mu - 1.0) + bottom * bottom
            d_ground = xp.where(disc_ground >= 0, 
                xp.maximum(0.0, -r_k * mu - xp.sqrt(xp.maximum(0.0, disc_ground))), 1e10)
            
            disc_top = r_k * r_k * (mu * mu - 1.0) + top * top
            d_top = xp.maximum(0.0, -r_k * mu + xp.sqrt(xp.maximum(0.0, disc_top)))
            
            d_max_ray = xp.where(ray_intersects_ground, d_ground, d_top)
            dx = d_max_ray / SAMPLE_COUNT
            
            # Integrate
            result = xp.zeros((SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_WIDTH, 3), dtype=xp.float64)
            
            for s in range(SAMPLE_COUNT + 1):
                d_i = s * dx
                
                r_i = xp.sqrt(d_i * d_i + 2.0 * r_k * mu * d_i + r_k * r_k)
                r_i = xp.clip(r_i, bottom, top)
                mu_i = xp.where(r_i > 0, (r_k * mu + d_i) / r_i, mu)
                mu_i = xp.clip(mu_i, -1.0, 1.0)
                mu_s_i = xp.where(r_i > 0, (r_k * mu_s + d_i * nu) / r_i, mu_s)
                mu_s_i = xp.clip(mu_s_i, -1.0, 1.0)
                
                scatter_i = self._sample_scattering_gpu(delta_density, r_i, mu_i, mu_s_i, nu)
                trans_i = self._get_transmittance_gpu(transmittance, r_k, mu, d_i, ray_intersects_ground)
                
                weight = 0.5 if (s == 0 or s == SAMPLE_COUNT) else 1.0
                result += scatter_i * trans_i * dx[..., None] * weight
            
            delta_multiple[k, :, :, :3] = result.astype(xp.float32)
    
    # Expose same interface as original model
    def get_shader_uniforms(self) -> dict:
        if not self._is_initialized:
            raise RuntimeError("Model not initialized.")
        
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
        """Save precomputed textures to NPZ file."""
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
        """Save precomputed textures as EXR files."""
        # Use original model's save method
        from .model import AtmosphereModel
        temp_model = AtmosphereModel(self.params)
        temp_model.textures = self.textures
        temp_model._is_initialized = True
        temp_model._solar_irradiance_rgb = self._solar_irradiance_rgb
        temp_model.save_textures_exr(output_dir)
