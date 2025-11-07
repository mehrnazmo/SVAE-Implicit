"""
Niño 3.4 and ONI Index Calculator for SST Time Series Data

This module provides functions to calculate Niño 3.4 and Oceanic Niño Index (ONI) 
from SST time series data stored as numpy arrays with shape (batch_size, time, height, width).

The Niño 3.4 region is defined as 5°N-5°S, 170°W-120°W (or 190°E-240°E in 0-360° longitude).
"""

import numpy as np
from typing import Tuple, Optional, Union


def create_coordinate_grid(lat_range: Tuple[float, float], 
                         lon_range: Tuple[float, float], 
                         h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create latitude and longitude coordinate arrays for the given grid.
    
    Args:
        lat_range: Tuple of (min_lat, max_lat) in degrees
        lon_range: Tuple of (min_lon, max_lon) in degrees  
        h: Number of latitude points (height)
        w: Number of longitude points (width)
        
    Returns:
        Tuple of (latitude_array, longitude_array) with shapes (h,) and (w,)
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    
    # Create coordinate arrays (centered on grid cells)
    latitudes = np.linspace(lat_max - 0.5, lat_min + 0.5, h)
    longitudes = np.linspace(lon_min + 0.5, lon_max - 0.5, w)
    
    return latitudes, longitudes


def find_nino34_indices(latitudes: np.ndarray, longitudes: np.ndarray) -> Tuple[slice, slice]:
    """
    Find the grid indices corresponding to the Niño 3.4 region.
    
    Niño 3.4 region: 5°N-5°S, 170°W-120°W (or 190°E-240°E in 0-360° longitude)
    
    Args:
        latitudes: Array of latitude values
        longitudes: Array of longitude values
        
    Returns:
        Tuple of (lat_slice, lon_slice) for indexing the Niño 3.4 region
    """
    # Niño 3.4 region boundaries
    lat_min, lat_max = -5.0, 5.0
    lon_min, lon_max = 190.0, 240.0
    
    # Find indices
    lat_mask = (latitudes >= lat_min) & (latitudes <= lat_max)
    lon_mask = (longitudes >= lon_min) & (longitudes <= lon_max)
    
    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]
    
    if len(lat_indices) == 0 or len(lon_indices) == 0:
        raise ValueError(f"Niño 3.4 region not found in the provided coordinates. "
                        f"Latitude range: {latitudes.min():.1f} to {latitudes.max():.1f}, "
                        f"Longitude range: {longitudes.min():.1f} to {longitudes.max():.1f}")
    
    lat_slice = slice(lat_indices[0], lat_indices[-1] + 1)
    lon_slice = slice(lon_indices[0], lon_indices[-1] + 1)
    
    return lat_slice, lon_slice


def compute_cosine_latitude_weights(latitudes: np.ndarray) -> np.ndarray:
    """
    Compute cosine latitude weights for area-weighted averaging.
    
    Args:
        latitudes: Array of latitude values in degrees
        
    Returns:
        Array of cosine latitude weights
    """
    # print('cos latitudes: ', latitudes)
    return np.cos(np.deg2rad(latitudes))



def compute_area_weighted_mean(sst_data: np.ndarray, 
                             nino34_lats) -> np.ndarray:
    """
    Compute area-weighted mean over the Niño 3.4 region.
    
    Args:
        sst_data: SST data array with shape (..., lat, lon)
        nino34_lats: Latitude coordinate array for Niño 3.4 region
        
    Returns:
        Area-weighted mean SST values
    """
    cos_weights = compute_cosine_latitude_weights(nino34_lats)
    # print('cos_weights: ', cos_weights.shape)
    
    # Broadcast weights to match SST data shape
    # For sst_data shape (..., lat, lon), we need weights shape (..., lat, lon)
    # where weights are constant along longitude dimension
    weight_shape = [1] * (sst_data.ndim - 2) + list(cos_weights.shape) + [sst_data.shape[-1]]
    weights = cos_weights.reshape([1] * (sst_data.ndim - 2) + list(cos_weights.shape) + [1])
    weights = np.broadcast_to(weights, sst_data.shape)
    # print('cos_weights: ', weights.shape)
    # print('cos weights: ', weights)
    
    # Compute weighted mean
    weighted_sum = np.sum(sst_data * weights, axis=(-2, -1))
    weight_sum = np.sum(weights, axis=(-2, -1))
    # print('weights_sum: ', weight_sum)
    
    return weighted_sum / weight_sum


def compute_monthly_climatology(sst_data: np.ndarray, 
                               months: np.ndarray,
                               baseline_years: Tuple[int, int] = (1991, 2020)) -> np.ndarray:
    """
    Compute monthly climatology from SST data.
    
    Args:
        sst_data: SST data array with shape (time, lat, lon)
        months: Array of month values (1-12) for each time step
        baseline_years: Tuple of (start_year, end_year) for climatology baseline
        
    Returns:
        Monthly climatology array with shape (12, lat, lon)
    """
    climatology = np.zeros((12, sst_data.shape[1], sst_data.shape[2]))
    
    for month in range(1, 13):
        month_mask = months == month
        if np.any(month_mask):
            climatology[month-1] = np.nanmean(sst_data[month_mask], axis=0)
    
    return climatology


def compute_nino34_and_oni(sst_array: np.ndarray,
                          lat_range: Tuple[float, float],
                          lon_range: Tuple[float, float], 
                          start_month: int,
                          time_stamps: Optional[np.ndarray] = None,
                          climatology: Optional[np.ndarray] = None,
                          baseline_years: Tuple[int, int] = (1991, 2020)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Niño 3.4 and ONI indices from SST time series data.
    
    Args:
        sst_array: SST data array with shape (batch_size, time, height, width)
        lat_range: Tuple of (min_lat, max_lat) in degrees
        lon_range: Tuple of (min_lon, max_lon) in degrees
        time_stamps: Optional array of timestamps (if None, assumes monthly data)
        climatology: Optional pre-computed climatology array with shape (12, lat, lon)
        baseline_years: Tuple of (start_year, end_year) for climatology baseline
        
    Returns:
        Tuple of (nino34_indices, oni_indices) with shapes (batch_size, time)
        
    Example:
        >>> sst_data = np.random.rand(5, 36, 26, 90)  # 5 batches, 36 months, 26x90 grid
        >>> lat_range = (-50, 70)
        >>> lon_range = (0, 360)
        >>> nino34, oni = compute_nino34_and_oni(sst_data, lat_range, lon_range)
        >>> print(f"Niño 3.4 shape: {nino34.shape}")  # (5, 36)
        >>> print(f"ONI shape: {oni.shape}")  # (5, 36)
    """
    bs, T, h, w = sst_array.shape
    
    # Create coordinate grid
    #(90, -90), (0,360)
    latitudes, longitudes = create_coordinate_grid(lat_range, lon_range, h, w)
    
    # Find Niño 3.4 region indices
    #slice(np.int64(85), np.int64(95), None), slice(np.int64(190), np.int64(240), None)
    lat_slice, lon_slice = find_nino34_indices(latitudes, longitudes)

    # Handle case where climatology is already spatially averaged (shape: (12,))
    if hasattr(climatology, 'values'):
        climatology = np.array(climatology.values)
        
    nino34_lats = latitudes[lat_slice]
    nino34_lons = longitudes[lon_slice]
    sst_box = sst_array[..., lat_slice, lon_slice]
    # print('='*100)
    # print('calculator: ')
    # print('sst_box: ', sst_box.shape)
    # print('sst_box: ', sst_box)
    
    # Initialize output arrays
    nino34_indices = np.zeros((bs, T))
    oni_indices = np.zeros((bs, T))
    
    # Process each batch
    for b in range(bs):
        batch_sst = sst_box[b]  # Shape: (T, h, w)
        
        # Compute monthly anomalies
        anomalies = np.zeros_like(batch_sst)
        for t in range(T):
            month_idx = (t + start_month - 1) % 12
            # print('batch_sst[t]: ', batch_sst.shape)
            # print('climatology[month_idx]: ', climatology.shape)
            anomalies[t] = batch_sst[t] - climatology[month_idx]
        # print('anomalies: ', anomalies)
        
        # Compute Niño 3.4 index (area-weighted mean of anomalies)
        nino34_values = compute_area_weighted_mean(anomalies, nino34_lats)
        nino34_indices[b] = nino34_values
        
        # Compute ONI (3-month running mean)
        oni_values = np.zeros_like(nino34_values)
        for t in range(T):
            if t == 0:
                # First month: average of months 0, 1
                oni_values[t] = np.mean(nino34_values[:2])
            elif t == T - 1:
                # Last month: average of months T-2, T-1
                oni_values[t] = np.mean(nino34_values[-2:])
            else:
                # Middle months: 3-month centered average
                oni_values[t] = np.mean(nino34_values[t-1:t+2])
        
        oni_indices[b] = oni_values
    # print('='*100)
    
    return nino34_indices, oni_indices


def compute_nino34_and_oni_single_batch(sst_array: np.ndarray,
                                       lat_range: Tuple[float, float],
                                       lon_range: Tuple[float, float],
                                       time_stamps: Optional[np.ndarray] = None,
                                       climatology: Optional[np.ndarray] = None,
                                       baseline_years: Tuple[int, int] = (1991, 2020)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Niño 3.4 and ONI indices for a single batch of SST data.
    
    Args:
        sst_array: SST data array with shape (time, height, width)
        lat_range: Tuple of (min_lat, max_lat) in degrees
        lon_range: Tuple of (min_lon, max_lon) in degrees
        time_stamps: Optional array of timestamps
        climatology: Optional pre-computed climatology array
        baseline_years: Tuple of (start_year, end_year) for climatology baseline
        
    Returns:
        Tuple of (nino34_index, oni_index) with shapes (time,)
    """
    # Add batch dimension
    sst_with_batch = sst_array[np.newaxis, ...]  # Shape: (1, T, h, w)
    
    # Compute indices
    nino34, oni = compute_nino34_and_oni(sst_with_batch, lat_range, lon_range, 
                                       time_stamps, climatology, baseline_years)
    
    # Remove batch dimension
    return nino34[0], oni[0]


def create_example_data(bs: int = 5, T: int = 36, h: int = 26, w: int = 90) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Create example SST data for testing.
    
    Args:
        bs: Batch size
        T: Number of time steps
        h: Height (latitude points)
        w: Width (longitude points)
        
    Returns:
        Tuple of (sst_data, lat_range, lon_range)
    """
    # Create realistic SST data with some spatial structure
    sst_data = np.random.randn(bs, T, h, w) * 2 + 20  # Mean SST around 20°C
    
    # Add some spatial structure (warmer in tropics)
    lat_indices = np.arange(h)
    lon_indices = np.arange(w)
    
    for b in range(bs):
        for t in range(T):
            # Add latitude-dependent temperature
            lat_effect = 10 * np.cos(np.pi * lat_indices / h)
            sst_data[b, t] += lat_effect[:, np.newaxis]
            
            # Add some random noise
            sst_data[b, t] += np.random.randn(h, w) * 0.5
    
    lat_range = (-50, 70)
    lon_range = (0, 360)
    
    return sst_data, lat_range, lon_range


if __name__ == "__main__":
    # Example usage
    print("Creating example SST data...")
    sst_data, lat_range, lon_range = create_example_data(bs=3, T=24, h=26, w=90)
    print(f"SST data shape: {sst_data.shape}")
    print(f"Latitude range: {lat_range}")
    print(f"Longitude range: {lon_range}")
    
    print("\nComputing Niño 3.4 and ONI indices...")
    nino34, oni = compute_nino34_and_oni(sst_data, lat_range, lon_range)
    
    print(f"Niño 3.4 indices shape: {nino34.shape}")
    print(f"ONI indices shape: {oni.shape}")
    print(f"Niño 3.4 range: {nino34.min():.3f} to {nino34.max():.3f}°C")
    print(f"ONI range: {oni.min():.3f} to {oni.max():.3f}°C")
    
    # Test single batch function
    print("\nTesting single batch function...")
    single_batch_sst = sst_data[0]  # Shape: (T, h, w)
    nino34_single, oni_single = compute_nino34_and_oni_single_batch(single_batch_sst, lat_range, lon_range)
    
    print(f"Single batch Niño 3.4 shape: {nino34_single.shape}")
    print(f"Single batch ONI shape: {oni_single.shape}")
    print(f"Values match first batch: {np.allclose(nino34_single, nino34[0])}")
