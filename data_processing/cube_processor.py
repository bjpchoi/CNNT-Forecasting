# data_processing/cube_processor.py

import os
import xarray as xr
import numpy as np


class CubeProcessor:
    def __init__(self, data_dir, variables, lat_bounds, lon_bounds, desired_levels_percentiles=[100, 75, 50], halo_size=4):
        """
        Initializes the CubeProcessor.

        Args:
            data_dir (str): Directory where NetCDF files are stored.
            variables (dict): Dictionary of variables with paths and internal names.
            lat_bounds (tuple): Latitude bounds (min, max).
            lon_bounds (tuple): Longitude bounds (min, max).
            desired_levels_percentiles (list): Percentiles to select levels.
            halo_size (int): Size of the halo region around each cube.
        """
        self.data_dir = data_dir
        self.variables = variables
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        self.desired_levels_percentiles = desired_levels_percentiles
        self.halo_size = halo_size
        self.processed_vars = {}

    def load_variable(self, var_key, file_paths):
        """
        Loads and subsets a variable from its NetCDF files.

        Args:
            var_key (str): Variable key.
            file_paths (list): List of file paths for the variable.

        Returns:
            xarray.DataArray: Subsetted data array.
        """
        var_info = self.variables[var_key]
        internal_var = var_info['internal_var']
        try:
            ds = xr.open_mfdataset(file_paths, combine='by_coords', engine='netcdf4')
            ds = ds.sel(lat=slice(*self.lat_bounds), lon=slice(*self.lon_bounds))
            if internal_var not in ds.variables:
                print(f"Variable '{internal_var}' not found in '{var_key}'. Skipping.")
                return None
            data_var = ds[internal_var].astype('float32')
            print(f"Loaded and subsetted variable '{var_key}' successfully.")
            return data_var
        except Exception as e:
            print(f"Error loading variable '{var_key}': {e}")
            return None

    def get_desired_level_indices(self, lev):
        """
        Identifies desired level indices based on percentiles.

        Args:
            lev (xarray.DataArray): Vertical levels.

        Returns:
            list: Desired level indices.
        """
        lev_values = lev.values
        num_levels = len(lev_values)
        half_index = num_levels // 2
        top_half_levels = lev_values[half_index:]
        desired_levels = []
        for p in self.desired_levels_percentiles:
            idx = int(np.floor((p / 100) * len(top_half_levels))) - 1
            idx = max(idx, 0)
            desired_levels.append(half_index + idx)
        desired_levels = sorted(list(set(desired_levels)))
        return desired_levels

    def process_variables(self, downloaded_files):
        """
        Processes all variables and stores them in processed_vars.

        Args:
            downloaded_files (dict): Dictionary of downloaded file paths per variable.
        """
        for var_key, file_paths in downloaded_files.items():
            print(f"Processing variable '{var_key}'...")
            data_var = self.load_variable(var_key, file_paths)
            if data_var is None:
                continue
            if 'lev' in data_var.dims:
                lev = data_var['lev']
                desired_levels = self.get_desired_level_indices(lev)
                data_var = data_var.isel(lev=desired_levels)
                print(f"Selected levels for '{var_key}': {data_var['lev'].values}")
            self.processed_vars[var_key] = data_var
        print("All variables processed successfully.")

    def extract_cubes(self, cube_size=64, stride=1):
        """
        Extracts data cubes with halo regions from the processed variables.

        Args:
            cube_size (int): Size of the data cube (height and width).
            stride (int): Stride for moving the cube extraction window.

        Returns:
            list: List of tuples containing input and target cubes.
        """
        if 'T' not in self.processed_vars:
            raise ValueError("Primary variable 'T' is missing.")

        T = self.processed_vars['T']
        lev = T['lev']
        lat = T['lat']
        lon = T['lon']
        n_lat = len(lat)
        n_lon = len(lon)
        num_levels = len(lev)

        # Define cube starting positions ensuring halos are within bounds
        lat_starts = [
            i for i in range(0, n_lat - cube_size + 1, stride)
            if (i - self.halo_size >= 0 and i + cube_size + self.halo_size <= n_lat)
        ]
        lon_starts = [
            i for i in range(0, n_lon - cube_size + 1, stride)
            if (i - self.halo_size >= 0 and i + cube_size + self.halo_size <= n_lon)
        ]
        starting_positions = [(lat, lon) for lat in lat_starts for lon in lon_starts]
        print(f"Total starting positions: {len(starting_positions)}")

        samples = []
        for lat_start, lon_start in starting_positions:
            input_samples = []
            for var_key in self.variables.keys():
                data_var = self.processed_vars[var_key]
                if 'lev' in data_var.dims:
                    # 4D variable: time, lev, lat, lon
                    if var_key == 'T':
                        time_index = 2  # t2
                    else:
                        time_index = 0  # t0
                    cube = data_var.isel(
                        time=time_index,
                        lev=slice(None),
                        lat=slice(lat_start - self.halo_size, lat_start + cube_size + self.halo_size),
                        lon=slice(lon_start - self.halo_size, lon_start + cube_size + self.halo_size)
                    ).values  # Shape: (lev, lat, lon)
                else:
                    # 3D variable: time, lat, lon
                    if var_key == 'T':
                        time_index = 2
                    else:
                        time_index = 0
                    cube = data_var.isel(
                        time=time_index,
                        lat=slice(lat_start - self.halo_size, lat_start + cube_size + self.halo_size),
                        lon=slice(lon_start - self.halo_size, lon_start + cube_size + self.halo_size)
                    ).values  # Shape: (lat, lon)
                    cube = cube[np.newaxis, :, :]  # Add lev dimension
                input_samples.append(cube)
            # Stack input samples along the channel dimension
            input_cube = np.concatenate(input_samples, axis=0)  # Shape: (channels, lev, lat, lon)
            # Remove halo to get the target region
            input_cube = input_cube[:, :, self.halo_size:self.halo_size + cube_size, self.halo_size:self.halo_size + cube_size]
            input_tensor = input_cube.astype(np.float32)

            # Prepare target: T at t3 minus T at t2 in the target region
            cube_T_t2 = self.processed_vars['T'].isel(
                time=2,
                lev=slice(None),
                lat=slice(lat_start, lat_start + cube_size),
                lon=slice(lon_start, lon_start + cube_size)
            ).values  # Shape: (lev, cube_size, cube_size)
            cube_T_t3 = self.processed_vars['T'].isel(
                time=3,
                lev=slice(None),
                lat=slice(lat_start, lat_start + cube_size),
                lon=slice(lon_start, lon_start + cube_size)
            ).values  # Shape: (lev, cube_size, cube_size)
            target_cube = cube_T_t3 - cube_T_t2  # Difference
            target_tensor = target_cube.astype(np.float32)

            samples.append((input_tensor, target_tensor))
        print(f"Total samples extracted: {len(samples)}")
        return samples
