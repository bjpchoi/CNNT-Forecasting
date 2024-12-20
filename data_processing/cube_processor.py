# data_processing/cube_processor.py

import os
import xarray as xr
import numpy as np


class CubeProcessor:
    def __init__(self, data_dir, variables, lat_bounds, lon_bounds, desired_levels=None, halo_size=4, lazy_load=False):
        """
        Initializes starter code for Cube Processor with optional arguments enabling adaptation.
        
        """
        self.data_dir = data_dir
        self.variables = variables
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        self.desired_levels = desired_levels
        self.halo_size = halo_size
        self.lazy_load = lazy_load
        self.processed_vars = {}

    def load_variable(self, var_key, file_paths):
        var_info = self.variables[var_key]
        internal_var = var_info['internal_var']
        try:
            if self.lazy_load:
                ds = xr.open_mfdataset(file_paths, combine='by_coords', engine='netcdf4', chunks={'time': 1})
            else:
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

    def select_levels(self, data_var):
        """
        Selects desired vertical levels from the data variable.

        Args:
            data_var (xarray.DataArray): Data array with vertical levels.

        Returns:
            xarray.DataArray: Data array with selected levels.
        """
        if self.desired_levels is not None and 'lev' in data_var.dims:
            available_levels = data_var['lev'].values
            selected_levels = [level for level in self.desired_levels if level in available_levels]
            if not selected_levels:
                print("No matching levels found. Using all available levels.")
                return data_var
            data_var = data_var.sel(lev=selected_levels)
            print(f"Selected levels: {selected_levels}")
        return data_var

    def process_variables(self, downloaded_files):
        """
        Processes all variables and stores them in processed_vars.

        """
        for var_key, file_paths in downloaded_files.items():
            print(f"Processing variable '{var_key}'...")
            data_var = self.load_variable(var_key, file_paths)
            if data_var is None:
                continue
            data_var = self.select_levels(data_var)
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
        lat = T['lat']
        lon = T['lon']
        n_lat = len(lat)
        n_lon = len(lon)
        n_time = len(T['time'])

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
        for time_index in range(n_time - 1):
            for lat_start, lon_start in starting_positions:
                input_samples = []
                for var_key in self.variables.keys():
                    data_var = self.processed_vars[var_key]
                    if 'lev' in data_var.dims:
                        cube = data_var.isel(
                            time=time_index,
                            lev=slice(None),
                            lat=slice(lat_start - self.halo_size, lat_start + cube_size + self.halo_size),
                            lon=slice(lon_start - self.halo_size, lon_start + cube_size + self.halo_size)
                        ).values
                    else:
                        cube = data_var.isel(
                            time=time_index,
                            lat=slice(lat_start - self.halo_size, lat_start + cube_size + self.halo_size),
                            lon=slice(lon_start - self.halo_size, lon_start + cube_size + self.halo_size)
                        ).values
                        cube = cube[np.newaxis, :, :]
                    input_samples.append(cube)

                input_cube = np.concatenate(input_samples, axis=0)
                input_cube = input_cube[:, :, self.halo_size:self.halo_size + cube_size, self.halo_size:self.halo_size + cube_size]
                input_tensor = input_cube.astype(np.float32)

                cube_T_n = self.processed_vars['T'].isel(
                    time=time_index,
                    lat=slice(lat_start, lat_start + cube_size),
                    lon=slice(lon_start, lon_start + cube_size)
                ).values
                cube_T_np1 = self.processed_vars['T'].isel(
                    time=time_index + 1,
                    lat=slice(lat_start, lat_start + cube_size),
                    lon=slice(lon_start, lon_start + cube_size)
                ).values
                target_cube = cube_T_np1 - cube_T_n
                target_tensor = target_cube.astype(np.float32)

                samples.append((input_tensor, target_tensor))
        print(f"Total samples extracted: {len(samples)}")
        return samples
