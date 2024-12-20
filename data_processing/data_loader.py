# data_processing.py

import os
import warnings
from urllib.parse import urljoin
import requests
from tqdm import tqdm
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
from contextlib import contextmanager

class DataLoader:
    def __init__(self, data_dir="weather_data", chunks=None, verbose=True):
        """
        Initializes the DataLoader.

        Args:
            data_dir (str): Directory where downloaded data is stored.
            chunks (dict, optional): Chunk sizes for dask. If None, uses xarray's default.
            verbose (bool): If True, displays progress bars during computation.
        """
        self.data_dir = data_dir
        self.chunks = chunks
        self.verbose = verbose

    def load_variable(self, variable, file_paths, lat_bounds=None, lon_bounds=None, time_slice=None):
        """
        Loads a specific variable from multiple NetCDF files with optional spatial and temporal slicing.

        Args:
            variable (str): Variable name to load.
            file_paths (list): List of file paths for the variable.
            lat_bounds (tuple, optional): Latitude bounds (min, max).
            lon_bounds (tuple, optional): Longitude bounds (min, max).
            time_slice (slice or tuple, optional): Time slice (start, end).

        Returns:
            xarray.DataArray or xarray.Dataset: Loaded data.
        """
        datasets = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}. Skipping.")
                continue
            try:
                ds = xr.open_dataset(file_path, chunks=self.chunks)
                if variable not in ds:
                    print(f"Variable '{variable}' not found in {file_path}. Skipping.")
                    ds.close()
                    continue
                da = ds[variable]
                # Apply spatial bounds
                if lat_bounds:
                    da = da.sel(lat=slice(lat_bounds[0], lat_bounds[1]))
                if lon_bounds:
                    da = da.sel(lon=slice(lon_bounds[0], lon_bounds[1]))
                # Apply time slicing
                if time_slice:
                    da = da.sel(time=time_slice)
                datasets.append(da)
                ds.close()
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        if not datasets:
            print(f"No data loaded for variable '{variable}'.")
            return None
        combined = xr.concat(datasets, dim='time')
        if self.verbose:
            print(f"Loaded variable '{variable}' with shape {combined.shape}.")
        return combined

    def load_all_variables(self, downloaded_files, lat_bounds=None, lon_bounds=None, time_slice=None):
        """
        Loads all specified variables from the downloaded files.

        Args:
            downloaded_files (dict): Dictionary of downloaded file paths per variable.
            lat_bounds (tuple, optional): Latitude bounds (min, max).
            lon_bounds (tuple, optional): Longitude bounds (min, max).
            time_slice (slice or tuple, optional): Time slice (start, end).

        Returns:
            dict: Dictionary of xarray DataArrays/Datasets per variable.
        """
        loaded_data = {}
        for variable, file_paths in downloaded_files.items():
            data = self.load_variable(variable, file_paths, lat_bounds, lon_bounds, time_slice)
            if data is not None:
                loaded_data[variable] = data
        return loaded_data

    def compute(self, data):
        """
        Computes the dask graph.

        Args:
            data (xarray.DataArray or xarray.Dataset): Data to compute.

        Returns:
            xarray.DataArray or xarray.Dataset: Computed data.
        """
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            with ProgressBar() if self.verbose else nullcontext():
                return data.compute()
        else:
            print("Data is not an xarray object. Skipping computation.")
            return data


class DataDownloader:
    def __init__(self, base_url, data_dir="weather_data", verbose=True):
        """
        Initializes the DataDownloader.

        Args:
            base_url (str): Base URL for data download.
            data_dir (str): Directory to store downloaded data.
            verbose (bool): If True, prints download status.
        """
        self.base_url = base_url
        self.data_dir = data_dir
        self.verbose = verbose
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def download_file(self, url, filename):
        """
        Downloads a single file from the specified URL.

        Args:
            url (str): URL of the file to download.
            filename (str): Local filename to save the downloaded file.
        """
        local_path = os.path.join(self.data_dir, filename)
        if os.path.exists(local_path):
            if self.verbose:
                print(f"{local_path} already exists. Skipping download.")
            return
        if self.verbose:
            print(f"Downloading {filename}...")
        try:
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                with open(local_path, 'wb') as f, tqdm(
                    total=total_size, unit='B', unit_scale=True, disable=not self.verbose
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            if self.verbose:
                print(f"Downloaded {filename} successfully.")
        except requests.HTTPError as e:
            print(f"HTTP Error: {e} for URL: {url}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

    def download_variables(self, variables, times, lat_bounds=None, lon_bounds=None):
        """
        Downloads the specified variables for given times and spatial bounds.

        Args:
            variables (dict): Dictionary of variables with their paths and internal names.
            times (dict): Dictionary containing 'primary' and 'supplementary' times.
            lat_bounds (tuple, optional): Latitude bounds (min, max).
            lon_bounds (tuple, optional): Longitude bounds (min, max).

        Returns:
            dict: Dictionary of downloaded file paths per variable.
        """
        downloaded_files = {}
        for var_key, var_info in variables.items():
            var_files = []
            var_times = times.get(var_key, [])
            if not var_times:
                if self.verbose:
                    print(f"No times specified for variable '{var_key}'. Skipping.")
                continue
            for time in var_times:
                if len(time) != 2:
                    print(f"Invalid time format for variable '{var_key}': {time}")
                    continue
                date_str, time_str = time
                if len(date_str) != 8 or not date_str.isdigit():
                    print(f"Invalid date string '{date_str}' for variable '{var_key}'. Skipping.")
                    continue
                if len(time_str) not in [1, 2, 3, 4]:
                    print(f"Invalid time string '{time_str}' for variable '{var_key}'. Skipping.")
                    continue
                year = date_str[:4]
                month = date_str[4:6]
                day = date_str[6:8]
                base_filename = var_info.get('filename', var_info['path'].split('/')[-1])
                file_name = f"c1440_NR.{base_filename}.{date_str}_{time_str.zfill(2)}z.nc4"
                url_path = f"{var_info['path']}/Y{year}/M{month}/D{day}/{file_name}"
                url = urljoin(self.base_url, url_path)
                self.download_file(url, file_name)
                var_files.append(os.path.join(self.data_dir, file_name))
            downloaded_files[var_key] = var_files
        return downloaded_files
