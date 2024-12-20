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

@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

class DataLoader:
    def __init__(self, data_dir="weather_data", chunks=None, verbose=True):
        """
        Initializes the DataLoader. Flexible starter implementation.

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
        Loads a specific variable from multiple NetCDF file; returns xarray.DataArray or xarray.Dataset.
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
        loaded_data = {}
        for variable, file_paths in downloaded_files.items():
            data = self.load_variable(variable, file_paths, lat_bounds, lon_bounds, time_slice)
            if data is not None:
                loaded_data[variable] = data
        return loaded_data

    def compute(self, data):
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            with ProgressBar() if self.verbose else nullcontext():
                return data.compute()
        else:
            print("Data is not an xarray object. Skipping computation.")
            return data


class DataDownloader:
    def __init__(self, base_url, data_dir="weather_data", verbose=True):
        """
        Initializes the DataDownloader; flexible starter implementation.
        
        """
        self.base_url = base_url
        self.data_dir = data_dir
        self.verbose = verbose
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def download_file(self, url, filename):
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

    def download_variables(self, variables, file_list):
        downloaded_files = {}
        for var_key, var_info in variables.items():
            var_files = []
            var_files_identifiers = file_list.get(var_key, [])
            if not var_files_identifiers:
                if self.verbose:
                    print(f"No file identifiers specified for variable '{var_key}'. Skipping.")
                continue
            for identifier in var_files_identifiers:
                file_name = var_info.get('filename', identifier)
                url = urljoin(self.base_url, var_info['path'] + '/' + file_name)
                self.download_file(url, file_name)
                var_files.append(os.path.join(self.data_dir, file_name))
            downloaded_files[var_key] = var_files
        return downloaded_files
