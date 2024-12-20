# analyses/global_test.py

import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tqdm import tqdm
import logging

class GlobalTester:
    """
    Sample script for performing global evaluation of forecasting model 
    across geographic regions.
    
    Attributes:
        model (torch.nn.Module): The trained PyTorch model to evaluate.
        processed_vars (dict): Dictionary of processed variables loaded from datasets.
        lat (np.ndarray): Array of latitude values.
        lon (np.ndarray): Array of longitude values.
        cube_size (int): Size of the data cube.
        halo_size (int): Size of the halo around the data cube.
        device (torch.device): Device to perform computations on.
        criterion (torch.nn.Module): Loss function for evaluation.
        num_past_steps (int): Number of past time steps to use for prediction.
    """
    
    def __init__(self, model, processed_vars, lat, lon, cube_size=20, halo_size=4, 
                 device=None, criterion=None, num_past_steps=1):
        """
        Initializes the GlobalTester with necessary components.
        
        Args:
            model (torch.nn.Module): The trained model.
            processed_vars (dict): Processed dataset variables.
            lat (np.ndarray): Latitude array.
            lon (np.ndarray): Longitude array.
            cube_size (int, optional): Size of the data cube. Defaults to 20.
            halo_size (int, optional): Size of the halo. Defaults to 4.
            device (torch.device, optional): Computation device. Defaults to CUDA if available.
            criterion (torch.nn.Module, optional): Loss function. Defaults to MSELoss.
            num_past_steps (int, optional): Number of past time steps to use for prediction. Defaults to 1.
        """
        self.model = model
        self.processed_vars = processed_vars
        self.lat = lat
        self.lon = lon
        self.cube_size = cube_size
        self.halo_size = halo_size
        self.num_past_steps = num_past_steps
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion if criterion else torch.nn.MSELoss()
        self.model.to(self.device)
        self.model.eval()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def sample_uniform_sphere(self, num_points, seed=None):
        """
        Samples points uniformly on the surface of a sphere.
        
        Args:
            num_points (int): Number of points to sample.
            seed (int, optional): Random seed for reproducibility.
        
        Returns:
            tuple: Arrays of latitudes and longitudes.
        """
        if seed is not None:
            np.random.seed(seed)
        
        z = np.random.uniform(-1, 1, num_points)
        theta = np.random.uniform(0, 2*np.pi, num_points)
        phi = np.arccos(z)
        lats = 90.0 - (phi * 180.0 / np.pi)
        lons = (theta * 180.0 / np.pi) - 180.0
        return lats, lons
    
    def find_nearest_index(self, arr, val):
        """
        Finds the nearest index in an array for a given value.
        
        Args:
            arr (np.ndarray): Array to search.
            val (float): Value to find.
        
        Returns:
            int: Index of the nearest value.
        """
        idx = np.abs(arr - val).argmin()
        return idx
    
    def get_data_cubes(self, time_indices, lat_start, lon_start):
        """
        Extracts input data cubes and target cube for a specific location.
        
        Args:
            time_indices (dict): Dictionary mapping variable keys to time indices.
            lat_start (int): Starting latitude index.
            lon_start (int): Starting longitude index.
        
        Returns:
            tuple: Input samples and target cube.
        """
        input_samples = {}
        for var_key, var_info in self.processed_vars.items():
            data_var = var_info['data']
            if 'lev' in data_var.dims:
                lat_indices = slice(lat_start - self.halo_size, lat_start + self.cube_size + self.halo_size)
                lon_indices = slice(lon_start - self.halo_size, lon_start + self.cube_size + self.halo_size)
                cube = data_var.isel(time=time_indices[var_key], lev=slice(None),
                                     lat=lat_indices, lon=lon_indices).values
            else:
                lat_indices = slice(lat_start - self.halo_size, lat_start + self.cube_size + self.halo_size)
                lon_indices = slice(lon_start - self.halo_size, lon_start + self.cube_size + self.halo_size)
                cube = data_var.isel(time=time_indices[var_key],
                                     lat=lat_indices, lon=lon_indices).values
                cube = cube[np.newaxis, :, :]
            input_samples[var_key] = cube
        
        # Target = difference between n+1 and n
        cube_T_tn = self.processed_vars['T']['data'].isel(time=self.num_past_steps, lev=slice(None),
                                                          lat=slice(lat_start, lat_start + self.cube_size),
                                                          lon=slice(lon_start, lon_start + self.cube_size)).values
        cube_T_tnp1 = self.processed_vars['T']['data'].isel(time=self.num_past_steps + 1, lev=slice(None),
                                                             lat=slice(lat_start, lat_start + self.cube_size),
                                                             lon=slice(lon_start, lon_start + self.cube_size)).values
        target_cube = cube_T_tnp1 - cube_T_tn
        return input_samples, target_cube
    
    def evaluate_point(self, glat, glon):
        """
        Evaluates the model at a single geographic point.
        
        Args:
            glat (float): Latitude of the point.
            glon (float): Longitude of the point.
        
        Returns:
            float: MSE loss for the evaluated point.
        """
        lat_idx = self.find_nearest_index(self.lat.values, glat)
        lon_idx = self.find_nearest_index(self.lon.values, glon)
        
        # Boundary check
        if (lat_idx - self.halo_size < 0 or 
            lat_idx + self.cube_size + self.halo_size > len(self.lat) or
            lon_idx - self.halo_size < 0 or 
            lon_idx + self.cube_size + self.halo_size > len(self.lon)):
            logging.warning(f"Skipping point ({glat}, {glon}) due to boundary constraints.")
            return None
        
        time_indices = {}
        for var_key in self.processed_vars:
            if var_key == 'T':
                # Using the first n past steps for prediction
                # Assuming time=0 is the earliest, adjust as needed based on your data
                time_indices[var_key] = np.arange(self.num_past_steps)
            else:
                time_indices[var_key] = 0  # Adjust if other variables have different time handling
        
        try:
            input_sample_dict, target_cube = self.get_data_cubes(time_indices, lat_idx, lon_idx)
            input_cubes = []
            for var_key in self.processed_vars:
                cube = input_sample_dict[var_key]
                input_cubes.append(cube)
            input_sample = np.concatenate(input_cubes, axis=0)
            input_sample = torch.tensor(input_sample, dtype=torch.float32, device=self.device).unsqueeze(0)
            target_sample = torch.tensor(target_cube, dtype=torch.float32, device=self.device).view(1, -1)
            
            with torch.no_grad():
                output = self.model(input_sample)
            loss = self.criterion(output, target_sample).item()
            return loss
        except Exception as e:
            logging.error(f"Error evaluating point ({glat}, {glon}): {e}")
            return None
    
    def evaluate_globally(self, num_samples=500, seed=42):
        """
        Performs global evaluation by sampling points uniformly on the sphere.
        
        Args:
            num_samples (int, optional): Number of global points to sample. Defaults to 500.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        
        Returns:
            tuple: Arrays of global points and their corresponding errors.
        """
        lats, lons = self.sample_uniform_sphere(num_samples, seed=seed)
        global_errors = []
        global_points = []
        
        logging.info("Starting global evaluation...")
        for glat, glon in tqdm(zip(lats, lons), total=num_samples, desc="Global Evaluation"):
            loss = self.evaluate_point(glat, glon)
            if loss is not None:
                global_points.append((glat, glon))
                global_errors.append(loss)
        
        self.global_points = np.array(global_points)
        self.global_errors = np.array(global_errors)
        logging.info("Global evaluation completed.")
        return self.global_points, self.global_errors
    
    def plot_global_errors(self):
        """
        Plots the global MSE errors on a world map.
        """
        if not hasattr(self, 'global_points') or not hasattr(self, 'global_errors'):
            logging.error("No global evaluation data to plot. Please run evaluate_globally() first.")
            return
        
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines()
        sc = ax.scatter(self.global_points[:,1], self.global_points[:,0], 
                        c=self.global_errors, cmap='coolwarm',
                        s=50, transform=ccrs.PlateCarree(), edgecolor='black')
        plt.colorbar(sc, ax=ax, orientation='vertical', label='MSE Error (K²)')
        plt.title(f"Model Performance Across the Globe (Uniform Sphere Sampling, n={self.num_past_steps})")
        plt.show()
