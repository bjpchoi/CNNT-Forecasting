# analyses/global_test.py

import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tqdm import tqdm
import logging
from typing import Callable, Dict, List, Tuple, Optional

class GlobalTester:
    """
    A flexible class for performing global evaluation of forecasting models
    across geographic regions using sequences of data cubes. Starter code that can be 
    modified as needed.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        processed_vars: Dict[str, Dict],
        lat: np.ndarray,
        lon: np.ndarray,
        cube_size: int = 20,
        halo_size: int = 4,
        device: Optional[torch.device] = None,
        criterion: Optional[torch.nn.Module] = None,
        num_past_steps: int = 1,
        preprocess_fn: Optional[Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]] = None
    ):
        """
        Initializes the GlobalTester with necessary components.
        
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
        self.preprocess_fn = preprocess_fn
        self.model.to(self.device)
        self.model.eval()

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def sample_uniform_sphere(self, num_points: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples points uniformly on the surface of a sphere.

        Args:
            num_points (int): Number of points to sample.
            seed (int, optional): Random seed for reproducibility.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of latitudes and longitudes.
        """
        if seed is not None:
            np.random.seed(seed)

        z = np.random.uniform(-1, 1, num_points)
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        phi = np.arccos(z)
        lats = 90.0 - (phi * 180.0 / np.pi)
        lons = (theta * 180.0 / np.pi) - 180.0
        return lats, lons

    def find_nearest_index(self, arr: np.ndarray, val: float) -> int:
        idx = np.abs(arr - val).argmin()
        return idx

    def get_data_cubes(
        self,
        point_idx: Tuple[int, int],
        sequence_idx: List[int]
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:

        lat_start, lon_start = point_idx
        input_samples = {}
        for var_key, var_info in self.processed_vars.items():
            data_var = var_info['data']
            dims = data_var.dims  # Assuming 'dims' attribute similar to xarray or similar library
            # Identify spatial dimensions dynamically (excluding 'time' and potential other non-spatial dims)
            spatial_dims = [dim for dim in dims if dim not in ['time']]
            # Determine slice objects based on spatial dimensions
            slices = {}
            for dim in spatial_dims:
                if dim == 'lat':
                    slices[dim] = slice(lat_start - self.halo_size, lat_start + self.cube_size + self.halo_size)
                elif dim == 'lon':
                    slices[dim] = slice(lon_start - self.halo_size, lon_start + self.cube_size + self.halo_size)
                else:
                    # Handle other spatial dimensions if any
                    slices[dim] = slice(None)
            # Extract cubes for each time index in the sequence
            cubes = [
                data_var.isel(time=idx, **slices).values
                for idx in sequence_idx
            ]
            input_samples[var_key] = np.stack(cubes, axis=0)  # Shape: (N, ...)
        
        # Extract target cube (N+1)
        target_var = self.processed_vars.get('T', None)
        if not target_var:
            raise ValueError("Target variable 'T' not found in processed_vars.")
        
        data_var = target_var['data']
        target_dims = data_var.dims
        target_spatial_dims = [dim for dim in target_dims if dim not in ['time']]
        target_slices = {}
        for dim in target_spatial_dims:
            if dim == 'lat':
                target_slices[dim] = slice(lat_start, lat_start + self.cube_size)
            elif dim == 'lon':
                target_slices[dim] = slice(lon_start, lon_start + self.cube_size)
            else:
                target_slices[dim] = slice(None)
        if 'lev' in data_var.dims:
            target_cube = data_var.isel(time=sequence_idx[-1] + 1, **target_slices).values
        else:
            target_cube = data_var.isel(time=sequence_idx[-1] + 1, **target_slices).values
            # Expand dimensions if necessary
            if target_cube.ndim == 2:
                target_cube = target_cube[np.newaxis, ...]  # Shape: (1, lat, lon)
        return input_samples, target_cube

    def preprocess_input(self, input_samples: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Applies external preprocessing to input samples if a preprocessing function is provided.

        """
        if self.preprocess_fn:
            try:
                input_samples = self.preprocess_fn(input_samples)
                logging.debug("External preprocessing applied successfully.")
            except Exception as e:
                logging.error(f"External preprocessing failed: {e}")
                raise e
        return input_samples

    def prepare_input_tensor(self, input_samples: Dict[str, np.ndarray]) -> torch.Tensor:
        input_cubes = []
        for var_key in sorted(input_samples.keys()):
            cube = input_samples[var_key]
            input_cubes.append(cube)
        input_sample = np.concatenate(input_cubes, axis=1)
        input_tensor = torch.tensor(input_sample, dtype=torch.float32, device=self.device).unsqueeze(0)
        return input_tensor

    def evaluate_point(self, glat: float, glon: float) -> Optional[float]:
        lat_idx = self.find_nearest_index(self.lat, glat)
        lon_idx = self.find_nearest_index(self.lon, glon)

        if (lat_idx - self.halo_size < 0 or 
            lat_idx + self.cube_size + self.halo_size > len(self.lat) or
            lon_idx - self.halo_size < 0 or 
            lon_idx + self.cube_size + self.halo_size > len(self.lon)):
            logging.warning(f"Skipping point ({glat}, {glon}) due to boundary constraints.")
            return None

        try:
            # Modify per data structure as needed
            max_time = min([var_info['data'].shape[0] for var_info in self.processed_vars.values()])
            if max_time < self.num_past_steps + 1:
                logging.error("Not enough time steps in the data for evaluation.")
                return None
            sequence_idx = list(range(max_time - self.num_past_steps, max_time))
            
            input_sample_dict, target_cube = self.get_data_cubes((lat_idx, lon_idx), sequence_idx)
            input_sample_dict = self.preprocess_input(input_sample_dict)
            input_tensor = self.prepare_input_tensor(input_sample_dict)
            target_sample = torch.tensor(target_cube, dtype=torch.float32, device=self.device).view(1, -1)
            
            with torch.no_grad():
                output = self.model(input_tensor)
            loss = self.criterion(output, target_sample).item()
            return loss
        except Exception as e:
            logging.error(f"Error evaluating point ({glat}, {glon}): {e}")
            return None

    def evaluate_globally(self, num_samples: int = 500, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs global evaluation by sampling points uniformly on the sphere.
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
        Plots global errors on a world map.
        """
        if not hasattr(self, 'global_points') or not hasattr(self, 'global_errors'):
            logging.error("No global evaluation data to plot. Please run evaluate_globally() first.")
            return

        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines()
        sc = ax.scatter(
            self.global_points[:, 1], self.global_points[:, 0],
            c=self.global_errors, cmap='coolwarm',
            s=50, transform=ccrs.PlateCarree(), edgecolor='black'
        )
        plt.colorbar(sc, ax=ax, orientation='vertical', label='MSE Error')
        plt.title(f"Model Performance Across the Globe (Uniform Sphere Sampling, N={self.num_past_steps})")
        plt.show()
