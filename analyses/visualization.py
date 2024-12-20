import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding, TSNE

class DataExplorer:
    """
    A class containing sample methods for exploring and visualizing data.
    
    Attributes:
        data (dict): Dictionary of xarray.DataArray objects.
        lat (np.ndarray): Array of latitude values.
        lon (np.ndarray): Array of longitude values.
    """
    
    def __init__(self, data, lat, lon):
        """
        Initializes the DataExplorer.
        
        Args:
            data (dict): Dictionary containing variables as xarray.DataArray.
            lat (np.ndarray): Latitude values.
            lon (np.ndarray): Longitude values.
        """
        self.data = data
        self.lat = lat
        self.lon = lon
    
    def plot_global_vectors(self, scalar_var, vector_vars, time_index=0, level_index=0, skip=10, title=None):
        """
        Plots vector fields over a scalar field on a global map.
        
        Args:
            scalar_var (str): Key for the scalar variable (e.g., 'T').
            vector_vars (list): List of keys for vector variables (e.g., ['U', 'V']).
            time_index (int, optional): Time index for the plot. Defaults to 0.
            level_index (int, optional): Vertical level index. Defaults to 0.
            skip (int, optional): Step size for quiver plotting. Defaults to 10.
            title (str, optional): Title of the plot.
        """
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines()
        
        scalar_data = self.data[scalar_var].isel(time=time_index, lev=level_index)
        contour = ax.contourf(self.lon, self.lat, scalar_data, 40, transform=ccrs.PlateCarree(), cmap='coolwarm')
        plt.colorbar(contour, orientation='vertical', label=f'{scalar_var} Units')
        
        U = self.data[vector_vars[0]].isel(time=time_index, lev=level_index).values
        V = self.data[vector_vars[1]].isel(time=time_index, lev=level_index).values
        lon_sub = self.lon.values[::skip]
        lat_sub = self.lat.values[::skip]
        UU = U[::skip, ::skip]
        VV = V[::skip, ::skip]
        
        ax.quiver(lon_sub, lat_sub, UU, VV, transform=ccrs.PlateCarree(), color='black', scale=700)
        plt.title(title or f"Vector Fields over {scalar_var} (Level {level_index}, Time {time_index})")
        plt.show()
    
    def plot_vertical_cross_section(self, var, fixed_coord, fixed_value, coord='lat', time_index=0, title=None):
        """
        Plots a vertical cross-section of a variable at a fixed latitude or longitude.
        
        Args:
            var (str): Key for the variable to plot.
            fixed_coord (str): Coordinate to fix ('lat' or 'lon').
            fixed_value (float): Value at which to fix the coordinate.
            coord (str, optional): 'lat' or 'lon'. Defaults to 'lat'.
            time_index (int, optional): Time index. Defaults to 0.
            title (str, optional): Title of the plot.
        """
        data = self.data[var].isel(time=time_index)
        if coord == 'lat':
            idx = np.abs(self.lat - fixed_value).argmin()
            cross_data = data.isel(lat=idx).values
            x, xlabel = self.lon.values, 'Longitude (°E)'
        elif coord == 'lon':
            idx = np.abs(self.lon - fixed_value).argmin()
            cross_data = data.isel(lon=idx).values
            x, xlabel = self.lat.values, 'Latitude (°N)'
        else:
            raise ValueError("coord must be 'lat' or 'lon'")
        
        y = data['lev'].values
        fig, ax = plt.subplots(figsize=(12, 5))
        cs = ax.contourf(x, y, cross_data, 40, cmap='coolwarm')
        plt.colorbar(cs, ax=ax, label=f'{var} Units')
        ax.set_title(title or f"Vertical Cross-Section of {var} at {coord.upper()} = {fixed_value}°")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Model Level")
        ax.invert_yaxis()
        plt.show()
    
    def plot_histogram(self, var, time_index=0, level_index=0, bins=50, title=None):
        """
        Plots a histogram of a variable's distribution.
        
        Args:
            var (str): Key for the variable to plot.
            time_index (int, optional): Time index. Defaults to 0.
            level_index (int, optional): Vertical level index. Defaults to 0.
            bins (int, optional): Number of histogram bins. Defaults to 50.
            title (str, optional): Title of the histogram.
        """
        data = self.data[var].isel(time=time_index, lev=level_index).values.flatten()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(data, bins=bins, color='orange', edgecolor='black')
        ax.set_title(title or f"Distribution of {var} (Level {level_index}, Time {time_index})")
        ax.set_xlabel(f'{var} Units')
        ax.set_ylabel("Frequency")
        plt.show()
    
    def perform_pca(self, var, level_index=0, n_components=3, time_indices=None, title=None):
        """
        Performs PCA on spatial data and plots the principal components.
        
        Args:
            var (str): Key for the variable to analyze.
            level_index (int, optional): Vertical level index. Defaults to 0.
            n_components (int, optional): Number of principal components. Defaults to 3.
            time_indices (list, optional): List of time indices to include. Defaults to all.
            title (str, optional): Title for the principal component plots.
        """
        data = self.data[var].isel(lev=level_index)
        data = data.isel(time=time_indices).values if time_indices else data.values
        nt, nlat, nlon = data.shape
        data_reshaped = data.reshape(nt, nlat * nlon)
        
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(data_reshaped)
        components = pca.components_
        
        for i, pc in enumerate(components[:n_components]):
            pc_map = pc.reshape(nlat, nlon)
            fig = plt.figure(figsize=(12, 6))
            ax = plt.axes(projection=ccrs.Robinson())
            ax.set_global()
            ax.coastlines()
            cs = ax.contourf(self.lon, self.lat, pc_map, 40, transform=ccrs.PlateCarree(), cmap='bwr')
            plt.colorbar(cs, orientation='vertical', label=f'PC{i+1} Loading')
            ax.set_title(title or f"Principal Component {i+1} of {var} (Level {level_index})")
            plt.show()
    
    def perform_spectral_embedding(self, var, level_index=0, n_components=2, time_indices=None, title=None):
        """
        Performs Spectral Embedding (Laplacian Eigenmaps) on spatial data and plots the results.
        
        Args:
            var (str): Key for the variable to analyze.
            level_index (int, optional): Vertical level index. Defaults to 0.
            n_components (int, optional): Number of embedding dimensions. Defaults to 2.
            time_indices (list, optional): List of time indices to include. Defaults to all.
            title (str, optional): Title for the embedding plot.
        """
        data = self.data[var].isel(lev=level_index)
        data = data.isel(time=time_indices).values if time_indices else data.values
        nt, nlat, nlon = data.shape
        data_reshaped = data.reshape(nt, nlat * nlon)
        
        embedding = SpectralEmbedding(n_components=n_components, random_state=42)
        embedded = embedding.fit_transform(data_reshaped)
        
        if n_components == 2:
            plt.figure(figsize=(8,6))
            plt.scatter(embedded[:,0], embedded[:,1], cmap='viridis', alpha=0.7)
            plt.title(title or f"Spectral Embedding of {var} (Level {level_index})")
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.grid(True)
            plt.show()
        else:
            print("Spectral Embedding visualization is available only for 2 components.")
    
    def perform_tsne(self, var, level_index=0, n_components=2, perplexity=30, time_indices=None, title=None):
        """
        Performs t-SNE on spatial data and plots the results.
        
        Args:
            var (str): Key for the variable to analyze.
            level_index (int, optional): Vertical level index. Defaults to 0.
            n_components (int, optional): Number of embedding dimensions. Defaults to 2.
            perplexity (float, optional): t-SNE perplexity parameter. Defaults to 30.
            time_indices (list, optional): List of time indices to include. Defaults to all.
            title (str, optional): Title for the t-SNE plot.
        """
        data = self.data[var].isel(lev=level_index)
        data = data.isel(time=time_indices).values if time_indices else data.values
        nt, nlat, nlon = data.shape
        data_reshaped = data.reshape(nt, nlat * nlon)
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        embedded = tsne.fit_transform(data_reshaped)
        
        if n_components == 2:
            plt.figure(figsize=(8,6))
            plt.scatter(embedded[:,0], embedded[:,1], cmap='viridis', alpha=0.7)
            plt.title(title or f"t-SNE of {var} (Level {level_index})")
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.grid(True)
            plt.show()
        else:
            print("t-SNE visualization is available only for 2 components.")
