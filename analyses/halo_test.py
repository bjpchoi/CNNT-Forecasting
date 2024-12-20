import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class HaloTester:
    """
    A sample class to test different halo sizes.
    
    Attributes:
        halo_sizes (list): List of halo sizes to evaluate.
        prepare_data (callable): Function to prepare data for a given halo size.
        evaluate_model (callable): Function to evaluate the model and return a performance metric.
        results (dict): Dictionary to store performance metrics for each halo size.
    """
    
    def __init__(self, halo_sizes, prepare_data, evaluate_model):
        """
        Initializes the HaloTester.
        
        Args:
            halo_sizes (list): List of halo sizes to test.
            prepare_data (callable): Function that takes halo_size and returns prepared data.
            evaluate_model (callable): Function that takes prepared data and returns a metric.
        """
        self.halo_sizes = halo_sizes
        self.prepare_data = prepare_data
        self.evaluate_model = evaluate_model
        self.results = {}
    
    def run_tests(self):
        """Iterates over halo sizes, evaluates the model, and stores the results."""
        for halo in tqdm(self.halo_sizes, desc="Testing Halo Sizes"):
            data = self.prepare_data(halo)
            metric = self.evaluate_model(data)
            self.results[halo] = metric
    
    def plot_results(self, xlabel='Halo Size', ylabel='Performance Metric', title='Halo Size vs Performance'):
        """Plots the performance metric against halo sizes."""
        halos = sorted(self.results.keys())
        metrics = [self.results[halo] for halo in halos]
        
        plt.figure(figsize=(8, 5))
        plt.plot(halos, metrics, marker='o', linestyle='-')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()
    
    def get_optimal_halo_size(self, minimize=True):
        if not self.results:
            raise ValueError("No results to determine optimal halo size. Run tests first.")
        
        halos, metrics = zip(*self.results.items())
        optimal_halo = halos[np.argmin(metrics)] if minimize else halos[np.argmax(metrics)]
        return optimal_halo
