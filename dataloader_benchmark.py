import itertools
import numpy as np
import time
import torch
from torch.utils.data import DataLoader

from config import TRAIN_DATA_PATHS, DATALOADER
from dataset import CIFAR10Dataset

NUM_WORKERS_LIST = [4, 8, 16, 32, 64]
BATCH_SIZE_LIST = [256, 512, 1024]
N_EPOCHS = 3

def benchmark_dataloader(dataset, num_workers, batch_size):
    loader_kwargs = {
        **DATALOADER,  # Copy all other specs
        'num_workers': num_workers,
        'batch_size': batch_size
    }
    loader = DataLoader(dataset, **loader_kwargs)
    start_time = time.time()
    for epoch in range(N_EPOCHS):
        for _ in loader:
            pass
    return time.time() - start_time

def run_grid_search(dataset):
    results = []
    for num_workers, batch_size in itertools.product(
        NUM_WORKERS_LIST, BATCH_SIZE_LIST
    ):
        try:
            print(f"\nTesting configuration: workers={num_workers}, batch={batch_size}")
            t = benchmark_dataloader(dataset, num_workers, batch_size)
            results.append((num_workers, batch_size, t))
            print(f"Success! Time: {t:.2f}s")
        except Exception as e:
            print(f"Error with workers={num_workers}, batch={batch_size}: {e}")
            results.append((num_workers, batch_size, None))
        time.sleep(0.5)
    return results

def main():
    """Main function to run the benchmark."""
    # Create dataset
    dataset = CIFAR10Dataset(data_paths=TRAIN_DATA_PATHS)
    
    # Create DataLoader with different worker counts
    worker_counts = [0, 2, 4, 8, 16]
    results = []
    
    for num_workers in worker_counts:
        print(f"\nBenchmarking with {num_workers} workers...")
        dataloader = DataLoader(
            dataset,
            num_workers=num_workers,
            **DATALOADER
        )
        
        # Run benchmark
        avg_time = benchmark_dataloader(dataloader)
        results.append((num_workers, avg_time))
        print(f"Average time per batch: {avg_time:.3f} seconds")
    
    # Plot results
    plot_benchmark_results(results)

if __name__ == "__main__":
    main()
