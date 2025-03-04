import itertools
import numpy as np
import time
import torch
from torch.utils.data import DataLoader

from config import DATA_PATHS
from dataset import CIFAR10Dataset

NUM_WORKERS_LIST = [4, 8, 16, 32, 48, 64]
BATCH_SIZE_LIST = [256, 512, 1024, 2048]
PREFETCH_FACTOR_LIST = [2, 4, 8, 16, 32]

def benchmark_dataloader(dataset, num_workers, batch_size, prefetch_factor):
    '''Benchmark the DataLoader: time to iterate over the dataset for 3 epochs.'''
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    
    try:
        start_time = time.time()
        # Iterate through the loader for 3 epochs
        for epoch in range(3):
            for _ in loader:
                pass
        return time.time() - start_time
    finally:
        # Clean up resources
        if hasattr(loader, '_iterator'):
            del loader._iterator
        if hasattr(loader, '_workers'):
            for w in loader._workers:
                if w.is_alive():
                    w.terminate()
            loader._workers = []
        torch.cuda.empty_cache()  # Clear GPU memory if using CUDA

def run_grid_search(dataset):
    results = []
    for num_workers, batch_size, prefetch_factor in itertools.product(
        NUM_WORKERS_LIST, BATCH_SIZE_LIST, PREFETCH_FACTOR_LIST
    ):
        try:
            print(f"\nTesting configuration: workers={num_workers}, batch={batch_size}, prefetch={prefetch_factor}")
            t = benchmark_dataloader(dataset, num_workers, batch_size, prefetch_factor)
            results.append((num_workers, batch_size, prefetch_factor, t))
            print(f"Success! Time: {t:.2f}s")
        except Exception as e:
            print(f"Error with workers={num_workers}, batch={batch_size}, prefetch={prefetch_factor}: {e}")
            # Add failed result with None time
            results.append((num_workers, batch_size, prefetch_factor, None))
    return results

def main():
    print("Starting dataloader benchmark...")
    dataset = CIFAR10Dataset(data_paths=DATA_PATHS, transform=None)
    results = run_grid_search(dataset)
    
    # Filter out failed results and find best configuration
    successful_results = [r for r in results if r[3] is not None]
    if successful_results:
        best_config = min(successful_results, key=lambda x: x[3])
        print(f"\nBest Configuration: Workers={best_config[0]}, Batch={best_config[1]}, Prefetch={best_config[2]} -> Time: {best_config[3]:.2f}s")
    else:
        print("\nNo successful configurations found!")
    
    # Save results, including failed attempts
    np.savetxt("benchmark_results.csv", results, delimiter=",", fmt='%s', 
               header="num_workers,batch_size,prefetch_factor,time_taken", 
               comments="")
    print("\nResults saved to benchmark_results.csv")

if __name__ == "__main__":
    main()
