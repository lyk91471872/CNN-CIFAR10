import itertools
import numpy as np
import time
import torch
import os
import resource
from torch.utils.data import DataLoader

from config import DATA_PATHS, PREFETCH_FACTOR
from dataset import CIFAR10Dataset

NUM_WORKERS_LIST = [4, 8, 16, 32, 64]
BATCH_SIZE_LIST = [256, 512, 1024, 2048]

def get_file_descriptor_limit():
    """Get the current file descriptor limit."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    return soft

def cleanup_loader(loader):
    """Clean up DataLoader resources."""
    try:
        if hasattr(loader, '_iterator'):
            del loader._iterator
        if hasattr(loader, '_workers'):
            for w in loader._workers:
                if w.is_alive():
                    w.terminate()
            loader._workers = []
        torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

def benchmark_dataloader(dataset, num_workers, batch_size):
    '''Benchmark the DataLoader: time to iterate over the dataset for 3 epochs.'''
    loader = None
    try:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=PREFETCH_FACTOR
        )
        
        start_time = time.time()
        # Iterate through the loader for 3 epochs
        for epoch in range(3):
            for _ in loader:
                pass
        return time.time() - start_time
    finally:
        if loader is not None:
            cleanup_loader(loader)
            time.sleep(1)  # Give system time to clean up resources

def run_grid_search(dataset):
    # Check system limits
    fd_limit = get_file_descriptor_limit()
    print(f"\nSystem file descriptor limit: {fd_limit}")
    
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
            # Add failed result with None time
            results.append((num_workers, batch_size, None))
        time.sleep(2)  # Add delay between tests
    return results

def main():
    print("Starting dataloader benchmark...")
    dataset = CIFAR10Dataset(data_paths=DATA_PATHS, transform=None)
    results = run_grid_search(dataset)
    
    # Filter out failed results and find best configuration
    successful_results = [r for r in results if r[2] is not None]
    if successful_results:
        best_config = min(successful_results, key=lambda x: x[2])
        print(f"\nBest Configuration: Workers={best_config[0]}, Batch={best_config[1]} -> Time: {best_config[2]:.2f}s")
    else:
        print("\nNo successful configurations found!")
    
    # Save results, including failed attempts
    np.savetxt("benchmark_results.csv", results, delimiter=",", fmt='%s', 
               header="num_workers,batch_size,time_taken", 
               comments="")
    print("\nResults saved to benchmark_results.csv")

if __name__ == "__main__":
    main()
