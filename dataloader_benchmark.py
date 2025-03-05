import itertools
import numpy as np
import time
import torch
from torch.utils.data import DataLoader

from config import TRAIN_DATA_PATHS, DATALOADER_KWARGS
from dataset import CIFAR10Dataset

NUM_WORKERS_LIST = [4, 8, 16, 32, 64]
BATCH_SIZE_LIST = [256, 512, 1024]
N_EPOCHS = 3

def benchmark_dataloader(dataset, num_workers, batch_size):
    loader_kwargs = {
        **DATALOADER_KWARGS,  # Copy all other specs
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
    print("Starting dataloader benchmark...")
    dataset = CIFAR10Dataset(data_paths=TRAIN_DATA_PATHS, transform=None)
    results = run_grid_search(dataset)
    successful_results = [r for r in results if r[2] is not None]
    if successful_results:
        best_config = min(successful_results, key=lambda x: x[2])
        print(f"\nBest Configuration: Workers={best_config[0]}, Batch={best_config[1]} -> Time: {best_config[2]:.2f}s")
    else:
        print("\nNo successful configurations found!")
    
    np.savetxt("benchmark_results.csv", results, delimiter=",", fmt='%s', 
               header="num_workers,batch_size,time_taken", 
               comments="")
    print("\nResults saved to benchmark_results.csv")

if __name__ == "__main__":
    main()
