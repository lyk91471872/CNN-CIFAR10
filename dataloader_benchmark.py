import itertools
import numpy as np
import time
from torch.utils.data import DataLoader

from config import NUM_WORKERS, BATCH_SIZE, PREFETCH_FACTOR, DATA_PATHS
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
    start_time = time.time()
    # Iterate through the loader for 3 epochs
    for epoch in range(3):
        for _ in loader:
            pass
    return time.time() - start_time

def run_grid_search(dataset):
    results = []
    for num_workers, batch_size, prefetch_factor in itertools.product(
        NUM_WORKERS_LIST, BATCH_SIZE_LIST, PREFETCH_FACTOR_LIST
    ):
        try:
            t = benchmark_dataloader(dataset, num_workers, batch_size, prefetch_factor)
            results.append((num_workers, batch_size, prefetch_factor, t))
            print(f"Workers: {num_workers}, Batch: {batch_size}, Prefetch: {prefetch_factor} -> Time: {t:.2f}s")
        except Exception as e:
            print(f"Error with workers={num_workers}, batch={batch_size}, prefetch={prefetch_factor}: {e}")
    return results

def main():
    dataset = CIFAR10Dataset(data_paths=DATA_PATHS, transform=None)
    results = run_grid_search(dataset)
    best_config = min(results, key=lambda x: x[3])
    print(f"\nBest Configuration: Workers={best_config[0]}, Batch={best_config[1]}, Prefetch={best_config[2]} -> Time: {best_config[3]:.2f}s")
    np.savetxt("benchmark_results.csv", results, delimiter=",", fmt='%s', header="num_workers,batch_size,prefetch_factor,time_taken", comments="")

if __name__ == "__main__":
    main()
