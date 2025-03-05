class EarlyStopping:
    """Early stopping to stop the training when the monitored metric does not improve after certain epochs."""
    def __init__(self, patience: int = 10, verbose: bool = False, delta: float = 0):
        """
        Args:
            patience (int): How many epochs to wait before stopping when metric is not improving.
            verbose (bool): If True, prints a message for each metric improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_metric = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_metric: float) -> None:
        """
        Args:
            val_metric (float): The validation metric to check (e.g., loss or accuracy).
        """
        if self.best_metric is None:
            self.best_metric = val_metric
        elif val_metric > self.best_metric - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_metric = val_metric
            self.counter = 0 