from time import perf_counter

import numpy as np


def reduce_sample_data(df, num_samples=1000, num_features=1000):
    """Takes ``df``, a sparse matrix, and reduces features by std"""
    sample = df[:num_samples]
    std = sample.power(2).mean(axis=0) - np.power(sample.mean(axis=0), 2)
    ret_idx = np.squeeze(np.array(np.argsort(-std)))[:num_features]
    return df[:, ret_idx]


class time_logger():
    """Class made for easy logging with toggleable verbosity"""
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.start_time = perf_counter()

    def log(self, str=''):
        """Print with message ``str`` if verbose.  Otherwise, skip"""
        if not self.verbose:
            return  # Cut timing for optimization
        self.end_time = perf_counter()
        print(f'{str}: {self.end_time - self.start_time}')
        self.start_time = perf_counter()  # Re-time to avoid CLI
