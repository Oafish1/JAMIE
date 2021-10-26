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
    def __init__(
        self,
        discard_first_sample=True,
        record=True,
        verbose=False,
    ):
        self.discard_first_sample = discard_first_sample
        self.record = record
        self.verbose = verbose

        self.history = {}
        self.start_time = perf_counter()

    def log(self, str=''):
        """Print with message ``str`` if verbose.  Otherwise, skip"""
        if not (self.verbose or self.record):
            return  # Cut timing for optimization
        self.end_time = perf_counter()

        # Perform any auxiliary operations here
        time_elapsed = self.end_time - self.start_time
        if self.record:
            if str not in self.history:
                self.history[str] = []
            self.history[str].append(time_elapsed)
        if self.verbose:
            print(f'{str}: {time_elapsed}')

        # Re-time to avoid extra runtime cost
        self.start_time = perf_counter()

    def aggregate(self):
        """Print mean times for all keys in ``self.history``"""
        running_total = 0
        for k, v in self.history.items():
            avg_time_elapsed = np.array(v)
            if self.discard_first_sample:
                avg_time_elapsed = avg_time_elapsed[1:]
            avg_time_elapsed = np.mean(np.array(v))

            running_total += avg_time_elapsed
            print(f'{k}: {avg_time_elapsed}')
        print(f'Total: {running_total}')
