import numpy as np

class Container:
    def __init__(self, samples, centre=None, *, copy=True):
        if isinstance(samples, np.ndarray):
            if samples.ndim != 1:
                raise ValueError("Multidimensional ndarray input is not supported. Pass a list of arrays to specify the bin direction.")
            samples = list(samples)
        self.samples = [np.array(x, copy=True) if copy else np.asarray(x) for x in samples]

        if not self.samples:
            raise ValueError("Container cannot be empty.")

        shape = self.samples[0].shape
        dtype = self.samples[0].dtype

        if any(x.shape != shape for x in self.samples):
            raise ValueError("All samples must have the same shape.")
        if any(x.dtype != dtype for x in self.samples):
            raise ValueError("All samples must have the same dtype.")

        self.centre = (np.array(centre, copy=True) if copy else np.asarray(centre)) if centre is not None else None
        if self.centre is not None:
            if self.centre.shape != shape:
                raise ValueError("Centre must have the same shape as samples.")

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    @property
    def shape(self):
        return self.samples[0].shape

    @property
    def size(self):
        return len(self.samples)
    
    @property
    def dtype(self):
        return self.samples[0].dtype

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(size={self.size}, shape={self.shape}, "
            f"has_centre={self.centre is not None})"
        )

    def eval_mean(self):
        return np.mean(self.samples, axis=0)